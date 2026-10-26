// True Middlebury flow pairs, sequence-family isolation and input-pixel units:
// a port of tflow's common/ml/flow_data.h (flownets, pwcnet). pairs.npz is made
// by tflow's `just optical-flow-assets` (32 fit / 8 validation / 24 test 64x64
// crops from disjoint scene families).
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <datamunge/nn/nn.hpp>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "common/checks.hpp"

namespace flow {

namespace nn = datamunge::nn;

struct FlowData {
  nn::Tensor first, second, flow, valid, scene_ids, family_ids;
};
struct FlowSplits {
  FlowData fit, validation, test;
};

inline FlowSplits LoadFlowPairs(const std::filesystem::path& root) {
  std::map<std::string, nn::Tensor> arrays;
  for (auto& [name, value] : nn::load_npz((root / "pairs.npz").string())) arrays[name] = value;
  std::set<int> prior_families;
  auto read = [&](const std::string& split, std::int64_t size) {
    FlowData r{arrays.at(split + "_first"), arrays.at(split + "_second"),    arrays.at(split + "_flow"),
               arrays.at(split + "_valid"), arrays.at(split + "_scene_ids"), arrays.at(split + "_family_ids")};
    check::Require(r.first.dtype() == nn::DType::Float32 && r.second.dtype() == nn::DType::Float32 && r.flow.dtype() == nn::DType::Float32 &&
                       r.valid.dtype() == nn::DType::Float32,
                   "float Middlebury RGB/flow/masks");
    check::Require(r.first.shape() == nn::Shape({size, 64, 64, 3}) && r.second.shape() == r.first.shape() &&
                       r.flow.shape() == nn::Shape({size, 64, 64, 2}) && r.valid.shape() == nn::Shape({size, 64, 64, 1}),
                   "original paired image/flow patch alignment");
    check::Require(r.scene_ids.dtype() == nn::DType::Int32 && r.family_ids.dtype() == nn::DType::Int32 &&
                       r.scene_ids.shape() == nn::Shape({size}) && r.family_ids.shape() == nn::Shape({size}),
                   "flow scene/family metadata");
    const auto    families_v = r.family_ids.to_vector<int>();
    const auto    valid      = r.valid.to_vector<float>();
    const auto    flowv      = r.flow.to_vector<float>();
    std::set<int> families;
    for (std::int64_t n = 0; n < size; ++n) {
      const int family = families_v[std::size_t(n)];
      check::Require(prior_families.count(family) == 0, "no related scene family crosses train/validation/test");
      families.insert(family);
      int valid_count = 0;
      for (int i = 0; i < 4096; ++i) {
        const float v = valid[std::size_t(n * 4096 + i)];
        check::Require(v == 0 || v == 1, "binary flow validity");
        valid_count += int(v);
        for (int c = 0; c < 2; ++c) {
          const float value = flowv[std::size_t((n * 4096 + i) * 2 + c)];
          check::Require(std::isfinite(value) && (v != 0 || value == 0), "finite true flow and zero invalid placeholders");
        }
      }
      check::Require(valid_count > 409, "sufficient published GT in every predetermined crop");
    }
    prior_families.insert(families.begin(), families.end());
    for (const auto& tensor : {r.first, r.second})
      for (float v : tensor.to_vector<float>()) check::Require(std::isfinite(v) && v >= 0 && v <= 1, "real observed RGB values");
    return r;
  };
  auto fit = read("fit", 32), validation = read("validation", 8), test = read("test", 24);
  return {fit, validation, test};
}

inline nn::Tensor FlowBatch(const nn::Tensor& input, const std::vector<std::size_t>& rows) {
  std::vector<std::int64_t> ids;
  for (auto r : rows) {
    check::Require(std::int64_t(r) < input.size(0), "paired flow row bounds");
    ids.push_back(std::int64_t(r));
  }
  return nn::take_rows(input, ids).contiguous();
}
inline FlowData FlowBatch(const FlowData& d, const std::vector<std::size_t>& rows) {
  return {FlowBatch(d.first, rows), FlowBatch(d.second, rows),    FlowBatch(d.flow, rows),
          FlowBatch(d.valid, rows), FlowBatch(d.scene_ids, rows), FlowBatch(d.family_ids, rows)};
}

struct FlowSupervision {
  std::array<nn::Tensor, 5> flow, valid;
};

/// Levels 2..6: mean of the valid GT vectors in each 2^level block, / 20 (input units).
inline FlowSupervision PrepareFlowSupervision(const FlowData& data) {
  FlowSupervision    result;
  const std::int64_t batch = data.flow.size(0);
  const auto         valid = data.valid.contiguous().to_vector<float>(), flowv = data.flow.contiguous().to_vector<float>();
  for (int level = 2; level <= 6; ++level) {
    const int          factor = 1 << level, side = 64 / factor, index = level - 2;
    std::vector<float> f(std::size_t(batch * side * side * 2)), v(std::size_t(batch * side * side));
    for (std::int64_t n = 0; n < batch; ++n)
      for (int y = 0; y < side; ++y)
        for (int x = 0; x < side; ++x) {
          double sum[2] = {};
          int    count  = 0;
          for (int dy = 0; dy < factor; ++dy)
            for (int dx = 0; dx < factor; ++dx) {
              const std::size_t p = std::size_t((n * 64 + y * factor + dy) * 64 + x * factor + dx);
              if (valid[p] > 0) {
                ++count;
                for (int c = 0; c < 2; ++c) sum[c] += flowv[p * 2 + std::size_t(c)];
              }
            }
          const std::size_t o = std::size_t((n * side + y) * side + x);
          v[o]                = count > 0;
          for (int c = 0; c < 2; ++c) f[o * 2 + std::size_t(c)] = count ? float(sum[c] / count / 20) : 0.f;
        }
    result.flow[std::size_t(index)]  = nn::from_vector(f, {batch, side, side, 2});
    result.valid[std::size_t(index)] = nn::from_vector(v, {batch, side, side, 1});
  }
  return result;
}

struct FlowPrepared {
  FlowData        data;
  FlowSupervision target;
};
inline FlowPrepared FlowBatch(const FlowPrepared& input, const std::vector<std::size_t>& rows) {
  FlowPrepared output;
  output.data = FlowBatch(input.data, rows);
  for (std::size_t i = 0; i < 5; ++i) {
    output.target.flow[i]  = FlowBatch(input.target.flow[i], rows);
    output.target.valid[i] = FlowBatch(input.target.valid[i], rows);
  }
  return output;
}

struct FlowMetrics {
  double       epe = 0, aae = 0, bad1 = 0, bad3 = 0, fl_all = 0;
  int          images = 0;
  std::int64_t pixels = 0;
};

/// Per-image EPE, angular error (augmented vectors), bad1/bad3 and KITTI Fl-all, averaged over images.
inline FlowMetrics ScoreFlow(const nn::Tensor& prediction, const nn::Tensor& truth, const nn::Tensor& valid) {
  check::Require(prediction.shape() == truth.shape() && truth.dim() == 4 && truth.size(3) == 2 && valid.numel() == truth.numel() / 2,
                 "aligned dx/dy flow score");
  FlowMetrics        score;
  const std::int64_t pixels = truth.size(1) * truth.size(2);
  const auto pr = prediction.contiguous().to_vector<float>(), tr = truth.contiguous().to_vector<float>(), va = valid.contiguous().to_vector<float>();
  for (std::int64_t n = 0; n < truth.size(0); ++n) {
    double epe = 0, angle = 0, bad1 = 0, bad3 = 0, fl = 0;
    int    count = 0;
    for (std::int64_t i = 0; i < pixels; ++i) {
      const std::size_t j = std::size_t(n * pixels + i);
      const double      u = tr[2 * j], v = tr[2 * j + 1];
      if (va[j] <= .5 || !std::isfinite(u) || !std::isfinite(v) || std::abs(u) >= 1e9 || std::abs(v) >= 1e9) continue;
      const double pu = pr[2 * j], pv = pr[2 * j + 1];
      check::Require(std::isfinite(pu) && std::isfinite(pv), "finite estimated optical flow");
      const double error = std::hypot(pu - u, pv - v), cosine = (pu * u + pv * v + 1) / std::sqrt((pu * pu + pv * pv + 1) * (u * u + v * v + 1));
      epe += error;
      angle += std::acos(std::clamp(cosine, -1., 1.)) * 180 / 3.14159265358979323846;
      bad1 += error > 1;
      bad3 += error > 3;
      fl += error > 3 && error > .05 * std::hypot(u, v);
      ++count;
    }
    if (!count) continue;
    ++score.images;
    score.pixels += count;
    score.epe += epe / count, score.aae += angle / count, score.bad1 += bad1 / count, score.bad3 += bad3 / count, score.fl_all += fl / count;
  }
  check::Require(score.images > 0, "flow score needs true valid pixels");
  for (double* value : {&score.epe, &score.aae, &score.bad1, &score.bad3, &score.fl_all}) *value /= score.images;
  return score;
}

inline std::array<double, 2> FitMeanFlow(const FlowData& data) {
  std::array<double, 2> mean{};
  std::int64_t          count = 0;
  const auto            va = data.valid.contiguous().to_vector<float>(), fl = data.flow.contiguous().to_vector<float>();
  for (std::size_t i = 0; i < va.size(); ++i)
    if (va[i] > 0) {
      ++count;
      for (int c = 0; c < 2; ++c) mean[std::size_t(c)] += fl[2 * i + std::size_t(c)];
    }
  check::Require(count > 0, "fit-only mean flow needs observations");
  for (auto& value : mean) value /= double(count);
  return mean;
}

inline nn::Tensor ConstantFlow(const FlowData& data, const std::array<double, 2>& values) {
  std::vector<float> out(std::size_t(data.flow.numel()));
  for (std::size_t i = 0; i < out.size(); ++i) out[i] = float(values[i % 2]);
  return nn::from_vector(out, data.flow.shape());
}

}  // namespace flow
