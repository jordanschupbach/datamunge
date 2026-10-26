// Real paired NYU raw-depth samples, scene-disjoint split checks and metric
// evaluation: a port of tflow's common/ml/depth_data.h. pairs.npz is read with
// nn::load_npz (float32 RGB/depth/valid, int32 room ids and source rows).
#pragma once

#include <algorithm>
#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "common/checks.hpp"

namespace depth {

namespace nn = datamunge::nn;

struct DepthData {
  nn::Tensor rgb, depth, valid, room_ids, source_rows;
};
struct DepthSplits {
  DepthData fit, validation, test;
};

inline DepthSplits LoadDepthPairs(const std::filesystem::path& root) {
  std::map<std::string, nn::Tensor> arrays;
  for (const auto& [name, value] : nn::load_npz((root / "pairs.npz").string())) arrays[name] = value;
  std::set<int> rooms, source_rows;
  auto          read = [&](const std::string& name, std::int64_t count) {
    DepthData result{arrays.at(name + "_rgb"), arrays.at(name + "_depth"), arrays.at(name + "_valid"), arrays.at(name + "_room_ids"),
                     arrays.at(name + "_source_rows")};
    check::Require(result.rgb.dtype() == nn::DType::Float32 && result.depth.dtype() == nn::DType::Float32 &&
                       result.valid.dtype() == nn::DType::Float32 && result.rgb.shape() == nn::Shape({count, 48, 64, 3}) &&
                       result.depth.shape() == nn::Shape({count, 48, 64, 1}) && result.valid.shape() == result.depth.shape(),
                   "NYU RGB/metric depth/validity shapes");
    check::Require(result.room_ids.dtype() == nn::DType::Int32 && result.source_rows.dtype() == nn::DType::Int32 &&
                       result.room_ids.shape() == nn::Shape({count}) && result.source_rows.shape() == nn::Shape({count}),
                   "NYU numeric source/room identities");
    const auto ids = result.room_ids.to_vector<std::int32_t>(), src = result.source_rows.to_vector<std::int32_t>();
    const auto valid = result.valid.to_vector<float>(), d = result.depth.to_vector<float>();
    for (std::int64_t n = 0; n < count; ++n) {
      check::Require(rooms.insert(ids[std::size_t(n)]).second && source_rows.insert(src[std::size_t(n)]).second,
                     "all frames from different canonical rooms across splits");
      int valid_count = 0;
      for (int i = 0; i < 3072; ++i) {
        const std::size_t j    = std::size_t(n * 3072 + i);
        const float       mask = valid[j], depth = d[j];
        check::Require(mask == 0 || mask == 1, "binary sensor validity mask");
        check::Require(mask ? (std::isfinite(depth) && depth > .1f && depth <= 10) : depth == 0, "metric sensor depth with explicit invalid zeros");
        valid_count += int(mask);
      }
      check::Require(valid_count > 307, "at least10% true depth observations in every room");
    }
    for (float v : result.rgb.to_vector<float>()) check::Require(std::isfinite(v) && v >= 0 && v <= 1, "real RGB in[0,1]");
    return result;
  };
  auto fit = read("fit", 32), validation = read("validation", 8), test = read("test", 16);
  return {fit, validation, test};
}

/// Rows of any-dtype tensor (dtype preserved).
inline nn::Tensor DepthBatch(const nn::Tensor& tensor, const std::vector<std::size_t>& rows) {
  std::vector<std::int64_t> ids;
  for (auto r : rows) {
    check::Require(r < std::size_t(tensor.size(0)), "depth paired row bounds");
    ids.push_back(std::int64_t(r));
  }
  return nn::take_rows(tensor, ids).contiguous();
}
inline DepthData DepthBatch(const DepthData& data, const std::vector<std::size_t>& rows) {
  return {DepthBatch(data.rgb, rows), DepthBatch(data.depth, rows), DepthBatch(data.valid, rows), DepthBatch(data.room_ids, rows),
          DepthBatch(data.source_rows, rows)};
}

struct DepthPriors {
  double     geometric = 1;
  nn::Tensor spatial;
};
inline DepthPriors FitDepthPriors(const DepthData& fitting) {
  DepthPriors         result;
  std::vector<double> sum(3072), count(3072);
  double              all = 0, total = 0;
  const auto          d = fitting.depth.to_vector<float>(), v = fitting.valid.to_vector<float>();
  for (std::int64_t n = 0; n < fitting.depth.size(0); ++n)
    for (int i = 0; i < 3072; ++i)
      if (v[std::size_t(n * 3072 + i)] != 0) {
        const double log = std::log(double(d[std::size_t(n * 3072 + i)]));
        sum[std::size_t(i)] += log, ++count[std::size_t(i)], all += log, ++total;
      }
  check::Require(total > 0, "fit-only depth priors need valid observations");
  result.geometric = std::exp(all / total);
  std::vector<float> spatial(3072);
  for (int i = 0; i < 3072; ++i) spatial[std::size_t(i)] = float(std::exp(count[std::size_t(i)] ? sum[std::size_t(i)] / count[std::size_t(i)] : all / total));
  result.spatial = nn::from_vector(spatial, {1, 48, 64, 1});
  return result;
}
inline nn::Tensor DepthPriorImages(const DepthData& data, const DepthPriors& prior, bool spatial) {
  const auto         s = prior.spatial.to_vector<float>();
  std::vector<float> result(std::size_t(data.depth.numel()));
  for (std::size_t i = 0; i < result.size(); ++i) result[i] = spatial ? s[i % 3072] : float(prior.geometric);
  return nn::from_vector(result, data.depth.shape());
}

struct DepthMetrics {
  double       abs_rel = 0, sq_rel = 0, rmse = 0, log_rmse = 0, log10 = 0, silog = 0, delta1 = 0, delta2 = 0, delta3 = 0;
  int          images = 0;
  std::int64_t pixels = 0;
};
inline DepthMetrics ScoreDepth(const nn::Tensor& prediction, const nn::Tensor& truth, const nn::Tensor& mask) {
  check::Require(prediction.dtype() == nn::DType::Float32 && prediction.shape() == truth.shape() && mask.shape() == truth.shape() &&
                     prediction.dim() == 4,
                 "aligned scalar metric depth scores");
  DepthMetrics       result;
  const auto         p = prediction.to_vector<float>(), t = truth.to_vector<float>(), m = mask.to_vector<float>();
  const std::int64_t pixels = truth.numel() / truth.size(0);
  for (std::int64_t n = 0; n < truth.size(0); ++n) {
    double       absolute = 0, squared = 0, square = 0, log_square = 0, log_sum = 0, log10 = 0, d1 = 0, d2 = 0, d3 = 0;
    std::int64_t count = 0;
    for (std::int64_t i = 0; i < pixels; ++i) {
      const std::size_t j      = std::size_t(n * pixels + i);
      const double      target = t[j];
      if (m[j] <= .5 || !std::isfinite(target) || target <= .1f || target > 10) continue;
      const double estimated = p[j];
      check::Require(std::isfinite(estimated) && estimated > 0, "positive finite unaligned depth estimate");
      const double difference = estimated - target, log = std::log(estimated) - std::log(target),
                   ratio = std::max(estimated / target, target / estimated);
      absolute += std::abs(difference) / target;
      squared += difference * difference / target;
      square += difference * difference;
      log_square += log * log;
      log_sum += log;
      log10 += std::abs(log) / std::log(10.);
      d1 += ratio < 1.25;
      d2 += ratio < 1.25 * 1.25;
      d3 += ratio < 1.25 * 1.25 * 1.25;
      ++count;
    }
    if (!count) continue;
    const double c = double(count);
    ++result.images;
    result.pixels += count;
    result.abs_rel += absolute / c;
    result.sq_rel += squared / c;
    result.rmse += std::sqrt(square / c);
    result.log_rmse += std::sqrt(log_square / c);
    result.log10 += log10 / c;
    result.silog += std::sqrt(std::max(0., log_square / c - std::pow(log_sum / c, 2)));
    result.delta1 += d1 / c;
    result.delta2 += d2 / c;
    result.delta3 += d3 / c;
  }
  check::Require(result.images > 0, "depth metrics need at least one valid image");
  for (double* value : {&result.abs_rel, &result.sq_rel, &result.rmse, &result.log_rmse, &result.log10, &result.silog, &result.delta1,
                        &result.delta2, &result.delta3})
    *value /= result.images;
  return result;
}

}  // namespace depth
