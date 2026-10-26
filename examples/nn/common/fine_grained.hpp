// True Flowers102 images/labels from separate official training/validation/
// test IDs: a port of tflow's common/ml/fine_grained.h. flowers.npz (made by
// tflow's `just fine-grained-assets`) holds {fit,validation,test}_{rgb,labels,ids}:
// 48 balanced six-class 64x64 RGB images per split.
#pragma once

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

namespace fine {

namespace nn = datamunge::nn;

struct FlowerData {
  nn::Tensor rgb, labels, ids;  // float32 [48, 64, 64, 3] in [0, 1]; int32 [48]; int32 [48]
};
struct FlowerSplits {
  FlowerData fit, validation, test;
};

inline nn::Tensor FlowerRows(const nn::Tensor& input, const std::vector<std::size_t>& rows) {
  std::vector<std::int64_t> ids;
  for (auto r : rows) {
    check::Require(std::int64_t(r) < input.size(0), "flower row bounds");
    ids.push_back(std::int64_t(r));
  }
  return nn::take_rows(input, ids).contiguous();
}
inline FlowerData FlowerRows(const FlowerData& data, const std::vector<std::size_t>& rows) {
  return {FlowerRows(data.rgb, rows), FlowerRows(data.labels, rows), FlowerRows(data.ids, rows)};
}

inline FlowerSplits LoadFlowers(const std::filesystem::path& root) {
  std::map<std::string, nn::Tensor> arrays;
  for (auto& [name, value] : nn::load_npz((root / "flowers.npz").string())) arrays[name] = value;
  std::set<int> previous;
  auto read = [&](const std::string& split) {
    FlowerData out{arrays.at(split + "_rgb"), arrays.at(split + "_labels"), arrays.at(split + "_ids")};
    check::Require(out.rgb.dtype() == nn::DType::Float32 && out.labels.dtype() == nn::DType::Int32 && out.ids.dtype() == nn::DType::Int32 &&
                       out.rgb.shape() == nn::Shape({48, 64, 64, 3}) && out.labels.shape() == nn::Shape({48}) && out.ids.shape() == nn::Shape({48}),
                   "flower numeric shapes/types");
    std::array<int, 6> counts{};
    const auto         labels = out.labels.to_vector<int>(), ids = out.ids.to_vector<int>();
    for (int n = 0; n < 48; ++n) {
      const int label = labels[n], id = ids[n];
      check::Require(label >= 0 && label < 6 && id >= 1 && id <= 8189 && previous.insert(id).second,
                     "unique official flower ID and valid six-class label");
      ++counts[label];
    }
    for (int count : counts) check::Require(count == 8, "balanced real flower source split");
    for (float v : out.rgb.to_vector<float>()) check::Require(std::isfinite(v) && v >= 0 && v <= 1, "observed RGB pixel range");
    return out;
  };
  auto fit = read("fit"), validation = read("validation"), test = read("test");
  return {fit, validation, test};
}

struct FlowerMetrics {
  double              nll = 0, accuracy = 0, macro_accuracy = 0, brier = 0;
  std::array<int, 36> confusion{};
};

/// NLL, accuracy (first maximum wins ties), macro accuracy and the six-way Brier sum.
inline FlowerMetrics ScoreFlowers(const nn::Tensor& probability, const nn::Tensor& labels) {
  check::Require(probability.shape() == nn::Shape({labels.numel(), 6}), "flower probability dimensions");
  FlowerMetrics      score;
  std::array<int, 6> total{}, correct{};
  const auto         p = probability.contiguous().to_vector<float>();
  const auto         y = labels.to_vector<int>();
  for (std::int64_t n = 0; n < labels.numel(); ++n) {
    int       predicted = 0;
    double    sum       = 0;
    const int target    = y[std::size_t(n)];
    check::Require(target >= 0 && target < 6, "flower metric labels");
    for (int c = 0; c < 6; ++c) {
      const double v = p[std::size_t(n * 6 + c)];
      check::Require(std::isfinite(v) && v >= 0 && v <= 1, "finite normalized probabilities");
      sum += v;
      if (v > p[std::size_t(n * 6 + predicted)]) predicted = c;
      score.brier += (v - (target == c)) * (v - (target == c));
    }
    check::Near(sum, 1, 2e-6, "flower probability row sum");
    score.nll -= std::log(std::max(double(p[std::size_t(n * 6 + target)]), 1e-30));
    score.accuracy += predicted == target;
    ++total[target];
    correct[target] += predicted == target;
    ++score.confusion[std::size_t(target * 6 + predicted)];
  }
  int classes = 0;
  for (int c = 0; c < 6; ++c)
    if (total[c]) score.macro_accuracy += double(correct[c]) / total[c], ++classes;
  score.nll /= double(labels.numel());
  score.accuracy /= double(labels.numel());
  score.brier /= double(labels.numel());
  score.macro_accuracy /= classes;
  return score;
}

/// Per-image mean RGB [N, 3].
inline std::vector<float> FlowerColorMeans(const FlowerData& data) {
  const std::int64_t n = data.rgb.size(0);
  const auto         v = data.rgb.contiguous().to_vector<float>();
  std::vector<float> out(std::size_t(n) * 3);
  for (std::int64_t b = 0; b < n; ++b)
    for (int c = 0; c < 3; ++c) {
      double sum = 0;
      for (int p = 0; p < 4096; ++p) sum += v[std::size_t((b * 4096 + p) * 3 + c)];
      out[std::size_t(b * 3 + c)] = float(sum / 4096);
    }
  return out;
}

/// Softmax(-|mean RGB - class prototype|^2 / .02) with fit-only prototypes.
inline nn::Tensor FlowerColorBaseline(const FlowerData& fitting, const FlowerData& testing) {
  const auto         fit = FlowerColorMeans(fitting), test = FlowerColorMeans(testing);
  const auto         labels = fitting.labels.to_vector<int>();
  double             centers[6][3] = {};
  std::array<int, 6> counts{};
  for (std::size_t n = 0; n < labels.size(); ++n) {
    ++counts[labels[n]];
    for (int c = 0; c < 3; ++c) centers[labels[n]][c] += fit[n * 3 + std::size_t(c)];
  }
  for (int label = 0; label < 6; ++label)
    for (int c = 0; c < 3; ++c) centers[label][c] /= counts[label];
  const std::int64_t n = testing.rgb.size(0);
  std::vector<float> out(std::size_t(n) * 6);
  for (std::int64_t b = 0; b < n; ++b) {
    double sum = 0;
    for (int label = 0; label < 6; ++label) {
      double distance = 0;
      for (int c = 0; c < 3; ++c) distance += std::pow(test[std::size_t(b * 3 + c)] - centers[label][c], 2);
      const double value              = std::exp(-distance / .02);
      out[std::size_t(b * 6 + label)] = float(value);
      sum += value;
    }
    for (int label = 0; label < 6; ++label) out[std::size_t(b * 6 + label)] = float(out[std::size_t(b * 6 + label)] / sum);
  }
  return nn::from_vector(out, {n, 6});
}

}  // namespace fine
