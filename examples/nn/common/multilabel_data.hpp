// Real COCO128 independent category-presence bits and fit-only class weights:
// a port of tflow's common/ml/multilabel_data.h. multilabel.npz (from tflow's
// `just multilabel-assets`) holds <split>_rgb float32 [N, 64, 64, 3],
// <split>_labels float32 [N, 80] and <split>_ids int32 [N] for fit (80),
// validation (24) and test (24).
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

namespace multilabel {

namespace nn = datamunge::nn;

struct MultilabelData {
  nn::Tensor rgb, labels, ids;
};
struct MultilabelSplits {
  MultilabelData fit, validation, test;
};

inline nn::Tensor MultilabelRows(const nn::Tensor& input, const std::vector<std::size_t>& rows) {
  std::vector<std::int64_t> index;
  for (std::size_t r : rows) {
    check::Require(std::int64_t(r) < input.size(0), "multi-label row bounds");
    index.push_back(std::int64_t(r));
  }
  return nn::take_rows(input, index).contiguous();
}
inline MultilabelData MultilabelRows(const MultilabelData& data, const std::vector<std::size_t>& rows) {
  return {MultilabelRows(data.rgb, rows), MultilabelRows(data.labels, rows), MultilabelRows(data.ids, rows)};
}

inline MultilabelSplits LoadMultilabel(const std::filesystem::path& root) {
  std::map<std::string, nn::Tensor> arrays;
  for (auto& [name, value] : nn::load_npz((root / "multilabel.npz").string())) arrays[name] = value;
  std::set<int> ids;
  auto          read = [&](const std::string& split, std::int64_t count) {
    MultilabelData result{arrays.at(split + "_rgb"), arrays.at(split + "_labels"), arrays.at(split + "_ids")};
    check::Require(result.rgb.dtype() == nn::DType::Float32 && result.labels.dtype() == nn::DType::Float32 &&
                       result.ids.dtype() == nn::DType::Int32 && result.rgb.shape() == nn::Shape{count, 64, 64, 3} &&
                       result.labels.shape() == nn::Shape{count, 80} && result.ids.shape() == nn::Shape{count},
                   "multi-label numeric shapes/types");
    const auto labels = result.labels.to_vector<float>();
    const auto id     = result.ids.to_vector<int>();
    int        multiple = 0;
    for (std::int64_t n = 0; n < count; ++n) {
      check::Require(ids.insert(id[std::size_t(n)]).second, "source image IDs never repeat across splits");
      int present = 0;
      for (int c = 0; c < 80; ++c) {
        const float y = labels[std::size_t(n * 80 + c)];
        check::Require(y == 0 || y == 1, "independent binary presence targets");
        present += int(y);
      }
      multiple += present > 1;
    }
    check::Require(multiple > 0, "genuine multi-object images in every split");
    for (float v : result.rgb.to_vector<float>()) check::Require(std::isfinite(v) && v >= 0 && v <= 1, "observed full-image RGB range");
    return result;
  };
  auto fit = read("fit", 80), validation = read("validation", 24), test = read("test", 24);
  return {fit, validation, test};
}

/// Positive-class weights clamp((N - pos + 1) / (pos + 1), 1, 20) from the fitting labels only.
inline nn::Tensor MultilabelWeights(const nn::Tensor& labels) {
  check::Require(labels.dim() == 2, "multi-label weight target matrix");
  const std::int64_t n = labels.size(0), classes = labels.size(1);
  const auto         y = labels.to_vector<float>();
  std::vector<float> out(static_cast<std::size_t>(classes));
  for (std::int64_t c = 0; c < classes; ++c) {
    int positive = 0;
    for (std::int64_t i = 0; i < n; ++i) positive += y[std::size_t(i * classes + c)] > .5;
    out[std::size_t(c)] = std::clamp(float(n - positive + 1) / float(positive + 1), 1.f, 20.f);
  }
  return nn::from_vector(out, {classes});
}

/// Laplace-smoothed fit prior (pos + 1) / (N + 2), repeated for `count` rows.
inline nn::Tensor MultilabelPrior(const nn::Tensor& fitting, int count) {
  const std::int64_t n = fitting.size(0), classes = fitting.size(1);
  const auto         y = fitting.to_vector<float>();
  std::vector<float> out(std::size_t(count * classes));
  for (std::int64_t c = 0; c < classes; ++c) {
    int positive = 0;
    for (std::int64_t i = 0; i < n; ++i) positive += y[std::size_t(i * classes + c)] > .5;
    const float value = float(positive + 1) / float(n + 2);
    for (int r = 0; r < count; ++r) out[std::size_t(r * classes + c)] = value;
  }
  return nn::from_vector(out, {count, classes});
}

}  // namespace multilabel
