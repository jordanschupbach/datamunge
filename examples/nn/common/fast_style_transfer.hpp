// Feed-forward Johnson training with frozen VGG-16 content/Gram losses: a port
// of tflow's common/ml/fast_style_transfer.h.
// TF -> datamunge: the TF session with training/inference graphs becomes
// eager calls of JohnsonTransform in training (batch statistics, moving BN
// updated only on optimizer steps) or inference mode. The checkpoint is
// best.npz with "transform/<name>" weights and "statistics/<name>" moving BN;
// JohnsonInference rebuilds a generator-only model from it (no VGG, targets
// or optimizer — TF restores a separate generator-only graph).
#pragma once

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>

#include "common/checks.hpp"
#include "common/cifar.hpp"
#include "common/johnson.hpp"
#include "common/style_transfer.hpp"

namespace style {

/// Gram / (H W C): [B, C, C].
inline nn::Tensor JohnsonGram(const nn::Tensor& x) { return StyleGram(x) / double(x.size(3)); }
/// Mean squared feature reconstruction.
inline nn::Tensor JohnsonContent(const nn::Tensor& features, const nn::Tensor& target) { return nn::mean(nn::squared_difference(features, target)); }
/// Sum over the four layers of the batch-mean squared Frobenius Gram distance.
inline nn::Tensor JohnsonStyle(const std::array<nn::Tensor, 4>& features, const std::array<nn::Tensor, 4>& targets) {
  nn::Tensor total;
  for (std::size_t i = 0; i < 4; ++i) {
    const nn::Tensor term = nn::mean(nn::sum(nn::squared_difference(JohnsonGram(features[i]), targets[i]), {1, 2}));
    total                 = i ? total + term : term;
  }
  return total;
}
/// .5 mean over the joint interior of (right^2 + down^2) neighbour differences.
inline nn::Tensor JohnsonTV(const nn::Tensor& image) {
  const std::int64_t b = image.size(0), h = image.size(1) - 1, w = image.size(2) - 1, c = image.size(3);
  const nn::Tensor   center = nn::slice(image, {0, 0, 0, 0}, {b, h, w, c}), right = nn::slice(image, {0, 0, 1, 0}, {b, h, w, c}),
                   down = nn::slice(image, {0, 1, 0, 0}, {b, h, w, c});
  return .5 * nn::mean(nn::squared_difference(center, right) + nn::squared_difference(center, down));
}

/// CIFAR rows / 255, legacy ResizeBilinear 32 -> 64.
inline nn::Tensor FastStyleImages(const data::CifarDataset& data, const std::vector<std::size_t>& rows) {
  const std::int32_t* px = data.images.contiguous().data_ptr<std::int32_t>();
  std::vector<float>  input(rows.size() * 3072);
  for (std::size_t i = 0; i < rows.size(); ++i) {
    check::Require(std::int64_t(rows[i]) < data.size(), "CIFAR style row bounds");
    for (std::size_t j = 0; j < 3072; ++j) input[i * 3072 + j] = float(px[rows[i] * 3072 + j]) / 255.f;
  }
  return nn::resize_bilinear(nn::from_vector(input, {std::int64_t(rows.size()), 32, 32, 3}), {64, 64}).contiguous();
}

inline nn::Tensor FastStyleBatch(const nn::Tensor& input, const std::vector<std::size_t>& rows) {
  check::Require(input.dim() == 4, "rank4 cached content/features");
  std::vector<std::int64_t> ids;
  for (auto r : rows) {
    check::Require(std::int64_t(r) < input.size(0), "cached content row bounds");
    ids.push_back(std::int64_t(r));
  }
  return nn::take_rows(input, ids).contiguous();
}

using JohnsonSnapshot = std::vector<std::vector<float>>;
inline JohnsonSnapshot JohnsonSnap(const nn::ParameterStore& p, const nn::ParameterStore& state) {
  JohnsonSnapshot out;
  for (const auto& v : p.parameters()) out.push_back(v.to_vector<float>());
  for (const auto& v : state.parameters()) out.push_back(v.to_vector<float>());
  return out;
}

inline void SaveJohnson(const nn::ParameterStore& p, const nn::ParameterStore& state, const std::string& path) {
  nn::NamedTensors arrays;
  for (std::size_t i = 0; i < p.size(); ++i) arrays.push_back({"transform/" + p.names()[i], p.parameters()[i]});
  for (std::size_t i = 0; i < state.size(); ++i) arrays.push_back({"statistics/" + state.names()[i], state.parameters()[i]});
  nn::save_npz(path, arrays);
}
inline void RestoreJohnson(nn::ParameterStore& p, nn::ParameterStore& state, const std::string& path) {
  nn::NoGradGuard guard;
  const auto      arrays = nn::load_npz(path);
  check::Require(arrays.size() == p.size() + state.size(), "complete Johnson generator checkpoint");
  for (const auto& [name, value] : arrays) {
    if (name.rfind("statistics/", 0) == 0)
      state.set(name.substr(11), value);
    else if (name.rfind("transform/", 0) == 0)
      p.set(name.substr(10), value);
    else
      throw std::runtime_error("unexpected Johnson checkpoint array " + name);
  }
}

class FastStyleModel {
 public:
  nn::ParameterStore             parameters{17}, state{18};
  FrozenVgg16                    vgg;
  std::unique_ptr<nn::Optimizer> optimizer;
  double                         content_denominator = 1, style_denominator = 1, tv_denominator = 1;
  std::array<nn::Tensor, 4>      target_grams;
  explicit FastStyleModel(const TensorMap& weights, std::uint64_t seed = 17) : parameters(seed), vgg(weights) {
    {
      nn::NoGradGuard guard;
      JohnsonTransform(parameters, state, nn::zeros({1, 64, 64, 3}), 8, false);
    }
    nn::OptimizerConfig config;
    config.kind = nn::OptimizerKind::Adam;
    optimizer   = std::make_unique<nn::Optimizer>(parameters, .001, config);
  }
  nn::Tensor Content(const nn::Tensor& x) {
    nn::NoGradGuard guard;
    return vgg.Extract(x).content;
  }
  void Style(const nn::Tensor& x) {
    nn::NoGradGuard guard;
    const auto      features = vgg.Extract(x);
    for (std::size_t i = 0; i < 4; ++i) target_grams[i] = JohnsonGram(features.style[i]);
  }
  /// {objective, content, style, TV}; `update` takes one Adam step and updates the moving BN.
  std::array<double, 4> Run(const nn::Tensor& x, const nn::Tensor& target, bool train = false, bool update = false) {
    std::unique_ptr<nn::NoGradGuard> guard;
    if (!update) guard = std::make_unique<nn::NoGradGuard>();
    const auto       output   = JohnsonTransform(parameters, state, x, 8, train, update).image;
    const auto       features = vgg.Extract(output);
    const nn::Tensor c = JohnsonContent(features.content, target), st = JohnsonStyle(features.style, target_grams), tv = JohnsonTV(output);
    const nn::Tensor total = c / content_denominator + st / style_denominator + tv / tv_denominator;
    const std::array<double, 4> result = {total.item(), c.item(), st.item(), tv.item()};
    for (double v : result) check::Require(std::isfinite(v) && v >= 0, "finite nonnegative feed-forward loss");
    if (update) optimizer->step(nn::grad(total, parameters.parameters()));
    return result;
  }
  nn::Tensor Predict(const nn::Tensor& x) {
    nn::NoGradGuard guard;
    return JohnsonTransform(parameters, state, x, 8, false).image;
  }
};

inline nn::Tensor CacheFastStyleContent(FastStyleModel& model, const nn::Tensor& images) {
  std::vector<nn::Tensor> parts;
  for (std::int64_t row = 0; row < images.size(0); row += 2)
    parts.push_back(model.Content(FastStyleBatch(images, {std::size_t(row), std::size_t(row + 1)})));
  return nn::concat(parts, 0).contiguous();
}

inline std::array<double, 4> ScoreFastStyle(FastStyleModel& model, const nn::Tensor& images, const nn::Tensor& targets) {
  std::array<double, 4> score{};
  for (std::int64_t row = 0; row < images.size(0); row += 2) {
    const std::vector<std::size_t> rows  = {std::size_t(row), std::size_t(row + 1)};
    const auto                     value = model.Run(FastStyleBatch(images, rows), FastStyleBatch(targets, rows));
    for (int i = 0; i < 4; ++i) score[std::size_t(i)] += value[std::size_t(i)] * 2 / double(images.size(0));
  }
  return score;
}

/// Generator-only deployment restored from best.npz.
class JohnsonInference {
 public:
  nn::ParameterStore parameters{17}, state{18};
  explicit JohnsonInference(const std::string& path) {
    {
      nn::NoGradGuard guard;
      JohnsonTransform(parameters, state, nn::zeros({1, 64, 64, 3}), 8, false);
    }
    RestoreJohnson(parameters, state, path);
  }
  nn::Tensor Predict(const nn::Tensor& image) {
    check::Require(image.size(1) > 40 && image.size(2) > 40 && image.size(1) % 4 == 0 && image.size(2) % 4 == 0,
                   "generator image dimensions >40 and divisible by4");
    nn::NoGradGuard  guard;
    const nn::Tensor out = JohnsonTransform(parameters, state, image, 8, false).image.contiguous();
    check::Require(out.shape() == image.shape(), "generator preserves image dimensions");
    for (float v : out.to_vector<float>()) check::Require(std::isfinite(v) && v >= 0 && v <= 1, "finite bounded generated RGB");
    return out;
  }
  JohnsonSnapshot Snapshot() const { return JohnsonSnap(parameters, state); }
};

}  // namespace style
