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
#include <datamunge/nn/johnson_losses.hpp>

namespace style {

using datamunge::nn::models::FastStyleBatch;
using datamunge::nn::models::JohnsonContent;
using datamunge::nn::models::JohnsonGram;
using datamunge::nn::models::JohnsonPreviewTiles;
using datamunge::nn::models::JohnsonStyle;
using datamunge::nn::models::JohnsonTV;
using datamunge::nn::models::RestoreJohnson;
using datamunge::nn::models::SaveJohnson;
inline nn::Tensor FastStyleImages(const data::CifarDataset &source,
                                  const std::vector<std::size_t> &rows) {
  return nn::models::FastStyleImages(source.images, rows);
}
using JohnsonSnapshot = std::vector<std::vector<float>>;
inline JohnsonSnapshot JohnsonSnap(const nn::ParameterStore &p,
                                   const nn::ParameterStore &state) {
  JohnsonSnapshot out;
  for (const auto &v : p.parameters())
    out.push_back(v.to_vector<float>());
  for (const auto &v : state.parameters())
    out.push_back(v.to_vector<float>());
  return out;
}

class FastStyleModel {
public:
  nn::ParameterStore parameters{17}, state{18};
  FrozenVgg16 vgg;
  std::unique_ptr<nn::Optimizer> optimizer;
  double content_denominator = 1, style_denominator = 1, tv_denominator = 1;
  std::array<nn::Tensor, 4> target_grams;
  explicit FastStyleModel(const TensorMap &weights, std::uint64_t seed = 17)
      : parameters(seed), vgg(weights) {
    {
      nn::NoGradGuard guard;
      JohnsonTransform(parameters, state, nn::zeros({1, 64, 64, 3}), 8, false);
    }
    nn::OptimizerConfig config;
    config.kind = nn::OptimizerKind::Adam;
    optimizer = std::make_unique<nn::Optimizer>(parameters, .001, config);
  }
  nn::Tensor Content(const nn::Tensor &x) {
    nn::NoGradGuard guard;
    return vgg.Extract(x).content;
  }
  void Style(const nn::Tensor &x) {
    nn::NoGradGuard guard;
    const auto features = vgg.Extract(x);
    for (std::size_t i = 0; i < 4; ++i)
      target_grams[i] = JohnsonGram(features.style[i]);
  }
  /// {objective, content, style, TV}; `update` takes one Adam step and updates the moving BN.
  std::array<double, 4> Run(const nn::Tensor &x, const nn::Tensor &target, bool train = false,
                            bool update = false) {
    std::unique_ptr<nn::NoGradGuard> guard;
    if (!update)
      guard = std::make_unique<nn::NoGradGuard>();
    const auto output = JohnsonTransform(parameters, state, x, 8, train, update).image;
    const auto features = vgg.Extract(output);
    const nn::Tensor c = JohnsonContent(features.content, target),
                     st = JohnsonStyle(features.style, target_grams), tv = JohnsonTV(output);
    const nn::Tensor total =
        c / content_denominator + st / style_denominator + tv / tv_denominator;
    const std::array<double, 4> result = {total.item(), c.item(), st.item(), tv.item()};
    for (double v : result)
      check::Require(std::isfinite(v) && v >= 0, "finite nonnegative feed-forward loss");
    if (update)
      optimizer->step(nn::grad(total, parameters.parameters()));
    return result;
  }
  nn::Tensor Predict(const nn::Tensor &x) {
    nn::NoGradGuard guard;
    return JohnsonTransform(parameters, state, x, 8, false).image;
  }
};

inline nn::Tensor CacheFastStyleContent(FastStyleModel &model, const nn::Tensor &images) {
  std::vector<nn::Tensor> parts;
  for (std::int64_t row = 0; row < images.size(0); row += 2)
    parts.push_back(
        model.Content(FastStyleBatch(images, {std::size_t(row), std::size_t(row + 1)})));
  return nn::concat(parts, 0).contiguous();
}

inline std::array<double, 4> ScoreFastStyle(FastStyleModel &model, const nn::Tensor &images,
                                            const nn::Tensor &targets) {
  std::array<double, 4> score{};
  for (std::int64_t row = 0; row < images.size(0); row += 2) {
    const std::vector<std::size_t> rows = {std::size_t(row), std::size_t(row + 1)};
    const auto value = model.Run(FastStyleBatch(images, rows), FastStyleBatch(targets, rows));
    for (int i = 0; i < 4; ++i)
      score[std::size_t(i)] += value[std::size_t(i)] * 2 / double(images.size(0));
  }
  return score;
}

/// Generator-only deployment restored from best.npz.
class JohnsonInference {
public:
  nn::ParameterStore parameters{17}, state{18};
  explicit JohnsonInference(const std::string &path) {
    {
      nn::NoGradGuard guard;
      JohnsonTransform(parameters, state, nn::zeros({1, 64, 64, 3}), 8, false);
    }
    RestoreJohnson(parameters, state, path);
  }
  nn::Tensor Predict(const nn::Tensor &image) {
    check::Require(image.size(1) > 40 && image.size(2) > 40 && image.size(1) % 4 == 0 &&
                       image.size(2) % 4 == 0,
                   "generator image dimensions >40 and divisible by4");
    nn::NoGradGuard guard;
    const nn::Tensor out =
        JohnsonTransform(parameters, state, image, 8, false).image.contiguous();
    check::Require(out.shape() == image.shape(), "generator preserves image dimensions");
    check::Require(
        nn::all(nn::logical_and(nn::is_finite(out), nn::logical_and(nn::greater_equal(out, 0.),
                                                                    nn::less_equal(out, 1.))))
            .item<bool>(),
        "finite bounded generated RGB");
    return out;
  }
  JohnsonSnapshot Snapshot() const {
    return JohnsonSnap(parameters, state);
  }
};

} // namespace style
