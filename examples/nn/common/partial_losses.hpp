// Full partial-inpainting objective with fixed genuine VGG-16 pool1/2/3
// features: a port of tflow's common/ml/partial_losses.h.
#pragma once

#include <array>

#include "common/johnson.hpp"
#include "common/partial_unet.hpp"

namespace inpaint {

using PartialFeatures = std::array<nn::Tensor, 3>;

/// Keras ImageNet VGG uses centred BGR255; pool1/2/3 features / 255 is an explicit unit adaptation.
inline PartialFeatures PartialVgg(const style::FrozenVgg16& vgg, const nn::Tensor& image) {
  const auto      features = vgg.Extract(image, 3);
  PartialFeatures pooled;
  for (std::size_t i = 0; i < 3; ++i) pooled[i] = nn::max_pool2d(features.style[i], 2, 2, nn::Padding::Valid) / 255.0;
  return pooled;
}

/// Per-image Gram / (H W C): [B, C, C].
inline nn::Tensor PartialGram(const nn::Tensor& value) {
  const nn::Tensor flat = nn::reshape(value, {value.size(0), -1, value.size(3)});
  return nn::matmul(flat, flat, true) / double(value.size(1) * value.size(2) * value.size(3));
}

/// Mean absolute difference over every element.
inline nn::Tensor PartialL1(const nn::Tensor& a, const nn::Tensor& b) { return nn::mean(nn::abs(a - b)); }

/// {perceptual, style(raw), style(composite)}, each summed over the three pools.
inline std::array<nn::Tensor, 3> PartialFeatureLosses(const PartialFeatures& raw, const PartialFeatures& composite, const PartialFeatures& truth) {
  nn::Tensor perceptual, style_raw, style_composite;
  for (std::size_t i = 0; i < 3; ++i) {
    const nn::Tensor target = PartialGram(truth[i]);
    const nn::Tensor p = PartialL1(raw[i], truth[i]) + PartialL1(composite[i], truth[i]), sr = PartialL1(PartialGram(raw[i]), target),
                     sc = PartialL1(PartialGram(composite[i]), target);
    perceptual      = i ? perceptual + p : p;
    style_raw       = i ? style_raw + sr : sr;
    style_composite = i ? style_composite + sc : sc;
  }
  return {perceptual, style_raw, style_composite};
}

/// L1 TV over neighbour pairs whose both endpoints lie in the 1-pixel-dilated hole, / full image size.
inline nn::Tensor PartialTV(const nn::Tensor& composite, const nn::Tensor& valid) {
  const nn::Tensor   region = nn::max_pool2d(1.0 - valid, 3, 1, nn::Padding::Same);
  const std::int64_t b = composite.size(0), h = composite.size(1), w = composite.size(2), c = composite.size(3);
  const nn::Shape    hs{b, h, w - 1, c}, vs{b, h - 1, w, c};
  const nn::Tensor   left = nn::slice(composite, {0, 0, 0, 0}, hs), right = nn::slice(composite, {0, 0, 1, 0}, hs),
                   top = nn::slice(composite, {0, 0, 0, 0}, vs), bottom = nn::slice(composite, {0, 1, 0, 0}, vs);
  const nn::Tensor horizontal = nn::slice(region, {0, 0, 0, 0}, hs) * nn::slice(region, {0, 0, 1, 0}, hs);
  const nn::Tensor vertical   = nn::slice(region, {0, 0, 0, 0}, vs) * nn::slice(region, {0, 1, 0, 0}, vs);
  const nn::Tensor hsum = nn::sum(horizontal * nn::abs(left - right)), vsum = nn::sum(vertical * nn::abs(top - bottom));
  return (hsum + vsum) / double(composite.numel());
}

/// {total, valid, hole, perceptual, style(raw), style(comp), TV}; pixel losses divide by the FULL image size.
inline std::array<nn::Tensor, 7> PartialObjective(const nn::Tensor& raw, const nn::Tensor& truth, const nn::Tensor& valid, const PartialFeatures& rf,
                                                  const PartialFeatures& cf, const PartialFeatures& target_features) {
  const nn::Tensor error    = nn::abs(raw - truth);
  const nn::Tensor observed = nn::mean(valid * error), hole = nn::mean((1.0 - valid) * error);
  const auto       features = PartialFeatureLosses(rf, cf, target_features);
  const nn::Tensor tv       = PartialTV(PartialComposite(truth, raw, valid), valid);
  const nn::Tensor total    = observed + 6.0 * hole + .05 * features[0] + 120.0 * (features[1] + features[2]) + .1 * tv;
  return {total, observed, hole, features[0], features[1], features[2], tv};
}

}  // namespace inpaint
