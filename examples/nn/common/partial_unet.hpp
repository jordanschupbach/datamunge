// Partial-convolution U-Net: propagate and concatenate both features and
// masks. A port of tflow's common/nn/partial_unet.h (partial_inpainting).
// Every layer is inpaint::PartialConv (multi-channel masks, SAME stride
// alignment) with He-normal kernels and zero bias; BN momentum .9/eps 1e-5
// (inpaint::InpaintNorm) on all but the first encoder and last decoder layer.
#pragma once

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/inpainting.hpp"

namespace inpaint {

struct PartialConfig {
  int                            depth = 8, width = 64;
  static PartialConfig           Compact() { return {5, 8}; }
  [[nodiscard]] std::vector<int> Widths() const {
    if (depth < 3 || depth > 8 || width < 1) throw std::invalid_argument("partial UNet depth3..8 and positive width required");
    std::vector<int> result;
    for (int i = 0; i < depth; ++i) result.push_back(width * (1 << std::min(i, 3)));
    return result;
  }
};

struct PartialUNetResult {
  nn::Tensor              image, mask;
  std::vector<nn::Tensor> stages, masks;
};

/// TF ResizeNearestNeighbor (legacy corners) of `input` to the spatial size of `like`.
inline nn::Tensor PartialResize(const nn::Tensor& input, const nn::Tensor& like) { return nn::resize_nearest(input, {like.size(1), like.size(2)}); }

enum class PartialAct { kNone, kRelu, kLeakyRelu };

/// Encoder: stride-2 partial convs (7, 5, 5, 3...), BN after the first, ReLU.
/// Decoder: nearest upsampling, concatenation of features AND masks with the
/// skip (the original image/mask at the last one), 3x3 partial conv, BN and
/// LeakyReLU .2 except the final linear RGB layer.
inline PartialUNetResult PartialUNet(nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& image, const nn::Tensor& valid,
                                     const PartialConfig& config = {}, bool training = false, bool update = true) {
  const auto              widths = config.Widths();
  PartialUNetResult       result;
  std::vector<nn::Tensor> features{image}, masks{valid};
  std::int64_t            channels = 3;
  auto layer = [&](const std::string& name, const nn::Tensor& x, const nn::Tensor& m, std::int64_t in, std::int64_t out, std::int64_t kernel,
                   std::int64_t stride, bool normalize, PartialAct activation) {
    const nn::Tensor weights = p.get(name + "/W", {kernel, kernel, in, out}, nn::init::he_normal());
    const nn::Tensor bias    = p.get(name + "/b", {out}, nn::init::zeros());
    auto             value   = PartialConv(x, m, weights, bias, in, out, kernel, stride);
    if (normalize) value.value = InpaintNorm(p, state, name + "/bn", value.value, training, update);
    if (activation == PartialAct::kRelu) value.value = nn::relu(value.value);
    if (activation == PartialAct::kLeakyRelu) value.value = nn::leaky_relu(value.value, .2);
    result.stages.push_back(value.value);
    result.masks.push_back(value.mask);
    return value;
  };
  for (int i = 0; i < config.depth; ++i) {
    const int  kernel = i == 0 ? 7 : i <= 2 ? 5 : 3;
    const auto value  = layer("encode" + std::to_string(i + 1), features.back(), masks.back(), channels, widths[std::size_t(i)], kernel, 2, i > 0,
                              PartialAct::kRelu);
    features.push_back(value.value);
    masks.push_back(value.mask);
    channels = widths[std::size_t(i)];
  }
  nn::Tensor x = features.back(), mask = masks.back();
  for (int i = config.depth - 1; i >= 0; --i) {
    const std::int64_t skip = i == 0 ? 3 : widths[std::size_t(i - 1)], out = i == 0 ? 3 : skip;
    const nn::Tensor   combined      = nn::concat({PartialResize(x, features[std::size_t(i)]), features[std::size_t(i)]}, 3);
    const nn::Tensor   combined_mask = nn::concat({PartialResize(mask, masks[std::size_t(i)]), masks[std::size_t(i)]}, 3);
    const auto         value = layer("decode" + std::to_string(config.depth - i), combined, combined_mask, channels + skip, out, 3, 1, i > 0,
                                     i > 0 ? PartialAct::kLeakyRelu : PartialAct::kNone);
    x = value.value, mask = value.mask, channels = out;
  }
  result.image = x;
  result.mask  = mask;
  return result;
}

/// Observed pixels where mask > .5, the network output elsewhere.
inline nn::Tensor PartialComposite(const nn::Tensor& observed, const nn::Tensor& output, const nn::Tensor& mask) {
  return nn::where(nn::greater(mask, .5), observed, output);
}

}  // namespace inpaint
