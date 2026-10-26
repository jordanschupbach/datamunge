// U-Net: symmetric contracting/expanding paths with concatenated cropped skips
// (port of tflow's common/nn/unet.h).
// https://arxiv.org/abs/1505.04597
// Full defaults retain original valid 3x3 pairs, four 2x2 pools, base 64,
// learned 2x2 transposed up-convolutions, grayscale input/two classes and
// 572->388 geometry. Compact teaching mode uses base 8, SAME pairs, RGB/four
// classes and no dropout; inputs divisible by 16 retain their spatial size.
// Parameter names follow tflow ("encoder<l>/conv1/W", "decoder<l>/up/W", ...).
#pragma once

#include <datamunge/nn/nn.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace unet {

namespace nn = datamunge::nn;

/// Output side of a U-Net with `levels` pools for a square input (throws when too small).
inline int UnetOutputSide(int side, int levels = 4, bool valid = true) {
  if (levels < 1 || levels > 6 || side < 1) throw std::invalid_argument("invalid U-Net geometry");
  for (int level = 0; level < levels; ++level) {
    if (valid) side -= 4;
    if (side < 2) throw std::invalid_argument("U-Net input too small");
    side /= 2;
  }
  if (valid) side -= 4;
  for (int level = 0; level < levels; ++level) {
    side *= 2;
    if (valid) side -= 4;
    if (side < 1) throw std::invalid_argument("U-Net input too small");
  }
  return side;
}

/// Floor-centred spatial crop of source [N, H, W, C] to the height/width of reference.
inline nn::Tensor CenterCropTo(const nn::Tensor& source, const nn::Tensor& reference) {
  const auto&        s  = source.shape();
  const std::int64_t th = reference.shape()[1], tw = reference.shape()[2];
  if (th > s[1] || tw > s[2]) throw std::invalid_argument("centre crop larger than source");
  return nn::slice(source, {0, (s[1] - th) / 2, (s[2] - tw) / 2, 0}, {s[0], th, tw, s[3]});
}

/// Two 3x3 ReLU convolutions "<name>/conv1", "<name>/conv2" (He uniform, bias).
inline nn::Tensor ConvPair(nn::ParameterStore& p, const std::string& name, const nn::Tensor& x, std::int64_t out,
                           nn::Padding padding = nn::Padding::Same) {
  nn::ConvLayerOptions conv;
  conv.padding    = padding;
  conv.activation = nn::Activation::Relu;
  conv.init       = nn::init::he_uniform();
  return nn::conv2d(p, name + "/conv2", nn::conv2d(p, name + "/conv1", x, out, 3, conv), out, 3, conv);
}

/// Learned 2x2 stride-2 transposed convolution (VALID, ReLU, He uniform) "<name>/W", "<name>/b".
inline nn::Tensor UpConv(nn::ParameterStore& p, const std::string& name, const nn::Tensor& x, std::int64_t out) {
  nn::ConvLayerOptions up;
  up.stride     = 2;
  up.padding    = nn::Padding::Valid;
  up.activation = nn::Activation::Relu;
  up.init       = nn::init::he_uniform();
  return nn::conv2d_transpose(p, name, x, out, 2, up);
}

struct UnetConfig {
  int   classes = 2, input_channels = 1, base_width = 64, levels = 4;
  bool  valid   = true;
  float dropout = .5f;
};
struct UnetOutputs {
  nn::Tensor              logits, bottleneck;
  std::vector<nn::Tensor> skips, decoder;
};

inline UnetOutputs Unet(nn::ParameterStore& p, const nn::Tensor& image, const UnetConfig& cfg = {},
                        bool training = false) {
  if (cfg.classes < 2 || cfg.input_channels < 1 || cfg.base_width < 1 || cfg.levels < 1 || cfg.levels > 6 ||
      !(cfg.dropout >= 0 && cfg.dropout < 1) || image.shape().back() != cfg.input_channels)
    throw std::invalid_argument("invalid U-Net configuration");
  const nn::Padding padding = cfg.valid ? nn::Padding::Valid : nn::Padding::Same;
  UnetOutputs       result;
  nn::Tensor        h = image;
  for (int level = 0; level < cfg.levels; ++level) {
    h = ConvPair(p, "encoder" + std::to_string(level), h, std::int64_t(cfg.base_width) << level, padding);
    if (training && cfg.dropout > 0 && level == cfg.levels - 1) h = nn::dropout(h, cfg.dropout, true, p.generator());
    result.skips.push_back(h);
    h = nn::max_pool2d(h, 2, 2, nn::Padding::Valid);
  }
  h = ConvPair(p, "bottleneck", h, 2 * h.shape().back(), padding);
  if (training && cfg.dropout > 0) h = nn::dropout(h, cfg.dropout, true, p.generator());
  result.bottleneck = h;
  for (int level = cfg.levels - 1; level >= 0; --level) {
    const std::int64_t width = std::int64_t(cfg.base_width) << level;
    const auto         name  = "decoder" + std::to_string(level);
    h = UpConv(p, name + "/up", h, width);
    h = nn::concat({CenterCropTo(result.skips[std::size_t(level)], h), h}, 3);
    h = ConvPair(p, name, h, width, padding);
    result.decoder.push_back(h);
  }
  nn::ConvLayerOptions output;  // linear 1x1, Glorot uniform
  result.logits = nn::conv2d(p, "output", h, cfg.classes, 1, output);
  return result;
}

}  // namespace unet
