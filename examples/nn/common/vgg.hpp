// VGG configurations A/D/E (11/16/19 learned layers) from Simonyan/Zisserman
// (port of tflow's common/nn/vgg.h): https://arxiv.org/abs/1409.1556
// Five blocks use only 3x3 SAME convolutions, ReLU and 2x2 stride-2 max pools.
// Widths 64/128/256/512/512 precede 4096/4096/1000 dense layers. There is no
// batch norm or LRN; dropout .5 applies to the two hidden classifier layers.
// Default 224-pixel inputs give a 7x7 final feature map. Explicit compact
// configurations alter widths/resolution/head size, preserving block depths.
// He/Glorot initialization is an adaptation to train from scratch. Dropout
// masks come from the parameter store's generator.
#pragma once

#include <array>
#include <datamunge/nn/nn.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace vgg {

namespace nn = datamunge::nn;

inline std::array<int, 5> VggBlockDepths(int depth) {
  if (depth == 11) return {1, 1, 2, 2, 2};
  if (depth == 16) return {2, 2, 3, 3, 3};
  if (depth == 19) return {2, 2, 4, 4, 4};
  throw std::invalid_argument("VGG depth must be 11, 16 or 19");
}

struct VggConfig {
  int    depth = 16, image_size = 224, classes = 1000, width_divisor = 1, dense_width = 4096;
  double keep_probability = .5;
  void   Validate() const {
    VggBlockDepths(depth);
    if (image_size < 32 || classes < 2 || dense_width < 1 || width_divisor <= 0 || 64 % width_divisor ||
        !(keep_probability > 0 && keep_probability <= 1))
      throw std::invalid_argument("invalid VGG configuration");
  }
  int FinalSpatialSize() const {
    Validate();
    return image_size / 32;
  }
};

struct VggOutputs {
  std::vector<nn::Tensor>   convolution;
  std::array<nn::Tensor, 5> pooling;
  nn::Tensor                fc6, fc7, logits;
};

inline VggOutputs Vgg(nn::ParameterStore& p, const nn::Tensor& image, const VggConfig& cfg = {}, bool training = false) {
  cfg.Validate();
  VggOutputs               out;
  const auto               depths = VggBlockDepths(cfg.depth);
  const std::array<int, 5> widths = {64, 128, 256, 512, 512};
  nn::ConvLayerOptions     options;
  options.activation = nn::Activation::Relu;
  options.init       = nn::init::he_uniform();
  nn::Tensor h       = image;
  std::int64_t channels = 3;
  for (int block = 0; block < 5; ++block) {
    const int width = widths[static_cast<std::size_t>(block)] / cfg.width_divisor;
    for (int layer = 0; layer < depths[static_cast<std::size_t>(block)]; ++layer) {
      h = nn::conv2d(p, "block" + std::to_string(block + 1) + "/conv" + std::to_string(layer + 1), h, width, 3, options);
      out.convolution.push_back(h);
      channels = width;
    }
    h                                            = nn::max_pool2d(h, 2, 2);
    out.pooling[static_cast<std::size_t>(block)] = h;
  }
  const std::int64_t features = std::int64_t(cfg.FinalSpatialSize()) * cfg.FinalSpatialSize() * channels;
  nn::DenseOptions   dense;
  dense.init       = nn::init::he_uniform();
  dense.activation = nn::Activation::Relu;
  auto drop = [&](const nn::Tensor& x) { return training ? nn::dropout(x, 1 - cfg.keep_probability, true, p.generator()) : x; };
  out.fc6    = nn::dense(p, "fc6", nn::reshape(h, {-1, features}), cfg.dense_width, dense);
  out.fc7    = nn::dense(p, "fc7", drop(out.fc6), cfg.dense_width, dense);
  out.logits = nn::dense(p, "fc8", drop(out.fc7), cfg.classes);
  return out;
}

// Reuse a previously built VGG classifier as a fully convolutional head.
// FC6 becomes a spatial kernel and FC7/8 become 1x1 kernels. An F x F feature
// map produces a 1x1 score map; larger maps produce sliding classifier scores.
// These are sliding windows of feature maps, not equivalent raw-image crops
// (convolution padding and surrounding image context can differ).
inline nn::Tensor VggConvolutionalHead(const nn::ParameterStore& p, const nn::Tensor& features, const VggConfig& cfg) {
  cfg.Validate();
  const std::int64_t spatial = cfg.FinalSpatialSize(), channels = 512 / cfg.width_divisor;
  const auto         valid   = nn::conv_options(1, nn::Padding::Valid);
  const auto first = nn::reshape(p.at("fc6/W"), {spatial, spatial, channels, cfg.dense_width});
  nn::Tensor h     = nn::relu(nn::conv2d(features, first, valid) + p.at("fc6/b"));
  const auto second = nn::reshape(p.at("fc7/W"), {1, 1, cfg.dense_width, cfg.dense_width});
  h                 = nn::relu(nn::conv2d(h, second, valid) + p.at("fc7/b"));
  const auto last   = nn::reshape(p.at("fc8/W"), {1, 1, cfg.dense_width, cfg.classes});
  return nn::conv2d(h, last, valid) + p.at("fc8/b");
}

}  // namespace vgg
