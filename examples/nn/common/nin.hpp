// Network in Network: spatial convolution followed by two shared per-location
// dense/ReLU layers implemented as 1x1 convolutions (port of tflow's
// common/nn/nin.h). Three such MLPconv blocks produce one final feature map
// per class; global averaging gives class logits.
// The full-width CIFAR-10 layout follows the authors' cuda-convnet definition,
// including max pooling after both early blocks and a final ReLU:
// https://github.com/mavenlin/cuda-convnet/blob/master/NIN/cifar-10_def
// Paper: https://arxiv.org/abs/1312.4400
//
// ReLU after every convolution, SAME spatial padding, 3x3 stride-2 SAME max
// pools, .5 dropout after each pool. He initialization, simple [-1,1] input
// normalization and the example's Adam training are explicit adaptations.
// Dropout masks come from the parameter store's generator.
#pragma once

#include <array>
#include <datamunge/nn/nn.hpp>
#include <stdexcept>
#include <string>

namespace nin {

namespace nn = datamunge::nn;

inline nn::Tensor MlpConv(nn::ParameterStore& p, const std::string& name, const nn::Tensor& input,
                          std::int64_t spatial_width, std::int64_t first_width, std::int64_t out, std::int64_t kernel) {
  nn::ConvLayerOptions options;
  options.activation = nn::Activation::Relu;
  options.init       = nn::init::he_uniform();
  auto h             = nn::conv2d(p, name + "/spatial", input, spatial_width, kernel, options);
  h                  = nn::conv2d(p, name + "/mix1", h, first_width, 1, options);
  return nn::conv2d(p, name + "/mix2", h, out, 1, options);
}

struct NinConfig {
  int    classes = 10, width_divisor = 1;
  double keep_probability = .5;
  void   Validate() const {
    if (classes < 2 || width_divisor <= 0 || 32 % width_divisor || !(keep_probability > 0 && keep_probability <= 1))
      throw std::invalid_argument("invalid NiN configuration");
  }
};

struct NinOutputs {
  std::array<nn::Tensor, 3> blocks;
  std::array<nn::Tensor, 2> pooling;
  nn::Tensor                logits;
};

inline NinOutputs NetworkInNetwork(nn::ParameterStore& p, const nn::Tensor& image, const NinConfig& cfg = {},
                                   bool training = false) {
  cfg.Validate();
  NinOutputs         out;
  const std::int64_t a = 192 / cfg.width_divisor, b = 160 / cfg.width_divisor, c = 96 / cfg.width_divisor;
  auto drop = [&](const nn::Tensor& x) { return training ? nn::dropout(x, 1 - cfg.keep_probability, true, p.generator()) : x; };
  out.blocks[0]  = MlpConv(p, "block1", image, a, b, c, 5);
  out.pooling[0] = nn::max_pool2d(out.blocks[0], 3, 2, nn::Padding::Same);
  out.blocks[1]  = MlpConv(p, "block2", drop(out.pooling[0]), a, a, a, 5);
  out.pooling[1] = nn::max_pool2d(out.blocks[1], 3, 2, nn::Padding::Same);
  out.blocks[2]  = MlpConv(p, "block3", drop(out.pooling[1]), a, a, cfg.classes, 3);
  out.logits     = nn::global_avg_pool2d(out.blocks[2]);
  return out;
}

}  // namespace nin
