// Post-activation ResNet basic and bottleneck blocks (port of tflow's
// common/nn/resnet.h and common/nn/normalized_conv.h), with projection
// shortcuts only when dimensions change. Full 18/34/50/101/152 layouts follow
// Table 1 of https://arxiv.org/abs/1512.03385. Symmetric spatial padding
// matches the original pixel alignment (an explicit-padding VALID conv here,
// tflow pads then convolves). Bottleneck downsampling defaults to the first
// 1x1; stride_on_3x3 selects the v1.5 variant.
//
// Convolutions omit bias and use He-uniform initialization, then batch norm
// (gamma/beta in the parameter store, moving mean/variance in the separate
// state store; eps 1e-5, configurable EMA momentum). A small-image stem (3x3
// stride 1, no pool) is an explicit CIFAR adaptation; all four stages and
// block counts remain. Dilated convolutions (output_stride 8/16) are not ported.
#pragma once

#include <array>
#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace resnet {

namespace nn = datamunge::nn;

struct ConvNormConfig {
  bool   training = false, scale = false;
  bool   update_stats = true;  // training mode: update the moving averages in `state`
  double momentum = .9997, epsilon = .001;
};

/// conv2d (no bias, He uniform) -> batch norm -> optional ReLU. `pad` adds
/// symmetric zero padding before a VALID convolution.
inline nn::Tensor NormalizedConv(nn::ParameterStore& p, nn::ParameterStore& state, const std::string& name,
                                 const nn::Tensor& input, std::int64_t out, std::int64_t kernel,
                                 const ConvNormConfig& cfg, std::int64_t stride, std::int64_t pad, bool relu) {
  const std::int64_t in = input.shape().back();
  const nn::Tensor   w  = p.get(name + "/W", {kernel, kernel, in, out}, nn::init::he_uniform());
  nn::Conv2DOptions  o  = nn::conv_options(stride, nn::Padding::Valid);
  if (pad > 0) {
    o.padding           = nn::Padding::Explicit;
    o.explicit_paddings = {pad, pad, pad, pad};
  }
  nn::BatchNormOptions bn;
  bn.momentum = cfg.momentum;
  bn.eps      = cfg.epsilon;
  bn.scale    = cfg.scale;
  bn.update_stats = cfg.update_stats;
  nn::Tensor y = nn::batch_norm(p, state, name + "/bn", nn::conv2d(input, w, o), cfg.training, bn);
  return relu ? nn::relu(y) : y;
}

inline std::array<int, 4> ResNetBlockCounts(int depth) {
  switch (depth) {
    case 18: return {2, 2, 2, 2};
    case 34: case 50: return {3, 4, 6, 3};
    case 101: return {3, 4, 23, 3};
    case 152: return {3, 8, 36, 3};
    default: throw std::invalid_argument("supported ResNet depths are 18/34/50/101/152");
  }
}

inline nn::Tensor ResNetConv(nn::ParameterStore& p, nn::ParameterStore& state, const std::string& name,
                             const nn::Tensor& input, std::int64_t out, std::int64_t kernel, const ConvNormConfig& cfg,
                             std::int64_t stride = 1, bool relu = true) {
  return NormalizedConv(p, state, name, input, out, kernel, cfg, stride, kernel / 2, relu);
}

struct ResNetBlockOutputs {
  nn::Tensor residual, shortcut, preactivation, output;
};
inline ResNetBlockOutputs ResNetBlock(nn::ParameterStore& p, nn::ParameterStore& state, const std::string& name,
                                      const nn::Tensor& input, std::int64_t width, const ConvNormConfig& cfg,
                                      int stride = 1, bool bottleneck = false, bool stride_on_3x3 = false) {
  const std::int64_t in = input.shape().back();
  if (in < 1 || width < 1 || (stride != 1 && stride != 2)) throw std::invalid_argument("invalid ResNet block dimensions");
  const std::int64_t out = width * (bottleneck ? 4 : 1);
  auto conv = [&](const std::string& part, const nn::Tensor& x, std::int64_t co, std::int64_t kernel, int step,
                  bool relu) { return ResNetConv(p, state, name + "/" + part, x, co, kernel, cfg, step, relu); };
  nn::Tensor h;
  if (bottleneck) {
    h = conv("conv1", input, width, 1, stride_on_3x3 ? 1 : stride, true);
    h = conv("conv2", h, width, 3, stride_on_3x3 ? stride : 1, true);
    h = conv("conv3", h, out, 1, 1, false);
  } else {
    h = conv("conv1", input, width, 3, stride, true);
    h = conv("conv2", h, out, 3, 1, false);
  }
  ResNetBlockOutputs result;
  result.residual      = h;
  result.shortcut      = in == out && stride == 1 ? input : conv("shortcut", input, out, 1, stride, false);
  result.preactivation = result.residual + result.shortcut;
  result.output        = nn::relu(result.preactivation);
  return result;
}

struct ResNetConfig {
  int    depth = 18, classes = 1000, width_divisor = 1;
  bool   small_stem = false, stride_on_3x3 = false;
  double momentum = .9;
  bool   include_classifier = true;  // false returns feature stages without dense parameters
  bool   update_stats       = true;  // false leaves the moving averages untouched in training mode
  void   Validate() const {
    ResNetBlockCounts(depth);
    if (classes < 2 || width_divisor < 1 || 64 % width_divisor || !(momentum >= 0 && momentum < 1))
      throw std::invalid_argument("invalid ResNet configuration");
  }
};
struct ResNetOutputs {
  nn::Tensor                stem, logits;
  std::array<nn::Tensor, 4> stages;
  std::vector<nn::Tensor>   blocks;
};

/// ResNet on NHWC images. `training` selects batch statistics (and updates
/// the moving averages in `state`) instead of the stored moving averages.
inline ResNetOutputs ResNet(nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& image,
                            const ResNetConfig& cfg = {}, bool training = false) {
  cfg.Validate();
  ResNetOutputs  out;
  ConvNormConfig norm;
  norm.training = training;
  norm.scale    = true;
  norm.momentum = cfg.momentum;
  norm.epsilon  = 1e-5;
  norm.update_stats = cfg.update_stats;
  auto h = ResNetConv(p, state, "stem", image, 64 / cfg.width_divisor, cfg.small_stem ? 3 : 7, norm, cfg.small_stem ? 1 : 2);
  if (!cfg.small_stem)
    h = nn::max_pool2d(nn::pad(h, {{0, 0}, {1, 1}, {1, 1}, {0, 0}}, nn::PadMode::Constant,
                               -std::numeric_limits<double>::infinity()),
                       3, 2, nn::Padding::Valid);
  out.stem              = h;
  const bool bottleneck = cfg.depth >= 50;
  const auto counts     = ResNetBlockCounts(cfg.depth);
  for (int stage = 0; stage < 4; ++stage) {
    const int          stage_stride = stage ? 2 : 1;
    const std::int64_t width        = (64 << stage) / cfg.width_divisor;
    for (int block = 0; block < counts[stage]; ++block) {
      const auto name = "stage" + std::to_string(stage + 1) + "/block" + std::to_string(block + 1);
      h = ResNetBlock(p, state, name, h, width, norm, block == 0 ? stage_stride : 1, bottleneck, cfg.stride_on_3x3)
              .output;
      out.blocks.push_back(h);
    }
    out.stages[stage] = h;
  }
  if (cfg.include_classifier) out.logits = nn::dense(p, "output", nn::global_avg_pool2d(h), cfg.classes);
  return out;
}

}  // namespace resnet
