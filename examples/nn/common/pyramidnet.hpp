// Additive pyramidal residual networks (PyramidNet), a port of tflow's
// common/nn/pyramidnet.h: unit widths grow by alpha/N per unit instead of
// doubling at each downsampling stage.
// https://arxiv.org/abs/1610.02915
// https://github.com/jhkim89/PyramidNet/blob/master/addpyramidnet.lua
//
// Follow the released Torch code. A running width starts at the stem width
// and gains alpha/N before every unit (N = total units); a unit's width is
// that value rounded half up, times four after a bottleneck. Units are
// BN/conv/BN/ReLU/conv/BN (basic) or BN/1x1/BN/ReLU/3x3/BN/ReLU/1x1/BN
// (bottleneck); the first BN has no ReLU and nothing follows the addition.
// Shortcuts are parameter-free: 2x2 average pooling when strided, then zero
// channels appended to reach the residual width. The CIFAR layout has a
// 3x3 16-channel stem (conv/BN, no ReLU) and three stages of (depth-2)/6
// basic or (depth-2)/9 bottleneck units; the ImageNet layout has a 7x7/2
// 64-channel stem with BN/ReLU/max-pool and ResNet stage counts (200 uses
// 3/24/36/3). Both end with BN/ReLU, global average pooling and a biased
// linear classifier. Convolutions are bias-free with He-uniform fan-in
// initialization (the release uses He-normal fan-out); BN epsilon is 1e-5.
#pragma once

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/preact_resnet.hpp"
#include "common/resnet.hpp"

namespace pyramidnet {

namespace nn = datamunge::nn;

/// Strided average pooling, then zero channels appended up to `out`.
inline nn::Tensor PyramidShortcut(const nn::Tensor& input, std::int64_t out, int stride) {
  const std::int64_t in = input.shape().back();
  if (out < in) throw std::invalid_argument("pyramidal shortcuts cannot remove channels");
  nn::Tensor h = stride == 2 ? nn::avg_pool2d(input, 2, 2, nn::Padding::Valid) : input;
  if (out == in) return h;
  return nn::pad(h, {{0, 0}, {0, 0}, {0, 0}, {0, out - in}});
}

inline resnet::ResNetBlockOutputs PyramidBlock(nn::ParameterStore& p, nn::ParameterStore& state, const std::string& name,
                                               const nn::Tensor& input, std::int64_t planes, bool bottleneck,
                                               int stride = 1, const preact::NormConfig& norm = {}) {
  const std::int64_t in = input.shape().back(), out = bottleneck ? 4 * planes : planes;
  if (in < 1 || planes < 1 || out < in || (stride != 1 && stride != 2)) throw std::invalid_argument("invalid pyramidal unit");
  auto bn = [&](const std::string& part, const nn::Tensor& x) {
    return preact::Normalize(p, state, name + "/" + part, x, norm);
  };
  nn::Tensor h = bn("bn1", input);
  if (bottleneck) {
    h = preact::RawResidualConv(p, name + "/conv1", h, planes, 1);
    h = preact::RawResidualConv(p, name + "/conv2", nn::relu(bn("bn2", h)), planes, 3, stride);
    h = preact::RawResidualConv(p, name + "/conv3", nn::relu(bn("bn3", h)), out, 1);
    h = bn("bn4", h);
  } else {
    h = preact::RawResidualConv(p, name + "/conv1", h, planes, 3, stride);
    h = preact::RawResidualConv(p, name + "/conv2", nn::relu(bn("bn2", h)), planes, 3);
    h = bn("bn3", h);
  }
  resnet::ResNetBlockOutputs result;
  result.residual      = h;
  result.shortcut      = PyramidShortcut(input, out, stride);
  result.output        = result.residual + result.shortcut;
  result.preactivation = result.output;  // no post-addition nonlinearity
  return result;
}

struct PyramidNetConfig {
  int    depth = 110, classes = 10;
  double alpha = 48;
  bool   bottleneck = false, imagenet = false;
  double momentum     = .9;
  bool   update_stats = true;
  std::vector<int> StageCounts() const {
    if (imagenet) {
      if (!bottleneck) throw std::invalid_argument("the released ImageNet PyramidNet uses bottlenecks");
      if (depth == 200) return {3, 24, 36, 3};
      if (depth < 50) throw std::invalid_argument("ImageNet PyramidNet depths are 50/101/152/200");
      const auto c = resnet::ResNetBlockCounts(depth);
      return {c[0], c[1], c[2], c[3]};
    }
    const int per = bottleneck ? 9 : 6;
    if (depth < per + 2 || (depth - 2) % per)
      throw std::invalid_argument("CIFAR PyramidNet depth must be 6n+2 (basic) or 9n+2 (bottleneck)");
    return std::vector<int>(3, (depth - 2) / per);
  }
  /// Released unit widths (before the bottleneck's fourfold expansion).
  std::vector<int> UnitWidths() const {
    const auto counts = StageCounts();
    int        units  = 0;
    for (int c : counts) units += c;
    if (!(alpha >= 0) || classes < 2 || !(momentum >= 0 && momentum < 1))
      throw std::invalid_argument("invalid PyramidNet configuration");
    std::vector<int> widths;
    double           running = imagenet ? 64 : 16;
    for (int i = 0; i < units; ++i) {
      running += alpha / units;
      widths.push_back(int(std::floor(running + .5)));
    }
    return widths;
  }
};
struct PyramidNetOutputs {
  nn::Tensor                              stem, features, logits;
  std::vector<nn::Tensor>                 stages;
  std::vector<resnet::ResNetBlockOutputs> blocks;
  std::vector<std::int64_t>               channels;  // output channels of every unit
};
inline PyramidNetOutputs PyramidNet(nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& image,
                                    const PyramidNetConfig& cfg = {}, bool training = false) {
  const auto               counts = cfg.StageCounts();
  const auto               widths = cfg.UnitWidths();
  PyramidNetOutputs        out;
  const preact::NormConfig norm{training, cfg.update_stats, cfg.momentum, 1e-5};
  resnet::ConvNormConfig   stem_norm;
  stem_norm.training     = training;
  stem_norm.scale        = true;
  stem_norm.epsilon      = 1e-5;
  stem_norm.momentum     = cfg.momentum;
  stem_norm.update_stats = cfg.update_stats;
  std::int64_t in        = cfg.imagenet ? 64 : 16;
  nn::Tensor   h = resnet::ResNetConv(p, state, "stem", image, in, cfg.imagenet ? 7 : 3, stem_norm, cfg.imagenet ? 2 : 1,
                                      cfg.imagenet);
  if (cfg.imagenet)
    h = nn::max_pool2d(nn::pad(h, {{0, 0}, {1, 1}, {1, 1}, {0, 0}}, nn::PadMode::Constant,
                               -std::numeric_limits<double>::infinity()),
                       3, 2, nn::Padding::Valid);
  out.stem         = h;
  std::size_t unit = 0;
  for (std::size_t stage = 0; stage < counts.size(); ++stage) {
    for (int block = 0; block < counts[stage]; ++block, ++unit) {
      const auto name = "stage" + std::to_string(stage + 1) + "/block" + std::to_string(block + 1);
      out.blocks.push_back(
          PyramidBlock(p, state, name, h, widths[unit], cfg.bottleneck, stage > 0 && block == 0 ? 2 : 1, norm));
      h  = out.blocks.back().output;
      in = std::int64_t(widths[unit]) * (cfg.bottleneck ? 4 : 1);
      out.channels.push_back(in);
    }
    out.stages.push_back(h);
  }
  out.features = nn::relu(preact::Normalize(p, state, "final/bn", h, norm));
  out.logits   = nn::dense(p, "output", nn::global_avg_pool2d(out.features), cfg.classes);
  return out;
}

}  // namespace pyramidnet
