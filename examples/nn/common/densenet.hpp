// DenseNet feature reuse by concatenation (port of tflow's common/nn/densenet.h),
// with optional bottlenecks and compression in transitions. A layer appends
// growth_rate new channels while preserving every previous channel. BN/ReLU
// precedes each convolution.
// https://arxiv.org/abs/1608.06993
// https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua
//
// Supports four-stage ImageNet BC layouts and three-stage CIFAR layouts.
// The initial width is twice the growth rate, as in the released code.
// Bottlenecks use 4k channels; transitions use BN/ReLU, bias-free 1x1 and
// 2x2 average pooling. The classifier follows final BN/ReLU and global mean.
// This straightforward eager model materializes concatenations; it does not
// use the authors' shared buffers or activation recomputation to save memory.
// He-uniform fan-in initialization and shared BN state are adaptations.
#pragma once

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/preact_resnet.hpp"

namespace densenet {

namespace nn = datamunge::nn;

struct DenseGrowthOutputs {
  nn::Tensor output, added;
};
inline DenseGrowthOutputs DenseGrowthLayer(nn::ParameterStore& p, nn::ParameterStore& state, const std::string& name,
                                           const nn::Tensor& input, std::int64_t growth, bool bottleneck = true,
                                           const preact::NormConfig& norm = {}) {
  const std::int64_t in = input.shape().back();
  if (in < 1 || growth < 1) throw std::invalid_argument("positive DenseNet channel counts required");
  nn::Tensor h = preact::PreActivate(p, state, name + "/bn1", input, norm);
  if (bottleneck) {
    h = preact::RawResidualConv(p, name + "/conv1", h, 4 * growth, 1);
    h = preact::PreActivate(p, state, name + "/bn2", h, norm);
  }
  DenseGrowthOutputs out;
  out.added  = preact::RawResidualConv(p, name + "/conv2", h, growth, 3);
  out.output = nn::concat({input, out.added}, 3);
  return out;
}

inline nn::Tensor DenseTransition(nn::ParameterStore& p, nn::ParameterStore& state, const std::string& name,
                                  const nn::Tensor& input, std::int64_t out, const preact::NormConfig& norm = {}) {
  const std::int64_t in = input.shape().back();
  if (in < 1 || out < 1 || out > in) throw std::invalid_argument("DenseNet compression must retain between one and all channels");
  nn::Tensor h = preact::PreActivate(p, state, name + "/bn", input, norm);
  h            = preact::RawResidualConv(p, name + "/conv", h, out, 1);
  return nn::avg_pool2d(h, 2, 2, nn::Padding::Valid);
}

struct DenseNetConfig {
  int    depth = 121, growth_rate = 32, classes = 1000;
  bool   cifar_layout = false, bottleneck = true;
  double compression = .5, momentum = .9;
  bool   update_stats = true;
  std::vector<int> BlockCounts() const {
    if (cifar_layout) {
      const int divisor = bottleneck ? 6 : 3;
      if (depth <= 4 || (depth - 4) % divisor)
        throw std::invalid_argument("CIFAR DenseNet depth must be 6n+4 (bottleneck) or 3n+4");
      return std::vector<int>(3, (depth - 4) / divisor);
    }
    if (!bottleneck) throw std::invalid_argument("ImageNet presets use bottleneck DenseNet");
    switch (depth) {
      case 121: return {6, 12, 24, 16};
      case 169: return {6, 12, 32, 32};
      case 201: return {6, 12, 48, 32};
      case 161: return {6, 12, 36, 24};
      default: throw std::invalid_argument("ImageNet DenseNet depth must be 121/169/201/161");
    }
  }
  void Validate() const {
    BlockCounts();
    if (growth_rate < 1 || classes < 2 || !(compression > 0 && compression <= 1) || !(momentum >= 0 && momentum < 1))
      throw std::invalid_argument("invalid DenseNet configuration");
  }
};
struct DenseNetOutputs {
  nn::Tensor              stem, features, logits;
  std::vector<nn::Tensor> stages, transitions, layers;
};
inline DenseNetOutputs DenseNet(nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& image,
                                const DenseNetConfig& cfg = {}, bool training = false) {
  cfg.Validate();
  DenseNetOutputs          out;
  const preact::NormConfig norm{training, cfg.update_stats, cfg.momentum, 1e-5};
  std::int64_t             channels = 2 * cfg.growth_rate;
  nn::Tensor h = preact::RawResidualConv(p, "stem", image, channels, cfg.cifar_layout ? 3 : 7, cfg.cifar_layout ? 1 : 2);
  if (!cfg.cifar_layout) {
    h = preact::PreActivate(p, state, "stem_bn", h, norm);
    h = nn::max_pool2d(nn::pad(h, {{0, 0}, {1, 1}, {1, 1}, {0, 0}}, nn::PadMode::Constant,
                               -std::numeric_limits<double>::infinity()),
                       3, 2, nn::Padding::Valid);
  }
  out.stem          = h;
  const auto counts = cfg.BlockCounts();
  for (std::size_t stage = 0; stage < counts.size(); ++stage) {
    for (int layer = 0; layer < counts[stage]; ++layer) {
      h = DenseGrowthLayer(p, state, "stage" + std::to_string(stage + 1) + "/layer" + std::to_string(layer + 1), h,
                           cfg.growth_rate, cfg.bottleneck, norm)
              .output;
      channels += cfg.growth_rate;
      out.layers.push_back(h);
    }
    out.stages.push_back(h);
    if (stage + 1 < counts.size()) {
      const auto compressed = static_cast<std::int64_t>(std::floor(double(channels) * cfg.compression));
      h = DenseTransition(p, state, "transition" + std::to_string(stage + 1), h, compressed, norm);
      channels = compressed;
      out.transitions.push_back(h);
    }
  }
  out.features = preact::PreActivate(p, state, "final_bn", h, norm);
  out.logits   = nn::dense(p, "output", nn::global_avg_pool2d(out.features), cfg.classes);
  return out;
}

}  // namespace densenet
