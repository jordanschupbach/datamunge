// ResNeXt grouped bottlenecks: cardinality independent of depth and width
// (port of tflow's common/nn/resnext.h).
// https://arxiv.org/abs/1611.05431
// https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua
//
// Form C uses a dense 1x1 reduction, grouped 3x3 convolution and dense 1x1
// expansion. Group width is floor(stage_width * base_width / 64); output
// width is four times stage_width. Downsampling is in the 3x3 convolution,
// as in the authors' code. BN/ReLU follows the first two convolutions; final
// BN precedes shortcut addition and ReLU. Projection shortcuts have BN.
// Both ImageNet four-stage and CIFAR 9n+2 three-stage layouts are supported.
// He-uniform fan-in initialization replaces the release's normal fan-out.
//
// Deviation: the grouped 3x3 is one nn::conv2d with Conv2DOptions::groups and
// a single filter "<name>/conv2/W" [3, 3, middle/cardinality, middle] (group g
// owns output columns [g*middle/C, (g+1)*middle/C)) instead of tflow's
// per-group "<name>/conv2/group<g>/W" filters. Same parameters and fan-in.
#pragma once

#include <datamunge/nn/nn.hpp>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/preact_resnet.hpp"
#include "common/resnet.hpp"

namespace resnext {

namespace nn = datamunge::nn;

inline resnet::ResNetBlockOutputs ResNeXtBlock(nn::ParameterStore& p, nn::ParameterStore& state, const std::string& name,
                                               const nn::Tensor& input, std::int64_t planes, std::int64_t cardinality,
                                               std::int64_t base_width, const resnet::ConvNormConfig& cfg,
                                               int stride = 1) {
  const std::int64_t in = input.shape().back(), group_width = planes * base_width / 64, middle = group_width * cardinality;
  const std::int64_t out = 4 * planes;
  if (in < 1 || planes < 1 || cardinality < 1 || base_width < 1 || group_width < 1 || (stride != 1 && stride != 2))
    throw std::invalid_argument("invalid ResNeXt bottleneck dimensions");
  resnet::ResNetBlockOutputs result;
  auto h = resnet::ResNetConv(p, state, name + "/conv1", input, middle, 1, cfg);
  h      = preact::RawResidualConv(p, name + "/conv2", h, middle, 3, stride, cardinality);
  nn::BatchNormOptions bn;
  bn.momentum     = cfg.momentum;
  bn.eps          = cfg.epsilon;
  bn.scale        = cfg.scale;
  bn.update_stats = cfg.update_stats;
  h               = nn::relu(nn::batch_norm(p, state, name + "/conv2/bn", h, cfg.training, bn));
  result.residual = resnet::ResNetConv(p, state, name + "/conv3", h, out, 1, cfg, 1, false);
  result.shortcut = in == out && stride == 1 ? input
                                             : resnet::ResNetConv(p, state, name + "/shortcut", input, out, 1, cfg, stride, false);
  result.preactivation = result.shortcut + result.residual;
  result.output        = nn::relu(result.preactivation);
  return result;
}

struct ResNeXtConfig {
  int    depth = 50, cardinality = 32, base_width = 4, stage_width = 64, classes = 1000;
  bool   cifar_layout = false;
  double momentum     = .9;
  bool   update_stats = true;  // false leaves the moving averages untouched in training mode
  void   Validate() const {
    if (cifar_layout ? (depth < 11 || (depth - 2) % 9) : (depth != 50 && depth != 101 && depth != 152))
      throw std::invalid_argument("ResNeXt requires ImageNet depth 50/101/152 or CIFAR depth 9n+2");
    if (cardinality < 1 || base_width < 1 || stage_width < 1 || stage_width * base_width < 64 || classes < 2 ||
        !(momentum >= 0 && momentum < 1))
      throw std::invalid_argument("invalid ResNeXt configuration");
  }
};
struct ResNeXtOutputs {
  nn::Tensor              stem, logits;
  std::vector<nn::Tensor> stages, blocks;
};
inline ResNeXtOutputs ResNeXt(nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& image,
                              const ResNeXtConfig& cfg = {}, bool training = false) {
  cfg.Validate();
  ResNeXtOutputs         out;
  resnet::ConvNormConfig norm;
  norm.training     = training;
  norm.scale        = true;
  norm.epsilon      = 1e-5;
  norm.momentum     = cfg.momentum;
  norm.update_stats = cfg.update_stats;
  auto h = resnet::ResNetConv(p, state, "stem", image, cfg.stage_width, cfg.cifar_layout ? 3 : 7, norm,
                              cfg.cifar_layout ? 1 : 2);
  if (!cfg.cifar_layout)
    h = nn::max_pool2d(nn::pad(h, {{0, 0}, {1, 1}, {1, 1}, {0, 0}}, nn::PadMode::Constant,
                               -std::numeric_limits<double>::infinity()),
                       3, 2, nn::Padding::Valid);
  out.stem         = h;
  const int  r      = (cfg.depth - 2) / 9;
  const auto counts = cfg.cifar_layout ? std::array<int, 4>{r, r, r, 0} : resnet::ResNetBlockCounts(cfg.depth);
  for (int stage = 0; stage < (cfg.cifar_layout ? 3 : 4); ++stage) {
    const std::int64_t planes = std::int64_t(cfg.stage_width) << stage;
    for (int block = 0; block < counts[static_cast<std::size_t>(stage)]; ++block) {
      h = ResNeXtBlock(p, state, "stage" + std::to_string(stage + 1) + "/block" + std::to_string(block + 1), h, planes,
                       cfg.cardinality, cfg.base_width, norm, stage > 0 && block == 0 ? 2 : 1)
              .output;
      out.blocks.push_back(h);
    }
    out.stages.push_back(h);
  }
  out.logits = nn::dense(p, "output", nn::global_avg_pool2d(h), cfg.classes);
  return out;
}

}  // namespace resnext
