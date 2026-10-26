// Res2Net bottlenecks following the authors' original (not v1b) release
// (port of tflow's common/nn/res2net.h).
// https://arxiv.org/abs/1904.01169
// https://github.com/Res2Net/Res2Net-PretrainedModels/blob/master/res2net.py
//
// Reduce to scale * floor(planes * base_width/64) channels, then split.
// Ordinary blocks process the first scale-1 groups hierarchically: each
// group adds the preceding processed group before its 3x3. The last group
// bypasses these convolutions. First blocks of EVERY stage instead process
// groups independently and average-pool the bypass group (including zeros
// in the padded 3x3 window), even when their stride is one. Scale=1 uses a
// single 3x3. Concatenate, expand, add the shortcut, then ReLU.
// Convolutions use bias-free He-uniform fan-in (an initialization adaptation),
// learned BN gamma/beta and symmetric padding. Optional small stem replaces
// 7x7/2 + max-pool by 3x3/1 for compact CIFAR training.
#pragma once

#include <datamunge/nn/nn.hpp>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/resnet.hpp"

namespace res2net {

namespace nn = datamunge::nn;

struct Res2ScaleOutputs {
  nn::Tensor              mixed;
  std::vector<nn::Tensor> groups;
};
/// The multi-scale 3x3 stage of one bottleneck: `input` has width*scale channels.
inline Res2ScaleOutputs Res2ScaleTransform(nn::ParameterStore& p, nn::ParameterStore& state, const std::string& name,
                                           const nn::Tensor& input, std::int64_t width, int scale,
                                           const resnet::ConvNormConfig& cfg, bool stage = false, int stride = 1) {
  if (width < 1 || scale < 1 || (stride != 1 && stride != 2) || (!stage && stride != 1) ||
      input.shape().back() != width * scale)
    throw std::invalid_argument("invalid Res2Net scale transform");
  const std::vector<nn::Tensor> split = scale == 1 ? std::vector<nn::Tensor>{input} : nn::split(input, scale, 3);
  Res2ScaleOutputs              out;
  const int                     processed = scale == 1 ? 1 : scale - 1;
  for (int group = 0; group < processed; ++group) {
    nn::Tensor h = split[static_cast<std::size_t>(group)];
    if (group > 0 && !stage) h = h + out.groups.back();
    h = resnet::ResNetConv(p, state, name + "/group" + std::to_string(group), h, width, 3, cfg, stride);
    out.groups.push_back(h);
  }
  if (scale > 1) {
    nn::Tensor h = split[static_cast<std::size_t>(scale - 1)];
    if (stage) h = nn::avg_pool2d(nn::pad(h, {{0, 0}, {1, 1}, {1, 1}, {0, 0}}), 3, stride, nn::Padding::Valid);
    out.groups.push_back(h);
  }
  out.mixed = out.groups.size() == 1 ? out.groups[0] : nn::concat(out.groups, 3);
  return out;
}

inline resnet::ResNetBlockOutputs Res2NetBlock(nn::ParameterStore& p, nn::ParameterStore& state, const std::string& name,
                                               const nn::Tensor& input, std::int64_t planes, std::int64_t base_width,
                                               int scale, const resnet::ConvNormConfig& cfg, bool stage = false,
                                               int stride = 1) {
  const std::int64_t in = input.shape().back(), width = planes * base_width / 64, middle = width * scale, out = 4 * planes;
  if (in < 1 || planes < 1 || base_width < 1 || width < 1 || scale < 1) throw std::invalid_argument("invalid Res2Net bottleneck");
  auto h = resnet::ResNetConv(p, state, name + "/conv1", input, middle, 1, cfg);
  h      = Res2ScaleTransform(p, state, name + "/scales", h, width, scale, cfg, stage, stride).mixed;
  resnet::ResNetBlockOutputs result;
  result.residual = resnet::ResNetConv(p, state, name + "/conv3", h, out, 1, cfg, 1, false);
  result.shortcut = in == out && stride == 1 ? input
                                             : resnet::ResNetConv(p, state, name + "/shortcut", input, out, 1, cfg, stride, false);
  result.preactivation = result.shortcut + result.residual;
  result.output        = nn::relu(result.preactivation);
  return result;
}

struct Res2NetConfig {
  int    depth = 50, base_width = 26, scale = 4, stage_width = 64, classes = 1000;
  bool   small_stem   = false;
  double momentum     = .9;
  bool   update_stats = true;  // false leaves the moving averages untouched in training mode
  void   Validate() const {
    if ((depth != 50 && depth != 101 && depth != 152) || base_width < 1 || scale < 1 || stage_width < 1 ||
        base_width * stage_width < 64 || classes < 2 || !(momentum >= 0 && momentum < 1))
      throw std::invalid_argument("invalid Res2Net configuration");
  }
};
inline resnet::ResNetOutputs Res2Net(nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& image,
                                     const Res2NetConfig& cfg = {}, bool training = false) {
  cfg.Validate();
  resnet::ResNetOutputs  out;
  const auto             counts = resnet::ResNetBlockCounts(cfg.depth);
  resnet::ConvNormConfig norm;
  norm.training     = training;
  norm.scale        = true;
  norm.epsilon      = 1e-5;
  norm.momentum     = cfg.momentum;
  norm.update_stats = cfg.update_stats;
  auto h = resnet::ResNetConv(p, state, "stem", image, cfg.stage_width, cfg.small_stem ? 3 : 7, norm, cfg.small_stem ? 1 : 2);
  if (!cfg.small_stem)
    h = nn::max_pool2d(nn::pad(h, {{0, 0}, {1, 1}, {1, 1}, {0, 0}}, nn::PadMode::Constant,
                               -std::numeric_limits<double>::infinity()),
                       3, 2, nn::Padding::Valid);
  out.stem = h;
  for (int stage = 0; stage < 4; ++stage) {
    const std::int64_t planes = std::int64_t(cfg.stage_width) << stage;
    for (int block = 0; block < counts[static_cast<std::size_t>(stage)]; ++block) {
      h = Res2NetBlock(p, state, "stage" + std::to_string(stage + 1) + "/block" + std::to_string(block + 1), h, planes,
                       cfg.base_width, cfg.scale, norm, block == 0, stage > 0 && block == 0 ? 2 : 1)
              .output;
      out.blocks.push_back(h);
    }
    out.stages[static_cast<std::size_t>(stage)] = h;
  }
  out.logits = nn::dense(p, "output", nn::global_avg_pool2d(h), cfg.classes);
  return out;
}

}  // namespace res2net
