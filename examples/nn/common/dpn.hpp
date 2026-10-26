// Dual path networks (port of tflow's common/nn/dpn.h): one shared bottleneck
// feeds a residual path (summed at fixed width, ResNet-like feature re-use)
// and a dense path (concatenated, growing by inc channels per unit,
// DenseNet-like feature exploration).
// https://arxiv.org/abs/1707.01629
// https://github.com/cypw/DPNs
// https://github.com/rwightman/pytorch-dpn-pretrained/blob/master/dpn.py
//
// Units are pre-activation: BN/ReLU/1x1, BN/ReLU/grouped 3x3, BN/ReLU/1x1 to
// width+inc channels. The first width channels are added to the residual
// path, the last inc channels are appended to the dense path. The first unit
// of each stage projects the concatenated input with its own BN/ReLU/1x1
// (stride two after stage one, as is the grouped 3x3) to width+2*inc
// channels, which starts both paths. The network concatenates both paths,
// then applies BN/ReLU, global average pooling and a biased linear classifier.
// Stage widths are 256/512/1024/2048 (64/128/256/512 in the small DPN-68
// layout) and bottleneck widths k_r*width/256 (k_r*width/64 for DPN-68).
// BN uses epsilon 1e-3 as in the released models. Convolutions are bias-free
// with He-uniform fan-in initialization (an adaptation). width_divisor and
// groups shrink models; small_stem replaces the strided stem and max-pool by
// a 3x3 stride-one convolution for CIFAR.
//
// Deviation: the grouped 3x3 is one nn::conv2d with Conv2DOptions::groups and
// a single filter "<name>/W" [3, 3, in/groups, out] instead of tflow's
// per-group "<name>/group<g>/W" filters (same parameter count and fan-in).
#pragma once

#include <array>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/preact_resnet.hpp"
#include "common/resnet.hpp"

namespace dpn {

namespace nn = datamunge::nn;

struct DPNLayout {
  bool               small;
  int                init, k_r, groups;
  std::array<int, 4> blocks, inc;
};
inline DPNLayout DPNLayoutFor(int depth) {
  switch (depth) {
    case 68: return {true, 10, 128, 32, {3, 4, 12, 3}, {16, 32, 32, 64}};
    case 92: return {false, 64, 96, 32, {3, 4, 20, 3}, {16, 32, 24, 128}};
    case 98: return {false, 96, 160, 40, {3, 6, 20, 3}, {16, 32, 32, 128}};
    case 107: return {false, 128, 200, 50, {4, 8, 20, 3}, {20, 64, 64, 128}};
    case 131: return {false, 128, 160, 40, {4, 8, 28, 3}, {16, 32, 32, 128}};
    default: throw std::invalid_argument("supported DPN depths are 68/92/98/107/131");
  }
}

/// Both paths of a feature map. The stem output has no dense channels yet.
struct DPNPaths {
  nn::Tensor   residual, dense;
  std::int64_t residual_channels = 0, dense_channels = 0;
  std::int64_t channels() const { return residual_channels + dense_channels; }
};
inline nn::Tensor DPNJoin(const DPNPaths& x) { return x.dense_channels ? nn::concat({x.residual, x.dense}, 3) : x.residual; }
inline nn::Tensor ChannelSlice(const nn::Tensor& x, std::int64_t start, std::int64_t count) {
  return nn::slice_axis(x, 3, start, start + count);
}

/// BN(eps 1e-3)/ReLU then a bias-free (grouped) convolution "<name>/W"; BN is "<name>/bn".
inline nn::Tensor DPNPreActConv(nn::ParameterStore& p, nn::ParameterStore& state, const std::string& name,
                                const nn::Tensor& input, std::int64_t out, std::int64_t kernel,
                                const preact::NormConfig& norm, std::int64_t stride = 1, std::int64_t groups = 1) {
  preact::NormConfig bn = norm;
  bn.epsilon            = 1e-3;  // the released DPN BN epsilon, whatever the caller's default
  const nn::Tensor h = preact::PreActivate(p, state, name + "/bn", input, bn);
  return preact::RawResidualConv(p, name, h, out, kernel, stride, groups);
}

struct DPNBlockOutputs {
  DPNPaths   input, output;                 // input paths as split by the projection, if any
  nn::Tensor residual_update, dense_update;  // the unit's new features
};
inline DPNBlockOutputs DPNBlock(nn::ParameterStore& p, nn::ParameterStore& state, const std::string& name,
                                const DPNPaths& x, std::int64_t width, std::int64_t bottleneck, std::int64_t inc,
                                std::int64_t groups, bool project, int stride = 1, const preact::NormConfig& norm = {}) {
  if (width < 1 || bottleneck < 1 || inc < 1 || groups < 1 || (stride != 1 && stride != 2) || (!project && stride != 1) ||
      (!project && (x.residual_channels != width || !x.dense_channels)))
    throw std::invalid_argument("invalid DPN unit dimensions");
  const nn::Tensor joined = DPNJoin(x);
  DPNBlockOutputs  out;
  out.input = x;
  if (project) {
    const nn::Tensor projection = DPNPreActConv(p, state, name + "/projection", joined, width + 2 * inc, 1, norm, stride);
    out.input = {ChannelSlice(projection, 0, width), ChannelSlice(projection, width, 2 * inc), width, 2 * inc};
  }
  nn::Tensor h        = DPNPreActConv(p, state, name + "/conv1", joined, bottleneck, 1, norm);
  h                   = DPNPreActConv(p, state, name + "/conv2", h, bottleneck, 3, norm, stride, groups);
  h                   = DPNPreActConv(p, state, name + "/conv3", h, width + inc, 1, norm);
  out.residual_update = ChannelSlice(h, 0, width);
  out.dense_update    = ChannelSlice(h, width, inc);
  out.output          = {out.input.residual + out.residual_update, nn::concat({out.input.dense, out.dense_update}, 3),
                         width, out.input.dense_channels + inc};
  return out;
}

struct DPNConfig {
  int    depth = 92, classes = 1000, width_divisor = 1, groups = 0;  // groups=0 keeps the release's cardinality
  bool   small_stem   = false;
  double momentum     = .9;
  bool   update_stats = true;
};
struct DPNOutputs {
  nn::Tensor                                stem, features, logits;
  std::array<DPNPaths, 4>                   stages;
  std::vector<std::vector<DPNBlockOutputs>> blocks;  // per stage
};
inline DPNOutputs DPN(nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& image, const DPNConfig& cfg = {},
                      bool training = false) {
  const auto layout  = DPNLayoutFor(cfg.depth);
  const int  divisor = cfg.width_divisor, groups = cfg.groups ? cfg.groups : layout.groups, factor = layout.small ? 1 : 4;
  if (divisor < 1 || layout.init % divisor || groups < 1 || cfg.classes < 2 || !(cfg.momentum >= 0 && cfg.momentum < 1))
    throw std::invalid_argument("invalid DPN configuration");
  DPNOutputs               out;
  const preact::NormConfig norm{training, cfg.update_stats, cfg.momentum, 1e-3};
  resnet::ConvNormConfig   stem_norm;
  stem_norm.training     = training;
  stem_norm.scale        = true;
  stem_norm.epsilon      = 1e-3;
  stem_norm.momentum     = cfg.momentum;
  stem_norm.update_stats = cfg.update_stats;
  const std::int64_t init = layout.init / divisor;
  nn::Tensor         h    = resnet::ResNetConv(p, state, "stem", image, init, cfg.small_stem || layout.small ? 3 : 7,
                                               stem_norm, cfg.small_stem ? 1 : 2);
  if (!cfg.small_stem)
    h = nn::max_pool2d(nn::pad(h, {{0, 0}, {1, 1}, {1, 1}, {0, 0}}, nn::PadMode::Constant,
                               -std::numeric_limits<double>::infinity()),
                       3, 2, nn::Padding::Valid);
  out.stem   = h;
  DPNPaths x = {h, {}, init, 0};
  for (int stage = 0; stage < 4; ++stage) {
    const int full_width = (64 << stage) * factor;
    if (full_width % divisor || layout.inc[stage] % divisor)
      throw std::invalid_argument("DPN width divisor must divide every width");
    const std::int64_t width = full_width / divisor, inc = layout.inc[stage] / divisor;
    const std::int64_t bottleneck = layout.k_r * width / (64 * factor);
    out.blocks.emplace_back();
    for (int block = 0; block < layout.blocks[stage]; ++block) {
      const auto name = "stage" + std::to_string(stage + 1) + "/block" + std::to_string(block + 1);
      out.blocks.back().push_back(
          DPNBlock(p, state, name, x, width, bottleneck, inc, groups, block == 0, stage > 0 && block == 0 ? 2 : 1, norm));
      x = out.blocks.back().back().output;
    }
    out.stages[stage] = x;
  }
  out.features = preact::PreActivate(p, state, "final/bn", DPNJoin(x), norm);
  out.logits   = nn::dense(p, "output", nn::global_avg_pool2d(out.features), cfg.classes);
  return out;
}

}  // namespace dpn
