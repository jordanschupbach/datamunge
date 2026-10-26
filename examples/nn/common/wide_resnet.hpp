// Wide Residual Networks with two preactivated 3x3 convolutions per block
// (port of tflow's common/nn/wide_resnet.h).
// https://arxiv.org/abs/1605.07146
// https://github.com/szagoruyko/wide-residual-networks/blob/master/models/wide-resnet.lua
//
// The released CIFAR layout has a fixed 16-channel stem, three stages with
// 16k/32k/64k channels and (depth-4)/6 blocks each, final BN/ReLU, global mean
// and a classifier. Equal-width shortcuts preserve raw inputs; projections
// use the first preactivation. Inverted elementwise dropout is applied only
// between the second BN/ReLU and second convolution during training.
// No convolution biases. He-uniform replaces the release's He-normal init;
// normalization uses preact::Normalize (epsilon 1e-5). Dropout masks come
// from an nn::Generator passed by the caller (one stream for the model)
// instead of tflow's per-block seeds 17+100*stage+block.
#pragma once

#include <array>
#include <datamunge/nn/nn.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/preact_resnet.hpp"

namespace wide_resnet {

namespace nn = datamunge::nn;

struct WideBlockOutputs {
  nn::Tensor residual, shortcut, output, before_dropout, after_dropout;
};
inline WideBlockOutputs WideBasicBlock(nn::ParameterStore& p, nn::ParameterStore& state, const std::string& name,
                                       const nn::Tensor& input, std::int64_t out, int stride,
                                       const preact::NormConfig& norm, double dropout, nn::Generator& gen) {
  const std::int64_t in = input.shape().back();
  if (in < 1 || out < 1 || (stride != 1 && stride != 2) || !(dropout >= 0 && dropout < 1))
    throw std::invalid_argument("invalid wide residual block");
  WideBlockOutputs result;
  auto             h = preact::PreActivate(p, state, name + "/bn1", input, norm);
  result.shortcut    = in == out && stride == 1 ? input : preact::RawResidualConv(p, name + "/shortcut", h, out, 1, stride);
  h                  = preact::RawResidualConv(p, name + "/conv1", h, out, 3, stride);
  result.before_dropout = preact::PreActivate(p, state, name + "/bn2", h, norm);
  result.after_dropout  = norm.training && dropout > 0 ? nn::dropout(result.before_dropout, dropout, true, gen)
                                                       : result.before_dropout;
  result.residual = preact::RawResidualConv(p, name + "/conv2", result.after_dropout, out, 3);
  result.output   = result.shortcut + result.residual;
  return result;
}

struct WideResNetConfig {
  int    depth = 28, widen_factor = 10, classes = 10;
  double dropout = .3, momentum = .9;
  bool   update_stats = true;  // false leaves the moving averages untouched in training mode
  void   Validate() const {
    if (depth < 10 || (depth - 4) % 6 || widen_factor < 1 || classes < 2 || !(dropout >= 0 && dropout < 1) ||
        !(momentum >= 0 && momentum < 1))
      throw std::invalid_argument("wide residual depth must be 6n+4, with positive width and valid dropout");
  }
};
struct WideResNetOutputs {
  nn::Tensor                stem, features, logits;
  std::array<nn::Tensor, 3> stages;
  std::vector<nn::Tensor>   blocks;
};
inline WideResNetOutputs WideResNet(nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& image,
                                    const WideResNetConfig& cfg, bool training, nn::Generator& gen) {
  cfg.Validate();
  WideResNetOutputs        out;
  const preact::NormConfig norm{training, cfg.update_stats, cfg.momentum, 1e-5};
  nn::Tensor               h = preact::RawResidualConv(p, "stem", image, 16, 3);
  out.stem                   = h;
  for (int stage = 0; stage < 3; ++stage) {
    const std::int64_t channels = (std::int64_t(16) << stage) * cfg.widen_factor;
    for (int block = 0; block < (cfg.depth - 4) / 6; ++block) {
      h = WideBasicBlock(p, state, "stage" + std::to_string(stage + 1) + "/block" + std::to_string(block + 1), h, channels,
                         stage > 0 && block == 0 ? 2 : 1, norm, cfg.dropout, gen)
              .output;
      out.blocks.push_back(h);
    }
    out.stages[static_cast<std::size_t>(stage)] = h;
  }
  out.features = preact::PreActivate(p, state, "final_bn", h, norm);
  out.logits   = nn::dense(p, "output", nn::global_avg_pool2d(out.features), cfg.classes);
  return out;
}

}  // namespace wide_resnet
