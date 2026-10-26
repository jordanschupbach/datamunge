// Full pre-activation residual units (port of tflow's common/nn/preact_resnet.h):
// BN/ReLU before each weight layer and no activation after addition.
// Equal-width shortcuts preserve the raw input; dimension-changing projections
// consume the first preactivation.
// https://arxiv.org/abs/1603.05027
// https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua
//
// The CIFAR bottleneck network follows the released 9n+2 layout: a raw 3x3
// stem, three stages at 32/16/8 pixels, final BN/ReLU, global mean and logits.
// Default widths are 16 in the stem and 64/128/256 after bottleneck expansion.
// Conv biases are absent; He-uniform fan-in initialization is an adaptation
// of the release's He-normal fan-out initialization. BN keeps gamma/beta in
// the parameter store and moving mean/variance in the separate state store,
// epsilon 1e-5. Convolutions pad symmetrically by kernel/2 and then run VALID
// (tflow's PadSpatial + VALID), so strided 3x3s keep the original alignment.
#pragma once

#include <array>
#include <datamunge/nn/nn.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/resnet.hpp"

namespace preact {

namespace nn = datamunge::nn;

/// Batch-norm settings shared by every unit of a model. update_stats=false
/// leaves the moving averages untouched in training mode (derivative checks).
struct NormConfig {
  bool   training = false, update_stats = true;
  double momentum = .9, epsilon = 1e-5;
};

/// The batch norm alone (no ReLU): "<name>/gamma|beta", state "<name>/moving_mean|var".
inline nn::Tensor Normalize(nn::ParameterStore& p, nn::ParameterStore& state, const std::string& name,
                            const nn::Tensor& input, const NormConfig& cfg) {
  nn::BatchNormOptions bn;
  bn.momentum     = cfg.momentum;
  bn.eps          = cfg.epsilon;
  bn.update_stats = cfg.update_stats;
  return nn::batch_norm(p, state, name, input, cfg.training, bn);
}

/// BN then ReLU (tflow PreActivate).
inline nn::Tensor PreActivate(nn::ParameterStore& p, nn::ParameterStore& state, const std::string& name,
                              const nn::Tensor& input, const NormConfig& cfg) {
  return nn::relu(Normalize(p, state, name, input, cfg));
}

/// Bias-free He-uniform convolution "<name>/W" with symmetric kernel/2 zero padding (tflow RawResidualConv).
inline nn::Tensor RawResidualConv(nn::ParameterStore& p, const std::string& name, const nn::Tensor& input,
                                  std::int64_t out, std::int64_t kernel, std::int64_t stride = 1,
                                  std::int64_t groups = 1) {
  const std::int64_t in = input.shape().back();
  if (in % groups || out % groups) throw std::invalid_argument("grouped convolution channels must divide by groups");
  const nn::Tensor  w = p.get(name + "/W", {kernel, kernel, in / groups, out}, nn::init::he_uniform());
  nn::Conv2DOptions o = nn::conv_options(stride, nn::Padding::Valid);
  o.groups            = groups;
  if (kernel > 1) {
    o.padding           = nn::Padding::Explicit;
    const std::int64_t pad = kernel / 2;
    o.explicit_paddings = {pad, pad, pad, pad};
  }
  return nn::conv2d(input, w, o);
}

/// Pre-activation bottleneck BN/ReLU/1x1 (stride), BN/ReLU/3x3, BN/ReLU/1x1 to 4*width.
inline resnet::ResNetBlockOutputs PreActBottleneck(nn::ParameterStore& p, nn::ParameterStore& state,
                                                   const std::string& name, const nn::Tensor& input,
                                                   std::int64_t width, int stride = 1, const NormConfig& cfg = {}) {
  const std::int64_t in = input.shape().back();
  if (in < 1 || width < 1 || (stride != 1 && stride != 2)) throw std::invalid_argument("invalid preactivation bottleneck");
  resnet::ResNetBlockOutputs result;
  const std::int64_t         out = width * 4;
  nn::Tensor                 h   = PreActivate(p, state, name + "/bn1", input, cfg);
  result.shortcut = in == out && stride == 1 ? input : RawResidualConv(p, name + "/shortcut", h, out, 1, stride);
  h               = RawResidualConv(p, name + "/conv1", h, width, 1, stride);
  h               = PreActivate(p, state, name + "/bn2", h, cfg);
  h               = RawResidualConv(p, name + "/conv2", h, width, 3);
  h               = PreActivate(p, state, name + "/bn3", h, cfg);
  result.residual = RawResidualConv(p, name + "/conv3", h, out, 1);
  result.output   = result.shortcut + result.residual;
  result.preactivation = result.output;  // no post-addition nonlinearity
  return result;
}

struct PreActResNetConfig {
  int    depth = 164, base_width = 16, classes = 10;
  double momentum     = .9;
  bool   update_stats = true;
  void   Validate() const {
    if (depth < 11 || (depth - 2) % 9 || base_width < 1 || classes < 2 || !(momentum >= 0 && momentum < 1))
      throw std::invalid_argument("preactivation CIFAR bottleneck depth must be 9n+2");
  }
};
struct PreActResNetOutputs {
  nn::Tensor                stem, features, logits;
  std::array<nn::Tensor, 3> stages;
  std::vector<nn::Tensor>   blocks;
};
inline PreActResNetOutputs PreActResNet(nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& image,
                                        const PreActResNetConfig& cfg = {}, bool training = false) {
  cfg.Validate();
  PreActResNetOutputs out;
  const NormConfig    norm{training, cfg.update_stats, cfg.momentum, 1e-5};
  const int           repetitions = (cfg.depth - 2) / 9;
  nn::Tensor          h           = RawResidualConv(p, "stem", image, cfg.base_width, 3);
  out.stem                        = h;
  for (int stage = 0; stage < 3; ++stage) {
    const std::int64_t width = std::int64_t(cfg.base_width) << stage;
    for (int block = 0; block < repetitions; ++block) {
      h = PreActBottleneck(p, state, "stage" + std::to_string(stage + 1) + "/block" + std::to_string(block + 1), h,
                           width, stage > 0 && block == 0 ? 2 : 1, norm)
              .output;
      out.blocks.push_back(h);
    }
    out.stages[stage] = h;
  }
  out.features = PreActivate(p, state, "final_bn", h, norm);
  out.logits   = nn::dense(p, "output", nn::global_avg_pool2d(out.features), cfg.classes);
  return out;
}

}  // namespace preact
