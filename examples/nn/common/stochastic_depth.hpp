// Stochastic depth / DropPath with independent whole-batch or per-row masks
// (port of tflow's common/nn/stochastic_depth.h).
// Original scaling: training b*F, inference p*F. Inverted scaling: training
// (b/p)*F, inference F. b is Bernoulli(p); endpoints never divide by zero.
// The linear schedule is p_l = 1 - l/L*(1-p_last), for l=1,...,L.
// https://arxiv.org/abs/1603.09382
// https://github.com/pytorch/vision/blob/main/torchvision/ops/stochastic_depth.py
//
// Every residual is computed and then masked; execution is not skipped.
// The CIFAR network uses the modern inverted convention and ordinary BN
// updates even for masked branches. It demonstrates regularization, not the
// original implementation's conditional execution or training-time speedup.
// The three-stage 6n+2 model uses bias-free convolutions, average-pool and
// zero-padded dimension-changing shortcuts, and post-addition ReLU.
// Masks come from an nn::Generator passed by the caller (one stream for the
// whole model) instead of tflow's per-block RandomUniform seeds 71+index.
#pragma once

#include <datamunge/nn/nn.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/resnet.hpp"

namespace stochastic_depth {

namespace nn = datamunge::nn;

enum class DropPathScaling { kOriginal, kInverted };

inline std::vector<float> SurvivalProbabilities(int blocks, float last) {
  if (blocks < 1 || !(last >= 0 && last <= 1)) throw std::invalid_argument("invalid survival probability schedule");
  std::vector<float> probabilities(static_cast<std::size_t>(blocks));
  for (int i = 0; i < blocks; ++i) probabilities[static_cast<std::size_t>(i)] = 1 - float(i + 1) / blocks * (1 - last);
  return probabilities;
}

struct DropPathOutputs {
  nn::Tensor value, gate, multiplier;
};
/// Masks `branch`: a scalar gate (whole batch) or a [batch, 1, ..., 1] gate (per example).
inline DropPathOutputs DropPath(const nn::Tensor& branch, double survival, bool training, bool per_example,
                                DropPathScaling scaling, nn::Generator& gen) {
  if (!(survival >= 0 && survival <= 1)) throw std::invalid_argument("survival probability must be in [0,1]");
  DropPathOutputs out;
  if (!training) {
    out.gate       = nn::scalar(1);
    out.multiplier = nn::scalar(scaling == DropPathScaling::kOriginal ? survival : 1.);
  } else {
    if (survival == 0 || survival == 1)
      out.gate = nn::scalar(survival);
    else {
      nn::Shape shape;
      if (per_example) {
        shape.assign(static_cast<std::size_t>(branch.dim()), 1);
        shape[0] = branch.shape()[0];
      }
      out.gate = nn::bernoulli(shape, survival, gen);
    }
    out.multiplier = scaling == DropPathScaling::kInverted && survival > 0 ? out.gate / survival : out.gate;
  }
  out.value = branch * out.multiplier;
  return out;
}

/// Option-A shortcut: 2x2 average pool when strided, zero channels appended when widened.
inline nn::Tensor CifarIdentityShortcut(const nn::Tensor& input, std::int64_t out, int stride) {
  const std::int64_t in = input.shape().back();
  if ((stride != 1 && stride != 2) || in < 1 || (out != in && out != 2 * in)) throw std::invalid_argument("invalid CIFAR option-A shortcut");
  nn::Tensor h = stride == 1 ? input : nn::avg_pool2d(input, 2, 2, nn::Padding::Valid);
  if (out != in) h = nn::concat({h, nn::zeros_like(h)}, 3);
  return h;
}

struct StochasticCifarConfig {
  int    depth = 110, base_width = 16, classes = 10;
  double last_survival = .5, momentum = .9;
  bool   per_example  = false;
  bool   update_stats = true;  // false leaves the moving averages untouched in training mode
  void   Validate() const {
    if (depth < 8 || (depth - 2) % 6 || base_width < 1 || classes < 2 || !(last_survival >= 0 && last_survival <= 1) ||
        !(momentum >= 0 && momentum < 1))
      throw std::invalid_argument("invalid 6n+2 stochastic CIFAR model");
  }
};
struct StochasticCifarOutputs {
  nn::Tensor              logits;
  std::vector<nn::Tensor> stages, gates;
};
inline StochasticCifarOutputs StochasticCifarResNet(nn::ParameterStore& p, nn::ParameterStore& state,
                                                    const nn::Tensor& image, const StochasticCifarConfig& cfg,
                                                    bool training, nn::Generator& gen) {
  cfg.Validate();
  StochasticCifarOutputs out;
  resnet::ConvNormConfig norm;
  norm.training     = training;
  norm.scale        = true;
  norm.epsilon      = 1e-5;
  norm.momentum     = cfg.momentum;
  norm.update_stats = cfg.update_stats;
  const int  repetitions = (cfg.depth - 2) / 6;
  const auto survival    = SurvivalProbabilities(3 * repetitions, float(cfg.last_survival));
  int        index       = 0;
  auto       h           = resnet::ResNetConv(p, state, "stem", image, cfg.base_width, 3, norm);
  for (int stage = 0; stage < 3; ++stage) {
    const std::int64_t channels = std::int64_t(cfg.base_width) << stage;
    for (int block = 0; block < repetitions; ++block, ++index) {
      const int  stride   = stage > 0 && block == 0 ? 2 : 1;
      const auto name     = "stage" + std::to_string(stage + 1) + "/block" + std::to_string(block + 1);
      const auto skip     = CifarIdentityShortcut(h, channels, stride);
      auto       residual = resnet::ResNetConv(p, state, name + "/conv1", h, channels, 3, norm, stride);
      residual            = resnet::ResNetConv(p, state, name + "/conv2", residual, channels, 3, norm, 1, false);
      const auto dropped  = DropPath(residual, survival[static_cast<std::size_t>(index)], training, cfg.per_example,
                                     DropPathScaling::kInverted, gen);
      h                   = nn::relu(skip + dropped.value);
      out.gates.push_back(dropped.gate);
    }
    out.stages.push_back(h);
  }
  out.logits = nn::dense(p, "output", nn::global_avg_pool2d(h), cfg.classes);
  return out;
}

}  // namespace stochastic_depth
