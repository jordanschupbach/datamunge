// RepVGG: multi-branch training blocks re-parameterized into a plain stack of
// 3x3 convolutions for inference (port of tflow's common/nn/repvgg.h).
// https://arxiv.org/abs/2101.03697
// https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
//
// A training block sums a 3x3 convolution with BN, a 1x1 convolution with BN
// (same stride, no padding, so its taps align with the 3x3 centres) and,
// when width and resolution are kept, a BN of the input, then applies ReLU.
// Each conv/BN pair fuses into a biased convolution, W*gamma/sigma and
// beta - mu*gamma/sigma; the 1x1 kernel pads to 3x3 and the identity becomes
// a 3x3 kernel with a one at each channel's own centre tap (within its group),
// so the deployed block is one biased 3x3 convolution and ReLU. The network
// is a stride-two block to min(64, 64a) channels, four stages of 2-4-14-1 (A)
// or 4-6-16-1 (B) blocks at widths 64a, 128a, 256a and 512b, each starting
// with stride two, then global pooling and a linear classifier. g2/g4
// variants group the even-numbered blocks 2..26 (counting from the first block
// after the stem). Convolutions are bias-free with symmetric padding during
// training; BN epsilon is 1e-5. He-uniform fan-in kernels are tflow's
// initialization adaptation; small_stem makes the first block stride one.
// Grouped layers use grouped.hpp's per-group filters "<name>/group<g>/W" (as
// tflow), so fused kernels map group by group.
#pragma once

#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/grouped.hpp"

namespace repvgg {

namespace nn = datamunge::nn;

struct RepVGGModel {
  std::vector<int>    blocks;
  std::vector<double> width;
  int                 groups;
};
inline RepVGGModel RepVGGLayout(const std::string& name) {
  static const std::map<std::string, RepVGGModel> models = {
      {"A0", {{2, 4, 14, 1}, {.75, .75, .75, 2.5}, 1}},  {"A1", {{2, 4, 14, 1}, {1, 1, 1, 2.5}, 1}},
      {"A2", {{2, 4, 14, 1}, {1.5, 1.5, 1.5, 2.75}, 1}}, {"B0", {{4, 6, 16, 1}, {1, 1, 1, 2.5}, 1}},
      {"B1", {{4, 6, 16, 1}, {2, 2, 2, 4}, 1}},          {"B1g2", {{4, 6, 16, 1}, {2, 2, 2, 4}, 2}},
      {"B1g4", {{4, 6, 16, 1}, {2, 2, 2, 4}, 4}},        {"B2", {{4, 6, 16, 1}, {2.5, 2.5, 2.5, 5}, 1}},
      {"B2g4", {{4, 6, 16, 1}, {2.5, 2.5, 2.5, 5}, 4}},  {"B3", {{4, 6, 16, 1}, {3, 3, 3, 5}, 1}}};
  if (!models.count(name)) throw std::invalid_argument("unknown RepVGG model " + name);
  return models.at(name);
}

struct RepVGGBlockSpec {
  std::string name;
  int         in, out, stride, groups;
  bool        identity;
};
struct RepVGGOutputs {
  nn::Tensor                   features, logits;
  std::vector<RepVGGBlockSpec> blocks;
  std::vector<nn::Tensor>      outputs;  // every block's output
};

inline nn::BatchNormOptions RepVGGNorm(double momentum, bool update_stats) {
  nn::BatchNormOptions bn;
  bn.momentum     = momentum;
  bn.eps          = 1e-5;
  bn.update_stats = update_stats;
  return bn;
}

inline nn::Tensor RepVGGBranch(nn::ParameterStore& p, nn::ParameterStore& state, const std::string& name, const nn::Tensor& x,
                               int out, int kernel, int stride, int groups, bool training, double momentum,
                               bool update_stats = true) {
  nn::ConvLayerOptions cfg;
  cfg.bias    = false;
  cfg.init    = nn::init::he_uniform();
  cfg.stride  = stride;
  cfg.padding = nn::Padding::Valid;
  const nn::Tensor in =
      kernel > 1 ? nn::pad(x, {{0, 0}, {kernel / 2, kernel / 2}, {kernel / 2, kernel / 2}, {0, 0}}) : x;
  const nn::Tensor h = grouped::GroupedConv2D(p, name, in, out, kernel, groups, cfg);
  return nn::batch_norm(p, state, name + "/bn", h, training, RepVGGNorm(momentum, update_stats));
}

/// One block, in training form (three branches) or deployed form (one biased 3x3).
inline nn::Tensor RepVGGBlock(nn::ParameterStore& p, nn::ParameterStore& state, const RepVGGBlockSpec& b, const nn::Tensor& x,
                              bool deploy, bool training, double momentum = .9, bool update_stats = true) {
  nn::Tensor y;
  if (deploy) {
    nn::ConvLayerOptions conv;
    conv.init    = nn::init::he_uniform();
    conv.stride  = b.stride;
    conv.padding = nn::Padding::Valid;
    y = grouped::GroupedConv2D(p, b.name + "/reparam", nn::pad(x, {{0, 0}, {1, 1}, {1, 1}, {0, 0}}), b.out, 3, b.groups, conv);
  } else {
    y = RepVGGBranch(p, state, b.name + "/dense", x, b.out, 3, b.stride, b.groups, training, momentum, update_stats) +
        RepVGGBranch(p, state, b.name + "/pointwise", x, b.out, 1, b.stride, b.groups, training, momentum, update_stats);
    if (b.identity)
      y = y + nn::batch_norm(p, state, b.name + "/identity/bn", x, training, RepVGGNorm(momentum, update_stats));
  }
  return nn::relu(y);
}

struct RepVGGConfig {
  std::string model = "A0";
  int         classes = 1000;
  bool        small_stem = false, deploy = false;
  double      momentum = .9;
  bool        update_stats = true;
};

inline RepVGGOutputs RepVGG(nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& image,
                            const RepVGGConfig& cfg = {}, bool training = false) {
  const auto layout = RepVGGLayout(cfg.model);
  if (cfg.classes < 2 || (cfg.deploy && training)) throw std::invalid_argument("invalid RepVGG configuration");
  RepVGGOutputs out;
  auto          block = [&](const RepVGGBlockSpec& b, const nn::Tensor& x) {
    out.blocks.push_back(b);
    out.outputs.push_back(RepVGGBlock(p, state, b, x, cfg.deploy, training, cfg.momentum, cfg.update_stats));
    return out.outputs.back();
  };
  int        in = std::min(64, int(64 * layout.width[0]));
  nn::Tensor h  = block({"stage0", 3, in, cfg.small_stem ? 1 : 2, 1, false}, image);
  int        index = 1;
  for (int stage = 0; stage < 4; ++stage) {
    const int planes = int((64 << stage) * layout.width[stage]);
    for (int i = 0; i < layout.blocks[stage]; ++i, ++index) {
      const int stride = i ? 1 : 2, groups = layout.groups > 1 && index % 2 == 0 && index <= 26 ? layout.groups : 1;
      h = block({"stage" + std::to_string(stage + 1) + "/block" + std::to_string(i + 1), in, planes, stride, groups,
                 in == planes && stride == 1},
                h);
      in = planes;
    }
  }
  out.features = h;
  out.logits   = nn::dense(p, "output", nn::global_avg_pool2d(h), cfg.classes);
  return out;
}

/// Fused kernel ([3, 3, in/groups, out/groups] per group) and bias [out] of
/// one training block, from its parameters `p` and BN statistics `state`.
struct FusedBlock {
  std::vector<nn::Tensor> kernels;
  nn::Tensor              bias;
};
inline FusedBlock FuseRepVGGBlock(const RepVGGBlockSpec& b, const nn::ParameterStore& p, const nn::ParameterStore& state) {
  const int ci = b.in / b.groups, co = b.out / b.groups;
  auto      get = [&](const nn::ParameterStore& store, const std::string& n) {
    if (!store.contains(n)) throw std::runtime_error("missing RepVGG value " + n);
    return store.at(n).to_vector<float>();
  };
  struct Norm {
    std::vector<float> gamma, beta, mean, var;
  };
  auto norm = [&](const std::string& bn) {
    return Norm{get(p, bn + "/gamma"), get(p, bn + "/beta"), get(state, bn + "/moving_mean"), get(state, bn + "/moving_var")};
  };
  auto scale = [](const Norm& n, int c, double& shift) {
    const double t = n.gamma[c] / std::sqrt(n.var[c] + 1e-5);
    shift += n.beta[c] - n.mean[c] * t;
    return t;
  };
  const Norm dense = norm(b.name + "/dense/bn"), pointwise = norm(b.name + "/pointwise/bn");
  const Norm identity = b.identity ? norm(b.name + "/identity/bn") : Norm{};
  auto       weight   = [&](const std::string& branch, int g) {
    return get(p, b.name + "/" + branch + (b.groups > 1 ? "/group" + std::to_string(g) : "") + "/W");
  };
  FusedBlock         f;
  std::vector<float> bias(b.out);
  for (int g = 0; g < b.groups; ++g) {
    std::vector<float> k(std::size_t(9) * ci * co, 0.f);
    const auto         w3 = weight("dense", g), w1 = weight("pointwise", g);
    for (int o = 0; o < co; ++o) {
      const int    channel = g * co + o;
      double       shift   = 0;
      const double t3 = scale(dense, channel, shift), t1 = scale(pointwise, channel, shift);
      for (int i = 0; i < 9 * ci; ++i) k[std::size_t(i) * co + o] += float(w3[std::size_t(i) * co + o] * t3);
      for (int i = 0; i < ci; ++i) k[(std::size_t(4) * ci + i) * co + o] += float(w1[std::size_t(i) * co + o] * t1);
      if (b.identity) k[(std::size_t(4) * ci + o) * co + o] += float(scale(identity, channel, shift));
      bias[channel] = float(shift);
    }
    f.kernels.push_back(nn::from_vector(k, {3, 3, ci, co}));
  }
  f.bias = nn::from_vector(bias, {b.out});
  return f;
}

/// Copies fused kernels/biases of `blocks` (and the classifier, when present)
/// into a deployed network's store.
inline void DeployRepVGG(nn::ParameterStore& deployed, const std::vector<RepVGGBlockSpec>& blocks,
                         const nn::ParameterStore& p, const nn::ParameterStore& state) {
  nn::NoGradGuard guard;
  for (const auto& b : blocks) {
    const auto fused = FuseRepVGGBlock(b, p, state);
    const int  co    = b.out / b.groups;
    for (int g = 0; g < b.groups; ++g) {
      const auto prefix = b.name + "/reparam" + (b.groups > 1 ? "/group" + std::to_string(g) : "");
      deployed.set(prefix + "/W", fused.kernels[g]);
      deployed.set(prefix + "/b", nn::slice(fused.bias, {g * co}, {co}));
    }
  }
  for (const auto& name : {"output/W", "output/b"})
    if (p.contains(name)) deployed.set(name, p.at(name));
}

}  // namespace repvgg
