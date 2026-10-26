// Attention modules for convolutional networks and a ResNet/ResNeXt
// bottleneck backbone that hosts them: a port of tflow's
// common/nn/attention_cnn.h (SE, CBAM, ECA, deformable stages, grouped 3x3).
//
// Squeeze-and-excitation (https://arxiv.org/abs/1709.01507): global mean,
// 1x1 to C/r with bias and ReLU, 1x1 back with bias, logistic gate.
// CBAM (https://arxiv.org/abs/1807.06521, Jongchan/attention-module): a
// channel gate whose shared biased MLP (C -> C/r -> C) sums its outputs on the
// mean- and max-pooled features, then a spatial gate: a 7x7 convolution
// (2 -> 1, no bias, BN) on the channel-wise max and mean.
// ECA (https://arxiv.org/abs/1910.03151, BangguWu/ECANet): global mean, a
// bias-free 1-D convolution of odd size k across channels, logistic gate;
// the paper's adaptive k is the odd number nearest above |log2(C)/2 + 1/2|.
// The backbone is the post-activation bottleneck of resnet.hpp (BN epsilon
// 1e-5, projection shortcuts with BN), optionally grouped (ResNeXt widths
// floor(planes*base/64)*cardinality, one grouped nn::conv2d) and with SE,
// CBAM or ECA applied to the residual branch after its last BN, before the
// addition. Deformable 3x3 convolutions (deformable.hpp, bias-free, then
// BN/ReLU) can replace the 3x3 of stages deformable_from..4.
// Convolutions are bias-free with symmetric padding and He-uniform fan-in
// kernels (tflow's initialization adaptation).
//
// tflow's selective-kernel and non-local options are not built in here: pass
// `conv2` (replaces the 3x3 stage of a block, e.g. a selective-kernel unit)
// and `after_block` (runs on every block output, e.g. non-local blocks) hooks
// instead, from the example's own header.
#pragma once

#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/deformable.hpp"
#include "common/resnet.hpp"

namespace attention_cnn {

namespace nn = datamunge::nn;

struct GateOutputs {
  nn::Tensor gate, output;
};

inline GateOutputs SqueezeExcitationModule(nn::ParameterStore& p, const std::string& name, const nn::Tensor& x,
                                           std::int64_t reduction) {
  const std::int64_t   channels = x.shape()[3];
  nn::ConvLayerOptions cfg;
  cfg.init               = nn::init::he_uniform();
  const nn::Tensor pooled = nn::mean(x, {1, 2}, true);
  const nn::Tensor h      = nn::relu(nn::conv2d(p, name + "/reduce", pooled, channels / reduction, 1, cfg));
  GateOutputs      g;
  g.gate   = nn::sigmoid(nn::conv2d(p, name + "/expand", h, channels, 1, cfg));
  g.output = x * g.gate;
  return g;
}

struct CbamOutputs {
  nn::Tensor channel_gate, spatial_gate, output;
};
/// CBAM on x [N, H, W, C]. While `training` the spatial BN uses batch
/// statistics and (with `update_stats`) updates its moving averages in `state`.
inline CbamOutputs Cbam(nn::ParameterStore& p, nn::ParameterStore& state, const std::string& name, const nn::Tensor& x,
                        std::int64_t reduction, bool training, double momentum, bool update_stats = true) {
  const std::int64_t channels = x.shape()[3];
  auto               mlp      = [&](const nn::Tensor& v) {
    const nn::Tensor h = nn::relu(nn::dense(p, name + "/mlp1", v, channels / reduction));
    return nn::dense(p, name + "/mlp2", h, channels);
  };
  CbamOutputs      r;
  const nn::Tensor logits = mlp(nn::mean(x, {1, 2})) + mlp(nn::max(x, {1, 2}));
  r.channel_gate          = nn::reshape(nn::sigmoid(logits), {-1, 1, 1, channels});
  const nn::Tensor refined = x * r.channel_gate;
  const nn::Tensor pooled  = nn::concat({nn::max(refined, {3}, true), nn::mean(refined, {3}, true)}, 3);
  const nn::Tensor w       = p.get(name + "/spatial/W", {7, 7, 2, 1}, nn::init::he_uniform());
  nn::Conv2DOptions o      = nn::conv_options(1, nn::Padding::Explicit);
  o.explicit_paddings      = {3, 3, 3, 3};
  nn::BatchNormOptions bn;
  bn.momentum     = momentum;
  bn.eps          = 1e-5;
  bn.update_stats = update_stats;
  r.spatial_gate  = nn::sigmoid(nn::batch_norm(p, state, name + "/spatial/bn", nn::conv2d(pooled, w, o), training, bn));
  r.output        = refined * r.spatial_gate;
  return r;
}

/// The paper's adaptive kernel size for C channels (gamma 2, b 1).
inline int EcaKernel(int channels) {
  const int t = int(std::abs(std::log2(double(channels)) / 2 + .5));
  return t % 2 ? t : t + 1;
}
inline GateOutputs Eca(nn::ParameterStore& p, const std::string& name, const nn::Tensor& x, std::int64_t kernel) {
  if (kernel < 1 || kernel % 2 == 0) throw std::invalid_argument("ECA kernels are odd");
  const std::int64_t channels = x.shape()[3];
  // Channels as a length-C sequence: [N, 1, C, 1], convolved by a [1, k] kernel.
  const nn::Tensor  pooled = nn::reshape(nn::mean(x, {1, 2}), {-1, 1, channels, 1});
  const nn::Tensor  w      = p.get(name + "/W", {1, kernel, 1, 1}, nn::init::he_uniform());
  nn::Conv2DOptions o      = nn::conv_options(1, nn::Padding::Explicit);
  o.explicit_paddings      = {0, 0, kernel / 2, kernel / 2};
  GateOutputs g;
  g.gate   = nn::reshape(nn::sigmoid(nn::conv2d(pooled, w, o)), {-1, 1, 1, channels});
  g.output = x * g.gate;
  return g;
}

enum class ChannelAttention { kNone, kSqueezeExcitation, kCbam, kEca };

/// Replaces a block's 3x3 stage: (p, state, name, input, width, stride, training) -> output (after BN/ReLU).
using Conv2Hook = std::function<nn::Tensor(nn::ParameterStore&, nn::ParameterStore&, const std::string&,
                                           const nn::Tensor&, std::int64_t, int, bool)>;
/// Runs after a block: (p, state, block name, output, stage 0..3, block index, training) -> new output.
using AfterBlockHook = std::function<nn::Tensor(nn::ParameterStore&, nn::ParameterStore&, const std::string&,
                                                const nn::Tensor&, int, int, bool)>;

struct AttentionResNetConfig {
  int              depth = 50, classes = 1000, width_divisor = 1, cardinality = 1, base_width = 64, reduction = 16;
  ChannelAttention attention = ChannelAttention::kNone;
  bool             small_stem = false, stride_on_3x3 = true, adaptive_eca = false;
  int              eca_kernel = 3;
  // Deformable 3x3 convolutions in stages deformable_from..4 (0: none).
  int    deformable_from = 0;
  bool   modulated       = false;
  double momentum        = .9;
  bool   update_stats    = true;  // false: training mode leaves the moving averages untouched
  Conv2Hook      conv2;            // optional replacement of the 3x3 stage
  AfterBlockHook after_block;      // optional extra module after each block
};

struct AttentionResNetOutputs {
  nn::Tensor              stem, features, logits;
  std::vector<nn::Tensor> blocks;
};

inline AttentionResNetOutputs AttentionResNet(nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& image,
                                              const AttentionResNetConfig& cfg = {}, bool training = false) {
  const auto counts = resnet::ResNetBlockCounts(cfg.depth);
  if (cfg.depth < 50 || cfg.classes < 2 || cfg.width_divisor < 1 || cfg.cardinality < 1 || cfg.reduction < 1)
    throw std::invalid_argument("invalid attention ResNet configuration");
  AttentionResNetOutputs out;
  resnet::ConvNormConfig norm;
  norm.training     = training;
  norm.scale        = true;
  norm.epsilon      = 1e-5;
  norm.momentum     = cfg.momentum;
  norm.update_stats = cfg.update_stats;
  nn::BatchNormOptions bn;
  bn.momentum     = cfg.momentum;
  bn.eps          = 1e-5;
  bn.update_stats = cfg.update_stats;
  std::int64_t in = 64 / cfg.width_divisor;
  nn::Tensor   h  = resnet::ResNetConv(p, state, "stem", image, in, cfg.small_stem ? 3 : 7, norm, cfg.small_stem ? 1 : 2);
  if (!cfg.small_stem)
    h = nn::max_pool2d(nn::pad(h, {{0, 0}, {1, 1}, {1, 1}, {0, 0}}, nn::PadMode::Constant,
                               -std::numeric_limits<double>::infinity()),
                       3, 2, nn::Padding::Valid);
  out.stem = h;
  for (int stage = 0; stage < 4; ++stage) {
    const std::int64_t planes = (64 << stage) / cfg.width_divisor,
                       width  = cfg.cardinality == 1 ? planes : planes * cfg.base_width / 64 * cfg.cardinality,
                       outputs = 4 * planes;
    for (int block = 0; block < counts[stage]; ++block) {
      const auto name   = "stage" + std::to_string(stage + 1) + "/block" + std::to_string(block + 1);
      const int  stride = stage > 0 && block == 0 ? 2 : 1;
      nn::Tensor r = resnet::ResNetConv(p, state, name + "/conv1", h, width, 1, norm, cfg.stride_on_3x3 ? 1 : stride);
      if (cfg.conv2) {
        r = cfg.conv2(p, state, name, r, width, stride, training);
      } else if (cfg.deformable_from > 0 && stage + 1 >= cfg.deformable_from) {
        if (cfg.cardinality != 1) throw std::invalid_argument("deformable layers here are ungrouped");
        r = deformable::DeformableConv2D(p, name + "/conv2", r, width, 3, cfg.stride_on_3x3 ? stride : 1, cfg.modulated)
                .output;
        r = nn::relu(nn::batch_norm(p, state, name + "/conv2/bn", r, training, bn));
      } else if (cfg.cardinality == 1) {
        r = resnet::ResNetConv(p, state, name + "/conv2", r, width, 3, norm, cfg.stride_on_3x3 ? stride : 1);
      } else {
        // One grouped convolution (tflow splits into per-group "<name>/conv2/group<g>/W" filters;
        // the parameter count and He-uniform fan-in are the same).
        const nn::Tensor  w = p.get(name + "/conv2/W", {3, 3, width / cfg.cardinality, width}, nn::init::he_uniform());
        nn::Conv2DOptions o = nn::conv_options(stride, nn::Padding::Explicit);
        o.explicit_paddings = {1, 1, 1, 1};
        o.groups            = cfg.cardinality;
        r = nn::relu(nn::batch_norm(p, state, name + "/conv2/bn", nn::conv2d(r, w, o), training, bn));
      }
      r = resnet::ResNetConv(p, state, name + "/conv3", r, outputs, 1, norm, 1, false);
      switch (cfg.attention) {
        case ChannelAttention::kSqueezeExcitation: r = SqueezeExcitationModule(p, name + "/se", r, cfg.reduction).output; break;
        case ChannelAttention::kCbam:
          r = Cbam(p, state, name + "/cbam", r, cfg.reduction, training, cfg.momentum, cfg.update_stats).output;
          break;
        case ChannelAttention::kEca:
          r = Eca(p, name + "/eca", r, cfg.adaptive_eca ? EcaKernel(int(outputs)) : cfg.eca_kernel).output;
          break;
        default: break;
      }
      const nn::Tensor shortcut =
          in == outputs && stride == 1 ? h : resnet::ResNetConv(p, state, name + "/shortcut", h, outputs, 1, norm, stride, false);
      h  = nn::relu(shortcut + r);
      in = outputs;
      if (cfg.after_block) h = cfg.after_block(p, state, name, h, stage, block, training);
      out.blocks.push_back(h);
    }
  }
  out.features = h;
  out.logits   = nn::dense(p, "output", nn::global_avg_pool2d(h), cfg.classes);
  return out;
}

// ---- Selective kernel (added by job 1.5b; tflow attention_cnn.h's SelectiveKernel) ----
// Selective kernel (https://arxiv.org/abs/1903.06586): two grouped 3x3
// branches (dilation one and two, i.e. 3x3 and 5x5 receptive fields) with
// BN/ReLU, their sum pooled and reduced to max(C/r, minimum) channels by a
// bias-free 1x1 with BN/ReLU, then per-branch bias-free 1x1 back to C and a
// softmax across branches weighting the branch outputs. The grouped branches
// are one grouped (dilated) nn::conv2d each, "<name>/branch<d>/W" (tflow:
// per-group filters "<name>/branch<d>/group<g>/W" with the same fans/count).
struct SelectiveKernelOutputs {
  std::vector<nn::Tensor> branches, weights;  // weights [N, 1, 1, C] each
  nn::Tensor              output;
};
inline SelectiveKernelOutputs SelectiveKernel(nn::ParameterStore& p, nn::ParameterStore& state, const std::string& name,
                                              const nn::Tensor& x, std::int64_t channels, std::int64_t groups, int stride,
                                              std::int64_t reduction, std::int64_t minimum, bool training,
                                              double momentum, bool update_stats = true) {
  nn::BatchNormOptions bn;
  bn.momentum     = momentum;
  bn.eps          = 1e-5;
  bn.update_stats = update_stats;
  auto norm = [&](const std::string& part, const nn::Tensor& h) {
    return nn::relu(nn::batch_norm(p, state, name + "/" + part + "/bn", h, training, bn));
  };
  const std::int64_t     in = x.shape()[3];
  SelectiveKernelOutputs r;
  for (std::int64_t dilation : {1, 2}) {
    const auto        part = "branch" + std::to_string(dilation);
    const nn::Tensor  w    = p.get(name + "/" + part + "/W", {3, 3, in / groups, channels}, nn::init::he_uniform());
    nn::Conv2DOptions o    = nn::conv_options(stride, nn::Padding::Explicit, dilation);
    o.explicit_paddings    = {dilation, dilation, dilation, dilation};
    o.groups               = groups;
    r.branches.push_back(norm(part, nn::conv2d(x, w, o)));
  }
  const std::int64_t   reduced = std::max(channels / reduction, minimum);
  const nn::Tensor     fused   = r.branches[0] + r.branches[1];
  nn::ConvLayerOptions cfg;
  cfg.bias               = false;
  cfg.init               = nn::init::he_uniform();
  const nn::Tensor z     = norm("fc", nn::conv2d(p, name + "/fc", nn::mean(fused, {1, 2}, true), reduced, 1, cfg));
  const nn::Tensor a     = nn::conv2d(p, name + "/select0", z, channels, 1, cfg),
                   b     = nn::conv2d(p, name + "/select1", z, channels, 1, cfg);
  const nn::Tensor soft  = nn::softmax(nn::stack({a, b}, 4), 4);  // [N, 1, 1, C, 2]
  const auto       parts = nn::unstack(soft, 4);
  r.weights              = {parts[0], parts[1]};
  r.output               = r.branches[0] * parts[0] + r.branches[1] * parts[1];
  return r;
}
/// AttentionResNetConfig::conv2 hook making every block's 3x3 stage an SK unit
/// "<block>/sk" (groups = cardinality, minimum 32 channels as in SKNet).
inline Conv2Hook SelectiveKernelHook(int cardinality, int reduction, double momentum = .9, bool update_stats = true,
                                     std::int64_t minimum = 32) {
  return [=](nn::ParameterStore& p, nn::ParameterStore& state, const std::string& name, const nn::Tensor& x,
             std::int64_t width, int stride, bool training) {
    return SelectiveKernel(p, state, name + "/sk", x, width, cardinality, stride, reduction, minimum, training, momentum,
                           update_stats)
        .output;
  };
}

}  // namespace attention_cnn
