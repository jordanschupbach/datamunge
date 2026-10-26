// Context encoder/GAN primitives and multi-channel partial convolution: a port
// of tflow's common/nn/inpainting.h (context_encoder, partial_inpainting).
// Parameters live in nn::ParameterStore (weights) plus a separate store for
// moving BN statistics, with tflow's names.
#pragma once

#include <cstdint>
#include <datamunge/nn/nn.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace inpaint {

namespace nn = datamunge::nn;

/// BN momentum .9, eps 1e-5 ("<name>/gamma|beta", state "<name>/moving_mean|moving_var").
inline nn::Tensor InpaintNorm(nn::ParameterStore& p, nn::ParameterStore& state, const std::string& name, const nn::Tensor& x,
                              bool training, bool update = true) {
  nn::BatchNormOptions o;
  o.momentum     = .9;
  o.eps          = 1e-5;
  o.update_stats = update;
  return nn::batch_norm(p, state, name, x, training, o);
}

/// 4x4 convolution, N(0, .02) weights, zero bias; stride 2 SAME (down) or stride 1 VALID.
inline nn::Tensor ContextConv(nn::ParameterStore& p, const std::string& name, const nn::Tensor& x, std::int64_t out, bool down) {
  const nn::Tensor w = p.get(name + "/W", {4, 4, x.size(3), out}, nn::init::normal(.02));
  const nn::Tensor b = p.get(name + "/b", {out}, nn::init::zeros());
  return nn::bias_add(nn::conv2d(x, w, nn::conv_options(down ? 2 : 1, down ? nn::Padding::Same : nn::Padding::Valid)), b);
}

/// 4x4 transposed convolution: 1x1 -> 4x4 VALID when `first`, else stride 2 SAME (x2).
inline nn::Tensor ContextUp(nn::ParameterStore& p, const std::string& name, const nn::Tensor& x, std::int64_t out, bool first) {
  const nn::Tensor w = p.get(name + "/W", {4, 4, out, x.size(3)}, nn::init::normal(.02));
  const nn::Tensor b = p.get(name + "/b", {out}, nn::init::zeros());
  const nn::Shape  shape{x.size(0), first ? 4 : x.size(1) * 2, first ? 4 : x.size(2) * 2, out};
  return nn::bias_add(
      nn::conv2d_transpose(x, w, shape, nn::conv_options(first ? 1 : 2, first ? nn::Padding::Valid : nn::Padding::Same)), b);
}

struct ContextConfig {
  int side = 128, width = 64, latent = 100;
  static ContextConfig Compact() { return {32, 8, 32}; }
  [[nodiscard]] std::vector<int> Encoder() const {
    if (side == 128) return {width, width, width * 2, width * 4, width * 8};
    if (side == 32) return {width, width, width * 2};
    throw std::invalid_argument("context encoder supports original128 or compact32");
  }
};

struct ContextResult {
  nn::Tensor              patch;
  std::vector<nn::Tensor> stages;
};

/// Encoder (LeakyReLU .2, BN after the first), 4x4 VALID bottleneck, learned
/// transposed decoder (BN/ReLU) and a tanh centre patch of half the side.
inline ContextResult ContextEncoder(nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& masked,
                                    const ContextConfig& config = {}, bool training = false, bool update = true) {
  const auto    widths = config.Encoder();
  ContextResult result;
  nn::Tensor    x = masked;
  for (std::size_t i = 0; i < widths.size(); ++i) {
    const auto name = "encode" + std::to_string(i + 1);
    x               = ContextConv(p, name, x, widths[i], true);
    if (i) x = InpaintNorm(p, state, name + "/bn", x, training, update);
    x = nn::leaky_relu(x, .2);
    result.stages.push_back(x);
  }
  x = ContextConv(p, "bottleneck", x, config.latent, false);
  x = nn::leaky_relu(InpaintNorm(p, state, "bottleneck/bn", x, training, update), .2);
  result.stages.push_back(x);
  std::int64_t in = widths.back();
  x = ContextUp(p, "decode0", x, in, true);
  x = nn::relu(InpaintNorm(p, state, "decode0/bn", x, training, update));
  result.stages.push_back(x);
  for (int i = int(widths.size()) - 2; i >= 1; --i) {
    const auto name = "decode" + std::to_string(int(widths.size()) - 1 - i);
    x = ContextUp(p, name, x, widths[std::size_t(i)], false);
    x = nn::relu(InpaintNorm(p, state, name + "/bn", x, training, update));
    result.stages.push_back(x);
  }
  result.patch = nn::tanh(ContextUp(p, "output", x, 3, false));
  result.stages.push_back(result.patch);
  return result;
}

struct ContextDiscriminatorResult {
  nn::Tensor              logits;
  std::vector<nn::Tensor> stages;
};

/// Patch adversary: stride-2 4x4 downs (BN after the first, LeakyReLU .2) and a 4x4 VALID scalar logit.
inline ContextDiscriminatorResult ContextDiscriminator(nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& patch,
                                                       const ContextConfig& config = {}, bool training = false,
                                                       bool update = true) {
  ContextDiscriminatorResult result;
  const int                  depth = config.side == 128 ? 4 : 2;
  nn::Tensor                 x     = patch;
  for (int i = 0; i < depth; ++i) {
    const auto name = "down" + std::to_string(i + 1);
    x               = ContextConv(p, name, x, config.width * (1 << i), true);
    if (i) x = InpaintNorm(p, state, name + "/bn", x, training, update);
    x = nn::leaky_relu(x, .2);
    result.stages.push_back(x);
  }
  result.logits = nn::reshape(ContextConv(p, "output", x, 1, false), {-1});
  result.stages.push_back(result.logits);
  return result;
}

/// Mean softplus(-logit) (real) or softplus(logit) (fake).
inline nn::Tensor ContextBCE(const nn::Tensor& logits, bool real) { return nn::mean(nn::softplus(real ? -logits : logits)); }

inline nn::Tensor ContextReconstruction(const nn::Tensor& patch, const nn::Tensor& truth) {
  return nn::mean(nn::squared_difference(patch, truth));
}

/// Observed pixels where valid > .5, the zero-padded centre patch elsewhere.
inline nn::Tensor ContextComposite(const nn::Tensor& masked, const nn::Tensor& patch, const nn::Tensor& valid, int side = 32) {
  const std::int64_t border = side / 4;
  const nn::Tensor   padded = nn::pad(patch, {{0, 0}, {border, border}, {border, border}, {0, 0}});
  return nn::where(nn::greater(valid, .5), masked, padded);
}

struct PartialConvResult {
  nn::Tensor value, mask, count;
};

/// Multi-channel partial convolution (Liu et al. 2018): renormalised by
/// k*k*in / (valid count), bias added, zero where the window saw nothing.
/// Each output channel receives an identical updated mask.
inline PartialConvResult PartialConv(const nn::Tensor& x, const nn::Tensor& mask, const nn::Tensor& weights, const nn::Tensor& bias,
                                     std::int64_t in, std::int64_t out, std::int64_t kernel, std::int64_t stride = 1) {
  const nn::Tensor fixed  = nn::stop_gradient(mask);
  const auto       opts   = nn::conv_options(stride, nn::Padding::Same);
  const nn::Tensor counts = nn::conv2d(fixed, nn::ones({kernel, kernel, in, out}, mask.dtype()), opts);
  const nn::Tensor valid  = nn::cast(nn::greater(counts, 0.0), x.dtype());
  const nn::Tensor ratio  = nn::div(double(kernel * kernel * in), nn::maximum(counts, nn::scalar_like(1.0, counts)));
  const nn::Tensor conv   = nn::conv2d(x * fixed, weights, opts);
  return {nn::bias_add(conv * ratio, bias) * valid, valid, counts};
}

}  // namespace inpaint
