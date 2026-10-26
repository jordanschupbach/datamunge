// Rectangular convolution followed by batch normalization and optional ReLU
// (port of tflow's common/nn/normalized_conv.h).
// Weights have no convolution bias: BN beta supplies the offset. Kernels are
// "<name>/W" [height, width, in, out] (He uniform); BN keeps gamma/beta in
// `p` ("<name>/bn/...") and its moving mean/variance in the separate `state`
// store. Training mode normalizes with batch statistics and (unless
// update_stats is off) updates the moving averages eagerly; tflow collects the
// equivalent update ops instead. Supports fixed gamma=1, as in TF-Slim's
// default Inception configuration. Dilation uses nn::conv2d's native atrous
// support (tflow expands the kernel because TF's CPU backprop rejects it).
// resnet.hpp keeps its own square, explicitly padded variant.
#pragma once

#include <datamunge/nn/nn.hpp>
#include <stdexcept>
#include <string>

namespace normalized_conv {

namespace nn = datamunge::nn;

struct ConvNormConfig {
  bool   training = false, scale = false;
  bool   update_stats = true;  // training mode: update the moving averages in `state`
  double momentum = .9997, epsilon = .001;
  int    dilation = 1;
};

/// "SAME"/"VALID" (TensorFlow padding names) to nn::Padding.
inline nn::Padding ParsePadding(const std::string& padding) {
  if (padding == "SAME") return nn::Padding::Same;
  if (padding == "VALID") return nn::Padding::Valid;
  throw std::invalid_argument("padding must be SAME or VALID");
}

inline nn::Tensor NormalizedConv(nn::ParameterStore& p, nn::ParameterStore& state, const std::string& name,
                                 const nn::Tensor& input, std::int64_t out, std::int64_t height, std::int64_t width,
                                 const ConvNormConfig& cfg = {}, std::int64_t stride = 1,
                                 const std::string& padding = "SAME", bool relu = true) {
  const std::int64_t in = input.dim() == 4 ? input.shape()[3] : 0;
  if (in < 1 || out < 1 || height < 1 || width < 1 || stride < 1 || cfg.dilation < 1 ||
      (cfg.dilation > 1 && (height > 1 || width > 1) && stride != 1))
    throw std::invalid_argument("positive normalized-convolution dimensions required");
  const nn::Tensor  kernel = p.get(name + "/W", {height, width, in, out}, nn::init::he_uniform());
  nn::Conv2DOptions o;
  o.strides   = {stride, stride};
  o.dilations = {cfg.dilation, cfg.dilation};
  o.padding   = ParsePadding(padding);
  nn::BatchNormOptions bn;
  bn.momentum     = cfg.momentum;
  bn.eps          = cfg.epsilon;
  bn.scale        = cfg.scale;
  bn.update_stats = cfg.update_stats;
  const nn::Tensor y = nn::batch_norm(p, state, name + "/bn", nn::conv2d(input, kernel, o), cfg.training, bn);
  return relu ? nn::relu(y) : y;
}

}  // namespace normalized_conv
