// GhostNet: ghost modules generate half their feature maps by cheap
// depthwise operations on the other half (port of tflow's common/nn/ghostnet.h).
// https://arxiv.org/abs/1911.11907
// https://github.com/huawei-noah/Efficient-AI-Backbones/blob/master/ghostnet_pytorch/ghostnet.py
//
// A ghost module computes ceil(out/2) "intrinsic" maps with a 1x1
// convolution/BN(/ReLU), derives as many "ghost" maps with a depthwise 3x3
// convolution/BN(/ReLU) of them, concatenates both and keeps the first out.
// A ghost bottleneck is ghost module (ReLU) -> depthwise kxk/BN when strided
// -> optional squeeze-and-excitation (reduced width MakeDivisible(mid/4, 4),
// ReLU, hard-sigmoid gate) -> ghost module (linear), plus the input or, when
// shape changes, a depthwise kxk/BN + 1x1/BN shortcut. The network is a
// 3x3/2 conv/BN/ReLU stem, sixteen bottlenecks, a 1x1 conv/BN/ReLU to 960,
// global pooling, a biased 1x1 to 1280 with ReLU, dropout(0.2) and the
// classifier. Widths are scaled and rounded to multiples of four; as in the
// release, convolutions are bias-free with symmetric padding and BN epsilon
// is 1e-5. He-uniform fan-in kernels are an initialization adaptation.
#pragma once

#include "common/mobilenet.hpp"

namespace ghostnet {

namespace nn = datamunge::nn;
using mobilenet::HardSigmoid;
using mobilenet::MakeDivisible;
using mobilenet::MobileContext;
using mobilenet::MobileNetOutputs;

/// Symmetric zero padding of H and W.
inline nn::Tensor PadSpatial(const nn::Tensor& x, std::int64_t pad) {
  return nn::pad(x, {{0, 0}, {pad, pad}, {pad, pad}, {0, 0}});
}
/// Bias-free convolution with symmetric padding, BN (eps 1e-5) and optional ReLU.
/// Depthwise layers use multiplier out/in.
inline nn::Tensor GhostConvBN(MobileContext& c, const std::string& name, nn::Tensor x, std::int64_t out,
                              std::int64_t kernel, std::int64_t stride, bool depthwise, bool relu) {
  const std::int64_t   in = x.shape()[3];
  nn::ConvLayerOptions o;
  o.bias    = false;
  o.init    = nn::init::he_uniform();
  o.stride  = stride;
  o.padding = nn::Padding::Valid;
  if (kernel > 1) x = PadSpatial(x, kernel / 2);
  nn::Tensor h = depthwise ? nn::depthwise_conv2d(c.p, name, x, out / in, kernel, o) : nn::conv2d(c.p, name, x, out, kernel, o);
  h = mobilenet::ContextBatchNorm(c, name + "/bn", h, 1e-5);
  c.layers.push_back({name, h.shape(), kernel * kernel * (depthwise ? out : in * out)});
  return relu ? nn::relu(h) : h;
}
struct GhostModuleOutputs {
  nn::Tensor intrinsic, ghost, output;
};
inline GhostModuleOutputs GhostModule(MobileContext& c, const std::string& name, const nn::Tensor& x, std::int64_t out,
                                      bool relu, int ratio = 2) {
  const std::int64_t init = (out + ratio - 1) / ratio;
  GhostModuleOutputs g;
  g.intrinsic = GhostConvBN(c, name + "/primary", x, init, 1, 1, false, relu);
  g.ghost     = GhostConvBN(c, name + "/cheap", g.intrinsic, init * (ratio - 1), 3, 1, true, relu);
  g.output    = nn::concat({g.intrinsic, g.ghost}, 3);
  if (init * ratio != out) g.output = nn::slice(g.output, {0, 0, 0, 0}, {-1, -1, -1, out});
  return g;
}
struct GhostBlock {
  int    kernel, expanded, out;
  double se;
  int    stride;
};
inline std::vector<GhostBlock> GhostNetBlocks() {
  return {{3, 16, 16, 0, 1},     {3, 48, 24, 0, 2},     {3, 72, 24, 0, 1},     {5, 72, 40, .25, 2},
          {5, 120, 40, .25, 1},  {3, 240, 80, 0, 2},    {3, 200, 80, 0, 1},    {3, 184, 80, 0, 1},
          {3, 184, 80, 0, 1},    {3, 480, 112, .25, 1}, {3, 672, 112, .25, 1}, {5, 672, 160, .25, 2},
          {5, 960, 160, 0, 1},   {5, 960, 160, .25, 1}, {5, 960, 160, 0, 1},   {5, 960, 160, .25, 1}};
}
inline nn::Tensor GhostBottleneck(MobileContext& c, const std::string& name, const nn::Tensor& x, std::int64_t mid,
                                  std::int64_t out, int kernel, int stride, double se_ratio) {
  const std::int64_t in = x.shape()[3];
  nn::Tensor         h  = GhostModule(c, name + "/ghost1", x, mid, true).output;
  if (stride > 1) h = GhostConvBN(c, name + "/depthwise", h, mid, kernel, stride, true, false);
  if (se_ratio > 0) {
    const int            reduced = MakeDivisible(mid * se_ratio, 4);
    nn::ConvLayerOptions o;
    o.init                    = nn::init::he_uniform();
    const nn::Tensor pooled   = nn::mean(h, {1, 2}, true);
    const nn::Tensor squeezed = nn::relu(nn::conv2d(c.p, name + "/se/reduce", pooled, reduced, 1, o));
    h = h * HardSigmoid(nn::conv2d(c.p, name + "/se/expand", squeezed, mid, 1, o));
    c.layers.push_back({name + "/se", {1, 1, 1, 1}, 2 * mid * reduced});
  }
  h = GhostModule(c, name + "/ghost2", h, out, false).output;
  nn::Tensor shortcut = x;
  if (in != out || stride != 1) {
    shortcut = GhostConvBN(c, name + "/shortcut/depthwise", x, in, kernel, stride, true, false);
    shortcut = GhostConvBN(c, name + "/shortcut/pointwise", shortcut, out, 1, 1, false, false);
  }
  return h + shortcut;
}
struct GhostNetConfig {
  double width   = 1;
  int    classes = 1000;
  bool   small_stem = false;
  double keep_probability = .8, momentum = .9;
  bool   update_stats = true;
};
/// Dropout masks draw from p.generator() (tflow: a seeded TF Dropout op).
inline MobileNetOutputs GhostNet(nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& image,
                                 const GhostNetConfig& cfg = {}, bool training = false) {
  if (!(cfg.width > 0) || cfg.classes < 2 || !(cfg.keep_probability > 0 && cfg.keep_probability <= 1))
    throw std::invalid_argument("invalid GhostNet configuration");
  MobileNetOutputs out;
  MobileContext    c{p, state, out.layers, training, cfg.momentum, cfg.update_stats};
  int              in = MakeDivisible(16 * cfg.width, 4);
  nn::Tensor       h  = GhostConvBN(c, "stem", image, in, 3, cfg.small_stem ? 1 : 2, false, true);
  int              index = 0;
  for (const auto& b : GhostNetBlocks()) {
    const int mid = MakeDivisible(b.expanded * cfg.width, 4), o = MakeDivisible(b.out * cfg.width, 4);
    h             = GhostBottleneck(c, "block" + std::to_string(++index), h, mid, o, b.kernel, b.stride, b.se);
    in            = o;
  }
  const int last = MakeDivisible(960 * cfg.width, 4);
  out.features   = GhostConvBN(c, "conv_last", h, last, 1, 1, false, true);
  nn::ConvLayerOptions head;
  head.init       = nn::init::he_uniform();
  head.activation = nn::Activation::Relu;
  nn::Tensor v = nn::reshape(nn::conv2d(p, "head", nn::mean(out.features, {1, 2}, true), 1280, 1, head), {-1, 1280});
  if (training && cfg.keep_probability < 1) v = nn::dropout(v, 1 - cfg.keep_probability, true, p.generator());
  out.logits          = nn::dense(p, "output", v, cfg.classes);
  out.classifier_macs = std::int64_t(last) * 1280 + 1280LL * cfg.classes;
  return out;
}

}  // namespace ghostnet
