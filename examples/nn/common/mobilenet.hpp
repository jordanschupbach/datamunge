// Shared mobile-network blocks and MobileNet v1, v2 and v3 (port of tflow's
// common/nn/mobilenet.h).
// https://arxiv.org/abs/1704.04861
// https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py
//
// ConvBN is a bias-free convolution (dense or depthwise) with TensorFlow SAME
// padding, batch normalization (epsilon 1e-3, as in the TF releases) and an
// optional activation. Every layer records its output shape and multiply-adds
// per output position so callers can tabulate cost (tflow records the output
// tensor and measures its shape in a session; eagerly the shape is known).
// MobileNet v1 (Table 1) is a 3x3/2 convolution followed by thirteen
// depthwise 3x3 + pointwise 1x1 pairs, each followed by BN and ReLU6 (the
// released activation; the paper writes ReLU), global average pooling and a
// biased classifier. The width multiplier alpha scales every channel count
// (int(c*alpha)); full_convolutions replaces each pair by one dense 3x3
// convolution (the paper's Table 4 baseline); small_stem makes the first
// convolution stride one for CIFAR. Kernels are He-uniform fan-in (depthwise:
// TF's k*k fan), an initialization adaptation. BN gamma/beta live in the
// parameter store, moving statistics in the separate state store.
#pragma once

#include <algorithm>
#include <datamunge/nn/nn.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace mobilenet {

namespace nn = datamunge::nn;

struct MobileLayer {
  std::string  name;
  nn::Shape    shape;  // output shape [N, H, W, C]
  std::int64_t macs_per_position;
  std::int64_t Macs() const { return shape[1] * shape[2] * macs_per_position; }
};
struct MobileContext {
  nn::ParameterStore&       p;
  nn::ParameterStore&       state;
  std::vector<MobileLayer>& layers;
  bool                      training;
  double                    momentum     = .9;
  bool                      update_stats = true;  // training mode: update the BN moving averages
};
/// Batch norm with the context's mode, momentum and the given epsilon.
inline nn::Tensor ContextBatchNorm(MobileContext& c, const std::string& name, const nn::Tensor& x, double epsilon) {
  nn::BatchNormOptions bn;
  bn.momentum     = c.momentum;
  bn.eps          = epsilon;
  bn.update_stats = c.update_stats;
  return nn::batch_norm(c.p, c.state, name, x, c.training, bn);
}

enum class MobileConv { kDense, kDepthwise };
/// Convolution (+BN, +activation). Depthwise layers require out == in.
inline nn::Tensor ConvBN(MobileContext& c, const std::string& name, const nn::Tensor& x, std::int64_t out,
                         std::int64_t kernel, std::int64_t stride, nn::Activation act,
                         MobileConv kind = MobileConv::kDense, bool normalize = true) {
  const std::int64_t in = x.dim() == 4 ? x.shape()[3] : 0;
  if (in < 1 || out < 1 || kernel < 1 || stride < 1 || (kind == MobileConv::kDepthwise && out != in))
    throw std::invalid_argument("invalid mobile convolution");
  nn::ConvLayerOptions o;
  o.bias    = !normalize;
  o.init    = nn::init::he_uniform();
  o.stride  = stride;
  o.padding = nn::Padding::Same;
  nn::Tensor h = kind == MobileConv::kDepthwise ? nn::depthwise_conv2d(c.p, name, x, 1, kernel, o)
                                                : nn::conv2d(c.p, name, x, out, kernel, o);
  if (normalize) h = ContextBatchNorm(c, name + "/bn", h, 1e-3);
  h = nn::activate(h, act);
  c.layers.push_back({name, h.shape(), kernel * kernel * (kind == MobileConv::kDepthwise ? 1 : in) * out});
  return h;
}

struct MobileNetV1Config {
  double alpha = 1;
  int    classes = 1000;
  bool   small_stem = false, full_convolutions = false;
  double momentum = .9;
  bool   update_stats = true;
};
struct MobileNetOutputs {
  nn::Tensor               features, logits;
  std::vector<MobileLayer> layers;
  std::int64_t             classifier_macs = 0;
  /// Sum of every layer's multiply-adds plus the classifier.
  std::int64_t Macs() const {
    std::int64_t total = classifier_macs;
    for (const auto& l : layers) total += l.Macs();
    return total;
  }
};
inline MobileNetOutputs MobileNetV1(nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& image,
                                    const MobileNetV1Config& cfg = {}, bool training = false) {
  if (!(cfg.alpha > 0) || cfg.classes < 2) throw std::invalid_argument("invalid MobileNet v1 configuration");
  MobileNetOutputs out;
  MobileContext    c{p, state, out.layers, training, cfg.momentum, cfg.update_stats};
  const auto       width     = [&](int channels) { return std::max(1, int(channels * cfg.alpha)); };
  const int        layers[13][2] = {{64, 1},  {128, 2}, {128, 1}, {256, 2}, {256, 1},  {512, 2}, {512, 1},
                                    {512, 1}, {512, 1}, {512, 1}, {512, 1}, {1024, 2}, {1024, 1}};
  int        in = width(32);
  nn::Tensor h  = ConvBN(c, "conv0", image, in, 3, cfg.small_stem ? 1 : 2, nn::Activation::Relu6);
  for (int i = 0; i < 13; ++i) {
    const int  o = width(layers[i][0]), stride = layers[i][1];
    const auto name = "block" + std::to_string(i + 1);
    if (cfg.full_convolutions)
      h = ConvBN(c, name + "/conv", h, o, 3, stride, nn::Activation::Relu6);
    else {
      h = ConvBN(c, name + "/depthwise", h, in, 3, stride, nn::Activation::Relu6, MobileConv::kDepthwise);
      h = ConvBN(c, name + "/pointwise", h, o, 1, 1, nn::Activation::Relu6);
    }
    in = o;
  }
  out.features        = h;
  out.logits          = nn::dense(p, "output", nn::global_avg_pool2d(h), cfg.classes);
  out.classifier_macs = std::int64_t(in) * cfg.classes;
  return out;
}

// MobileNet v2 (https://arxiv.org/abs/1801.04381, torchvision/TF-slim
// releases): inverted residuals expand by t with a 1x1 (skipped when t=1),
// filter with a depthwise 3x3 (stride on the first repeat), and project with
// a linear 1x1 (BN, no activation), adding the input when shape is kept.
// Channels are alpha-scaled and rounded to multiples of eight (never down by
// more than 10%); the final 1x1 to 1280 channels only grows (alpha > 1). The
// head is global average pooling, dropout(0.2) in training and a classifier.
inline int MakeDivisible(double value, int divisor = 8) {
  int rounded = std::max(divisor, int(value + divisor / 2.) / divisor * divisor);
  return rounded < .9 * value ? rounded + divisor : rounded;
}
struct InvertedResidualOutputs {
  nn::Tensor expanded, filtered, projected, output;
};
inline InvertedResidualOutputs InvertedResidual(MobileContext& c, const std::string& name, const nn::Tensor& x,
                                                std::int64_t out, int expansion, int stride) {
  InvertedResidualOutputs r;
  const std::int64_t      in = x.shape()[3], hidden = in * expansion;
  r.expanded  = expansion == 1 ? x : ConvBN(c, name + "/expand", x, hidden, 1, 1, nn::Activation::Relu6);
  r.filtered  = ConvBN(c, name + "/depthwise", r.expanded, hidden, 3, stride, nn::Activation::Relu6, MobileConv::kDepthwise);
  r.projected = ConvBN(c, name + "/project", r.filtered, out, 1, 1, nn::Activation::None);
  r.output    = stride == 1 && in == out ? x + r.projected : r.projected;
  return r;
}
struct MobileNetV2Config {
  double alpha = 1;
  int    classes = 1000;
  bool   small_stem = false;
  double keep_probability = .8, momentum = .9;
  bool   update_stats = true;
};
/// Dropout masks draw from p.generator() (tflow: seeded TF Dropout ops).
inline MobileNetOutputs MobileNetV2(nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& image,
                                    const MobileNetV2Config& cfg = {}, bool training = false) {
  if (!(cfg.alpha > 0) || cfg.classes < 2 || !(cfg.keep_probability > 0 && cfg.keep_probability <= 1))
    throw std::invalid_argument("invalid MobileNet v2 configuration");
  MobileNetOutputs out;
  MobileContext    c{p, state, out.layers, training, cfg.momentum, cfg.update_stats};
  // (expansion t, channels c, repeats n, first stride s), Table 2.
  const int  stages[7][4] = {{1, 16, 1, 1}, {6, 24, 2, 2}, {6, 32, 3, 2}, {6, 64, 4, 2}, {6, 96, 3, 1}, {6, 160, 3, 2}, {6, 320, 1, 1}};
  int        in   = MakeDivisible(32 * cfg.alpha);
  const int  last = MakeDivisible(1280 * std::max(1., cfg.alpha));
  nn::Tensor h    = ConvBN(c, "conv0", image, in, 3, cfg.small_stem ? 1 : 2, nn::Activation::Relu6);
  int        block = 0;
  for (const auto& stage : stages) {
    const int o = MakeDivisible(stage[1] * cfg.alpha);
    for (int i = 0; i < stage[2]; ++i, in = o)
      h = InvertedResidual(c, "block" + std::to_string(++block), h, o, stage[0], i ? 1 : stage[3]).output;
  }
  out.features      = ConvBN(c, "conv_last", h, last, 1, 1, nn::Activation::Relu6);
  nn::Tensor pooled = nn::global_avg_pool2d(out.features);
  if (training && cfg.keep_probability < 1) pooled = nn::dropout(pooled, 1 - cfg.keep_probability, true, p.generator());
  out.logits          = nn::dense(p, "output", pooled, cfg.classes);
  out.classifier_macs = std::int64_t(last) * cfg.classes;
  return out;
}

// MobileNet v3 (https://arxiv.org/abs/1905.02244, torchvision release):
// searched inverted residuals with 3x3 or 5x5 depthwise filters, ReLU or
// hard-swish, and optional squeeze-and-excitation on the depthwise output
// (global mean, 1x1 to MakeDivisible(expanded/4) with ReLU, 1x1 back with a
// hard sigmoid, channel scaling). The stem is 3x3/2 to 16 channels with
// hard-swish; the last stage is a 1x1 to six times the last width
// (BN, hard-swish), pooling, a biased dense layer to 1280 (Large) or 1024
// (Small) with hard-swish, dropout(0.2) in training and the classifier.
inline nn::Tensor HardSigmoid(const nn::Tensor& x) { return nn::relu6(x + 3.0) / 6.0; }
inline nn::Tensor HardSwish(const nn::Tensor& x) { return x * HardSigmoid(x); }
struct SqueezeExcitation {
  nn::Tensor gate, output;
};
inline SqueezeExcitation SqueezeExcite(nn::ParameterStore& p, const std::string& name, const nn::Tensor& x,
                                       std::int64_t squeeze) {
  const std::int64_t   channels = x.shape()[3];
  nn::ConvLayerOptions o;
  o.init                  = nn::init::he_uniform();
  const nn::Tensor pooled = nn::mean(x, {1, 2}, true);
  const nn::Tensor h      = nn::relu(nn::conv2d(p, name + "/reduce", pooled, squeeze, 1, o));
  SqueezeExcitation se;
  se.gate   = HardSigmoid(nn::conv2d(p, name + "/expand", h, channels, 1, o));
  se.output = x * se.gate;
  return se;
}
struct MobileNetV3Block {
  int  in, kernel, expanded, out;
  bool se, hard_swish;
  int  stride;
};
inline std::vector<MobileNetV3Block> MobileNetV3Blocks(bool large) {
  if (large)
    return {{16, 3, 16, 16, false, false, 1},   {16, 3, 64, 24, false, false, 2},   {24, 3, 72, 24, false, false, 1},
            {24, 5, 72, 40, true, false, 2},    {40, 5, 120, 40, true, false, 1},   {40, 5, 120, 40, true, false, 1},
            {40, 3, 240, 80, false, true, 2},   {80, 3, 200, 80, false, true, 1},   {80, 3, 184, 80, false, true, 1},
            {80, 3, 184, 80, false, true, 1},   {80, 3, 480, 112, true, true, 1},   {112, 3, 672, 112, true, true, 1},
            {112, 5, 672, 160, true, true, 2},  {160, 5, 960, 160, true, true, 1},  {160, 5, 960, 160, true, true, 1}};
  return {{16, 3, 16, 16, true, false, 2},  {16, 3, 72, 24, false, false, 2}, {24, 3, 88, 24, false, false, 1},
          {24, 5, 96, 40, true, true, 2},   {40, 5, 240, 40, true, true, 1},  {40, 5, 240, 40, true, true, 1},
          {40, 5, 120, 48, true, true, 1},  {48, 5, 144, 48, true, true, 1},  {48, 5, 288, 96, true, true, 2},
          {96, 5, 576, 96, true, true, 1},  {96, 5, 576, 96, true, true, 1}};
}
struct MobileNetV3Config {
  bool   large = true, small_stem = false;
  int    classes = 1000;
  double keep_probability = .8, momentum = .9;
  bool   update_stats = true;
};
inline MobileNetOutputs MobileNetV3(nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& image,
                                    const MobileNetV3Config& cfg = {}, bool training = false) {
  if (cfg.classes < 2 || !(cfg.keep_probability > 0 && cfg.keep_probability <= 1))
    throw std::invalid_argument("invalid MobileNet v3 configuration");
  MobileNetOutputs out;
  MobileContext    c{p, state, out.layers, training, cfg.momentum, cfg.update_stats};
  auto act = [](const nn::Tensor& x, bool hard_swish) { return hard_swish ? HardSwish(x) : nn::relu(x); };
  nn::Tensor h      = HardSwish(ConvBN(c, "conv0", image, 16, 3, cfg.small_stem ? 1 : 2, nn::Activation::None));
  const auto blocks = MobileNetV3Blocks(cfg.large);
  int        index  = 0;
  for (const auto& b : blocks) {
    const auto name = "block" + std::to_string(++index);
    nn::Tensor x    = h;
    if (b.expanded != b.in) h = act(ConvBN(c, name + "/expand", h, b.expanded, 1, 1, nn::Activation::None), b.hard_swish);
    h = act(ConvBN(c, name + "/depthwise", h, b.expanded, b.kernel, b.stride, nn::Activation::None, MobileConv::kDepthwise),
            b.hard_swish);
    if (b.se) {
      const int squeeze = MakeDivisible(b.expanded / 4);
      h                 = SqueezeExcite(p, name + "/se", h, squeeze).output;
      // Two 1x1 layers on the pooled vector (the channel scaling is not counted).
      c.layers.push_back({name + "/se", {1, 1, 1, 1}, 2 * std::int64_t(b.expanded) * squeeze});
    }
    h = ConvBN(c, name + "/project", h, b.out, 1, 1, nn::Activation::None);
    if (b.stride == 1 && b.in == b.out) h = x + h;
  }
  const int width = blocks.back().out, last = 6 * width, head = cfg.large ? 1280 : 1024;
  out.features  = HardSwish(ConvBN(c, "conv_last", h, last, 1, 1, nn::Activation::None));
  nn::Tensor v  = HardSwish(nn::dense(p, "head", nn::global_avg_pool2d(out.features), head));
  if (training && cfg.keep_probability < 1) v = nn::dropout(v, 1 - cfg.keep_probability, true, p.generator());
  out.logits          = nn::dense(p, "output", v, cfg.classes);
  out.classifier_macs = std::int64_t(last) * head + std::int64_t(head) * cfg.classes;
  return out;
}

}  // namespace mobilenet
