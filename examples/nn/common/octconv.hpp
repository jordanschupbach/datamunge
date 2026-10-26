// Octave convolution: high- and low-frequency feature maps, the low ones at
// half resolution, exchanging information at every layer (port of tflow's
// common/nn/octconv.h).
// https://arxiv.org/abs/1904.05049
// https://github.com/facebookresearch/OctConv
//
// A feature map with C channels and ratio alpha holds C - floor(alpha*C)
// channels at full resolution (H) and floor(alpha*C) at half resolution (L).
// An octave convolution computes
//   Y_H = f(X_H; W_HH) + upsample(f(X_L; W_LH)),
//   Y_L = f(X_L; W_LL) + f(pool(X_H); W_HL),
// with 2x2 average pooling and nearest-neighbour 2x upsampling; the four
// kernels partition one ordinary kernel, so the parameter count is unchanged.
// alpha_in = 0 (first layer) or alpha_out = 0 (last layer) drop the missing
// paths. A strided layer average-pools both inputs first. BN and ReLU apply to
// each frequency's output channels. The Oct-ResNet bottleneck makes all three
// convolutions and the projection octave layers; the trunk enters with
// alpha_in = 0 in its first block and returns to a single frequency in the
// first block of the last stage (whose 7x7 maps at 224 pixels have no even
// half resolution), which then continues with ordinary convolutions.
// Convolutions are bias-free with symmetric padding and He-uniform kernels
// (fan-in of the full layer); BN epsilon is 1e-5. Multiply-adds are counted
// from the nominal sizes (high_size / image_size), as in tflow, so a model
// can be built on a smaller input for parameter counts.
#pragma once

#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <limits>
#include <stdexcept>
#include <string>

#include "common/resnet.hpp"

namespace octconv {

namespace nn = datamunge::nn;

struct OctaveMaps {
  nn::Tensor high, low;
  int        high_channels = 0, low_channels = 0;
};
inline int LowChannels(int channels, double alpha) { return int(alpha * channels); }

struct OctaveCost {
  std::int64_t macs = 0;  // multiply-adds per example, accumulated by OctaveConv
};

inline OctaveMaps OctaveConv(nn::ParameterStore& p, const std::string& name, const OctaveMaps& x, int out, double alpha_out,
                             int kernel, int stride, int high_size, OctaveCost* cost = nullptr) {
  const int in = x.high_channels + x.low_channels, out_low = LowChannels(out, alpha_out), out_high = out - out_low;
  if (stride != 1 && stride != 2) throw std::invalid_argument("octave convolutions use stride one or two");
  const auto init = nn::init::uniform(std::sqrt(6.0 / double(kernel * kernel * in)));  // He uniform, full-layer fan-in
  auto conv = [&](const std::string& part, const nn::Tensor& v, int from, int to, int size) {
    const nn::Tensor w = p.get(name + "/" + part + "/W", {kernel, kernel, from, to}, init);
    if (cost) cost->macs += std::int64_t(size) * size * kernel * kernel * from * to;
    nn::Conv2DOptions o = nn::conv_options(1, nn::Padding::Valid);
    if (kernel > 1) {
      o.padding           = nn::Padding::Explicit;
      o.explicit_paddings = {kernel / 2, kernel / 2, kernel / 2, kernel / 2};
    }
    return nn::conv2d(v, w, o);
  };
  nn::Tensor xh = x.high, xl = x.low;
  int        size = high_size;
  if (stride == 2) {
    xh = nn::avg_pool2d(xh, 2, 2, nn::Padding::Valid);
    if (x.low_channels) xl = nn::avg_pool2d(xl, 2, 2, nn::Padding::Valid);
    size /= 2;
  }
  OctaveMaps y;
  y.high_channels = out_high;
  y.low_channels  = out_low;
  y.high          = conv("hh", xh, x.high_channels, out_high, size);
  if (x.low_channels) y.high = y.high + nn::upsample_nearest(conv("lh", xl, x.low_channels, out_high, size / 2), 2);
  if (out_low) {
    y.low = conv("hl", nn::avg_pool2d(xh, 2, 2, nn::Padding::Valid), x.high_channels, out_low, size / 2);
    if (x.low_channels) y.low = y.low + conv("ll", xl, x.low_channels, out_low, size / 2);
  }
  return y;
}

inline OctaveMaps OctaveNorm(nn::ParameterStore& p, nn::ParameterStore& state, const std::string& name, const OctaveMaps& x,
                             bool relu, bool training, double momentum, bool update_stats = true) {
  nn::BatchNormOptions bn;
  bn.momentum     = momentum;
  bn.eps          = 1e-5;
  bn.update_stats = update_stats;
  auto norm = [&](const std::string& part, const nn::Tensor& v) {
    const nn::Tensor h = nn::batch_norm(p, state, name + "/" + part + "/bn", v, training, bn);
    return relu ? nn::relu(h) : h;
  };
  OctaveMaps y = x;
  y.high       = norm("high", x.high);
  if (x.low_channels) y.low = norm("low", x.low);
  return y;
}

struct OctResNetConfig {
  int    depth = 50, classes = 1000, width_divisor = 1, image_size = 224;
  double alpha = .25;
  bool   small_stem = false;
  double momentum = .9;
  bool   update_stats = true;
};
struct OctResNetOutputs {
  nn::Tensor   features, logits;
  std::int64_t macs = 0;
};

inline OctResNetOutputs OctResNet(nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& image,
                                  const OctResNetConfig& cfg = {}, bool training = false) {
  const auto counts = resnet::ResNetBlockCounts(cfg.depth);
  if (cfg.depth < 50 || !(cfg.alpha >= 0 && cfg.alpha < 1) || cfg.classes < 2)
    throw std::invalid_argument("invalid Oct-ResNet configuration");
  OctResNetOutputs       out;
  OctaveCost             cost;
  resnet::ConvNormConfig norm;
  norm.training     = training;
  norm.scale        = true;
  norm.epsilon      = 1e-5;
  norm.momentum     = cfg.momentum;
  norm.update_stats = cfg.update_stats;
  int        in = 64 / cfg.width_divisor, size = cfg.small_stem ? cfg.image_size : cfg.image_size / 4;
  nn::Tensor stem = resnet::ResNetConv(p, state, "stem", image, in, cfg.small_stem ? 3 : 7, norm, cfg.small_stem ? 1 : 2);
  if (!cfg.small_stem)
    stem = nn::max_pool2d(nn::pad(stem, {{0, 0}, {1, 1}, {1, 1}, {0, 0}}, nn::PadMode::Constant,
                                  -std::numeric_limits<double>::infinity()),
                          3, 2, nn::Padding::Valid);
  cost.macs += std::int64_t(cfg.small_stem ? size : 2 * size) * (cfg.small_stem ? size : 2 * size) *
               (cfg.small_stem ? 27 : 147) * in;
  OctaveMaps h;
  h.high          = stem;
  h.high_channels = in;
  for (int stage = 0; stage < 4; ++stage) {
    const double alpha  = stage < 3 ? cfg.alpha : 0;
    const int    planes = (64 << stage) / cfg.width_divisor, outputs = 4 * planes;
    for (int block = 0; block < counts[stage]; ++block) {
      const auto name   = "stage" + std::to_string(stage + 1) + "/block" + std::to_string(block + 1);
      const int  stride = stage > 0 && block == 0 ? 2 : 1;
      auto       layer  = [&](const std::string& part, const OctaveMaps& x, int to, int kernel, int step, int at, bool relu) {
        return OctaveNorm(p, state, name + "/" + part, OctaveConv(p, name + "/" + part, x, to, alpha, kernel, step, at, &cost),
                          relu, training, cfg.momentum, cfg.update_stats);
      };
      OctaveMaps r     = layer("conv1", h, planes, 1, 1, size, true);
      r                = layer("conv2", r, planes, 3, stride, size, true);
      const int next   = stride == 2 ? size / 2 : size;
      r                = layer("conv3", r, outputs, 1, 1, next, false);
      OctaveMaps shortcut = h;
      if (in != outputs || stride != 1 || (h.low_channels == 0) != (r.low_channels == 0))
        shortcut = layer("shortcut", h, outputs, 1, stride, size, false);
      h.high          = nn::relu(shortcut.high + r.high);
      h.high_channels = r.high_channels;
      h.low_channels  = r.low_channels;
      h.low           = r.low_channels ? nn::relu(shortcut.low + r.low) : nn::Tensor();
      in              = outputs;
      size            = next;
    }
  }
  out.features = h.high;
  out.logits   = nn::dense(p, "output", nn::global_avg_pool2d(h.high), cfg.classes);
  out.macs     = cost.macs + std::int64_t(in) * cfg.classes;
  return out;
}

}  // namespace octconv
