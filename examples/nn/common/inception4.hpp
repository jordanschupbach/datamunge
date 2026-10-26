// Inception v4 and Inception-ResNet-v2, with complete reference block counts
// (port of tflow's common/nn/inception4.h).
// https://arxiv.org/abs/1602.07261
// https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v4.py
// https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_resnet_v2.py
//
// v4 repeats A/B/C blocks 4/7/3 times. Inception-ResNet-v2 repeats them
// 10/20/10 times, scales residuals by .17/.10/.20, and uses an unscaled final
// C block without its final ReLU (as in TF-Slim). Residual projections are
// linear, have biases and no BN. Other convolutions use beta-only BN/ReLU.
// Preserve reference widths, reductions, optional auxiliary classifiers and
// .8 dropout retention. He initialization and compact training are adaptations.
#pragma once

#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <stdexcept>
#include <string>

#include "common/inception23.hpp"

namespace inception4 {

namespace nn = datamunge::nn;
using inception23::ConvFeature;
using inception23::Inception23Config;
using inception23::Inception23Outputs;
using inception23::InceptionBuilder;

struct Inception4Config {
  bool              residual   = false;
  int               image_size = 299, classes = 1000, width_divisor = 1;
  double            momentum   = .9997;
  Inception23Config Layers() const {
    Inception23Config cfg;
    cfg.image_size    = image_size;
    cfg.classes       = classes;
    cfg.width_divisor = width_divisor;
    cfg.momentum      = momentum;
    cfg.Validate();
    return cfg;
  }
};

inline ConvFeature Inception4B(InceptionBuilder& b, const std::string& name, const ConvFeature& x) {
  auto a = b.Conv(name + "/one", x, 384);
  auto c = b.Conv(name + "/seven_reduce", x, 192);
  c      = b.Conv(name + "/seven_h", c, 224, 1, 7);
  c      = b.Conv(name + "/seven_v", c, 256, 7, 1);
  auto d = b.Conv(name + "/double_reduce", x, 192);
  d      = b.Conv(name + "/double_v1", d, 192, 7, 1);
  d      = b.Conv(name + "/double_h1", d, 224, 1, 7);
  d      = b.Conv(name + "/double_v2", d, 224, 7, 1);
  d      = b.Conv(name + "/double_h2", d, 256, 1, 7);
  return b.Stage(name, b.Cat({a, c, d, b.Conv(name + "/pool", b.Pool(x), 128)}));
}

inline ConvFeature Inception4C(InceptionBuilder& b, const std::string& name, const ConvFeature& x) {
  auto a = b.Conv(name + "/one", x, 256), c = b.Conv(name + "/split_reduce", x, 384);
  auto d = b.Conv(name + "/double_reduce", x, 384);
  d      = b.Conv(name + "/double_v", d, 448, 3, 1);
  d      = b.Conv(name + "/double_h", d, 512, 1, 3);
  auto pooled = b.Conv(name + "/pool", b.Pool(x), 256);
  return b.Stage(name, b.Cat({a, b.Conv(name + "/split_h", c, 256, 1, 3), b.Conv(name + "/split_v", c, 256, 3, 1),
                              b.Conv(name + "/double_split_h", d, 256, 1, 3), b.Conv(name + "/double_split_v", d, 256, 3, 1),
                              pooled}));
}

/// x + scale * project(branches(x)), then ReLU unless `relu` is false.
inline ConvFeature InceptionResidual(InceptionBuilder& b, const std::string& name, const ConvFeature& x, char kind,
                                     double scale, bool relu = true) {
  if (!std::isfinite(scale)) throw std::invalid_argument("finite residual scale required");
  ConvFeature joined;
  if (kind == 'A') {
    auto a = b.Conv(name + "/one", x, 32);
    auto c = b.Conv(name + "/three_reduce", x, 32);
    c      = b.Conv(name + "/three", c, 32, 3, 3);
    auto d = b.Conv(name + "/double_reduce", x, 32);
    d      = b.Conv(name + "/double1", d, 48, 3, 3);
    d      = b.Conv(name + "/double2", d, 64, 3, 3);
    joined = b.Cat({a, c, d});
  } else if (kind == 'B' || kind == 'C') {
    auto a = b.Conv(name + "/one", x, 192);
    auto c = b.Conv(name + "/reduce", x, kind == 'B' ? 128 : 192);
    c      = b.Conv(name + "/horizontal", c, kind == 'B' ? 160 : 224, 1, kind == 'B' ? 7 : 3);
    c      = b.Conv(name + "/vertical", c, kind == 'B' ? 192 : 256, kind == 'B' ? 7 : 3, 1);
    joined = b.Cat({a, c});
  } else
    throw std::invalid_argument("Inception residual kind must be A, B or C");
  nn::ConvLayerOptions linear;
  linear.init          = nn::init::he_uniform();
  const auto projected = nn::conv2d(b.params(), name + "/project", joined.value, x.channels, 1, linear);
  nn::Tensor result    = x.value + scale * projected;
  if (relu) result = nn::relu(result);
  return b.Stage(name, {result, x.channels});
}

inline Inception23Outputs Inception4(nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& image,
                                     const Inception4Config& cfg = {}, bool training = false, bool auxiliary = false) {
  const auto         layers = cfg.Layers();
  Inception23Outputs out;
  InceptionBuilder   b(p, state, out, layers, training);
  ConvFeature        x{image, 3}, aux_input;
  x = b.Stage("stem1", b.Conv("stem1", x, 32, 3, 3, 2, "VALID"));
  x = b.Stage("stem2", b.Conv("stem2", x, 32, 3, 3, 1, "VALID"));
  x = b.Stage("stem3", b.Conv("stem3", x, 64, 3, 3));
  if (cfg.residual) {
    x      = b.Stage("pool1", b.Pool(x, true, 3, 2, "VALID"));
    x      = b.Stage("stem4", b.Conv("stem4", x, 80));
    x      = b.Stage("stem5", b.Conv("stem5", x, 192, 3, 3, 1, "VALID"));
    x      = b.Stage("pool2", b.Pool(x, true, 3, 2, "VALID"));
    auto a = b.Conv("mixed/one", x, 96);
    auto c = b.Conv("mixed/five_reduce", x, 48);
    c      = b.Conv("mixed/five", c, 64, 5, 5);
    auto d = b.Conv("mixed/double_reduce", x, 64);
    d      = b.Conv("mixed/double1", d, 96, 3, 3);
    d      = b.Conv("mixed/double2", d, 96, 3, 3);
    x      = b.Stage("mixed", b.Cat({a, c, d, b.Conv("mixed/pool", b.Pool(x), 64)}));
    for (int i = 0; i < 10; ++i) x = InceptionResidual(b, "A" + std::to_string(i), x, 'A', .17);
    a = b.Conv("reductionA/three", x, 384, 3, 3, 2, "VALID");
    c = b.Conv("reductionA/double_reduce", x, 256);
    c = b.Conv("reductionA/double1", c, 256, 3, 3);
    c = b.Conv("reductionA/double2", c, 384, 3, 3, 2, "VALID");
    x = b.Stage("reductionA", b.Cat({a, c, b.Pool(x, true, 3, 2, "VALID")}));
    for (int i = 0; i < 20; ++i) x = InceptionResidual(b, "B" + std::to_string(i), x, 'B', .1);
    aux_input = x;
    a = b.Conv("reductionB/first_reduce", x, 256);
    a = b.Conv("reductionB/first", a, 384, 3, 3, 2, "VALID");
    c = b.Conv("reductionB/second_reduce", x, 256);
    c = b.Conv("reductionB/second", c, 288, 3, 3, 2, "VALID");
    d = b.Conv("reductionB/third_reduce", x, 256);
    d = b.Conv("reductionB/third1", d, 288, 3, 3);
    d = b.Conv("reductionB/third2", d, 320, 3, 3, 2, "VALID");
    x = b.Stage("reductionB", b.Cat({a, c, d, b.Pool(x, true, 3, 2, "VALID")}));
    for (int i = 0; i < 9; ++i) x = InceptionResidual(b, "C" + std::to_string(i), x, 'C', .2);
    x = InceptionResidual(b, "C9", x, 'C', 1., false);
    x = b.Stage("final", b.Conv("final", x, 1536));
  } else {
    x = b.Stage("stem_mix1", b.Cat({b.Pool(x, true, 3, 2, "VALID"), b.Conv("stem_mix1/conv", x, 96, 3, 3, 2, "VALID")}));
    auto a = b.Conv("stem_mix2/short_reduce", x, 64);
    a      = b.Conv("stem_mix2/short", a, 96, 3, 3, 1, "VALID");
    auto c = b.Conv("stem_mix2/long_reduce", x, 64);
    c      = b.Conv("stem_mix2/long_h", c, 64, 1, 7);
    c      = b.Conv("stem_mix2/long_v", c, 64, 7, 1);
    c      = b.Conv("stem_mix2/long", c, 96, 3, 3, 1, "VALID");
    x      = b.Stage("stem_mix2", b.Cat({a, c}));
    x = b.Stage("stem_mix3", b.Cat({b.Conv("stem_mix3/conv", x, 192, 3, 3, 2, "VALID"), b.Pool(x, true, 3, 2, "VALID")}));
    for (int i = 0; i < 4; ++i) x = b.V2Module("A" + std::to_string(i), x, {96, 64, 96, 64, 96, 96, 96});
    a = b.Conv("reductionA/three", x, 384, 3, 3, 2, "VALID");
    c = b.Conv("reductionA/double_reduce", x, 192);
    c = b.Conv("reductionA/double1", c, 224, 3, 3);
    c = b.Conv("reductionA/double2", c, 256, 3, 3, 2, "VALID");
    x = b.Stage("reductionA", b.Cat({a, c, b.Pool(x, true, 3, 2, "VALID")}));
    for (int i = 0; i < 7; ++i) x = Inception4B(b, "B" + std::to_string(i), x);
    aux_input = x;
    a = b.Conv("reductionB/three_reduce", x, 192);
    a = b.Conv("reductionB/three", a, 192, 3, 3, 2, "VALID");
    c = b.Conv("reductionB/seven_reduce", x, 256);
    c = b.Conv("reductionB/seven_h", c, 256, 1, 7);
    c = b.Conv("reductionB/seven_v", c, 320, 7, 1);
    c = b.Conv("reductionB/seven_three", c, 320, 3, 3, 2, "VALID");
    x = b.Stage("reductionB", b.Cat({a, c, b.Pool(x, true, 3, 2, "VALID")}));
    for (int i = 0; i < 3; ++i) x = Inception4C(b, "C" + std::to_string(i), x);
  }
  out.logits = inception23::InceptionClassifier(p, x.value, cfg.classes, training);
  if (auxiliary) out.auxiliary.push_back(inception23::InceptionAuxiliary(b, aux_input, layers));
  return out;
}

}  // namespace inception4
