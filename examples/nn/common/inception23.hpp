// Inception v2 (BN-Inception) and v3, following TF-Slim's public definitions
// (port of tflow's common/nn/inception23.h):
// https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v2.py
// https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v3.py
// https://arxiv.org/abs/1502.03167 and https://arxiv.org/abs/1512.00567
//
// v2 has a separable 7x7 stem, eight ordinary modules and two reduction modules.
// v3 has three A modules, four asymmetric B modules, two split-branch C modules,
// two reductions and one optional auxiliary head after the final B module.
// Match TF-Slim's branch widths, padding, beta-only BN (eps .001), and .8
// dropout retention. The v2 separable stem has bias/ReLU without BN, matching
// TF-Slim's distinct separable-convolution defaults. He initialization and
// configurable EMA momentum are explicit adaptations; weights are not imported.
// Batch-norm moving averages update eagerly in training mode (tflow collects
// update ops in Inception23Outputs::updates; nn needs no such list).
#pragma once

#include <algorithm>
#include <array>
#include <datamunge/nn/nn.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common/normalized_conv.hpp"

namespace inception23 {

namespace nn = datamunge::nn;

struct Inception23Config {
  int    version = 3, image_size = 299, classes = 1000, width_divisor = 1;
  double momentum = .9997;
  void   Validate() const {
    if ((version != 2 && version != 3) || classes < 2 || image_size < (version == 2 ? 32 : 107) || image_size > 299 ||
        (width_divisor != 1 && width_divisor != 2 && width_divisor != 4 && width_divisor != 8) ||
        !(momentum >= 0 && momentum < 1))
      throw std::invalid_argument("invalid Inception v2/v3 configuration");
  }
  int AuxiliaryMapSize() const {
    int size = (image_size - 3) / 2 + 1;
    size -= 2;
    size = (size - 3) / 2 + 1;
    size -= 2;
    size = (size - 3) / 2 + 1;
    return (size - 3) / 2 + 1;
  }
};

struct Inception23Outputs {
  std::vector<std::pair<std::string, nn::Tensor>> stages;
  nn::Tensor                                      logits;
  std::vector<nn::Tensor>                         auxiliary;
};

// Carry channel counts with tensors so branch construction never guesses the
// input width after concatenation. Width scaling applies only to learned outputs.
struct ConvFeature {
  nn::Tensor value;
  int        channels = 0;
};

class InceptionBuilder {
 public:
  InceptionBuilder(nn::ParameterStore& p, nn::ParameterStore& state, Inception23Outputs& out,
                   const Inception23Config& cfg, bool training)
      : p_(p), state_(state), out_(out), cfg_(cfg) {
    norm_.training = training;
    norm_.momentum = cfg.momentum;
  }
  nn::ParameterStore& params() { return p_; }
  ConvFeature Conv(const std::string& name, const ConvFeature& x, int channels, int h = 1, int w = 1, int stride = 1,
                   const std::string& padding = "SAME") {
    channels /= cfg_.width_divisor;
    return {normalized_conv::NormalizedConv(p_, state_, name, x.value, channels, h, w, norm_, stride, padding), channels};
  }
  ConvFeature Pool(const ConvFeature& x, bool maximum = false, int window = 3, int stride = 1,
                   const std::string& padding = "SAME") {
    const auto pad = normalized_conv::ParsePadding(padding);
    return {maximum ? nn::max_pool2d(x.value, window, stride, pad) : nn::avg_pool2d(x.value, window, stride, pad),
            x.channels};
  }
  ConvFeature Cat(const std::vector<ConvFeature>& features) {
    std::vector<nn::Tensor> values;
    int                     channels = 0;
    for (const auto& feature : features) {
      values.push_back(feature.value);
      channels += feature.channels;
    }
    return {nn::concat(values, 3), channels};
  }
  ConvFeature Stage(const std::string& name, const ConvFeature& x) {
    out_.stages.emplace_back(name, x.value);
    return x;
  }
  ConvFeature V2Module(const std::string& name, const ConvFeature& x, const std::array<int, 7>& w, bool max_pool = false) {
    const auto a = Conv(name + "/one", x, w[0]);
    auto       b = Conv(name + "/three_reduce", x, w[1]);
    b            = Conv(name + "/three", b, w[2], 3, 3);
    auto c       = Conv(name + "/double_reduce", x, w[3]);
    c            = Conv(name + "/double1", c, w[4], 3, 3);
    c            = Conv(name + "/double2", c, w[5], 3, 3);
    const auto d = Conv(name + "/pool", Pool(x, max_pool), w[6]);
    return Stage(name, Cat({a, b, c, d}));
  }
  ConvFeature V2Reduction(const std::string& name, const ConvFeature& x, const std::array<int, 5>& w) {
    auto a = Conv(name + "/three_reduce", x, w[0]);
    a      = Conv(name + "/three", a, w[1], 3, 3, 2);
    auto b = Conv(name + "/double_reduce", x, w[2]);
    b      = Conv(name + "/double1", b, w[3], 3, 3);
    b      = Conv(name + "/double2", b, w[4], 3, 3, 2);
    return Stage(name, Cat({a, b, Pool(x, true, 3, 2)}));
  }
  ConvFeature V3A(const std::string& name, const ConvFeature& x, int pool) {
    const auto a = Conv(name + "/one", x, 64);
    auto       b = Conv(name + "/five_reduce", x, 48);
    b            = Conv(name + "/five", b, 64, 5, 5);
    auto c       = Conv(name + "/double_reduce", x, 64);
    c            = Conv(name + "/double1", c, 96, 3, 3);
    c            = Conv(name + "/double2", c, 96, 3, 3);
    const auto d = Conv(name + "/pool", Pool(x), pool);
    return Stage(name, Cat({a, b, c, d}));
  }
  ConvFeature V3B(const std::string& name, const ConvFeature& x, int width) {
    const auto a = Conv(name + "/one", x, 192);
    auto       b = Conv(name + "/seven_reduce", x, width);
    b            = Conv(name + "/seven_h", b, width, 1, 7);
    b            = Conv(name + "/seven_v", b, 192, 7, 1);
    auto c       = Conv(name + "/double_reduce", x, width);
    c            = Conv(name + "/double_v1", c, width, 7, 1);
    c            = Conv(name + "/double_h1", c, width, 1, 7);
    c            = Conv(name + "/double_v2", c, width, 7, 1);
    c            = Conv(name + "/double_h2", c, 192, 1, 7);
    const auto d = Conv(name + "/pool", Pool(x), 192);
    return Stage(name, Cat({a, b, c, d}));
  }
  ConvFeature V3C(const std::string& name, const ConvFeature& x) {
    const auto a = Conv(name + "/one", x, 320), b = Conv(name + "/split_reduce", x, 384);
    auto       c = Conv(name + "/double_reduce", x, 448);
    c            = Conv(name + "/double", c, 384, 3, 3);
    const auto d = Conv(name + "/pool", Pool(x), 192);
    return Stage(name, Cat({a, Conv(name + "/split_h", b, 384, 1, 3), Conv(name + "/split_v", b, 384, 3, 1),
                            Conv(name + "/double_h", c, 384, 1, 3), Conv(name + "/double_v", c, 384, 3, 1), d}));
  }

 private:
  nn::ParameterStore&             p_;
  nn::ParameterStore&             state_;
  Inception23Outputs&             out_;
  const Inception23Config&        cfg_;
  normalized_conv::ConvNormConfig norm_;
};

/// Global average pool, training-time dropout (keep .8) and the linear classifier.
inline nn::Tensor InceptionClassifier(nn::ParameterStore& p, const nn::Tensor& features, int classes, bool training) {
  nn::Tensor pooled = nn::global_avg_pool2d(features);
  if (training) pooled = nn::dropout(pooled, .2, true, p.generator());
  return nn::dense(p, "output", pooled, classes);
}

/// The reference auxiliary head: 5x5/3 average pool, 1x1 128, a VALID
/// convolution covering the remaining map (768 wide), and linear logits.
inline nn::Tensor InceptionAuxiliary(InceptionBuilder& b, const ConvFeature& input, const Inception23Config& cfg) {
  auto      h      = b.Pool(input, false, 5, 3, "VALID");
  h                = b.Conv("aux/projection", h, 128);
  const int kernel = (cfg.AuxiliaryMapSize() - 5) / 3 + 1;
  h                = b.Conv("aux/spatial", h, 768, kernel, kernel, 1, "VALID");
  return nn::dense(b.params(), "aux/output", nn::reshape(h.value, {-1, h.channels}), cfg.classes);
}

inline Inception23Outputs Inception23(nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& image,
                                      const Inception23Config& cfg = {}, bool training = false, bool auxiliary = false) {
  cfg.Validate();
  Inception23Outputs out;
  InceptionBuilder   b(p, state, out, cfg, training);
  ConvFeature        x{image, 3}, aux_input{};
  if (cfg.version == 2) {
    nn::ConvLayerOptions stem;
    stem.stride           = 2;
    stem.activation       = nn::Activation::Relu;
    stem.init             = nn::init::he_uniform();
    const int channels    = 64 / cfg.width_divisor;
    const int multiplier  = std::min(channels / 3, 8);
    x = b.Stage("stem7", {nn::separable_conv2d(p, "stem7", x.value, channels, 7, stem, multiplier), channels});
    x = b.Stage("pool1", b.Pool(x, true, 3, 2));
    x = b.Stage("stem1", b.Conv("stem1", x, 64));
    x = b.Stage("stem3", b.Conv("stem3", x, 192, 3, 3));
    x = b.Stage("pool2", b.Pool(x, true, 3, 2));
    x = b.V2Module("3b", x, {64, 64, 64, 64, 96, 96, 32});
    x = b.V2Module("3c", x, {64, 64, 96, 64, 96, 96, 64});
    x = b.V2Reduction("4a", x, {128, 160, 64, 96, 96});
    x = b.V2Module("4b", x, {224, 64, 96, 96, 128, 128, 128});
    x = b.V2Module("4c", x, {192, 96, 128, 96, 128, 128, 128});
    x = b.V2Module("4d", x, {160, 128, 160, 128, 160, 160, 96});
    x = b.V2Module("4e", x, {96, 128, 192, 160, 192, 192, 96});
    x = b.V2Reduction("5a", x, {128, 192, 192, 256, 256});
    x = b.V2Module("5b", x, {352, 192, 320, 160, 224, 224, 128});
    x = b.V2Module("5c", x, {352, 192, 320, 192, 224, 224, 128}, true);
  } else {
    x = b.Stage("stem1", b.Conv("stem1", x, 32, 3, 3, 2, "VALID"));
    x = b.Stage("stem2", b.Conv("stem2", x, 32, 3, 3, 1, "VALID"));
    x = b.Stage("stem3", b.Conv("stem3", x, 64, 3, 3));
    x = b.Stage("pool1", b.Pool(x, true, 3, 2, "VALID"));
    x = b.Stage("stem4", b.Conv("stem4", x, 80));
    x = b.Stage("stem5", b.Conv("stem5", x, 192, 3, 3, 1, "VALID"));
    x = b.Stage("pool2", b.Pool(x, true, 3, 2, "VALID"));
    x = b.V3A("5b", x, 32);
    x = b.V3A("5c", x, 64);
    x = b.V3A("5d", x, 64);
    auto a = b.Conv("6a/three", x, 384, 3, 3, 2, "VALID");
    auto c = b.Conv("6a/double_reduce", x, 64);
    c      = b.Conv("6a/double1", c, 96, 3, 3);
    c      = b.Conv("6a/double2", c, 96, 3, 3, 2, "VALID");
    x      = b.Stage("6a", b.Cat({a, c, b.Pool(x, true, 3, 2, "VALID")}));
    x      = b.V3B("6b", x, 128);
    x      = b.V3B("6c", x, 160);
    x      = b.V3B("6d", x, 160);
    x      = b.V3B("6e", x, 192);
    aux_input = x;
    a = b.Conv("7a/three_reduce", x, 192);
    a = b.Conv("7a/three", a, 320, 3, 3, 2, "VALID");
    c = b.Conv("7a/seven_reduce", x, 192);
    c = b.Conv("7a/seven_h", c, 192, 1, 7);
    c = b.Conv("7a/seven_v", c, 192, 7, 1);
    c = b.Conv("7a/seven_three", c, 192, 3, 3, 2, "VALID");
    x = b.Stage("7a", b.Cat({a, c, b.Pool(x, true, 3, 2, "VALID")}));
    x = b.V3C("7b", x);
    x = b.V3C("7c", x);
  }
  out.logits = InceptionClassifier(p, x.value, cfg.classes, training);
  if (cfg.version == 3 && auxiliary) out.auxiliary.push_back(InceptionAuxiliary(b, aux_input, cfg));
  return out;
}

}  // namespace inception23
