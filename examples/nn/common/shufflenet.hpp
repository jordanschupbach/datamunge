// ShuffleNet v1 and v2, following Megvii's official PyTorch release (port of
// tflow's common/nn/shufflenet.h; job 1.4a ported it for CondenseNet's channel
// shuffle and checked only the shuffles, the networks are untested here).
// https://arxiv.org/abs/1707.01083 (v1), https://arxiv.org/abs/1807.11164 (v2)
// https://github.com/megvii-model/ShuffleNet-Series
//
// v1 units: grouped 1x1 to out/4 channels (ungrouped in the very first
// unit), BN/ReLU, channel shuffle, depthwise 3x3, BN, grouped 1x1, BN; stride
// one adds the input and applies ReLU, stride two concatenates a 3x3/2
// average-pooled input (padding counted in the mean) with ReLU of a branch
// producing out-in channels. The release shuffles channel c to position
// (c mod g)*(C/g) + floor(c/g), the inverse of the paper's transpose; both
// give every group of the next grouped layer channels from every group.
// v2 units: stride one sends the odd channels through 1x1/BN/ReLU,
// depthwise 3x3/BN, 1x1/BN/ReLU and concatenates them after the even
// channels (the release's form of split, concat and shuffle); stride two
// feeds the whole input to that branch (out-in outputs) and to a
// depthwise 3x3/2, BN, 1x1, BN, ReLU projection that keeps its width. Both
// use a 3x3/2 conv/BN/ReLU stem and a 3x3/2 max-pool, stages of 4/8/4 units
// (v2 adds a 1x1 conv/BN/ReLU), global average pooling and a bias-free
// classifier (dropout 0.2 before it for v2 2.0x). Convolutions are bias-free
// with symmetric padding; BN epsilon is 1e-5. He-uniform fan-in kernels are
// an initialization adaptation; small_stem drops the stem stride and pool.
// Grouped convolutions use common/grouped.hpp ("<name>/group<g>/W").
#pragma once

#include <datamunge/nn/nn.hpp>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common/grouped.hpp"

namespace shufflenet {

namespace nn = datamunge::nn;

/// Reshape the channel axis of an NHWC tensor, keeping N, H and W.
inline nn::Tensor ReshapeChannels(const nn::Tensor& x, const std::vector<std::int64_t>& channels) {
  nn::Shape shape(x.shape().begin(), x.shape().begin() + 3);
  shape.insert(shape.end(), channels.begin(), channels.end());
  return nn::reshape(x, shape);
}
/// Release order: channel c moves to (c mod g)*(C/g) + floor(c/g).
inline nn::Tensor ChannelShuffle(const nn::Tensor& x, std::int64_t channels, std::int64_t groups) {
  if (groups < 1 || channels % groups) throw std::invalid_argument("channel shuffle needs channels divisible by groups");
  if (groups == 1) return x;
  return ReshapeChannels(nn::transpose(ReshapeChannels(x, {channels / groups, groups}), {0, 1, 2, 4, 3}), {channels});
}
/// The paper's order: reshape to (groups, C/groups), transpose, flatten.
inline nn::Tensor PaperChannelShuffle(const nn::Tensor& x, std::int64_t channels, std::int64_t groups) {
  if (groups < 1 || channels % groups) throw std::invalid_argument("channel shuffle needs channels divisible by groups");
  return ReshapeChannels(nn::transpose(ReshapeChannels(x, {groups, channels / groups}), {0, 1, 2, 4, 3}), {channels});
}
inline nn::Tensor PadSpatial(const nn::Tensor& x, std::int64_t pad, double value = 0) {
  return nn::pad(x, {{0, 0}, {pad, pad}, {pad, pad}, {0, 0}}, nn::PadMode::Constant, value);
}
struct ShuffleContext {
  nn::ParameterStore& p;
  nn::ParameterStore& state;
  bool                training;
  double              momentum     = .9;
  bool                update_stats = true;
  // Optional record of every convolution's output shape and multiply-adds per output position.
  std::vector<std::pair<nn::Shape, std::int64_t>>* layers = nullptr;
};
inline nn::Tensor ShuffleConv(ShuffleContext& c, const std::string& name, nn::Tensor x, std::int64_t out,
                              std::int64_t kernel, std::int64_t stride, std::int64_t groups, bool depthwise, bool relu) {
  const std::int64_t   in = x.shape()[3];
  nn::ConvLayerOptions o;
  o.bias    = false;
  o.init    = nn::init::he_uniform();
  o.stride  = stride;
  o.padding = nn::Padding::Valid;
  if (kernel > 1) x = PadSpatial(x, kernel / 2);
  nn::Tensor h = depthwise ? nn::depthwise_conv2d(c.p, name, x, 1, kernel, o)
                           : grouped::GroupedConv2D(c.p, name, x, out, kernel, groups, o);
  nn::BatchNormOptions bn;
  bn.momentum     = c.momentum;
  bn.eps          = 1e-5;
  bn.update_stats = c.update_stats;
  h               = nn::batch_norm(c.p, c.state, name + "/bn", h, c.training, bn);
  if (c.layers) c.layers->push_back({h.shape(), kernel * kernel * (depthwise ? in : in * out / groups)});
  return relu ? nn::relu(h) : h;
}
/// 3x3/2 average pooling with one row of zero padding counted in the mean.
inline nn::Tensor AveragePoolWithPadding(const nn::Tensor& x) {
  return nn::avg_pool2d(PadSpatial(x, 1), 3, 2, nn::Padding::Valid);
}
inline nn::Tensor ShuffleV1Unit(ShuffleContext& c, const std::string& name, const nn::Tensor& x, std::int64_t out,
                                std::int64_t groups, bool first, int stride) {
  const std::int64_t in = x.shape()[3], mid = out / 4, branch = stride == 2 ? out - in : out;
  if (mid % groups || branch % groups || (stride == 1 && in != out)) throw std::invalid_argument("invalid ShuffleNet v1 unit");
  nn::Tensor h = ShuffleConv(c, name + "/pointwise1", x, mid, 1, 1, first ? 1 : groups, false, true);
  h            = ShuffleConv(c, name + "/depthwise", ChannelShuffle(h, mid, groups), mid, 3, stride, 1, true, false);
  h            = ShuffleConv(c, name + "/pointwise2", h, branch, 1, 1, groups, false, false);
  if (stride == 1) return nn::relu(h + x);
  return nn::concat({AveragePoolWithPadding(x), nn::relu(h)}, 3);
}
inline nn::Tensor ShuffleV2Unit(ShuffleContext& c, const std::string& name, const nn::Tensor& x, std::int64_t out,
                                int stride) {
  const std::int64_t in = x.shape()[3], mid = out / 2;
  if (stride == 1) {
    if (in != out || in % 4) throw std::invalid_argument("invalid ShuffleNet v2 unit");
    const auto parts = nn::split(ReshapeChannels(x, {in / 2, 2}), 2, 4);
    const auto even = ReshapeChannels(parts[0], {in / 2}), odd = ReshapeChannels(parts[1], {in / 2});
    nn::Tensor h = ShuffleConv(c, name + "/pointwise1", odd, mid, 1, 1, 1, false, true);
    h            = ShuffleConv(c, name + "/depthwise", h, mid, 3, 1, 1, true, false);
    h            = ShuffleConv(c, name + "/pointwise2", h, out - in / 2, 1, 1, 1, false, true);
    return nn::concat({even, h}, 3);
  }
  nn::Tensor projection = ShuffleConv(c, name + "/projection/depthwise", x, in, 3, 2, 1, true, false);
  projection            = ShuffleConv(c, name + "/projection/pointwise", projection, in, 1, 1, 1, false, true);
  nn::Tensor h          = ShuffleConv(c, name + "/pointwise1", x, mid, 1, 1, 1, false, true);
  h                     = ShuffleConv(c, name + "/depthwise", h, mid, 3, 2, 1, true, false);
  h                     = ShuffleConv(c, name + "/pointwise2", h, out - in, 1, 1, 1, false, true);
  return nn::concat({projection, h}, 3);
}
struct ShuffleNetConfig {
  int         version = 2, groups = 3, classes = 1000;
  std::string size       = "1.0x";
  bool        small_stem = false;
  double      momentum   = .9;
  bool        update_stats = true;
  // Stem width, stage widths and (v2) the final 1x1 width, as released.
  std::vector<int> Widths() const {
    static const std::map<std::string, std::vector<int>> v1g3 = {
        {"0.5x", {12, 120, 240, 480}}, {"1.0x", {24, 240, 480, 960}}, {"1.5x", {24, 360, 720, 1440}}, {"2.0x", {48, 480, 960, 1920}}};
    static const std::map<std::string, std::vector<int>> v1g8 = {{"0.5x", {16, 192, 384, 768}},
                                                                 {"1.0x", {24, 384, 768, 1536}},
                                                                 {"1.5x", {24, 576, 1152, 2304}},
                                                                 {"2.0x", {48, 768, 1536, 3072}}};
    static const std::map<std::string, std::vector<int>> v2 = {{"0.5x", {24, 48, 96, 192, 1024}},
                                                               {"1.0x", {24, 116, 232, 464, 1024}},
                                                               {"1.5x", {24, 176, 352, 704, 1024}},
                                                               {"2.0x", {24, 244, 488, 976, 2048}}};
    if (version != 1 && version != 2) throw std::invalid_argument("invalid ShuffleNet configuration");
    if (version == 1 && groups != 3 && groups != 8) throw std::invalid_argument("ShuffleNet v1 releases use 3 or 8 groups");
    const auto& table = version == 2 ? v2 : groups == 3 ? v1g3 : v1g8;
    if (classes < 2 || !table.count(size)) throw std::invalid_argument("invalid ShuffleNet configuration");
    return table.at(size);
  }
};
struct ShuffleNetOutputs {
  nn::Tensor                                      features, logits;
  std::vector<nn::Tensor>                         stages;
  std::vector<std::pair<nn::Shape, std::int64_t>> layers;  // convolution output shapes and multiply-adds per position
  std::int64_t                                    classifier_macs = 0;
};
/// Dropout (v2 2.0x) draws from p.generator().
inline ShuffleNetOutputs ShuffleNet(nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& image,
                                    const ShuffleNetConfig& cfg = {}, bool training = false) {
  const auto        widths = cfg.Widths();
  ShuffleNetOutputs out;
  ShuffleContext    c{p, state, training, cfg.momentum, cfg.update_stats, &out.layers};
  nn::Tensor        h = ShuffleConv(c, "conv0", image, widths[0], 3, cfg.small_stem ? 1 : 2, 1, false, true);
  if (!cfg.small_stem)
    h = nn::max_pool2d(PadSpatial(h, 1, -std::numeric_limits<double>::infinity()), 3, 2, nn::Padding::Valid);
  std::int64_t in          = widths[0];
  const int    repeats[3] = {4, 8, 4};
  for (int stage = 0; stage < 3; ++stage) {
    const int width = widths[stage + 1];
    for (int unit = 0; unit < repeats[stage]; ++unit, in = width) {
      const auto name = "stage" + std::to_string(stage + 2) + "/unit" + std::to_string(unit + 1);
      h = cfg.version == 1 ? ShuffleV1Unit(c, name, h, width, cfg.groups, stage == 0 && unit == 0, unit ? 1 : 2)
                           : ShuffleV2Unit(c, name, h, width, unit ? 1 : 2);
    }
    out.stages.push_back(h);
  }
  if (cfg.version == 2) {
    h  = ShuffleConv(c, "conv_last", h, widths[4], 1, 1, 1, false, true);
    in = widths[4];
  }
  out.features      = h;
  nn::Tensor pooled = nn::global_avg_pool2d(h);
  if (training && cfg.version == 2 && cfg.size == "2.0x") pooled = nn::dropout(pooled, .2, true, p.generator());
  nn::DenseOptions classifier;
  classifier.bias     = false;
  out.logits          = nn::dense(p, "output", pooled, cfg.classes, classifier);
  out.classifier_macs = in * cfg.classes;
  return out;
}

}  // namespace shufflenet
