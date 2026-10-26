// CondenseNet: densely connected networks with learned group convolutions
// (port of tflow's common/nn/condensenet.h).
// https://arxiv.org/abs/1711.09224
// https://github.com/ShichenLiu/CondenseNet (layers.py, models/condensenet.py)
//
// A learned group convolution (LGC) is BN/ReLU and a 1x1 convolution whose
// weights are multiplied by a mask. Output channel o belongs to group
// o mod G. Training progress p = step / total runs through C-1 condensing
// stages: stage floor(2(C-1)p), capped at C-1, so condensation finishes
// halfway. Entering a stage, each group drops the in/C still-active input
// channels with the smallest summed |w| over its outputs; finally each group
// reads in/C channels. Until then a group lasso, the sum over groups and
// inputs of sqrt(max(sum of the group's squared weights, 1e-6)), is added
// to the loss. The mask, the stage and a step counter are kept in the state
// store; each training-mode forward pass reads the current mask, then (as
// tflow's per-step update ops) advances the stage/mask from the pre-step
// weights and counts the step. A dense layer is LGC to 4k channels, then
// BN/ReLU and a 3x3 group convolution to k channels, concatenated to its
// input. Stages are separated by 2x2 average pooling only (no transition
// convolutions), and growth increases from stage to stage. The stem is a
// bias-free 3x3 convolution to 2k0 channels (stride two on ImageNet); the
// head is BN/ReLU, global pooling and a classifier. For inference an LGC
// converts to an index selection of each group's surviving channels (BN
// statistics copied), BN/ReLU, a G-group 1x1 convolution and a channel
// shuffle back to the original output order.
#pragma once

#include <algorithm>
#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common/grouped.hpp"
#include "common/mobilenet.hpp"

namespace condensenet {

namespace nn = datamunge::nn;
using mobilenet::MobileLayer;

struct CondenseContext {
  nn::ParameterStore&       p;
  nn::ParameterStore&       state;
  std::vector<MobileLayer>& layers;
  bool                      training;
  double                    momentum = .9;
  double                    progress = 0;  // training progress in [0,1] (training passes)
  std::vector<nn::Tensor>*  penalties = nullptr;
  double                    lasso     = 0;
  bool                      update_stats = true;  // training mode: update BN statistics and the condensing state
};
struct LearnedGroupConvOutputs {
  nn::Tensor normalized, output, mask;
};
inline nn::Tensor BNReLU(CondenseContext& c, const std::string& name, const nn::Tensor& x) {
  nn::BatchNormOptions bn;
  bn.momentum     = c.momentum;
  bn.eps          = 1e-5;
  bn.update_stats = c.update_stats;
  return nn::relu(nn::batch_norm(c.p, c.state, name, x, c.training, bn));
}
/// The stage/mask update of one LGC "<name>" for the given progress (tflow's
/// update op): entering a new stage drops in/C still-active channels per group
/// with the smallest summed |masked w| (ties: lower index first, as TopK).
inline void UpdateCondensation(nn::ParameterStore& p, nn::ParameterStore& state, const std::string& name, int groups,
                               int condense, double progress) {
  nn::NoGradGuard    guard;
  const nn::Tensor   mask_t = state.at(name + "/mask");
  const std::int64_t in = mask_t.shape()[0], out = mask_t.shape()[1], per = out / groups;
  const double       now   = std::min<double>(std::floor(float(progress) * 2.f * float(condense - 1)), double(condense - 1));
  const double       stage = state.at(name + "/stage").item<float>();
  if (!(now > stage)) return;
  auto       mask   = mask_t.to_vector<float>();
  const auto weight = p.at(name + "/W").to_vector<float>();
  for (int g = 0; g < groups; ++g) {
    std::vector<std::pair<double, std::int64_t>> scores;
    for (std::int64_t i = 0; i < in; ++i)
      if (mask[static_cast<std::size_t>(i * out + g)] > .5f) {
        float sum = 0;
        for (std::int64_t k = 0; k < per; ++k) {
          const auto j = static_cast<std::size_t>(i * out + k * groups + g);
          sum += std::abs(weight[j] * mask[j]);
        }
        scores.push_back({sum, i});
      }
    std::sort(scores.begin(), scores.end());
    for (std::int64_t d = 0; d < in / condense && d < std::int64_t(scores.size()); ++d)
      for (std::int64_t k = 0; k < per; ++k) mask[static_cast<std::size_t>(scores[d].second * out + k * groups + g)] = 0;
  }
  state.set(name + "/mask", nn::from_vector(mask, {in, out}));
  state.set(name + "/stage", nn::scalar(std::max(stage, now)));
}
/// The masked 1x1 convolution after BN/ReLU; the weights times the mask form an [in, out] matrix.
inline LearnedGroupConvOutputs LearnedGroupConv(CondenseContext& c, const std::string& name, const nn::Tensor& x,
                                                std::int64_t out, int groups, int condense) {
  const std::int64_t in = x.shape()[3];
  if (in % groups || out % groups || in % condense || condense < 2)
    throw std::invalid_argument("invalid learned group convolution");
  LearnedGroupConvOutputs r;
  r.normalized            = BNReLU(c, name + "/bn", x);
  const nn::Tensor weight = c.p.get(name + "/W", {1, 1, in, out}, nn::init::he_uniform());
  // A detached copy: the condensing update below rewrites the stored mask in place.
  r.mask = c.state.get(name + "/mask", {in, out}, nn::init::ones()).detach().clone();
  c.state.get(name + "/stage", {}, nn::init::zeros());
  const nn::Tensor masked = nn::reshape(weight, {in, out}) * r.mask;
  r.output = nn::conv2d(r.normalized, nn::reshape(masked, {1, 1, in, out}), nn::conv_options(1, nn::Padding::Valid));
  c.layers.push_back({name, r.output.shape(), in * out});
  if (!c.training) return r;
  if (c.penalties && c.lasso > 0 && c.state.at(name + "/stage").item<float>() < condense - 1) {
    const nn::Tensor squared = nn::sum(nn::reshape(nn::square(masked), {in, out / groups, groups}), {1});
    const nn::Tensor lasso   = nn::sum(nn::sqrt(nn::maximum(squared, nn::scalar(1e-6))));
    c.penalties->push_back(lasso * c.lasso);
  }
  if (c.update_stats) UpdateCondensation(c.p, c.state, name, groups, condense, c.progress);
  return r;
}
struct CondenseNetConfig {
  std::vector<int> stages = {14, 14, 14}, growth = {8, 16, 32};
  int              groups = 4, condense = 4, bottleneck = 4, classes = 10;
  bool             imagenet = false;
  double           lasso = 1e-5, momentum = .9;
  bool             update_stats = true;
};
struct CondenseNetOutputs {
  nn::Tensor               features, logits;
  std::vector<MobileLayer> layers;
  std::vector<nn::Tensor>  penalties;
  int                      channels = 0;
};
/// Training passes need the run's total number of steps for the schedule.
inline CondenseNetOutputs CondenseNet(nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& image,
                                      const CondenseNetConfig& cfg = {}, bool training = false,
                                      std::int64_t total_steps = 0) {
  if (cfg.stages.empty() || cfg.stages.size() != cfg.growth.size() || cfg.classes < 2 || (training && total_steps < 1))
    throw std::invalid_argument("invalid CondenseNet configuration");
  CondenseNetOutputs out;
  CondenseContext    c{p, state, out.layers, training, cfg.momentum, 0, &out.penalties, cfg.lasso, cfg.update_stats};
  if (training) {
    // One training step per training-mode pass; progress is the fraction completed.
    const nn::Tensor step = state.get("condense/step", {}, nn::init::zeros());
    c.progress            = double(step.item<float>()) / double(total_steps);
  }
  int                  channels = 2 * cfg.growth[0];
  nn::ConvLayerOptions stem;
  stem.bias    = false;
  stem.init    = nn::init::he_uniform();
  stem.stride  = cfg.imagenet ? 2 : 1;
  stem.padding = nn::Padding::Valid;
  nn::Tensor h = nn::conv2d(p, "stem", nn::pad(image, {{0, 0}, {1, 1}, {1, 1}, {0, 0}}), channels, 3, stem);
  c.layers.push_back({"stem", h.shape(), 27LL * channels});
  for (std::size_t stage = 0; stage < cfg.stages.size(); ++stage) {
    const int k = cfg.growth[stage], wide = cfg.bottleneck * k;
    for (int layer = 0; layer < cfg.stages[stage]; ++layer) {
      const auto name = "stage" + std::to_string(stage + 1) + "/layer" + std::to_string(layer + 1);
      nn::Tensor y    = LearnedGroupConv(c, name + "/lgc", h, wide, cfg.groups, cfg.condense).output;
      nn::ConvLayerOptions conv;
      conv.bias    = false;
      conv.init    = nn::init::he_uniform();
      conv.padding = nn::Padding::Valid;
      y = grouped::GroupedConv2D(p, name + "/conv", nn::pad(BNReLU(c, name + "/conv/bn", y), {{0, 0}, {1, 1}, {1, 1}, {0, 0}}),
                                 k, 3, cfg.groups, conv);
      c.layers.push_back({name + "/conv", y.shape(), 9LL * wide * k / cfg.groups});
      h = nn::concat({h, y}, 3);
      channels += k;
    }
    if (stage + 1 < cfg.stages.size()) h = nn::avg_pool2d(h, 2, 2, nn::Padding::Valid);
  }
  out.features = BNReLU(c, "final/bn", h);
  out.logits   = nn::dense(p, "output", nn::global_avg_pool2d(out.features), cfg.classes);
  out.channels = channels;
  if (training && cfg.update_stats) {
    nn::NoGradGuard guard;
    state.set("condense/step", state.at("condense/step") + 1.0);
  }
  return out;
}
/// Inference form of a condensed LGC from its weights [in,out] (any shape with
/// in*out values), mask [in,out]: each group's surviving input channels in
/// increasing order and the [1,1,width,out] grouped kernel (group g's outputs
/// are g*per .. g*per+per-1, i.e. before the channel shuffle).
struct CondensedConv {
  std::vector<std::int64_t> index;
  nn::Tensor                weight;
};
inline CondensedConv Condense(const nn::Tensor& weight, const nn::Tensor& mask, int groups) {
  const std::int64_t in = mask.shape()[0], out = mask.shape()[1], per = out / groups;
  const auto         m = mask.to_vector<float>(), w = weight.to_vector<float>();
  CondensedConv      r;
  std::vector<std::vector<std::int64_t>> kept(static_cast<std::size_t>(groups));
  for (int g = 0; g < groups; ++g)
    for (std::int64_t i = 0; i < in; ++i)
      if (m[static_cast<std::size_t>(i * out + g)] > .5f) kept[g].push_back(i);
  const std::int64_t width = std::int64_t(kept[0].size());
  for (const auto& k : kept)
    if (std::int64_t(k.size()) != width) throw std::logic_error("unevenly condensed groups");
  std::vector<float> kernel(static_cast<std::size_t>(width * out));
  for (int g = 0; g < groups; ++g)
    for (std::int64_t j = 0; j < width; ++j) {
      r.index.push_back(kept[g][j]);
      for (std::int64_t k = 0; k < per; ++k)
        kernel[static_cast<std::size_t>(j * out + g * per + k)] = w[static_cast<std::size_t>(kept[g][j] * out + k * groups + g)];
    }
  r.weight = nn::from_vector(kernel, {1, 1, width, out});
  return r;
}

}  // namespace condensenet
