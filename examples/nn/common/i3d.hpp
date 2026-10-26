// RGB Inception-v1 I3D (Carreira & Zisserman 2017): temporal kernel inflation,
// nine four-branch Mixed blocks, and the compact real-video model trained by
// i3d. A port of tflow's common/nn/i3d.h (model) and common/ml/i3d.h
// (training; shared with C3D/R(2+1)D/P3D as c3d::TrainVideoModel).
//
// TF -> datamunge: each unit is nn::conv3d (SAME) plus Sonnet-v1 batch norm
// written out with nn ops (offset only, epsilon .001, population moments over
// N,T,H,W and population-variance moving averages with decay .999), so it is
// not nn::batch_norm (which keeps TF FusedBatchNorm's unbiased moving variance).
// Moving averages are updated only with `update_stats` (TF runs them with the
// training step only). The inflated initial filter is a custom nn::Initializer.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <datamunge/nn/conv.hpp>
#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/nn.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/c3d.hpp"
#include "common/checks.hpp"

namespace i3d {

namespace nn = datamunge::nn;
using nn::Padding;
using nn::Tensor;

/// Inflates a spatial HWIO filter [kh, kw, in, out] to [time, kh, kw, in, out] by
/// repeating it over time and dividing by `time`.
inline Tensor InflateI3d(const Tensor& spatial, int time) {
  if (spatial.dtype() != nn::DType::Float32 || spatial.dim() != 4 || time < 1)
    throw std::invalid_argument("I3D inflation expects float HWIO and positive time");
  return (nn::tile(nn::expand_dims(spatial, 0), {time, 1, 1, 1, 1}) / static_cast<double>(time)).contiguous();
}

/// Random spatial Glorot-uniform filters inflated as W[t] = W2D / kernel (no pretrained weights).
inline nn::Initializer InflatedGlorot() {
  return [](const nn::Shape& shape, nn::DType dtype, nn::Generator& gen) {
    const std::int64_t k = shape[0], in = shape[3], out = shape[4];
    const double       limit = std::sqrt(6.0 / static_cast<double>(k * k * (in + out)));
    const Tensor spatial = nn::uniform({shape[1], shape[2], in, out}, -limit, limit, gen, dtype);
    return InflateI3d(spatial, static_cast<int>(k));
  };
}

struct I3dConfig {
  int   divisor = 1, classes = 400, frames = 64, image = 224;
  float decay   = .999f;
};

struct I3dOutput {
  Tensor              logits, pooled;
  std::vector<Tensor> stages;
};

/// Conv3D "<name>/W" [k, k, k, in, out] (SAME, equal strides, no bias), Sonnet-v1 batch norm with
/// offset "<name>/beta" and moving "<name>/mean", "<name>/variance" in `state`, then ReLU.
inline Tensor I3dUnit(nn::ParameterStore& p, nn::ParameterStore& state, const std::string& name, const Tensor& x,
                      int out, int kernel, int stride, bool training, double decay, bool update_stats) {
  const std::int64_t in = x.shape().back();
  const Tensor w    = p.get(name + "/W", {kernel, kernel, kernel, in, out}, InflatedGlorot());
  const Tensor y    = nn::conv3d(x, w, nn::conv3d_options(stride, Padding::Same));
  const Tensor beta = p.get(name + "/beta", {out}, nn::init::zeros());
  const Tensor mean = state.get(name + "/mean", {out}, nn::init::zeros());
  const Tensor variance = state.get(name + "/variance", {out}, nn::init::ones());
  Tensor mu = mean.detach(), var = variance.detach();
  if (training) {
    mu  = nn::mean(y, {0, 1, 2, 3});
    var = nn::mean(nn::squared_difference(y, mu), {0, 1, 2, 3});
    if (update_stats) {
      nn::NoGradGuard guard;
      state.set(name + "/mean", mean.detach() * decay + mu.detach() * (1 - decay));
      state.set(name + "/variance", variance.detach() * decay + var.detach() * (1 - decay));
    }
  }
  return nn::relu((y - mu) * nn::rsqrt(var + .001) + beta);
}

/// I3D on RGB clips in [0, 1] of shape [N, frames, image, image, 3]. With `training`,
/// batch moments and head dropout (keep .5) drawn from `dropout_gen`.
inline I3dOutput I3d(nn::ParameterStore& p, nn::ParameterStore& state, const Tensor& rgb, const I3dConfig& cfg = {},
                     bool training = false, bool update_stats = false, nn::Generator* dropout_gen = nullptr) {
  if (cfg.divisor < 1 || cfg.classes < 2 || cfg.frames < 16 || cfg.image < 32 || cfg.decay < 0 || cfg.decay >= 1)
    throw std::invalid_argument("I3D dimensions/decay");
  I3dOutput  result;
  const auto width = [&](int c) { return std::max(1, c / cfg.divisor); };
  const auto unit  = [&](const std::string& name, const Tensor& x, int out, int k, int stride = 1) {
    return I3dUnit(p, state, name, x, out, k, stride, training, cfg.decay, training && update_stats);
  };
  Tensor x = rgb * 2.0 - 1.0;
  x        = unit("Conv3d_1a_7x7", x, width(64), 7, 2);
  result.stages.push_back(x);
  x = nn::max_pool3d(x, {1, 3, 3}, {1, 2, 2}, Padding::Same);
  result.stages.push_back(x);
  x = unit("Conv3d_2b_1x1", x, width(64), 1);
  result.stages.push_back(x);
  x = unit("Conv3d_2c_3x3", x, width(192), 3);
  result.stages.push_back(x);
  x = nn::max_pool3d(x, {1, 3, 3}, {1, 2, 2}, Padding::Same);
  result.stages.push_back(x);
  const int channels[9][6] = {{64, 96, 128, 16, 32, 32},     {128, 128, 192, 32, 96, 64},
                              {192, 96, 208, 16, 48, 64},    {160, 112, 224, 24, 64, 64},
                              {128, 128, 256, 24, 64, 64},   {112, 144, 288, 32, 64, 64},
                              {256, 160, 320, 32, 128, 128}, {256, 160, 320, 32, 128, 128},
                              {384, 192, 384, 48, 128, 128}};
  const char* names[] = {"Mixed_3b", "Mixed_3c", "Mixed_4b", "Mixed_4c", "Mixed_4d",
                         "Mixed_4e", "Mixed_4f", "Mixed_5b", "Mixed_5c"};
  for (int block = 0; block < 9; ++block) {
    if (block == 2) {
      x = nn::max_pool3d(x, {3, 3, 3}, {2, 2, 2}, Padding::Same);
      result.stages.push_back(x);
    }
    if (block == 7) {
      x = nn::max_pool3d(x, {2, 2, 2}, {2, 2, 2}, Padding::Same);
      result.stages.push_back(x);
    }
    const auto*       c    = channels[block];
    const std::string name = names[block];
    const Tensor      a    = unit(name + "/b0", x, width(c[0]), 1);
    Tensor            b    = unit(name + "/b1a", x, width(c[1]), 1);
    b                      = unit(name + "/b1b", b, width(c[2]), 3);
    Tensor d               = unit(name + "/b2a", x, width(c[3]), 1);
    d                      = unit(name + "/b2b", d, width(c[4]), 3);
    const Tensor pooled    = nn::max_pool3d(x, {3, 3, 3}, {1, 1, 1}, Padding::Same);
    const Tensor e         = unit(name + "/b3", pooled, width(c[5]), 1);
    x                      = nn::concat({a, b, d, e}, 4);
    result.stages.push_back(x);
  }
  const std::int64_t incoming = x.shape().back();
  const int          side     = (cfg.image + 31) / 32;
  result.pooled = nn::avg_pool3d(x, {2, side, side}, {1, 1, 1}, Padding::Valid);
  x             = result.pooled;
  if (training) x = nn::dropout(x, .5, true, *dropout_gen);
  const Tensor weight = p.get("Logits/W", {1, 1, 1, incoming, cfg.classes}, nn::init::glorot_uniform());
  const Tensor score  = nn::conv3d(x, weight, nn::conv3d_options(1, Padding::Same));
  const Tensor bias   = p.get("Logits/b", {cfg.classes}, nn::init::zeros());
  result.logits       = nn::mean(score + bias, {1, 2, 3});
  return result;
}

// ---- compact real-video model (tflow ml/i3d.h) ---------------------------------------------

/// divisor 8, three KTH classes, 16x32x32 clips; parameters seed 677, dropout seed 673.
struct I3dModel {
  nn::ParameterStore parameters{677}, state{679};
  nn::Generator      dropout{673};
  I3dModel() {
    nn::NoGradGuard guard;  // create every learned and moving tensor
    Logits(nn::zeros({1, 16, 32, 32, 3}), false, false);
  }
  Tensor Logits(const Tensor& rgb, bool training, bool update_stats) {
    return I3d(parameters, state, rgb, {8, 3, 16, 32}, training, update_stats, &dropout).logits;
  }
};

}  // namespace i3d
