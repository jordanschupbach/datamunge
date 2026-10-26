// R(2+1)D-18 (Tran et al. 2018) and P3D-199 (Qiu et al. 2017): separate
// spatial/temporal filters and residual paths, plus the compact real-video
// model trained by factorized_video. A port of tflow's
// common/nn/factorized_video.h (model) and common/ml/factorized_video.h
// (training; shared with C3D/I3D as c3d::TrainVideoModel).
//
// TF -> datamunge: FactorConv's explicit Pad + VALID Conv3D is nn::conv3d with
// Padding::Explicit (symmetric k/2 borders, also at stride 2); tflow's
// BatchNorm (FusedBatchNormV3 + moving averages) is nn::batch_norm with
// momentum .9 and epsilon 1e-5; the P3D stem's PadV2(-inf) + VALID MaxPool3D
// "ceil" pool is nn::pad(-inf) + nn::max_pool3d. Moving-statistic updates run
// only when `update_stats` is set (TF runs them with the training step only).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <datamunge/nn/conv.hpp>
#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/nn.hpp>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/c3d.hpp"

namespace factorized_video {

namespace nn = datamunge::nn;
using nn::Padding;
using nn::Tensor;

/// Parameter-matched intermediate width M = floor(27 in out / (9 in + 3 out)) (at least 1).
inline int FactorMidplanes(int in, int out) {
  if (in < 1 || out < 1) throw std::invalid_argument("positive factorized widths");
  return std::max(1, static_cast<int>(std::int64_t{27} * in * out / (9 * in + 3 * out)));
}

/// Convolution with explicit symmetric (k/2, k/2) borders on T, H, W and the given
/// strides (PyTorch/Caffe padding, including stride-2 borders). kernel [kt, kh, kw, C, O].
inline Tensor FactorConv(const Tensor& x, const Tensor& kernel, int st = 1, int sh = 1, int sw = 1) {
  const auto&        k = kernel.shape();
  nn::Conv3DOptions o;
  o.strides           = {st, sh, sw};
  o.padding           = Padding::Explicit;
  o.explicit_paddings = {k[0] / 2, k[0] / 2, k[1] / 2, k[1] / 2, k[2] / 2, k[2] / 2};
  return nn::conv3d(x, kernel, o);
}

enum class P3dKind { A, B, C };

/// Authors' released Caffe topology: A = T(S(x)), B = relu(S(x) + T(x)),
/// C = relu(.5 (S(x) + T(S(x)))). The callbacks include their own normalization/nonlinearities.
inline Tensor P3dMix(const Tensor& x, const std::function<Tensor(const Tensor&)>& spatial,
                     const std::function<Tensor(const Tensor&)>& temporal, P3dKind kind) {
  const Tensor a = spatial(x);
  const Tensor b = temporal(kind == P3dKind::B ? x : a);
  if (kind == P3dKind::A) return b;
  Tensor combined = a + b;
  if (kind == P3dKind::C) combined = combined * .5;
  return nn::relu(combined);
}

struct FactorVideoConfig {
  bool p3d     = false;
  int  base    = 64;
  int  classes = 400;
};

struct FactorVideoOutput {
  Tensor              logits, pooled;
  std::vector<Tensor> stages;
};

/// Factorized conv "<name>/W" [kt, kh, kw, in, out] (Kaiming normal, fan-out), then batch
/// norm "<name>/bn" (momentum .9, epsilon 1e-5, learned gamma/beta), optional ReLU.
inline Tensor FactorUnit(nn::ParameterStore& p, nn::ParameterStore& state, const std::string& name, const Tensor& x,
                         int out, int kt, int kh, int kw, int st, int sh, int sw, bool relu, bool training,
                         bool update_stats) {
  const std::int64_t in = x.shape().back();
  const Tensor w = p.get(name + "/W", {kt, kh, kw, in, out}, nn::init::normal(std::sqrt(2.0 / (kt * kh * kw * out))));
  const Tensor y = FactorConv(x, w, st, sh, sw);
  nn::BatchNormOptions bn;
  bn.momentum     = .9;
  bn.eps          = 1e-5;
  bn.update_stats = update_stats;
  const Tensor normalized = nn::batch_norm(p, state, name + "/bn", y, training, bn);
  return relu ? nn::relu(normalized) : normalized;
}

/// R(2+1)D-18 (cfg.p3d = false) or P3D-199 on RGB clips in [0, 1] of shape [N, T, H, W, 3].
/// With `training`, batch statistics (and P3D's head dropout, keep .5, from `dropout_gen`).
inline FactorVideoOutput FactorVideo(nn::ParameterStore& p, nn::ParameterStore& state, const Tensor& rgb,
                                     const FactorVideoConfig& cfg = {}, bool training = false,
                                     bool update_stats = false, nn::Generator* dropout_gen = nullptr) {
  if (cfg.base < 2 || cfg.classes < 2) throw std::invalid_argument("factorized video configuration");
  FactorVideoOutput result;
  const auto unit = [&](const std::string& name, const Tensor& x, int out, int kt, int kh, int kw, int st, int sh,
                        int sw, bool relu = true) {
    return FactorUnit(p, state, name, x, out, kt, kh, kw, st, sh, sw, relu, training, training && update_stats);
  };
  Tensor x        = rgb * 2.0 - 1.0;
  int    incoming = cfg.base;
  if (!cfg.p3d) {
    const int middle = std::max(1, 45 * cfg.base / 64);
    x = unit("stem/spatial", x, middle, 1, 7, 7, 1, 2, 2);
    x = unit("stem/temporal", x, cfg.base, 3, 1, 1, 1, 1, 1);
    result.stages.push_back(x);
    for (int stage = 0; stage < 4; ++stage)
      for (int block = 0; block < 2; ++block) {
        const int out = cfg.base << stage, stride = stage && block == 0 ? 2 : 1, mid = FactorMidplanes(incoming, out);
        const std::string name = "stage" + std::to_string(stage) + "/block" + std::to_string(block);
        Tensor skip = x;
        Tensor y    = unit(name + "/first/spatial", x, mid, 1, 3, 3, 1, stride, stride);
        y           = unit(name + "/first/temporal", y, out, 3, 1, 1, stride, 1, 1);
        y           = unit(name + "/second/spatial", y, mid, 1, 3, 3, 1, 1, 1);
        y           = unit(name + "/second/temporal", y, out, 3, 1, 1, 1, 1, 1, false);
        if (stride != 1 || incoming != out) skip = unit(name + "/skip", x, out, 1, 1, 1, stride, stride, stride, false);
        x        = nn::relu(y + skip);
        incoming = out;
        result.stages.push_back(x);
      }
  } else {
    x = unit("stem", x, cfg.base, 1, 7, 7, 1, 2, 2);
    // Caffe CEIL spatial 3/stride 2 pool from an even stem grid: partial end window.
    x = nn::pad(x, {{0, 0}, {0, 0}, {0, 1}, {0, 1}, {0, 0}}, nn::PadMode::Constant,
                -std::numeric_limits<double>::infinity());
    x = nn::max_pool3d(x, {2, 3, 3}, {2, 2, 2}, Padding::Valid);
    result.stages.push_back(x);
    const int depth[] = {3, 8, 36, 3};
    int       index   = 0;
    for (int stage = 0; stage < 4; ++stage) {
      if (stage) x = nn::max_pool3d(x, {2, 1, 1}, {2, 1, 1}, Padding::Valid);
      for (int block = 0; block < depth[stage]; ++block, ++index) {
        const int width = cfg.base << stage, out = 4 * width, stride = stage && block == 0 ? 2 : 1;
        const std::string name = "stage" + std::to_string(stage) + "/block" + std::to_string(block);
        Tensor skip = x;
        Tensor y    = unit(name + "/reduce", x, width, 1, 1, 1, 1, stride, stride);
        if (stage < 3) {
          const auto kind     = static_cast<P3dKind>(index % 3);
          const auto spatial  = [&](const Tensor& v) { return unit(name + "/spatial", v, width, 1, 3, 3, 1, 1, 1); };
          const auto temporal = [&](const Tensor& v) {
            return unit(name + "/temporal", v, width, 3, 1, 1, 1, 1, 1, kind == P3dKind::A);
          };
          y = P3dMix(y, spatial, temporal, kind);
        } else {
          y = unit(name + "/spatial", y, width, 1, 3, 3, 1, 1, 1);  // T = 1: the authors' final 2D stage.
        }
        y = unit(name + "/expand", y, out, 1, 1, 1, 1, 1, 1, false);
        if (incoming != out || stride != 1) skip = unit(name + "/skip", x, out, 1, 1, 1, 1, stride, stride, false);
        x        = nn::relu(skip + y);
        incoming = out;
        result.stages.push_back(x);
      }
    }
  }
  result.pooled = nn::mean(x, {1, 2, 3});
  x             = result.pooled;
  if (cfg.p3d && training) x = nn::dropout(x, .5, true, *dropout_gen);
  const Tensor w = p.get("head/W", {incoming, cfg.classes}, nn::init::normal(.01));
  const Tensor b = p.get("head/b", {cfg.classes}, nn::init::zeros());
  result.logits  = nn::matmul(x, w) + b;
  return result;
}

// ---- compact real-video model (tflow ml/factorized_video.h) ------------------------------

/// base 4, three KTH classes on 16x32x32 clips; parameters seed 727, dropout seed 719.
struct FactorVideoModel {
  bool               p3d;
  nn::ParameterStore parameters{727}, state{729};
  nn::Generator      dropout{719};
  explicit FactorVideoModel(bool p3d_model) : p3d(p3d_model) {
    nn::NoGradGuard guard;  // create every learned and moving tensor
    Logits(nn::zeros({1, 16, 32, 32, 3}), false, false);
  }
  Tensor Logits(const Tensor& rgb, bool training, bool update_stats) {
    return FactorVideo(parameters, state, rgb, {p3d, 4, 3}, training, update_stats, &dropout).logits;
  }
};
struct R2Plus1dModel : FactorVideoModel {
  R2Plus1dModel() : FactorVideoModel(false) {}
};
struct P3dModel : FactorVideoModel {
  P3dModel() : FactorVideoModel(true) {}
};

}  // namespace factorized_video
