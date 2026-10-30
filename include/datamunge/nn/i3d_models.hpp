#pragma once
/// @file i3d_models.hpp
/// @brief Complete inflated Inception-v1 RGB3D graph and Sonnet population-moment units.
#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief Inflate an existing spatial filter into equal temporal slices.
///
/// Tensor repeat/division retain the spatial filter graph. Interior repeated-image
/// response matches2D convolution;finite-window temporal zero borders reduce
/// the available tap count. No fitting,pretrained loading or source mutation.
///
/// @param spatial Float32[kh,kw,in,out] spatial kernel,positive axes.
/// @param time Positive temporal tap count.
/// @return Contiguous Float32[time,kh,kw,in,out],each slice spatial/time.
/// @throws std::invalid_argument For non-Float32,invalid rank/empty axes or nonpositive time.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto inflated = m::InflateI3d(n::ones({3, 3, 2, 4}), 3);
/// @endcode
/// @par References
/// - Carreira and Zisserman(2017),Quo Vadis,Action Recognition? A New Model and the Kinetics Dataset, https://arxiv.org/abs/1705.07750. Source RGB-only teaching adaptation inflates RANDOM spatial filters for all57units,uses Sonnet-v1 offset-only population BN,and compact16frame/32px/divisor8/three-action fits;not ImageNet/Kinetics pretraining or paper accuracy.
/// @see I3d()
inline Tensor InflateI3d(const Tensor &spatial, int time) {
  if (spatial.dtype() != nn::DType::Float32 || spatial.dim() != 4 || time < 1 ||
      std::any_of(
          spatial.shape().begin(), spatial.shape().end(),
          [](auto value) {
            return value < 1;
          }))
    throw std::invalid_argument("I3D inflation expects float HWIO and positive time");
  return (nn::tile(nn::expand_dims(spatial, 0), {time, 1, 1, 1, 1}) /
          static_cast<double>(time))
      .contiguous();
}

/// @brief Create a random spatial Glorot initializer followed by equal temporal inflation.
///
/// Require positive cubic[time,time,time,in,out] and Float32. Draw original
/// spatial HWIO uniform radius sqrt(6/(time*time*(in+out))),then repeat/time.
/// Consumes caller RNG state;no pretrained arrays or fitted weights are loaded.
///
/// @return Stateless callable(shape,dtype,generator) returning a graph-free inflated Float32 kernel.
/// @throws std::invalid_argument When invoked with invalid rank,empty/noncubic shape or non-Float32 dtype.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::Generator gen(673);
/// const auto kernel = m::InflatedGlorot()({3, 3, 3, 2, 4}, n::DType::Float32, gen);
/// @endcode
/// @par References
/// - Carreira and Zisserman(2017),Quo Vadis,Action Recognition? A New Model and the Kinetics Dataset, https://arxiv.org/abs/1705.07750. Source RGB-only teaching adaptation inflates RANDOM spatial filters for all57units,uses Sonnet-v1 offset-only population BN,and compact16frame/32px/divisor8/three-action fits;not ImageNet/Kinetics pretraining or paper accuracy.
/// @see I3d()
inline nn::Initializer InflatedGlorot() {
  return [](const nn::Shape &shape, nn::DType dtype, nn::Generator &gen) {
    if (shape.size() != 5 || dtype != nn::DType::Float32 || shape[0] != shape[1] ||
        shape[0] != shape[2] || std::any_of(shape.begin(), shape.end(), [](auto value) {
          return value < 1;
        }))
      throw std::invalid_argument("positive cubic Float32 THWIO inflation shape required");
    const std::int64_t k = shape[0], in = shape[3], out = shape[4];
    const double limit = std::sqrt(6.0 / static_cast<double>(k * k * (in + out)));
    const Tensor spatial =
        nn::uniform({shape[1], shape[2], in, out}, -limit, limit, gen, dtype);
    return InflateI3d(spatial, static_cast<int>(k));
  };
}

/// @brief Configure the complete RGB Inception-v1 inflated3D graph.
///
/// Integer widths/geometry and moving-decay metadata only. Scaling retains
/// 57normalized units,nine4branchMixed modules and116learned/114moving leaves.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::I3dConfig c{8, 3, 16, 32};
/// @endcode
/// @par References
/// - Carreira and Zisserman(2017),Quo Vadis,Action Recognition? A New Model and the Kinetics Dataset, https://arxiv.org/abs/1705.07750. Source RGB-only teaching adaptation inflates RANDOM spatial filters for all57units,uses Sonnet-v1 offset-only population BN,and compact16frame/32px/divisor8/three-action fits;not ImageNet/Kinetics pretraining or paper accuracy.
/// @see I3d()
struct I3dConfig {
  int divisor = 1;     ///< Positive width divisor,each source branch at least1;compact8.
  int classes = 400;   ///< Positive head classes>=2;compact3.
  int frames = 64;     ///< Exact observed clip length>=16;compact16.
  int image = 224;     ///< Exact square RGB side>=32;compact32.
  float decay = .999f; ///< Finite moving-population decay[0,1),source.999.
};

/// @brief Own complete I3D endpoints,pooled features and categorical logits.
///
/// Owning tensor handles retain graph/storage. BN moments are in the separate
/// caller state store and are excluded from the learned optimizer.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(677), state(679);
/// const auto out = m::I3d(p, state, n::zeros({1, 16, 32, 32, 3}), {8, 3, 16, 32});
/// @endcode
/// @par References
/// - Carreira and Zisserman(2017),Quo Vadis,Action Recognition? A New Model and the Kinetics Dataset, https://arxiv.org/abs/1705.07750. Source RGB-only teaching adaptation inflates RANDOM spatial filters for all57units,uses Sonnet-v1 offset-only population BN,and compact16frame/32px/divisor8/three-action fits;not ImageNet/Kinetics pretraining or paper accuracy.
/// @see I3d()
struct I3dOutput {
  Tensor logits; ///< Raw[N,classes] scores averaged over surviving temporal logits.
  Tensor pooled; ///< Temporal2/spatial-final average pooled activation before dropout.
  std::vector<Tensor> stages; ///< All16stem/pool/Mixed endpoints in source order.
};

/// @brief Apply inflated SAME convolution and offset-only Sonnet-v1 population BN.
///
/// All moments/updates are tensor operations. Unlike fused batch normalization,
/// moving variance is the POPULATION variance without unbiased correction.
/// Mean0/variance1/beta0 initial states;no separate calibration. Frozen moving
/// arrays are detached during inference,and matching learned/state checkpoints
/// are required. Creates leaves but performs no optimizer update.
///
/// @param p Mutable Float32 learned store with name/W and name/beta;no gamma or conv bias.
/// @param state Mutable Float32 name/mean and name/variance moving-population arrays.
/// @param name Caller prefix for normalized convolution unit.
/// @param x Float32NDHWC input with positive channel count.
/// @param out Positive output width.
/// @param kernel Positive cubic tap width;each temporal slice initializes from one random spatial filter.
/// @param stride Positive equal T/H/W stride.
/// @param training Use batch population moments iftrue,moving moments iffalse.
/// @param decay Finite moving-population decay[0,1),source.999.
/// @param update_stats Update moving arrays only during training&&true.
/// @return Float32ReLU((conv-mean)/sqrt(population_variance+.001)+beta).
/// @throws std::invalid_argument For invalid shape,dtype,width/kernel/stride,decay or stored tensors.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(677), s(679);
/// const auto y = m::I3dUnit(p, s, "u", n::ones({2, 3, 3, 3, 1}), 2, 1, 1, true, .5, false);
/// @endcode
/// @par References
/// - Carreira and Zisserman(2017),Quo Vadis,Action Recognition? A New Model and the Kinetics Dataset, https://arxiv.org/abs/1705.07750. Source RGB-only teaching adaptation inflates RANDOM spatial filters for all57units,uses Sonnet-v1 offset-only population BN,and compact16frame/32px/divisor8/three-action fits;not ImageNet/Kinetics pretraining or paper accuracy.
/// @see I3d()
inline Tensor I3dUnit(nn::ParameterStore &p, nn::ParameterStore &state,
                      const std::string &name, const Tensor &x, int out, int kernel,
                      int stride, bool training, double decay, bool update_stats) {
  if (x.dim() != 5 || x.size(4) < 1 || out < 1 || kernel < 1 || stride < 1 ||
      !std::isfinite(decay) || decay < 0 || decay >= 1 || p.dtype() != nn::DType::Float32 ||
      state.dtype() != p.dtype() || x.dtype() != p.dtype())
    throw std::invalid_argument("positive Float32 I3D unit geometry and decay[0,1) required");
  const std::int64_t in = x.shape().back();
  const Tensor w = p.get(name + "/W", {kernel, kernel, kernel, in, out}, InflatedGlorot());
  const Tensor y = nn::conv3d(x, w, nn::conv3d_options(stride, nn::Padding::Same));
  const Tensor beta = p.get(name + "/beta", {out}, nn::init::zeros());
  const Tensor mean = state.get(name + "/mean", {out}, nn::init::zeros());
  const Tensor variance = state.get(name + "/variance", {out}, nn::init::ones());
  Tensor mu = mean.detach(), var = variance.detach();
  if (training) {
    mu = nn::mean(y, {0, 1, 2, 3});
    var = nn::mean(nn::squared_difference(y, mu), {0, 1, 2, 3});
    if (update_stats) {
      nn::NoGradGuard guard;
      state.set(name + "/mean", mean.detach() * decay + mu.detach() * (1 - decay));
      state.set(name + "/variance", variance.detach() * decay + var.detach() * (1 - decay));
    }
  }
  return nn::relu((y - mu) * nn::rsqrt(var + .001) + beta);
}

/// @brief Classify video with all57inflated units and nine four-branch Mixed modules.
///
/// Scale RGB*2-1. Three stem convolutions/four pools/nine4branchMixed blocks;
/// all57unit kernels initialize by inflating RANDOM2DGlorot filters,not pretrained
/// image weights. Sonnet-v1 BN is offset-only,eps.001,population EMA.
/// Temporal2/finalspatial VALID average pool,keep.5dropout,linear1x1x1head
/// then mean remaining temporal logits. Head Glorot/bias0,no optimizer hidden.
///
/// @param p Mutable Float32 learned store;warm all116leaves before optimizer construction.
/// @param state Mutable Float32 moving store;save/restore all114arrays with p.
/// @param rgb Float32[N,frames,image,image,3] RGB normally[0,1],Npositive.
/// @param cfg Width/geometry/decay configuration;full64frame224px400class defaults.
/// @param training False uses moving moments;true uses batch moments and keep.5head dropout.
/// @param update_stats False by default;training&&true alone updates moving population moments.
/// @param dropout_gen Required mutable generator for training,unused for inference.
/// @return Raw categorical logits,pooled activations and16complete source endpoints.
/// @throws std::invalid_argument For invalid architecture,shape,dtype,decay,dropout or stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(677), state(679);
/// const auto out = m::I3d(p, state, n::zeros({1, 16, 32, 32, 3}), {8, 3, 16, 32});
/// @endcode
/// @par References
/// - Carreira and Zisserman(2017),Quo Vadis,Action Recognition? A New Model and the Kinetics Dataset, https://arxiv.org/abs/1705.07750. Source RGB-only teaching adaptation inflates RANDOM spatial filters for all57units,uses Sonnet-v1 offset-only population BN,and compact16frame/32px/divisor8/three-action fits;not ImageNet/Kinetics pretraining or paper accuracy.
/// @see I3d()
inline I3dOutput I3d(nn::ParameterStore &p, nn::ParameterStore &state, const Tensor &rgb,
                     const I3dConfig &cfg = {}, bool training = false,
                     bool update_stats = false, nn::Generator *dropout_gen = nullptr) {
  if (cfg.divisor < 1 || cfg.classes < 2 || cfg.frames < 16 || cfg.image < 32 ||
      !std::isfinite(cfg.decay) || cfg.decay < 0 || cfg.decay >= 1)
    throw std::invalid_argument("I3D dimensions/decay");
  if (rgb.dim() != 5 || rgb.size(0) < 1 ||
      rgb.shape() != nn::Shape{rgb.size(0), cfg.frames, cfg.image, cfg.image, 3} ||
      p.dtype() != nn::DType::Float32 || state.dtype() != p.dtype() ||
      rgb.dtype() != p.dtype() || (training && !dropout_gen))
    throw std::invalid_argument(
        "matching Float32 RGB clip geometry and explicit training dropout required");
  I3dOutput result;
  const auto width = [&](int c) {
    return std::max(1, c / cfg.divisor);
  };
  const auto unit = [&](const std::string &name, const Tensor &x, int out, int k,
                        int stride = 1) {
    return I3dUnit(p, state, name, x, out, k, stride, training, cfg.decay,
                   training && update_stats);
  };
  Tensor x = rgb * 2.0 - 1.0;
  x = unit("Conv3d_1a_7x7", x, width(64), 7, 2);
  result.stages.push_back(x);
  x = nn::max_pool3d(x, {1, 3, 3}, {1, 2, 2}, nn::Padding::Same);
  result.stages.push_back(x);
  x = unit("Conv3d_2b_1x1", x, width(64), 1);
  result.stages.push_back(x);
  x = unit("Conv3d_2c_3x3", x, width(192), 3);
  result.stages.push_back(x);
  x = nn::max_pool3d(x, {1, 3, 3}, {1, 2, 2}, nn::Padding::Same);
  result.stages.push_back(x);
  const int channels[9][6] = {{64, 96, 128, 16, 32, 32},     {128, 128, 192, 32, 96, 64},
                              {192, 96, 208, 16, 48, 64},    {160, 112, 224, 24, 64, 64},
                              {128, 128, 256, 24, 64, 64},   {112, 144, 288, 32, 64, 64},
                              {256, 160, 320, 32, 128, 128}, {256, 160, 320, 32, 128, 128},
                              {384, 192, 384, 48, 128, 128}};
  const char *names[] = {"Mixed_3b", "Mixed_3c", "Mixed_4b", "Mixed_4c", "Mixed_4d",
                         "Mixed_4e", "Mixed_4f", "Mixed_5b", "Mixed_5c"};
  for (int block = 0; block < 9; ++block) {
    if (block == 2) {
      x = nn::max_pool3d(x, {3, 3, 3}, {2, 2, 2}, nn::Padding::Same);
      result.stages.push_back(x);
    }
    if (block == 7) {
      x = nn::max_pool3d(x, {2, 2, 2}, {2, 2, 2}, nn::Padding::Same);
      result.stages.push_back(x);
    }
    const auto *c = channels[block];
    const std::string name = names[block];
    const Tensor a = unit(name + "/b0", x, width(c[0]), 1);
    Tensor b = unit(name + "/b1a", x, width(c[1]), 1);
    b = unit(name + "/b1b", b, width(c[2]), 3);
    Tensor d = unit(name + "/b2a", x, width(c[3]), 1);
    d = unit(name + "/b2b", d, width(c[4]), 3);
    const Tensor pooled = nn::max_pool3d(x, {3, 3, 3}, {1, 1, 1}, nn::Padding::Same);
    const Tensor e = unit(name + "/b3", pooled, width(c[5]), 1);
    x = nn::concat({a, b, d, e}, 4);
    result.stages.push_back(x);
  }
  const std::int64_t incoming = x.shape().back();
  const int side = (cfg.image + 31) / 32;
  result.pooled = nn::avg_pool3d(x, {2, side, side}, {1, 1, 1}, nn::Padding::Valid);
  x = result.pooled;
  if (training)
    x = nn::dropout(x, .5, true, *dropout_gen);
  const Tensor weight =
      p.get("Logits/W", {1, 1, 1, incoming, cfg.classes}, nn::init::glorot_uniform());
  const Tensor score = nn::conv3d(x, weight, nn::conv3d_options(1, nn::Padding::Same));
  const Tensor bias = p.get("Logits/b", {cfg.classes}, nn::init::zeros());
  result.logits = nn::mean(score + bias, {1, 2, 3});
  return result;
}

} // namespace datamunge::nn::models
