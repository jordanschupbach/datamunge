#pragma once
/// @file c3d_models.hpp
/// @brief Complete differentiable C3D topology and source ceil pooling.
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief Configure all eight C3D convolutions,five pools and two dense layers.
///
/// Architecture metadata;full Sports1M shape64/4096/487/112.
/// Width/image/head adaptation keeps22 learned leaves and every layer.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::C3dConfig compact{4, 64, 3, 32};
/// @endcode
/// @par References
/// - Tran et al.(2015),Learning Spatiotemporal Features with 3D Convolutional Networks, https://arxiv.org/abs/1412.0767. Source teaching C3D uses RGB*2-1,He-uniform convolution,library Glorot dense,ceil pools and a compact32px/base4/FC64/three-action head;not pretrained Sports1M or benchmark training.
/// @see C3d()
struct C3dConfig {
  int base = 64;     ///< Positive base convolution width;compact4.
  int dense = 4096;  ///< Positive width of both hidden dense layers;compact64.
  int classes = 487; ///< At least2 head classes;compact3 original actions.
  int image = 112;   ///< Fixed square side>=16;compact32.
};

/// @brief Own every C3D endpoint and its differentiation graph.
///
/// No BN or mutable inference statistics;handles own graph/storage lifetimes.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(643);
/// const auto out = m::C3d(p, n::zeros({1, 16, 32, 32, 3}), {4, 64, 3, 32});
/// @endcode
/// @par References
/// - Tran et al.(2015),Learning Spatiotemporal Features with 3D Convolutional Networks, https://arxiv.org/abs/1412.0767. Source teaching C3D uses RGB*2-1,He-uniform convolution,library Glorot dense,ceil pools and a compact32px/base4/FC64/three-action head;not pretrained Sports1M or benchmark training.
/// @see C3d()
struct C3dOutput {
  Tensor logits;             ///< Raw[N,classes] head scores.
  Tensor flat;               ///< Last pool flattened in time/height/width/channel order.
  Tensor fc6;                ///< First dense/ReLU before optional dropout.
  Tensor fc7;                ///< Second dense/ReLU before optional dropout.
  std::vector<Tensor> conv;  ///< Eight SAME3x3x3 convolution/ReLU endpoints.
  std::vector<Tensor> pools; ///< Five Caffe-ceil endpoints in execution order.
};

/// @brief Pool real3D samples with the source Caffe-ceil trailing-edge rule.
///
/// For equal kernel/stride2,SAME pads only the trailing odd edge.
/// This matches explicit negative-infinity trailing pad then VALID pool.
/// No fitting or input mutation.
///
/// @param x Floating[N,T,H,W,C],positive axes.
/// @param first True selects1x2x2 first pool;false2x2x2.
/// @return Floating pooled tensor,ceil-divided axes,negative padded values cannot win.
/// @throws std::invalid_argument For invalid rank/shape/pooling dtype.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto pooled = m::C3dPool(-n::ones({1, 3, 5, 5, 1}), false);
/// @endcode
/// @par References
/// - Tran et al.(2015),Learning Spatiotemporal Features with 3D Convolutional Networks, https://arxiv.org/abs/1412.0767. Source teaching C3D uses RGB*2-1,He-uniform convolution,library Glorot dense,ceil pools and a compact32px/base4/FC64/three-action head;not pretrained Sports1M or benchmark training.
/// @see C3d()
inline Tensor C3dPool(const Tensor &x, bool first) {
  if (x.dim() != 5 || std::any_of(x.shape().begin(), x.shape().end(), [](auto value) {
        return value < 1;
      }))
    throw std::invalid_argument("positive NDHWC pool axes required");
  const std::int64_t temporal = first ? 1 : 2;
  return nn::max_pool3d(x, {temporal, 2, 2}, {temporal, 2, 2}, Padding::Same);
}

/// @brief Classify16-frame RGB clips with the complete source C3D topology.
///
/// Tensor scaling RGB*2-1 precedes eight SAME He-uniform3x3x3/ReLU
/// layers,width multiples1/2/4/4/8/8/8/8. Pools after1/2/4/6/8,first
/// keeps time. Two Glorot dense/ReLU/dropouts then linear head. No BN,
/// pretrained weights,hidden optimizer or data fitting.
///
/// @param p Mutable learned store;warm all22 leaves before optimizer construction.
/// @param rgb Finite floating[N,16,image,image,3] in[0,1],dtype matching p,N>0.
/// @param cfg Valid positive widths,classes>=2,image>=16;full defaults.
/// @param training False by default;true enables dense-layer dropout.
/// @param dropout_gen Required mutable generator if training and keep<1;otherwise unused.
/// @param keep Finite keep probability(0,1],default.5 after each dense ReLU.
/// @return Raw logits,flat/dense features and every convolution/pool activation.
/// @throws std::invalid_argument For invalid geometry,dtype,range,dropout policy or store leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(643);
/// const auto out = m::C3d(p, n::zeros({1, 16, 32, 32, 3}), {4, 64, 3, 32});
/// @endcode
/// @par References
/// - Tran et al.(2015),Learning Spatiotemporal Features with 3D Convolutional Networks, https://arxiv.org/abs/1412.0767. Source teaching C3D uses RGB*2-1,He-uniform convolution,library Glorot dense,ceil pools and a compact32px/base4/FC64/three-action head;not pretrained Sports1M or benchmark training.
/// @see C3d()
inline C3dOutput C3d(nn::ParameterStore &p, const Tensor &rgb, const C3dConfig &cfg = {},
                     bool training = false, nn::Generator *dropout_gen = nullptr,
                     double keep = .5) {
  if (cfg.base < 1 || cfg.dense < 1 || cfg.classes < 2 || cfg.image < 16)
    throw std::invalid_argument("C3D configuration");
  if (rgb.dim() != 5 || rgb.size(0) < 1 ||
      rgb.shape() != nn::Shape{rgb.size(0), 16, cfg.image, cfg.image, 3} ||
      rgb.dtype() != p.dtype() || !std::isfinite(keep) || keep <= 0 || keep > 1 ||
      (training && keep < 1 && !dropout_gen) || !nn::all(nn::is_finite(rgb)).item<bool>() ||
      !nn::all(nn::logical_and(nn::greater_equal(rgb, 0.), nn::less_equal(rgb, 1.)))
           .item<bool>())
    throw std::invalid_argument(
        "finite unit RGB[N,16,image,image,3] and explicit dropout policy required");
  C3dOutput result;
  Tensor x = rgb * 2.0 - 1.0;
  const int multiples[] = {1, 2, 4, 4, 8, 8, 8, 8};
  const char *names[] = {"conv1a", "conv2a", "conv3a", "conv3b",
                         "conv4a", "conv4b", "conv5a", "conv5b"};
  for (int layer = 0; layer < 8; ++layer) {
    nn::ConvLayerOptions opt;
    opt.init = nn::init::he_uniform();
    opt.activation = nn::Activation::Relu;
    x = nn::conv3d(p, names[layer], x, cfg.base * multiples[layer], 3, opt);
    result.conv.push_back(x);
    if (layer == 0 || layer == 1 || layer == 3 || layer == 5 || layer == 7) {
      x = C3dPool(x, layer == 0);
      result.pools.push_back(x);
    }
  }
  const auto &s = x.shape();
  result.flat = nn::reshape(x, {s[0], s[1] * s[2] * s[3] * s[4]});
  result.fc6 = nn::dense(p, "fc6", result.flat, cfg.dense, nn::Activation::Relu);
  Tensor feature = result.fc6;
  if (training && keep < 1)
    feature = nn::dropout(feature, 1 - keep, true, *dropout_gen);
  result.fc7 = nn::dense(p, "fc7", feature, cfg.dense, nn::Activation::Relu);
  feature = result.fc7;
  if (training && keep < 1)
    feature = nn::dropout(feature, 1 - keep, true, *dropout_gen);
  result.logits = nn::dense(p, "fc8", feature, cfg.classes);
  return result;
}

} // namespace datamunge::nn::models
