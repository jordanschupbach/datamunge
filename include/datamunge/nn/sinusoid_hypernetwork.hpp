#pragma once

/// @file sinusoid_hypernetwork.hpp
/// @brief Support-conditioned weights for a functional scalar regressor.

#include <cmath>
#include <stdexcept>
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Predict scalar values with explicitly supplied one-hidden-layer weights.
/// @param x Floating query matrix [rows,1].
/// @param weights Floating vector [3*width+1] containing input weights, hidden
/// biases, output weights and one output bias, in that order.
/// @return Predictions [rows,1], differentiable with respect to inputs and weights.
/// @throws std::invalid_argument For incompatible dimensions or dtype.
/// @par Example
/// @code
/// auto y=dm::nn::scalar_regressor(dm::nn::ones({2,1}),dm::nn::zeros({37}));
/// @endcode
/// @see sinusoid_hypernetwork_weights()
inline Tensor scalar_regressor(const Tensor& x,const Tensor& weights) {
  if (!x.defined() || !weights.defined() || x.dim()!=2 || x.size(1)!=1 ||
      weights.dim()!=1 || weights.size(0)<4 || (weights.size(0)-1)%3!=0 ||
      x.dtype()!=weights.dtype() || (x.dtype()!=DType::Float32 && x.dtype()!=DType::Float64))
    throw std::invalid_argument("Invalid scalar regressor input or weights");
  const auto width=(weights.size(0)-1)/3;
  const auto incoming=slice_axis(weights,0,0,width), offset=slice_axis(weights,0,width,2*width);
  const auto head=slice_axis(weights,0,2*width,3*width), bias=select(weights,0,3*width);
  return sum(tanh(x*incoming+offset)*head,{1},true)+bias;
}

/// @brief Generate a task regressor from an unordered set of sinusoid observations.
/// The encoder pools learned features of each input, response and their
/// Fourier cross-products. Every generated parameter receives query gradients.
/// @param model Parameter store initialized on first use.
/// @param x Support inputs [rows,1], with at least one row.
/// @param y Support responses matching x and the store dtype.
/// @return Packed 37-parameter vector for scalar_regressor().
/// @throws std::invalid_argument For invalid shapes, dtypes or parameters.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(7);
/// auto w=dm::nn::sinusoid_hypernetwork_weights(model,dm::nn::ones({2,1}),dm::nn::zeros({2,1}));
/// @endcode
/// @par References
/// - Ha, Dai and Le (2017), HyperNetworks. https://arxiv.org/abs/1609.09106
/// @see scalar_regressor()
inline Tensor sinusoid_hypernetwork_weights(ParameterStore& model,const Tensor& x,const Tensor& y) {
  if (!x.defined() || !y.defined() || x.dim()!=2 || x.size(0)<1 || x.size(1)!=1 ||
      y.shape()!=x.shape() || x.dtype()!=model.dtype() || y.dtype()!=model.dtype())
    throw std::invalid_argument("Invalid sinusoid support set");
  const auto pairs=concat({x,y,y*sin(x),y*cos(x)},1);
  const auto encoded=dense(model,"support",pairs,32,Activation::Tanh);
  const auto summary=mean(encoded,{0},true);
  const auto hidden=dense(model,"hyper",summary,32,Activation::Tanh);
  return reshape(dense(model,"weights",hidden,37),{37});
}

/// @brief Compute a differentiable mean-square gradient for a scalar regressor.
/// @param x Support inputs [rows,1], with at least one row.
/// @param y Support targets matching x.
/// @param weights Packed weights as in scalar_regressor().
/// @return Packed support gradient with history retained for meta-gradients.
/// @throws std::invalid_argument For incompatible dimensions, dtypes.
/// @par Example
/// @code
/// auto w=dm::nn::scalar_regressor_gradient(dm::nn::ones({2,1}),dm::nn::ones({2,1}),
///                                      dm::nn::zeros({37}));
/// @endcode
/// @see scalar_regressor()
inline Tensor scalar_regressor_gradient(const Tensor& x,const Tensor& y,const Tensor& weights) {
  if (!x.defined() || !y.defined() || !weights.defined() || x.dim()!=2 || x.size(0)<1 ||
      x.size(1)!=1 || y.shape()!=x.shape() || x.dtype()!=y.dtype() || x.dtype()!=weights.dtype() ||
      weights.dim()!=1 || weights.size(0)<4 || (weights.size(0)-1)%3!=0 ||
      (x.dtype()!=DType::Float32 && x.dtype()!=DType::Float64))
    throw std::invalid_argument("Invalid scalar regressor support step");
  const auto width=(weights.size(0)-1)/3;
  const auto incoming=slice_axis(weights,0,0,width),offset=slice_axis(weights,0,width,2*width);
  const auto head=slice_axis(weights,0,2*width,3*width),bias=select(weights,0,3*width);
  const auto hidden=tanh(x*incoming+offset);
  const auto residual=(2./x.size(0))*(sum(hidden*head,{1},true)+bias-y);
  const auto dh=residual*head*(1-square(hidden));
  return concat({sum(dh*x,{0}),sum(dh,{0}),sum(hidden*residual,{0}),sum(residual,{0})},0);
}

/// @brief Take one differentiable mean-square SGD step on a scalar regressor.
/// @param x Support inputs [rows,1], with at least one row.
/// @param y Support targets matching x.
/// @param weights Packed weights as in scalar_regressor().
/// @param rate Finite nonnegative step size.
/// @return Updated packed weights with history retained for meta-gradients.
/// @throws std::invalid_argument For incompatible dimensions, dtypes or rate.
/// @par Example
/// @code
/// auto w=dm::nn::scalar_regressor_step(dm::nn::ones({2,1}),dm::nn::ones({2,1}),
///                                      dm::nn::zeros({37}),.045);
/// @endcode
/// @see scalar_regressor()
inline Tensor scalar_regressor_step(const Tensor& x,const Tensor& y,const Tensor& weights,double rate) {
  if (!std::isfinite(rate) || rate<0)
    throw std::invalid_argument("Invalid scalar regressor step size");
  return weights-rate*scalar_regressor_gradient(x,y,weights);
}

/// @brief Interpolate shared initialization toward weights adapted to one task.
/// @param initial Shared initialization.
/// @param adapted Support-trained weights of matching shape and dtype.
/// @param rate Interpolation coefficient in [0,1].
/// @return Updated initialization.
/// @throws std::invalid_argument For incompatible parameters or an invalid coefficient.
/// @par Example
/// @code
/// auto w=dm::nn::reptile_update(dm::nn::zeros({37}),dm::nn::ones({37}),.12);
/// @endcode
/// @par References
/// - Nichol et al. (2018), On First-Order Meta-Learning Algorithms. https://arxiv.org/abs/1803.02999
/// @see scalar_regressor_step()
inline Tensor reptile_update(const Tensor& initial,const Tensor& adapted,double rate) {
  if (!initial.defined() || !adapted.defined() || initial.shape()!=adapted.shape() ||
      initial.dtype()!=adapted.dtype() || (initial.dtype()!=DType::Float32 &&
      initial.dtype()!=DType::Float64) || !(rate>=0 && rate<=1))
    throw std::invalid_argument("Invalid Reptile interpolation inputs");
  return initial+rate*(adapted-initial);
}

}  // namespace datamunge::nn
