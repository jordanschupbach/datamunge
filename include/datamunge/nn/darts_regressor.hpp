#pragma once

/// @file darts_regressor.hpp
/// @brief Mixed and discrete regressors for differentiable architecture search.

#include <array>
#include <stdexcept>
#include <string>

#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Evaluate a two-layer mixed-operation or discrete DARTS regressor.
/// @param model Trainable operation weights initialized on first use.
/// @param x Floating [batch,features] inputs in the store dtype.
/// @param alpha Architecture logits [2,2] for tanh and ReLU branches.
/// Pass an undefined tensor to select the discrete child described by choices.
/// @param choices Activation indices for each layer, used in discrete mode.
/// @return Regression predictions [batch,1].
/// @throws std::invalid_argument For invalid inputs, choices or stored parameters.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(7);
/// auto y=dm::nn::darts_regression(model,dm::nn::ones({2,3}),dm::nn::zeros({2,2}));
/// @endcode
/// @par References
/// - Liu, Simonyan and Yang (2019), DARTS. https://arxiv.org/abs/1806.09055
/// @see dense(), softmax()
inline Tensor darts_regression(ParameterStore& model,const Tensor& x,const Tensor& alpha,
                               std::array<int,2> choices={0,0}) {
  if (!x.defined() || x.dim()!=2 || x.size(0)<1 || x.size(1)<1 || x.dtype()!=model.dtype() ||
      (alpha.defined() && (alpha.shape()!=Shape{2,2} || alpha.dtype()!=x.dtype())))
    throw std::invalid_argument("Invalid DARTS regression inputs");
  if (!alpha.defined() && (choices[0]<0 || choices[0]>1 || choices[1]<0 || choices[1]>1))
    throw std::invalid_argument("Invalid DARTS discrete operation");
  const auto probability=alpha.defined() ? softmax(alpha) : Tensor();
  auto state=x;
  for (int layer=0;layer<2;++layer) {
    const auto name="layer_"+std::to_string(layer)+"_op_";
    if (probability.defined()) {
      const auto a=dense(model,name+"0",state,16,Activation::Tanh);
      const auto b=dense(model,name+"1",state,16,Activation::Relu);
      const auto p=select(probability,0,layer);
      state=a*select(p,0,0)+b*select(p,0,1);
    } else {
      const auto op=choices[layer];
      state=dense(model,name+std::to_string(op),state,16,op==0?Activation::Tanh:Activation::Relu);
    }
  }
  return dense(model,"head",state,1);
}

}  // namespace datamunge::nn
