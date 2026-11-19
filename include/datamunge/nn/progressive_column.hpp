#pragma once

/// @file progressive_column.hpp
/// @brief Dense progressive transfer with frozen lateral sources.

#include <cmath>
#include <stdexcept>

#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Hidden states and predictions exposed by a progressive column.
/// @par Example
/// @code
/// dm::nn::ProgressiveColumn output;
/// @endcode
struct ProgressiveColumn {
  Tensor first; ///< First hidden activation.
  Tensor second; ///< Second hidden activation.
  Tensor prediction; ///< Linear output.
};

/// @brief Evaluate a two-layer column with optional frozen lateral features.
/// The lateral projections remain trainable in this column. Their source
/// activations are detached so training does not differentiate into earlier columns.
/// @param model Store initialized on first use.
/// @param x Floating input [batch,features] in store dtype.
/// @param previous Optional result supplying same-batch hidden activations.
/// @param gate Finite multiplier for lateral contributions. Use zero to ablate them.
/// @param width Positive hidden width, default 16.
/// @param outputs Positive output width, default one.
/// @return Hidden activations and output of this column.
/// @throws std::invalid_argument For invalid inputs, gate or stored parameters.
/// @par Example
/// @code
/// dm::nn::ParameterStore a(1), b(2);
/// auto x=dm::nn::zeros({2,3});
/// auto old=dm::nn::progressive_column(a,x);
/// auto next=dm::nn::progressive_column(b,x,&old);
/// @endcode
/// @par References
/// - Rusu et al. (2016), Progressive Neural Networks. https://arxiv.org/abs/1606.04671
/// @see dense()
inline ProgressiveColumn progressive_column(ParameterStore& model,const Tensor& x,
    const ProgressiveColumn* previous=nullptr,double gate=1,int width=16,int outputs=1) {
  if (!x.defined() || x.dim()!=2 || x.size(0)<1 || x.size(1)<1 ||
      x.dtype()!=model.dtype() || width<1 || outputs<1 || !std::isfinite(gate))
    throw std::invalid_argument("Invalid progressive column input");
  if (previous && (!previous->first.defined() || !previous->second.defined() ||
      previous->first.dim()!=2 || previous->second.dim()!=2 ||
      previous->first.size(0)!=x.size(0) || previous->second.size(0)!=x.size(0) ||
      previous->first.size(1)<1 || previous->second.size(1)<1 ||
      previous->first.dtype()!=x.dtype() || previous->second.dtype()!=x.dtype()))
    throw std::invalid_argument("Invalid progressive lateral activations");
  const auto first=dense(model,"hidden1",x,width,Activation::Tanh);
  auto second=dense(model,"hidden2",first,width);
  if (previous) second=second+gate*dense(model,"lateral1",previous->first.detach(),width);
  second=tanh(second);
  auto prediction=dense(model,"head",second,outputs);
  if (previous) prediction=prediction+gate*dense(model,"lateral2",previous->second.detach(),outputs);
  return {first,second,prediction};
}

}  // namespace datamunge::nn
