#pragma once

/// @file tcav.hpp
/// @brief Linear concept probes and directional sensitivities.

#include <cmath>
#include <stdexcept>

#include "datamunge/nn/linalg.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Fit a concept direction by ridge regression on hidden activations.
/// @param activation Floating [rows,features] matrix.
/// @param concept_labels Concept indicators [rows,1], typically -1 and +1.
/// @param ridge Finite positive slope penalty.
/// @return Double-precision coefficients [features+1,1], unpenalized intercept first.
/// @throws std::invalid_argument For invalid dimensions, dtype or penalty.
/// @par Example
/// @code
/// auto probe=dm::nn::fit_concept_probe(dm::nn::ones({3,2}),dm::nn::ones({3,1}),.15);
/// @endcode
/// @par References
/// - Kim et al. (2018), Interpretability Beyond Feature Attribution. https://proceedings.mlr.press/v80/kim18d.html
/// @see concept_sensitivity()
inline Tensor fit_concept_probe(const Tensor& activation,const Tensor& concept_labels,double ridge) {
  if (!activation.defined() || activation.dim()!=2 || activation.size(0)<1 ||
      activation.size(1)<1 || !concept_labels.defined() ||
      concept_labels.shape()!=Shape{activation.size(0),1} ||
      (activation.dtype()!=DType::Float32 && activation.dtype()!=DType::Float64) ||
      !std::isfinite(ridge) || ridge<=0)
    throw std::invalid_argument("Invalid concept probe inputs");
  const auto width=activation.size(1);
  const auto design=concat({ones({activation.size(0),1},DType::Float64),activation.to(DType::Float64)},1);
  const auto transposed=transpose(design);
  const auto penalty=eye(width+1,DType::Float64)*concat({zeros({1},DType::Float64),
                                                          full({width},ridge,DType::Float64)},0);
  return solve(matmul(transposed,design)+penalty,matmul(transposed,concept_labels.to(DType::Float64)));
}

/// @brief Project activation gradients onto the fitted concept direction.
/// @param gradients Floating [rows,features] derivatives of a class score.
/// @param probe Concept coefficients [features+1,1], intercept first.
/// @return Directional derivative [rows,1] in double precision.
/// @throws std::invalid_argument If shapes are incompatible.
/// @par Example
/// @code
/// auto dy=dm::nn::concept_sensitivity(dm::nn::ones({2,3}),dm::nn::ones({4,1}));
/// @endcode
/// @see fit_concept_probe()
inline Tensor concept_sensitivity(const Tensor& gradients,const Tensor& probe) {
  if (!gradients.defined() || !probe.defined() || gradients.dim()!=2 ||
      gradients.size(1)<1 || probe.shape()!=Shape{gradients.size(1)+1,1})
    throw std::invalid_argument("Invalid concept sensitivity shapes");
  return matmul(gradients.to(DType::Float64),slice_axis(probe.to(DType::Float64),0,1,probe.size(0)));
}

}  // namespace datamunge::nn
