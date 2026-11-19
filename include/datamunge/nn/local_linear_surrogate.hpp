#pragma once

/// @file local_linear_surrogate.hpp
/// @brief Gaussian-weighted local ridge surrogates for numerical inputs.

#include <cmath>
#include <stdexcept>

#include "datamunge/nn/linalg.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Compute Gaussian neighborhood weights from centered feature rows.
/// @param displacement Floating [rows,features] offsets from the anchor.
/// @param width Finite positive kernel width.
/// @return Weights [rows,1], equal to exp(-squared distance / width squared).
/// @throws std::invalid_argument For invalid shape, dtype or width.
/// @par Example
/// @code
/// auto weights=dm::nn::local_kernel_weights(dm::nn::zeros({2,3}),.45);
/// @endcode
/// @see fit_local_linear_surrogate()
inline Tensor local_kernel_weights(const Tensor& displacement,double width) {
  if (!displacement.defined() || displacement.dim()!=2 || displacement.size(1)<1 ||
      (displacement.dtype()!=DType::Float32 && displacement.dtype()!=DType::Float64) ||
      !std::isfinite(width) || width<=0)
    throw std::invalid_argument("Invalid local kernel displacement or width");
  return exp(-sum(square(displacement/width),{1},true));
}

/// @brief Fit a locally weighted linear surrogate with an unpenalized intercept.
/// Fits in double precision with the Gaussian kernel applied to distances from the anchor.
/// @param displacement Floating offsets [rows,features] with at least one row.
/// @param response Black-box response [rows,1].
/// @param width Positive kernel width.
/// @param ridge Finite positive slope penalty.
/// @return Coefficients [features+1,1], intercept first.
/// @throws std::invalid_argument For invalid shapes, width or ridge.
/// @par Example
/// @code
/// auto beta=dm::nn::fit_local_linear_surrogate(dm::nn::zeros({2,3}),dm::nn::ones({2,1}),.45,.03);
/// @endcode
/// @par References
/// - Ribeiro, Singh and Guestrin (2016), Why Should I Trust You? https://arxiv.org/abs/1602.04938
/// @see local_kernel_weights()
inline Tensor fit_local_linear_surrogate(const Tensor& displacement,const Tensor& response,
                                          double width,double ridge) {
  if (!displacement.defined() || displacement.dim()!=2 || displacement.size(0)<1 ||
      !response.defined() || response.shape()!=Shape{displacement.size(0),1} ||
      !std::isfinite(ridge) || ridge<=0)
    throw std::invalid_argument("Invalid local linear fit operands");
  const auto weight=local_kernel_weights(displacement.to(DType::Float64),width);
  const auto design=concat({ones({displacement.size(0),1},DType::Float64),
                            displacement.to(DType::Float64)},1);
  const auto slopes=displacement.size(1);
  const auto penalty=eye(slopes+1,DType::Float64)*concat({zeros({1},DType::Float64),
                                                         full({slopes},ridge,DType::Float64)},0);
  const auto transposed=transpose(design);
  return solve(matmul(transposed,weight*design)+penalty,
               matmul(transposed,weight*response.to(DType::Float64)));
}

}  // namespace datamunge::nn
