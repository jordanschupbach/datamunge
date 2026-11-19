#pragma once

/// @file forecast_losses.hpp
/// @brief Ordered quantile outputs and a mean pinball objective.

#include <datamunge/nn/ops.hpp>

#include <stdexcept>

namespace datamunge::nn {

/// @brief Decode a center and two softplus gaps into ordered quantiles.
/// @param raw CPU Float32 or Float64 matrix [rows,3] in center, lower-gap, upper-gap order.
/// @return Differentiable [rows,3] matrix in lower, center, upper order.
/// @throws std::invalid_argument Incompatible shape, dtype, or device.
/// @par Example
/// @code
/// const auto q = dm::nn::ordered_quantile_forecast(dm::nn::zeros({1,3}));
/// std::println("{}", q.size(1)); // 3
/// @endcode
/// @see quantile_forecast_loss(), softplus()
[[nodiscard]] inline Tensor ordered_quantile_forecast(const Tensor& raw) {
  if (!raw.defined() || raw.dim()!=2 || raw.size(0)<1 || raw.size(1)!=3 ||
      raw.device().backend!=compute::Backend::CPU ||
      (raw.dtype()!=DType::Float32 && raw.dtype()!=DType::Float64))
    throw std::invalid_argument("ordered_quantile_forecast: expected CPU float matrix [rows,3]");
  const auto middle=slice(raw,{0,0},{raw.size(0),1});
  const auto below=softplus(slice(raw,{0,1},{raw.size(0),1}));
  const auto above=softplus(slice(raw,{0,2},{raw.size(0),1}));
  return concat({middle-below,middle,middle+above},1);
}

/// @brief Average the pinball loss over rows and quantile levels.
/// @param forecast CPU Float32 or Float64 matrix [rows, quantiles].
/// @param actual Observed values [rows,1] with the same dtype.
/// @param levels Same-dtype CPU vector of quantile probabilities strictly between zero and one.
/// @return Scalar loss with a gradient with respect to forecast.
/// @throws std::invalid_argument Incompatible dimensions, devices, or levels.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto loss=n::quantile_forecast_loss(n::zeros({1,1}),n::ones({1,1}),n::tensor({.5}));
/// std::println("{}", loss.item()); // 0.5
/// @endcode
/// @par References
/// - Koenker and Bassett (1978), Regression Quantiles, https://doi.org/10.2307/1913643
/// @see ordered_quantile_forecast()
[[nodiscard]] inline Tensor quantile_forecast_loss(
    const Tensor& forecast, const Tensor& actual, const Tensor& levels) {
  if (!forecast.defined() || !actual.defined() || !levels.defined() ||
      forecast.dim()!=2 || forecast.numel()==0 || actual.shape()!=Shape{forecast.size(0),1} ||
      levels.shape()!=Shape{forecast.size(1)} || actual.dtype()!=forecast.dtype() ||
      levels.dtype()!=forecast.dtype() || forecast.device().backend!=compute::Backend::CPU ||
      actual.device().backend!=compute::Backend::CPU || levels.device().backend!=compute::Backend::CPU ||
      (forecast.dtype()!=DType::Float32 && forecast.dtype()!=DType::Float64) ||
      !all(logical_and(greater(levels,0.),less(levels,1.))).item<bool>())
    throw std::invalid_argument("quantile_forecast_loss: invalid quantile forecast or levels");
  const auto error=actual-forecast;
  return mean(maximum(levels*error,(levels-1.)*error));
}

} // namespace datamunge::nn
