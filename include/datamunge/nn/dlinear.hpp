#pragma once

/// @file dlinear.hpp
/// @brief Direct multi-step forecasts from a moving-average decomposition.

#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/ops.hpp>

#include <algorithm>
#include <stdexcept>

namespace datamunge::nn {

/// @brief Fitted projections and the two observed-history components.
/// @par Example
/// @code
/// dm::nn::ParameterStore parameters;
/// const auto result = dm::nn::dlinear(parameters, "series", dm::nn::ones({1, 5}), 2, 3);
/// std::println("{}", result.trend.size(1)); // 5
/// @endcode
/// @see dlinear()
struct DLinearOutput {
  Tensor forecast; ///< Sum of the two fitted affine projections [batch, horizon].
  Tensor trend; ///< Centered moving average with replicated edges [batch, lookback].
  Tensor seasonal; ///< History minus the moving average [batch, lookback].
};

/// @brief Project the trend and residual history directly to the forecast horizon.
///
/// Each projection starts with weights 1/lookback and zero bias. Subsequent
/// calls with the same name reuse the parameters and preserve autograd.
/// @param store CPU parameter store with the same floating dtype as history.
/// @param name Prefix for the trend and seasonal projections.
/// @param history CPU Float32 or Float64 matrix [batch, lookback].
/// @param horizon Positive number of future samples to predict.
/// @param kernel Odd positive moving-average width at most the lookback, default 25.
/// @return Forecast and observed-history components.
/// @throws std::invalid_argument Invalid history, dimensions, or incompatible existing parameters.
/// @par Example
/// @code
/// dm::nn::ParameterStore parameters;
/// const auto result = dm::nn::dlinear(parameters, "test", dm::nn::ones({1, 5}), 2, 3);
/// std::println("{}", result.forecast.size(1)); // 2
/// @endcode
/// @par References
/// - Zeng et al., Are Transformers Effective for Time Series Forecasting?, https://arxiv.org/abs/2205.13504
/// @see dense()
[[nodiscard]] inline DLinearOutput dlinear(
    ParameterStore& store, const std::string& name, const Tensor& history,
    std::int64_t horizon, std::int64_t kernel = 25) {
  if (!history.defined() || history.dim() != 2 || history.size(0) < 1 ||
      history.device().backend != compute::Backend::CPU || history.dtype() != store.dtype() ||
      horizon < 1 || kernel < 1 || kernel % 2 == 0 || kernel > history.size(1))
    throw std::invalid_argument("dlinear: invalid history, horizon, or moving-average width");

  const auto lookback = history.size(1);
  auto smoothing = zeros({lookback, lookback}, history.dtype());
  /// @brief Populate the moving-average matrix with replicated edge weights.
  auto fill_weights = [&]<class T>(T* weights) {
    const auto weight = T{1} / static_cast<T>(kernel);
    for (std::int64_t time = 0; time < lookback; ++time) {
      for (auto offset = -kernel / 2; offset <= kernel / 2; ++offset) {
        const auto source = std::clamp(time + offset, std::int64_t{0}, lookback - 1);
        weights[source * lookback + time] += weight;
      }
    }
  };
  if (history.dtype() == DType::Float32)
    fill_weights(smoothing.data_ptr<float>());
  else
    fill_weights(smoothing.data_ptr<double>());

  const auto trend = matmul(history, smoothing);
  const auto seasonal = history - trend;
  DenseOptions options;
  options.init = init::constant(1.0 / static_cast<double>(lookback));
  const auto trend_forecast = dense(store, name + "/trend", trend, horizon, options);
  const auto seasonal_forecast = dense(store, name + "/seasonal", seasonal, horizon, options);
  return {trend_forecast + seasonal_forecast, trend, seasonal};
}

}  // namespace datamunge::nn
