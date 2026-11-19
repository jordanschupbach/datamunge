#pragma once

/// @file nbeats.hpp
/// @brief Generic learned-basis N-BEATS blocks for direct forecasting.

#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/ops.hpp>

#include <stdexcept>
#include <vector>

namespace datamunge::nn {

/// @brief Additive prediction and per-block residual traces.
/// @par Example
/// @code
/// dm::nn::ParameterStore p;
/// const auto y = dm::nn::nbeats(p, "series", dm::nn::ones({2, 24}), 6);
/// std::println("{}", y.backcasts.size()); // 2
/// @endcode
/// @see nbeats()
struct NBeatsOutput {
  Tensor forecast; ///< Sum of forecast contributions [batch, horizon].
  std::vector<Tensor> residuals; ///< Input to each block [batch, lookback].
  std::vector<Tensor> backcasts; ///< History removed by each nonfinal block [batch, lookback].
  std::vector<Tensor> forecasts; ///< Contribution from each block [batch, horizon].
};

/// @brief Forecast with a sequence of generic learned-basis N-BEATS blocks.
///
/// Each block uses four ReLU layers and separate linear coefficient heads
/// and learned bases for forecast and backcast. Backcasts are subtracted
/// from the residual history. The final block only contributes a forecast.
/// @param store CPU parameter store matching the history dtype.
/// @param name Prefix of the model parameters.
/// @param history CPU Float32 or Float64 matrix [batch, lookback].
/// @param horizon Positive number of forecast leads.
/// @param blocks Positive number of blocks, default 3.
/// @param width Positive hidden layer width, default 32.
/// @param coefficients Positive basis rank, default 8.
/// @return Differentiable forecast and ordered block traces.
/// @throws std::invalid_argument Invalid dimensions, dtype, or incompatible parameters.
/// @par Example
/// @code
/// dm::nn::ParameterStore p;
/// const auto y = dm::nn::nbeats(p, "series", dm::nn::ones({2, 24}), 6);
/// std::println("{}", y.forecast.size(1)); // 6
/// @endcode
/// @par References
/// - Oreshkin et al., N-BEATS, https://arxiv.org/abs/1905.10437
/// @see dense()
[[nodiscard]] inline NBeatsOutput nbeats(
    ParameterStore& store, const std::string& name, const Tensor& history,
    std::int64_t horizon, std::int64_t blocks = 3, std::int64_t width = 32,
    std::int64_t coefficients = 8) {
  if (!history.defined() || history.dim() != 2 || history.size(0) < 1 || history.size(1) < 1 ||
      history.device().backend != compute::Backend::CPU || history.dtype() != store.dtype() ||
      horizon < 1 || blocks < 1 || width < 1 || coefficients < 1)
    throw std::invalid_argument("nbeats: invalid history or architecture");

  /// @brief Map the block's coefficients through a reusable learned basis.
  const auto project = [&](const std::string& prefix, const Tensor& hidden, std::int64_t length) {
    const auto theta = dense(store, prefix + "/theta", hidden, coefficients);
    const auto basis = store.get(prefix + "/basis", {coefficients, length}, init::glorot_uniform());
    return matmul(theta, basis);
  };
  NBeatsOutput result{zeros({history.size(0), horizon}, history.dtype()), {}, {}, {}};
  auto residual = history;
  for (std::int64_t block = 0; block < blocks; ++block) {
    const auto prefix = name + "/block" + std::to_string(block);
    result.residuals.push_back(residual);
    auto hidden = residual;
    for (int layer = 0; layer < 4; ++layer)
      hidden = dense(store, prefix + "/fc" + std::to_string(layer), hidden, width, Activation::Relu);
    result.forecasts.push_back(project(prefix + "/forecast", hidden, horizon));
    result.forecast = result.forecast + result.forecasts.back();
    if (block + 1 < blocks) {
      result.backcasts.push_back(project(prefix + "/backcast", hidden, history.size(1)));
      residual = residual - result.backcasts.back();
    }
  }
  return result;
}

}  // namespace datamunge::nn
