#pragma once

/// @file irregular_ode.hpp
/// @brief Differentiable ODE-RNN forecasts at irregular query times.

#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/ops.hpp>

#include <stdexcept>
#include <vector>

namespace datamunge::nn {

/// @brief Predictions and state traces before and after each integration interval.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p;
/// const auto result = n::irregular_ode_forecast(p, "model", n::ones({1, 3}), n::zeros({1, 2}));
/// std::println("{}", result.start.size(1)); // 3
/// @endcode
/// @see irregular_ode_forecast()
struct IrregularODEForecast {
  Tensor forecast; ///< Scalar readout at each query [batch, queries].
  Tensor start; ///< States before integration [batch, observations + queries, width].
  Tensor end; ///< Integrated states before any observation update, same shape as start.
};

/// @brief Integrate a neural vector field between observations and future queries.
///
/// Two fixed RK4 substeps advance each interval through a shared autonomous
/// tanh field. A GRU assimilates each observation after integration. At
/// queries, only the integrated state and a shared linear readout are used.
/// The initial hidden state is zero. The first gap can be zero.
/// @param store CPU parameter store matching the input dtype.
/// @param name Model parameter prefix.
/// @param gaps CPU floating matrix [batch, observations + queries] of finite nonnegative elapsed times.
/// @param observed CPU floating matrix [batch, observations] with at least one observation and one later query.
/// @param width Positive hidden state width, default 8.
/// @return Fully differentiable forecasts and interval traces.
/// @throws std::invalid_argument For incompatible inputs, invalid gaps, or stored parameter shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p;
/// const auto result = n::irregular_ode_forecast(p, "model", n::ones({1, 3}), n::zeros({1, 2}));
/// std::println("{}", result.forecast.size(1)); // 1
/// @endcode
/// @par References
/// - Rubanova et al., Latent ODEs for Irregularly-Sampled Time Series, https://arxiv.org/abs/1907.03907
/// @see gru_step(), dense()
[[nodiscard]] inline IrregularODEForecast irregular_ode_forecast(
    ParameterStore& store, const std::string& name, const Tensor& gaps,
    const Tensor& observed, std::int64_t width = 8) {
  if (!gaps.defined() || !observed.defined() || gaps.dim() != 2 || observed.dim() != 2 ||
      gaps.size(0) < 1 || observed.size(0) != gaps.size(0) || observed.size(1) < 1 ||
      observed.size(1) >= gaps.size(1) || width < 1 ||
      gaps.dtype() != store.dtype() || observed.dtype() != store.dtype() ||
      gaps.device().backend != compute::Backend::CPU || observed.device().backend != compute::Backend::CPU ||
      !all(logical_and(is_finite(gaps), greater_equal(gaps, 0.))).item<bool>())
    throw std::invalid_argument("irregular_ode_forecast: invalid observations or elapsed times");

  /// @brief Evaluate the learned autonomous vector field at a batch of states.
  const auto field = [&](const Tensor& x) {
    const auto hidden = dense(store, name + "/field_hidden", x, width, Activation::Tanh);
    return dense(store, name + "/field_output", hidden, width);
  };
  const auto batch = gaps.size(0);
  auto state = zeros({batch, width}, gaps.dtype());
  std::vector<Tensor> starts, ends, forecasts;
  for (std::int64_t step = 0; step < gaps.size(1); ++step) {
    starts.push_back(state);
    const auto dt = .5 * slice(gaps, {0, step}, {batch, 1});
    const auto half = .5 * dt;
    for (int substep = 0; substep < 2; ++substep) {
      const auto a = field(state);
      const auto b = field(state + half * a);
      const auto c = field(state + half * b);
      const auto d = field(state + dt * c);
      state = state + (dt / 6.) * (a + 2. * (b + c) + d);
    }
    ends.push_back(state);
    if (step < observed.size(1))
      state = gru_step(store, name + "/observe", slice(observed, {0, step}, {batch, 1}), state);
    else
      forecasts.push_back(dense(store, name + "/readout", state, 1));
  }
  return {concat(forecasts, 1), stack(starts, 1), stack(ends, 1)};
}

}  // namespace datamunge::nn
