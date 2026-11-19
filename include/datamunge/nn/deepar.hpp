#pragma once

/// @file deepar.hpp
/// @brief Gaussian DeepAR rollouts with one shared recurrent model.

#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/ops.hpp>

#include <stdexcept>
#include <vector>

namespace datamunge::nn {

/// @brief Control which values the recurrent decoder receives after each lead.
enum class DeepARMode {
  Teacher, ///< Feed back the observed target.
  Sample, ///< Feed back a draw using the supplied standard normal noise.
};

/// @brief Conditional parameters and recurrent inputs and outputs, all [batch, horizon].
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p;
/// const auto result = n::deepar(p, "panel", n::zeros({1, 4, 1}), n::zeros({1, 2}), n::DeepARMode::Sample);
/// std::println("{}", result.mean.size(1)); // 2
/// @endcode
/// @see deepar()
struct DeepAROutput {
  Tensor mean; ///< Conditional Gaussian means.
  Tensor scale; ///< Conditional standard deviations, with a 0.02 floor.
  Tensor values; ///< Target values in teacher mode and sampled observations in sample mode.
  Tensor previous; ///< Values fed into the decoder at each lead.
};

/// @brief Roll a shared univariate GRU forward with Gaussian emissions.
///
/// The last observation seeds the decoder. The same GRU encodes the earlier
/// history and updates the state at each lead. Rows evolve independently.
/// Repeat a history in multiple rows and supply independent standard normal
/// draws to generate multiple paths for that history.
/// @param store CPU parameter store matching both inputs.
/// @param name Parameter prefix.
/// @param history CPU floating tensor [batch, lookback, 1].
/// @param future CPU floating tensor [batch, horizon] of targets or standard normal draws.
/// @param mode Whether to feed back observed or sampled observations.
/// @param width Positive recurrent width, default 12.
/// @return Differentiable conditional parameters and rollout traces.
/// @throws std::invalid_argument Invalid shapes, dtype, mode, or stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p;
/// const auto result = n::deepar(p, "panel", n::zeros({1, 4, 1}), n::zeros({1, 2}), n::DeepARMode::Sample);
/// std::println("{}", result.values.size(1)); // 2
/// @endcode
/// @par References
/// - Salinas et al., DeepAR, https://arxiv.org/abs/1704.04110
/// @see gru_step(), GaussianRegressionNll()
[[nodiscard]] inline DeepAROutput deepar(
    ParameterStore& store, const std::string& name, const Tensor& history,
    const Tensor& future, DeepARMode mode, std::int64_t width = 12) {
  if (!history.defined() || !future.defined() || history.dim() != 3 || future.dim() != 2 ||
      history.size(0) < 1 || history.size(1) < 1 || history.size(2) != 1 ||
      future.size(0) != history.size(0) || future.size(1) < 1 || width < 1 ||
      history.dtype() != store.dtype() || future.dtype() != store.dtype() ||
      history.device().backend != compute::Backend::CPU || future.device().backend != compute::Backend::CPU ||
      (mode != DeepARMode::Teacher && mode != DeepARMode::Sample))
    throw std::invalid_argument("deepar: invalid history, future inputs, or architecture");

  const auto steps = time_steps(history);
  auto state = zeros({history.size(0), width}, history.dtype());
  const auto recurrent = name + "/shared_gru", mean_head = name + "/mean", scale_head = name + "/scale";
  for (std::size_t t = 0; t + 1 < steps.size(); ++t)
    state = gru_step(store, recurrent, steps[t], state);
  auto previous = steps.back();
  std::vector<Tensor> means, scales, values, inputs;
  for (const auto& step : time_steps(future.reshape({future.size(0), future.size(1), 1}))) {
    inputs.push_back(previous);
    state = gru_step(store, recurrent, previous, state);
    const auto mu = dense(store, mean_head, state, 1);
    const auto sigma = softplus(dense(store, scale_head, state, 1)) + .02;
    previous = mode == DeepARMode::Teacher ? step : mu + sigma * step;
    means.push_back(mu);
    scales.push_back(sigma);
    values.push_back(previous);
  }
  return {concat(means, 1), concat(scales, 1), concat(values, 1), concat(inputs, 1)};
}

}  // namespace datamunge::nn
