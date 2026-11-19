#pragma once

/// @file seq2seq_forecast.hpp
/// @brief Autoregressive multistep GRU forecasting with additive attention.

#include <datamunge/nn/attention.hpp>
#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/ops.hpp>

#include <stdexcept>
#include <vector>

namespace datamunge::nn {

/// @brief Forecast values and per-lead attention over the observed history.
/// @par Example
/// @code
/// dm::nn::ParameterStore p;
/// const auto y = dm::nn::seq2seq_forecast(p, "series", dm::nn::zeros({1, 8, 1}), 3);
/// std::println("{}", y.weights.size(2)); // 8
/// @endcode
/// @see seq2seq_forecast()
struct Seq2SeqForecastOutput {
  Tensor forecast; ///< Autoregressive predictions [batch, horizon].
  Tensor weights; ///< Attention over observed times [batch, horizon, lookback].
};

/// @brief Encode a univariate history and decode successive leads with additive attention.
///
/// The decoder starts with the final encoder state and last observation.
/// Each step queries the encoder memory, predicts a scalar, and feeds it
/// back into the next step. Training and inference use the same rollout.
/// @param store CPU parameter store matching the history dtype.
/// @param name Parameter prefix.
/// @param history CPU Float32 or Float64 tensor [batch, lookback, 1].
/// @param horizon Positive number of steps to generate.
/// @param width Positive GRU state width, default 12.
/// @return Differentiable forecast and attention trace.
/// @throws std::invalid_argument Invalid history, dimensions, dtype, or stored parameters.
/// @par Example
/// @code
/// dm::nn::ParameterStore p;
/// const auto y = dm::nn::seq2seq_forecast(p, "series", dm::nn::zeros({1, 8, 1}), 3);
/// std::println("{}", y.forecast.size(1)); // 3
/// @endcode
/// @par References
/// - Bahdanau, Cho, and Bengio, Neural Machine Translation by Jointly Learning to Align and Translate, https://arxiv.org/abs/1409.0473
/// @see gru_step(), additive_attention()
[[nodiscard]] inline Seq2SeqForecastOutput seq2seq_forecast(
    ParameterStore& store, const std::string& name, const Tensor& history,
    std::int64_t horizon, std::int64_t width = 12) {
  if (!history.defined() || history.dim() != 3 || history.size(0) < 1 ||
      history.size(1) < 1 || history.size(2) != 1 || history.dtype() != store.dtype() ||
      history.device().backend != compute::Backend::CPU || horizon < 1 || width < 1)
    throw std::invalid_argument("seq2seq_forecast: invalid history or dimensions");

  const auto steps = time_steps(history);
  auto state = zeros({history.size(0), width}, history.dtype());
  std::vector<Tensor> states;
  const auto encoder = name + "/encoder";
  for (const auto& step : steps) {
    state = gru_step(store, encoder, step, state);
    states.push_back(state);
  }
  const auto memory = stack(states, 1);
  auto previous = steps.back();
  const auto attention = name + "/attention", decoder = name + "/decoder", head = name + "/head";
  std::vector<Tensor> predictions, weights;
  for (std::int64_t lead = 0; lead < horizon; ++lead) {
    const auto attended = additive_attention(store, attention, state, memory, memory, width);
    state = gru_step(store, decoder, concat({previous, attended.output}, 1), state);
    previous = dense(store, head, state, 1);
    predictions.push_back(previous);
    weights.push_back(attended.weights);
  }
  return {concat(predictions, 1), stack(weights, 1)};
}

}  // namespace datamunge::nn
