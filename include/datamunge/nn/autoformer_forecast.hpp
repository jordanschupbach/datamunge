#pragma once

/// @file autoformer_forecast.hpp
/// @brief Compact Autoformer with progressive decomposition and circular delay mixing.

#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/ops.hpp>

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace datamunge::nn {

/// @brief Projected operands and top-delay aggregation for one attention layer.
/// @par Example
/// @code
/// dm::nn::ParameterStore p;
/// const auto y = dm::nn::autoformer_forecast(p, "model", dm::nn::ones({1, 12, 1}), 4);
/// std::println("{}", y.encoder.weights.size(1)); // 3
/// @endcode
/// @see autoformer_forecast()
struct AutoformerDelayTrace {
  Tensor mixed; ///< Value mixture [batch, time, width].
  Tensor query; ///< Projected queries [batch, time, width].
  Tensor key; ///< Projected keys [batch, time, width].
  Tensor value; ///< Projected values [batch, time, width].
  Tensor scores; ///< Circular correlation by delay [batch, time].
  Tensor indices; ///< Selected zero-based delays [batch, delays].
  Tensor weights; ///< Softmax weights over selected delays [batch, delays].
};

/// @brief Forecast components and intermediate traces from an Autoformer forward pass.
/// @par Example
/// @code
/// dm::nn::ParameterStore p;
/// const auto y = dm::nn::autoformer_forecast(p, "model", dm::nn::ones({1, 12, 1}), 4);
/// std::println("{}", y.trend.size(1)); // 8
/// @endcode
/// @see autoformer_forecast()
struct AutoformerForecast {
  Tensor forecast; ///< Summed future trend and seasonal components [batch, horizon].
  Tensor initial_trend; ///< Smoothed input history [batch, lookback, 1].
  Tensor seed; ///< Seasonal decoder seed [batch, 2 * horizon, 1].
  Tensor trend_seed; ///< Initial trend decoder seed [batch, 2 * horizon, 1].
  Tensor seasonal; ///< Final seasonal component [batch, 2 * horizon, 1].
  Tensor trend; ///< Accumulated trend component [batch, 2 * horizon, 1].
  AutoformerDelayTrace encoder; ///< Circular mixing of encoder input.
  AutoformerDelayTrace decoder; ///< Circular mixing of decoder input.
  AutoformerDelayTrace cross; ///< Circular mixing of encoder memory into decoder.
};

/// @brief Forecast from observed history using progressive decomposition and circular delays.
///
/// Three-point moving averages replicate endpoint values. The decoder uses
/// the most recent horizon of seasonal history, followed by zeros. Its trend
/// seed combines the recent smoothed history and the history mean. This
/// compact model computes correlations directly for short windows.
/// @param store CPU parameter store matching history dtype.
/// @param name Prefix for learned layers.
/// @param history Floating CPU tensor [batch, lookback, 1], lookback at least twice horizon.
/// @param horizon Positive number of forecast leads.
/// @param width Positive embedding width, default 8.
/// @param delays Number of selected delays, positive and at most twice horizon (default 3).
/// @return Differentiable forecast and decomposition and correlation traces.
/// @throws std::invalid_argument Invalid architecture or shapes.
/// @par Example
/// @code
/// dm::nn::ParameterStore p;
/// auto y = dm::nn::autoformer_forecast(p, "model", dm::nn::ones({1, 12, 1}), 4);
/// std::println("{}", y.forecast.size(1)); // 4
/// @endcode
/// @par References
/// - Wu et al., Autoformer, https://arxiv.org/abs/2106.13008
/// @see dense(), top_k()
[[nodiscard]] inline AutoformerForecast autoformer_forecast(
    ParameterStore& store, const std::string& name, const Tensor& history,
    std::int64_t horizon, std::int64_t width = 8, std::int64_t delays = 3) {
  if (!history.defined() || history.dim() != 3 || history.size(0) < 1 || history.size(2) != 1 ||
      history.dtype() != store.dtype() || history.device().backend != compute::Backend::CPU ||
      horizon < 1 || horizon > history.size(1) / 2 || delays < 1 || delays > 2 * horizon ||
      width < 1 || width > std::numeric_limits<std::int64_t>::max() / 2)
    throw std::invalid_argument("autoformer_forecast: invalid history or dimensions");

  /// @brief Split a sequence into seasonal residual and endpoint-replicated moving average.
  const auto decompose = [](const Tensor& x) {
    const auto length = x.size(1);
    std::vector<std::int64_t> before(length), after(length);
    for (std::int64_t t = 0; t < length; ++t) {
      before[t] = std::max<std::int64_t>(0, t - 1);
      after[t] = std::min(t + 1, length - 1);
    }
    const auto weight = x.dtype() == DType::Float32 ? double(1.f / 3) : 1. / 3;
    const auto trend = (gather(x, from_vector(before), 1) + x + gather(x, from_vector(after), 1)) * weight;
    return std::pair{x - trend, trend};
  };
  /// @brief Mix circularly shifted values using the top delays of each sequence.
  const auto correlate = [&](const std::string& prefix, const Tensor& input, const Tensor& memory) {
    const auto q = dense(store, prefix + "/query", input, width);
    const auto k = dense(store, prefix + "/key", memory, width);
    const auto v = dense(store, prefix + "/value", memory, width);
    const auto length = input.size(1);
    std::vector<Tensor> scores, shifts;
    std::vector<std::int64_t> index(length);
    std::iota(index.begin(), index.end(), 0);
    for (std::int64_t lag = 0; lag < length; ++lag) {
      const auto shift = from_vector(index);
      scores.push_back(mean(q * gather(k, shift, 1), {1, 2}));
      shifts.push_back(gather(v, shift, 1));
      std::rotate(index.begin(), index.begin() + 1, index.end());
    }
    const auto all_scores = stack(scores, 1);
    const auto indices = top_k(all_scores, delays).second;
    const auto mask = one_hot(indices, length, 1., 0., input.dtype());
    const auto weights = softmax(sum(mask * expand_dims(all_scores, 1), {2}), 1);
    const auto per_delay = sum(mask * expand_dims(weights, 2), {1});
    const auto mixed = sum(stack(shifts, 1) * per_delay.reshape({input.size(0), length, 1, 1}), {1});
    return AutoformerDelayTrace{mixed, q, k, v, all_scores, indices, weights};
  };
  /// @brief Apply a residual feed-forward projection and remove its smooth component.
  const auto feedforward = [&](const std::string& prefix, const Tensor& x) {
    const auto expanded = dense(store, prefix + "/expand", x, 2 * width, Activation::Relu);
    return decompose(x + dense(store, prefix + "/contract", expanded, width));
  };
  /// @brief Select the newest count time positions.
  const auto tail = [](const Tensor& x, std::int64_t count) {
    return slice_axis(x, 1, x.size(1) - count, x.size(1));
  };

  const auto [initial_seasonal, initial_trend] = decompose(history);
  const auto tokens = dense(store, name + "/encoder_input", history, width);
  auto encoder = correlate(name + "/encoder_corr", tokens, tokens);
  const auto memory = feedforward(name + "/encoder_ff", decompose(tokens + encoder.mixed).first).first;
  const auto observed = tail(initial_seasonal, horizon);
  const auto future = zeros_like(observed);
  const auto seed = concat({observed, future}, 1);
  const auto decoder_input = dense(store, name + "/decoder_input", seed, width);
  auto decoder = correlate(name + "/decoder_corr", decoder_input, decoder_input);
  const auto [s1, t1] = decompose(decoder_input + decoder.mixed);
  auto cross = correlate(name + "/cross_corr", s1, tail(memory, 2 * horizon));
  const auto [s2, t2] = decompose(s1 + cross.mixed);
  const auto [s3, t3] = feedforward(name + "/decoder_ff", s2);
  const auto trend_seed = concat({tail(initial_trend, horizon), future + mean(history, {1}, true)}, 1);
  const auto trend = trend_seed + dense(store, name + "/trend", t1 + t2 + t3, 1);
  const auto seasonal = dense(store, name + "/seasonal", s3, 1);
  return {squeeze(tail(trend + seasonal, horizon), {2}), initial_trend, seed, trend_seed,
          seasonal, trend, encoder, decoder, cross};
}

}  // namespace datamunge::nn
