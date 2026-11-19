#pragma once

/// @file recurrent_forecast.hpp
/// @brief Chronological training windows and direct recurrent forecasts.

#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/ops.hpp>
#include <datamunge/nn/training.hpp>
#include <datamunge/nn/views.hpp>
#include <stdexcept>

namespace datamunge::nn {

/// @brief A batch of observed histories and their subsequent forecast targets.
struct ForecastWindowBatch {
  Tensor history; ///< Shape [origins, lag, 1], ending immediately before each origin.
  Tensor target;  ///< Shape [origins, horizon], starting at each origin.
};

/// @brief Gather univariate rolling-origin windows in the order supplied by the caller.
/// Split the origin lists first to keep complete targets inside their assigned partitions.
/// @param series Rank-one series tensor. Normalize it using fitting observations if desired.
/// @param origins Indices of the first target observations.
/// @param lag Positive number of preceding observations.
/// @param horizon Positive number of target observations.
/// @return Histories and targets for all supplied origins.
/// @throws std::invalid_argument Invalid dimensions or a window outside the series.
/// @code
/// auto batch = dm::nn::ForecastWindows(dm::nn::arange(0, 10), {3, 6}, 3, 2);
/// @endcode
inline ForecastWindowBatch ForecastWindows(
    const Tensor& series, const std::vector<std::int64_t>& origins,
    std::int64_t lag, std::int64_t horizon) {
  if (!series.defined() || series.dim() != 1 || origins.empty() || lag <= 0 || horizon <= 0 ||
      lag > series.size(0) || horizon > series.size(0) - lag) {
    throw std::invalid_argument("forecast window dimensions");
  }
  std::vector<std::int64_t> starts;
  starts.reserve(origins.size());
  for (const auto origin : origins) {
    if (origin < lag || origin > series.size(0) - horizon)
      throw std::invalid_argument("forecast window outside series");
    starts.push_back(origin - lag);
  }
  const auto window = take_rows(sliding_window(series, 0, lag + horizon), starts);
  return {reshape(slice_axis(window, 1, 0, lag), {-1, lag, 1}),
          slice_axis(window, 1, lag, lag + horizon)};
}

/// @brief Forecast all leads directly from the last recurrent state of each history.
/// @param parameters Caller-owned layer parameters, initialized on first use.
/// @param history Input shape [batch, lag, features], with positive lag.
/// @param units Number of recurrent state coordinates.
/// @param horizon Number of forecast leads in the direct output.
/// @param kind Recurrent cell to unroll through the history.
/// @return A differentiable [batch, horizon] tensor, with no future input.
/// @throws std::invalid_argument Invalid history dimensions or model widths.
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(41380);
/// auto prediction = n::RecurrentDirectForecast(p, n::zeros({5, 24, 1}), 12, 6, n::RnnKind::Gru);
/// @endcode
inline Tensor RecurrentDirectForecast(
    ParameterStore& parameters, const Tensor& history,
    std::int64_t units, std::int64_t horizon, RnnKind kind) {
  if (!history.defined() || history.dim() != 3 || history.size(1) <= 0 ||
      history.size(0) <= 0 || history.size(2) <= 0 || units <= 0 || horizon <= 0 ||
      (kind != RnnKind::Simple && kind != RnnKind::Lstm && kind != RnnKind::Gru)) {
    throw std::invalid_argument("recurrent forecast dimensions");
  }
  const auto hidden = recurrent(parameters, "encoder", history, units, kind);
  return dense(parameters, "head", select(hidden, 1, history.size(1) - 1), horizon);
}

}  // namespace datamunge::nn
