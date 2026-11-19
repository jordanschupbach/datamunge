#pragma once

/// @file graph_forecast.hpp
/// @brief Multivariate rolling windows and directed graph diffusion forecasts.

#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/ops.hpp>
#include <datamunge/nn/training.hpp>
#include <datamunge/nn/views.hpp>

#include <stdexcept>
#include <vector>

namespace datamunge::nn {

/// @brief Node-major histories and their immediately subsequent observations.
/// @par Example
/// @code
/// auto batch = dm::nn::graph_forecast_windows(dm::nn::ones({12, 3}), {6}, 4, 2);
/// std::println("{}", batch.history.size(1)); // 3
/// @endcode
/// @see graph_forecast_windows()
struct GraphForecastWindows {
  Tensor history; ///< Observed tensor [origins, nodes, lag].
  Tensor target; ///< Future tensor [origins, nodes, horizon].
};

/// @brief Gather rolling-origin windows without mixing nodes or crossing series bounds.
/// @param series CPU observations [time, nodes].
/// @param origins First target indices in requested batch order.
/// @param lag Number of observations before each origin.
/// @param horizon Number of target observations from each origin.
/// @return Node-major histories and targets.
/// @throws std::invalid_argument Invalid dimensions, empty origins, or an incomplete window.
/// @par Example
/// @code
/// auto batch = dm::nn::graph_forecast_windows(dm::nn::ones({12, 3}), {6}, 4, 2);
/// std::println("{}", batch.target.size(2)); // 2
/// @endcode
/// @see sliding_window()
[[nodiscard]] inline GraphForecastWindows graph_forecast_windows(
    const Tensor& series, const std::vector<std::int64_t>& origins,
    std::int64_t lag, std::int64_t horizon) {
  if (!series.defined() || series.dim() != 2 || series.size(1) < 1 ||
      series.device().backend != compute::Backend::CPU || origins.empty() ||
      lag < 1 || horizon < 1 || lag > series.size(0) || horizon > series.size(0) - lag)
    throw std::invalid_argument("graph_forecast_windows: invalid window dimensions");
  std::vector<std::int64_t> starts;
  starts.reserve(origins.size());
  for (const auto origin : origins) {
    if (origin < lag || origin > series.size(0) - horizon)
      throw std::invalid_argument("graph_forecast_windows: incomplete window");
    starts.push_back(origin - lag);
  }
  const auto windows = take_rows(sliding_window(series, 0, lag + horizon), starts);
  return {slice_axis(windows, 2, 0, lag), slice_axis(windows, 2, lag, lag + horizon)};
}

/// @brief Features from each diffusion order and the associated direct predictions.
/// @par Example
/// @code
/// dm::nn::ParameterStore p;
/// auto model = dm::nn::graph_diffusion_forecast(p, "shared", dm::nn::ones({1, 3, 4}),
///                                               dm::nn::eye(3), 2, 1);
/// std::println("{}", model.features.size(2)); // 8
/// @endcode
/// @see graph_diffusion_forecast()
struct GraphDiffusionForecast {
  Tensor forecast; ///< Forecast per node [batch, nodes, horizon].
  Tensor features; ///< Concatenated hop features [batch, nodes, (hops + 1) * lag].
};

/// @brief Apply a shared affine forecast to local and graph-propagated histories.
///
/// Entry adjacency[destination, source] weights the source history at the
/// destination. Supply weights already scaled as desired. At zero hops only
/// the local history enters the head. A parameter prefix fixes feature width
/// and horizon across calls.
/// @param store CPU parameter store matching the tensors' floating dtype.
/// @param name Prefix of the reusable affine head.
/// @param history CPU Float32 or Float64 histories [batch, nodes, lag].
/// @param adjacency Same-dtype CPU square node adjacency.
/// @param horizon Positive forecast length.
/// @param hops Nonnegative number of diffusion steps.
/// @return Differentiable forecast and history features in increasing hop order.
/// @throws std::invalid_argument Invalid shapes, dtypes, or settings.
/// @par Example
/// @code
/// dm::nn::ParameterStore p;
/// auto model = dm::nn::graph_diffusion_forecast(p, "shared", dm::nn::ones({1, 3, 4}),
///                                               dm::nn::eye(3), 2, 1);
/// std::println("{}", model.forecast.size(2)); // 2
/// @endcode
/// @see graph_forecast_windows(), matmul(), dense()
[[nodiscard]] inline GraphDiffusionForecast graph_diffusion_forecast(
    ParameterStore& store, const std::string& name, const Tensor& history,
    const Tensor& adjacency, std::int64_t horizon, std::int64_t hops) {
  if (!history.defined() || !adjacency.defined() || history.dim() != 3 ||
      history.size(0) < 1 || history.size(1) < 1 || history.size(2) < 1 ||
      adjacency.shape() != Shape{history.size(1), history.size(1)} ||
      history.dtype() != store.dtype() || adjacency.dtype() != store.dtype() ||
      history.device().backend != compute::Backend::CPU ||
      adjacency.device().backend != compute::Backend::CPU || horizon < 1 || hops < 0)
    throw std::invalid_argument("graph_diffusion_forecast: invalid inputs");

  std::vector<Tensor> channels{history};
  for (std::int64_t step = 0; step < hops; ++step)
    channels.push_back(matmul(adjacency, channels.back()));
  const auto features = hops == 0 ? history : concat(channels, 2);
  return {dense(store, name, features, horizon), features};
}

}  // namespace datamunge::nn
