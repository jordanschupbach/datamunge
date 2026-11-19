#pragma once

/// @file forecast_baselines.hpp
/// @brief Chronological point forecast baselines with a ridge first-difference model.

#include <datamunge/nn/linalg.hpp>
#include <datamunge/nn/ops.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace datamunge::nn {

/// @brief Fits, forecasts, and lead-wise errors for four point baselines.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto result = n::forecast_baselines(n::arange(0, 20).to(n::DType::Float64),
///                                          {12}, 10, 4, 2, 3, 2);
/// std::println("{}", result.forecasts.size(1)); // 4
/// @endcode
/// @see forecast_baselines()
struct ForecastBaselines {
  Tensor coefficients; ///< Ridge autoregression coefficients [order, 1], most recent lag first.
  Tensor actual; ///< Observed targets [origins, leads].
  Tensor forecasts; ///< Levels [origins, 4, leads] in persistence, seasonal, mean, recurrence order.
  Tensor mae; ///< Mean absolute error over origins [4, leads].
};

/// @brief Fit a ridge recurrence once and compare four baselines over forecast origins.
///
/// The autoregression has no intercept and uses first differences from the
/// fitting prefix. Each forecast integrates recursive differences from the
/// last observed level. All models see only observations before each origin.
/// @param series Finite CPU Float32 or Float64 rank-one observations.
/// @param origins Strictly increasing first target indices at or after fit_end.
/// @param fit_end Exclusive fitting prefix endpoint, at least order plus two.
/// @param season Positive seasonal cycle length.
/// @param leads Positive forecast horizon.
/// @param mean_window Positive trailing mean window.
/// @param order Positive autoregression order.
/// @param ridge Nonnegative finite squared-norm penalty, default 0.01.
/// @return Float64 coefficients, predictions, targets and lead-wise MAE.
/// @throws std::invalid_argument Invalid windows, parameters, or nonfinite data or results.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto result = n::forecast_baselines(n::arange(0, 20).to(n::DType::Float64),
///                                          {12, 13}, 10, 4, 2, 3, 2);
/// std::println("{}", result.actual.size(0)); // 2
/// @endcode
/// @par References
/// - Hyndman and Athanasopoulos, Forecasting: Principles and Practice, https://otexts.com/fpp3/arima.html
/// @see naive_forecast_backtest(), lstsq()
[[nodiscard]] inline ForecastBaselines forecast_baselines(
    const Tensor& series, const std::vector<std::int64_t>& origins, std::int64_t fit_end,
    std::int64_t season, std::int64_t leads, std::int64_t mean_window, std::int64_t order,
    double ridge = .01) {
  if (!series.defined() || series.dim() != 1 || series.device().backend != compute::Backend::CPU ||
      (series.dtype() != DType::Float32 && series.dtype() != DType::Float64) ||
      origins.empty() || season < 1 || leads < 1 || mean_window < 1 || order < 1 ||
      fit_end < 3 || fit_end > series.size(0) || order > fit_end - 2 ||
      !(ridge >= 0. && std::isfinite(ridge)) || !std::ranges::is_sorted(origins) ||
      std::adjacent_find(origins.begin(), origins.end()) != origins.end())
    throw std::invalid_argument("forecast_baselines: invalid series or model settings");
  for (const auto origin : origins)
    if (origin < std::max({fit_end, season, mean_window}) || origin > series.size(0) - leads)
      throw std::invalid_argument("forecast_baselines: incomplete forecast window");
  const auto values = series.to_vector<double>();
  if (!std::ranges::all_of(values, [](double value) { return std::isfinite(value); }))
    throw std::invalid_argument("forecast_baselines: expected finite observations");

  const auto samples = fit_end - order - 1;
  auto design = empty({samples, order}, DType::Float64);
  auto response = empty({samples, 1}, DType::Float64);
  auto* predictors = design.data_ptr<double>();
  auto* targets = response.data_ptr<double>();
  for (std::int64_t row = 0; row < samples; ++row) {
    const auto t = row + order + 1;
    targets[row] = values[t] - values[t - 1];
    for (std::int64_t lag = 0; lag < order; ++lag)
      predictors[row * order + lag] = values[t - lag - 1] - values[t - lag - 2];
  }
  if (!all(is_finite(design)).item<bool>() || !all(is_finite(response)).item<bool>())
    throw std::invalid_argument("forecast_baselines: nonfinite fitting differences");
  if (ridge > 0.) {
    design = concat({design, eye(order, DType::Float64) * std::sqrt(ridge)}, 0);
    response = concat({response, zeros({order, 1}, DType::Float64)}, 0);
  }
  auto coefficients = lstsq(design, response, 0., false);
  const auto beta = coefficients.to_vector<double>();

  const auto rows = static_cast<std::int64_t>(origins.size());
  ForecastBaselines out{coefficients, empty({rows, leads}, DType::Float64),
                        empty({rows, 4, leads}, DType::Float64), zeros({4, leads}, DType::Float64)};
  auto* actual = out.actual.data_ptr<double>();
  auto* forecasts = out.forecasts.data_ptr<double>();
  auto* errors = out.mae.data_ptr<double>();
  std::vector<double> lags(order);
  for (std::int64_t r = 0; r < rows; ++r) {
    const auto origin = origins[r];
    const double last = values[origin - 1];
    const double trailing = std::accumulate(values.begin() + origin - mean_window,
                                            values.begin() + origin, 0.) / mean_window;
    for (std::int64_t j = 0; j < order; ++j)
      lags[j] = values[origin - 1 - j] - values[origin - 2 - j];
    double level = last;
    for (std::int64_t lead = 0; lead < leads; ++lead) {
      const double diff = std::inner_product(beta.begin(), beta.end(), lags.begin(), 0.);
      std::copy_backward(lags.begin(), lags.end() - 1, lags.end());
      lags.front() = diff;
      level += diff;
      const double y = values[origin + lead];
      actual[r * leads + lead] = y;
      const double predictions[]{last, values[origin - season + lead % season], trailing, level};
      for (int model = 0; model < 4; ++model) {
        const auto index = model * leads + lead;
        const auto value = predictions[model];
        if (!std::isfinite(value)) throw std::invalid_argument("forecast_baselines: nonfinite forecast");
        forecasts[r * (4 * leads) + index] = value;
        errors[index] += std::abs(y - value) / rows;
        if (!std::isfinite(errors[index]))
          throw std::invalid_argument("forecast_baselines: nonfinite MAE");
      }
    }
  }
  return out;
}

} // namespace datamunge::nn
