#pragma once

/// @file forecast_evaluation.hpp
/// @brief Leakage-free rolling-origin scores for naive forecasts.

#include <datamunge/nn/tensor.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace datamunge::nn {

/// @brief Point forecasts, residual ensembles, and mean scores by model and lead.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto fit = n::naive_forecast_backtest(n::arange(0,10).to(n::DType::Float64), {5}, 2, 2, 2);
/// std::println("{}",fit.mase.size(0)); // 2
/// @endcode
/// @see naive_forecast_backtest()
struct NaiveForecastBacktest {
  Tensor actual; ///< Observations [origins, 1, leads].
  Tensor point; ///< Persistence, then seasonal naive [origins, 2, leads].
  Tensor scale; ///< Expanding seasonal absolute error [origins, 1, 1].
  Tensor draws; ///< Predictions plus prior residuals [origins, 2, leads, draws].
  Tensor mase; ///< Mean absolute scaled error [2, leads].
  Tensor smape; ///< Mean symmetric absolute percentage error in [0, 200] [2, leads].
  Tensor crps; ///< Mean equally weighted ensemble CRPS [2, leads].
};

/// @brief Evaluate persistence and seasonal-naive forecasts at chronological origins.
///
/// Each origin is the first unobserved index. A residual draw uses a
/// same-lead hindcast whose target occurs strictly before that origin.
/// MASE uses its expanding seasonal scale. A zero-over-zero sMAPE term
/// contributes zero. CRPS includes the diagonal of the empirical pair sum.
/// @param series Finite rank-one CPU Float32 or Float64 observations.
/// @param origins Nonempty strictly increasing indices of the first target values.
/// @param season Positive seasonal lag.
/// @param leads Positive forecast horizon, at most the season.
/// @param draws Positive number of hindcast errors per predictive ensemble.
/// @return Float64 predictions, ensembles and scores with model order persistence then seasonal naive.
/// @throws std::invalid_argument Invalid input, insufficient history, or undefined scores.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto fit = n::naive_forecast_backtest(n::from_vector(std::vector<double>{0,1,2,3,4,5,6}),
///                                            {5}, 2, 2, 2);
/// std::println("{}", fit.mase.to_vector<double>()[0]); // 0.5
/// @endcode
/// @par References
/// - Hyndman and Koehler (2006), https://doi.org/10.1016/j.ijforecast.2006.03.001
/// - Gneiting and Raftery (2007), https://doi.org/10.1198/016214506000001437
/// @see ForecastWindows()
[[nodiscard]] inline NaiveForecastBacktest naive_forecast_backtest(
    const Tensor& series, const std::vector<std::int64_t>& origins,
    std::int64_t season, std::int64_t leads, std::int64_t draws) {
  if (!series.defined() || series.dim() != 1 || series.device().backend != compute::Backend::CPU ||
      (series.dtype() != DType::Float32 && series.dtype() != DType::Float64) ||
      origins.empty() || season <= 0 || leads <= 0 || leads > season || draws <= 0 ||
      !std::ranges::is_sorted(origins) || std::adjacent_find(origins.begin(), origins.end()) != origins.end())
    throw std::invalid_argument("naive_forecast_backtest: invalid series or window specification");
  const auto size = series.size(0);
  for (auto origin : origins)
    if (origin <= season || origin > size - leads || draws > origin - season - (leads - 1))
      throw std::invalid_argument("naive_forecast_backtest: origin has insufficient history or targets");
  const auto values = series.to_vector<double>();
  if (!std::ranges::all_of(values, [](double x) { return std::isfinite(x); }))
    throw std::invalid_argument("naive_forecast_backtest: expected finite observations");

  const auto rows = static_cast<std::int64_t>(origins.size());
  NaiveForecastBacktest out{
      empty({rows, 1, leads}, DType::Float64), empty({rows, 2, leads}, DType::Float64),
      empty({rows, 1, 1}, DType::Float64), empty({rows, 2, leads, draws}, DType::Float64),
      zeros({2, leads}, DType::Float64), zeros({2, leads}, DType::Float64), zeros({2, leads}, DType::Float64)};
  auto* actual = out.actual.data_ptr<double>();
  auto* point = out.point.data_ptr<double>();
  auto* scale = out.scale.data_ptr<double>();
  auto* ensemble = out.draws.data_ptr<double>();
  auto* mase = out.mase.data_ptr<double>();
  auto* smape = out.smape.data_ptr<double>();
  auto* crps = out.crps.data_ptr<double>();
  /// @brief Retrieve the unadjusted naive forecast using pre-origin observations.
  auto predict = [&](int model, std::int64_t origin, std::int64_t lead) {
    return values[model == 0 ? origin - 1 : origin - season + lead];
  };
  std::int64_t cursor = season;
  double seasonal_error = 0.;
  std::vector<double> sorted(draws);
  for (std::int64_t r = 0; r < rows; ++r) {
    const auto origin = origins[r];
    for (; cursor < origin; ++cursor)
      seasonal_error += std::abs(values[cursor] - values[cursor - season]);
    const double denominator = seasonal_error / (origin - season);
    if (!(denominator > 0. && std::isfinite(denominator)))
      throw std::invalid_argument("naive_forecast_backtest: undefined MASE scale");
    scale[r] = denominator;
    for (std::int64_t lead = 0; lead < leads; ++lead) {
      const double y = values[origin + lead];
      actual[r * leads + lead] = y;
      for (int model = 0; model < 2; ++model) {
        const auto score = model * leads + lead;
        const auto pos = r * (2 * leads) + score;
        const double forecast = predict(model, origin, lead);
        point[pos] = forecast;
        const double error = std::abs(forecast - y);
        const double magnitude = std::abs(forecast) + std::abs(y);
        mase[score] += error / denominator;
        if (magnitude > 0.) smape[score] += 200. * (error / magnitude);

        double deviation = 0.;
        for (std::int64_t j = 0; j < draws; ++j) {
          const auto past = origin - lead - draws + j;
          sorted[j] = forecast + (values[past + lead] - predict(model, past, lead));
          if (!std::isfinite(sorted[j]))
            throw std::invalid_argument("naive_forecast_backtest: nonfinite residual draw");
          ensemble[pos * draws + j] = sorted[j];
          deviation += std::abs(sorted[j] - y);
        }
        std::ranges::sort(sorted);
        double spread = 0.;
        for (std::int64_t j = 1; j < draws; ++j)
          spread += (sorted[j] - sorted[j - 1]) * double(j) * double(draws - j);
        const double empirical = deviation / draws - spread / draws / draws;
        if (!std::isfinite(empirical))
          throw std::invalid_argument("naive_forecast_backtest: undefined CRPS");
        crps[score] += std::max(0., empirical);
      }
    }
  }
  for (std::int64_t i = 0; i < 2 * leads; ++i) {
    mase[i] /= rows;
    smape[i] /= rows;
    crps[i] /= rows;
    if (!std::isfinite(mase[i]) || !std::isfinite(smape[i]) || !std::isfinite(crps[i]))
      throw std::invalid_argument("naive_forecast_backtest: nonfinite score");
  }
  return out;
}

} // namespace datamunge::nn
