/// @file anova_measures.cpp
/// @brief ANOVA variance ratios and mean-square reliability estimates.

#include <datamunge/stats/anova_measures.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace datamunge::stats {

/// @brief Validation and conversion for ANOVA-based estimands.
namespace {

/// @brief Converts a wide effect estimate to double.
/// @param value Estimate in coefficient units.
/// @return Finite rounded estimate.
/// @throws std::overflow_error Estimate is outside the finite double range.
double checked_estimate(long double value) {
  if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<double>::max())
    throw std::overflow_error("ANOVA measure: estimate exceeds finite double range");
  return static_cast<double>(value);
}

} // namespace

AnovaEffectSizes anova_effect_sizes(double term_ss, double residual_ss, double total_ss,
                                   double term_df, double residual_df) {
  if (!std::isfinite(total_ss) || !(total_ss > 0) ||
      !(term_ss >= 0 && term_ss <= total_ss && residual_ss >= 0 && residual_ss <= total_ss) ||
      (term_ss == 0 && residual_ss == 0) ||
      !(term_df > 0 && std::isfinite(term_df) && residual_df > 0 && std::isfinite(residual_df)))
    throw std::invalid_argument("ANOVA effect sizes: invalid sum of squares or degrees of freedom");
  const auto total = static_cast<long double>(total_ss);
  const auto term = term_ss / total, residual = residual_ss / total;
  const auto mean_error = residual / residual_df;
  const auto corrected_term = term - term_df * mean_error;
  return {checked_estimate(term), checked_estimate(term / (term + residual)),
          checked_estimate(corrected_term / (1 + mean_error)),
          checked_estimate(corrected_term / (term + residual + mean_error))};
}

IntraclassCorrelation intraclass_correlation(const linalg::DenseMatrix<double>& ratings, IccModel model,
                                             IccDefinition definition, IccUnit unit) {
  const bool one_way = model == IccModel::OneWayRandom;
  if ((!one_way && model != IccModel::TwoWayRandom && model != IccModel::TwoWayMixed) ||
      (definition != IccDefinition::AbsoluteAgreement && definition != IccDefinition::Consistency) ||
      (unit != IccUnit::Single && unit != IccUnit::Average) ||
      (one_way && definition == IccDefinition::Consistency) || ratings.rows() < 2 || ratings.cols() < 2)
    throw std::invalid_argument("ICC: invalid model, definition, unit or dimensions");

  const long double origin = ratings(0, 0);
  const bool agreement = definition == IccDefinition::AbsoluteAgreement;
  long double scale = 0;
  for (std::size_t j = 0; j < ratings.cols(); ++j) {
    if (agreement)
      scale = std::max(scale, std::abs(static_cast<long double>(ratings(0, j)) - origin));
    for (std::size_t i = 0; i < ratings.rows(); ++i) {
      const auto value = ratings(i, j);
      if (!std::isfinite(value))
        throw std::invalid_argument("ICC: ratings must be finite");
      scale = std::max(scale, std::abs(static_cast<long double>(value) - ratings(0, j)));
    }
  }
  if (scale == 0)
    throw std::invalid_argument("ICC: no variation for the selected definition");

  const long double n = ratings.rows(), k = ratings.cols();
  std::vector<long double> target(ratings.rows()), rater(ratings.cols()), baseline(ratings.cols());
  /// @brief Reads a rating relative to its column's first value on the common scale.
  const auto value = [&](std::size_t i, std::size_t j) {
    return (static_cast<long double>(ratings(i, j)) - ratings(0, j)) / scale;
  };
  for (std::size_t j = 0; j < rater.size(); ++j) {
    if (agreement)
      baseline[j] = (static_cast<long double>(ratings(0, j)) - origin) / scale;
    for (std::size_t i = 0; i < target.size(); ++i)
      rater[j] += value(i, j) / n;
    for (std::size_t i = 0; i < target.size(); ++i)
      target[i] += (value(i, j) - rater[j]) / k;
  }
  const auto grand_baseline = std::accumulate(baseline.begin(), baseline.end(), 0.0L) / k;
  const auto grand_rater = std::accumulate(rater.begin(), rater.end(), 0.0L) / k;
  const auto grand = std::accumulate(target.begin(), target.end(), 0.0L) / n;
  long double target_ss = 0, rater_ss = 0, error_ss = 0;
  for (const auto mean : target)
    target_ss += (mean - grand) * (mean - grand);
  for (std::size_t j = 0; j < rater.size(); ++j) {
    const auto rater_shift = (baseline[j] - grand_baseline) + (rater[j] - grand_rater);
    rater_ss += rater_shift * rater_shift;
    for (std::size_t i = 0; i < target.size(); ++i) {
      const auto two_way_error = (value(i, j) - rater[j]) - (target[i] - grand);
      const auto residual = two_way_error + (one_way ? rater_shift : 0);
      error_ss += residual * residual;
    }
  }
  const auto ms_target = k * target_ss / (n - 1);
  const auto ms_rater = n * rater_ss / (k - 1);
  const auto ms_error = error_ss / ((one_way ? n : n - 1) * (k - 1));
  const auto correction = !one_way && agreement ?
                          (ms_rater - ms_error) / n : 0;
  const auto denominator = unit == IccUnit::Single ?
                           ms_target + (k - 1) * ms_error + k * correction : ms_target + correction;
  if (denominator == 0)
    throw std::invalid_argument("ICC: zero coefficient denominator");
  return {checked_estimate((ms_target - ms_error) / denominator), model, definition, unit,
          ratings.rows(), ratings.cols()};
}

} // namespace datamunge::stats
