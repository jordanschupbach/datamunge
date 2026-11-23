/// @file robust_tests.cpp
/// @brief Trimmed means and projected rank variances for independent samples.

#include <datamunge/stats/robust_tests.hpp>

#include <datamunge/stats/detail/ranking.hpp>
#include <datamunge/stats/detail/test_common.hpp>
#include <datamunge/stats/robust_summary.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Shared calculations for unequal-variance comparisons.
namespace {

/// @brief Computes Satterthwaite degrees of freedom without squaring raw variances.
/// @param a First component of the squared standard error.
/// @param b Second component, with positive finite a+b.
/// @param na First effective sample size.
/// @param nb Second effective sample size.
/// @return Positive reference degrees of freedom.
double satterthwaite(long double a, long double b, std::size_t na, std::size_t nb) {
  const auto wa = a / (a + b);
  const auto wb = b / (a + b);
  return static_cast<double>(1 / (wa * wa / (na - 1) + wb * wb / (nb - 1)));
}

/// @brief Collects rank-based superiority estimates and variance terms.
struct SuperiorityMoments {
  long double probability; ///< Estimated P(X > Y) + P(X = Y)/2.
  long double variance_x;  ///< Contribution of the first sample to variance of the estimate.
  long double variance_y;  ///< Contribution of the second sample to variance of the estimate.
};

/// @brief Computes tie-aware placements from pooled and within-sample midranks.
/// @param x First finite sample of length at least two.
/// @param y Second finite sample of length at least two.
/// @return Superiority and both independent-group variance contributions.
SuperiorityMoments superiority_moments(const std::vector<double>& x, const std::vector<double>& y) {
  detail::check_sample(x, 2);
  detail::check_sample(y, 2);
  auto pooled = x;
  pooled.insert(pooled.end(), y.begin(), y.end());
  const auto pooled_ranks = detail::rank_with_ties(pooled);
  auto x_placements = detail::rank_with_ties(x);
  auto y_placements = detail::rank_with_ties(y);

  // Subtracting the within-group rank leaves wins plus half the cross-group ties.
  // Center on the first placement so identical placements have exact zero variance.
  const auto first_x = pooled_ranks.front() - x_placements.front();
  const auto first_y = pooled_ranks[x.size()] - y_placements.front();
  for (std::size_t i = 0; i < x.size(); ++i)
    x_placements[i] = (pooled_ranks[i] - x_placements[i] - first_x) / y.size();
  for (std::size_t i = 0; i < y.size(); ++i)
    y_placements[i] = (pooled_ranks[x.size() + i] - y_placements[i] - first_y) / x.size();

  const auto mx = detail::test_moments(x_placements);
  const auto my = detail::test_moments(y_placements);
  const auto probability = std::clamp(mx.mean + static_cast<long double>(first_x) / y.size(), 0.0L, 1.0L);
  return {probability, mx.variance / x.size(), my.variance / y.size()};
}

} // namespace

HypothesisTestResult yuen_test(const std::vector<double>& x, const std::vector<double>& y, double trim,
                              Alternative alternative, double conf_level) {
  detail::check_alternative(alternative);
  detail::check_confidence(conf_level);
  const auto mx = trimmed_summary(x, trim);
  const auto my = trimmed_summary(y, trim);
  const auto hx = x.size() - 2 * static_cast<std::size_t>(trim * static_cast<double>(x.size()));
  const auto hy = y.size() - 2 * static_cast<std::size_t>(trim * static_cast<double>(y.size()));
  if (hx < 2 || hy < 2)
    throw std::invalid_argument("yuen_test: need at least two retained observations per group");

  const auto vx = mx.winsorized_variance * static_cast<long double>(x.size() - 1) / hx / (hx - 1);
  const auto vy = my.winsorized_variance * static_cast<long double>(y.size() - 1) / hy / (hy - 1);
  const auto se = std::sqrt(vx + vy);
  if (!(se > 0 && std::isfinite(se)))
    throw std::invalid_argument("yuen_test: standard error must be positive and finite");

  const auto difference = static_cast<long double>(mx.trimmed_mean) - my.trimmed_mean;
  const auto df = satterthwaite(vx, vy, hx, hy);
  const auto statistic = static_cast<double>(difference / se);
  const auto alpha = (1 - conf_level) / (alternative == Alternative::TwoSided ? 2 : 1);
  const auto margin = -random::student_t_quantile(alpha, df) * se;
  const auto infinity = std::numeric_limits<double>::infinity();

  HypothesisTestResult result;
  result.statistic = statistic;
  result.parameter1 = df;
  result.p_value = detail::student_test_p(statistic, df, alternative);
  result.estimate1 = mx.trimmed_mean;
  result.estimate2 = my.trimmed_mean;
  result.has_conf_int = true;
  result.conf_int_lower = alternative == Alternative::Less ? -infinity : static_cast<double>(difference - margin);
  result.conf_int_upper = alternative == Alternative::Greater ? infinity : static_cast<double>(difference + margin);
  result.alternative = alternative;
  result.method = "Yuen's unequal-variance trimmed-mean t test";
  return result;
}

HypothesisTestResult brunner_munzel_test(const std::vector<double>& x, const std::vector<double>& y,
                                        Alternative alternative) {
  detail::check_alternative(alternative);
  const auto effect = superiority_moments(x, y);
  const auto variance = effect.variance_x + effect.variance_y;
  if (!(variance > 0))
    throw std::invalid_argument("brunner_munzel_test: rank-placement variance is zero");
  const auto df = satterthwaite(effect.variance_x, effect.variance_y, x.size(), y.size());
  const auto statistic = static_cast<double>((effect.probability - .5L) / std::sqrt(variance));

  HypothesisTestResult result;
  result.statistic = statistic;
  result.parameter1 = df;
  result.p_value = detail::student_test_p(statistic, df, alternative);
  result.estimate1 = static_cast<double>(effect.probability);
  result.alternative = alternative;
  result.method = "Brunner–Munzel test with Student t reference";
  return result;
}

CliffsDeltaResult cliffs_delta(const std::vector<double>& x, const std::vector<double>& y, double conf_level) {
  detail::check_confidence(conf_level);
  const auto effect = superiority_moments(x, y);
  const auto variance = effect.variance_x + effect.variance_y;
  const auto delta = static_cast<double>(2 * effect.probability - 1);
  const auto standard_error = static_cast<double>(2 * std::sqrt(variance));
  if (variance == 0)
    return {delta, standard_error, std::nullopt};

  const auto df = satterthwaite(effect.variance_x, effect.variance_y, x.size(), y.size());
  const auto margin = -random::student_t_quantile((1 - conf_level) / 2, df) * standard_error;
  return {delta, standard_error, std::pair{std::max(-1.0, delta - margin), std::min(1.0, delta + margin)}};
}

} // namespace datamunge::stats
