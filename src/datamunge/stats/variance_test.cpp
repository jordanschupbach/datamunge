/// @file variance_test.cpp
/// @brief Classical variance test statistics, domain checks and stable probabilities.
#include <algorithm>
#include <datamunge/random/distributions.hpp>
#include <datamunge/stats/detail/test_common.hpp>
#include <datamunge/stats/variance_test.hpp>
#include <limits>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Checked sample variance calculation.
namespace {

/// @brief Computes a positive finite sample variance.
/// @param x At least two finite observations.
/// @return Variance with n-1 denominator.
double sample_variance(const std::vector<double>& x) {
  const double v = static_cast<double>(detail::test_moments(x).variance);
  if (!(v > 0) || !std::isfinite(v))
    throw std::invalid_argument("f_test_variance: positive finite variances required");
  return v;
}

/// @brief Inverts a lower F tail with a zero lower bracket, including very small probabilities.
/// @param p Lower-tail probability in (0,1).
/// @param df1 Positive numerator degrees of freedom.
/// @param df2 Positive denominator degrees of freedom.
/// @return Positive F quantile; bisection stops when the floating bracket no longer changes.
double lower_f_quantile(double p, double df1, double df2) {
  double lo = 0, hi = 1;
  while (random::f_cdf(hi, df1, df2) < p)
    hi *= 2;
  for (int i = 0; i < 200; ++i) {
    const double mid = lo + (hi - lo) / 2;
    if (mid == lo || mid == hi)
      break;
    if (random::f_cdf(mid, df1, df2) < p)
      lo = mid;
    else
      hi = mid;
  }
  return lo + (hi - lo) / 2;
}

} // namespace

HypothesisTestResult f_test_variance(
    const std::vector<double>& x,
    const std::vector<double>& y,
    Alternative                alternative,
    double                     conf_level) {
  detail::check_alternative(alternative);
  detail::check_confidence(conf_level);
  if (x.size() < 2 || y.size() < 2)
    throw std::invalid_argument("f_test_variance: each sample needs at least 2 observations");

  const double df1      = static_cast<double>(x.size() - 1);
  const double df2      = static_cast<double>(y.size() - 1);
  const double estimate = sample_variance(x) / sample_variance(y);

  HypothesisTestResult result;
  result.statistic   = estimate;
  result.parameter1  = df1;
  result.parameter2  = df2;
  result.estimate1   = estimate;
  result.alternative = alternative;
  result.method      = "F test to compare two variances";

  const double upper  = detail::f_test_survival(estimate, df1, df2);
  const double lower  = detail::f_test_survival(1 / estimate, df2, df1);
  const double alpha  = 1.0 - conf_level;
  const double inf    = std::numeric_limits<double>::infinity();
  result.has_conf_int = true;
  switch (alternative) {
    case Alternative::Less:
      result.p_value        = lower;
      result.conf_int_lower = 0.0;
      result.conf_int_upper = estimate / lower_f_quantile(alpha, df1, df2);
      break;
    case Alternative::Greater:
      result.p_value        = upper;
      result.conf_int_lower = estimate * lower_f_quantile(alpha, df2, df1);
      result.conf_int_upper = inf;
      break;
    case Alternative::TwoSided:
    default:
      result.p_value        = 2.0 * std::min(lower, upper);
      result.conf_int_lower = estimate * lower_f_quantile(.5 * alpha, df2, df1);
      result.conf_int_upper = estimate / lower_f_quantile(.5 * alpha, df1, df2);
      break;
  }
  if (result.p_value > 1.0)
    result.p_value = 1.0;
  return result;
}

} // namespace datamunge::stats
