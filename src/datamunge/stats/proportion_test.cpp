/// @file proportion_test.cpp
/// @brief Classical proportion test statistics, domain checks and stable probabilities.
#include <algorithm>
#include <cmath>
#include <datamunge/random/distributions.hpp>
#include <datamunge/stats/detail/test_common.hpp>
#include <datamunge/stats/proportion_test.hpp>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Internal sign, tail and interval calculations.
namespace {

/// @brief Returns -1, zero or one according to the finite value sign.
/// @param v Signed value.
/// @return Sign of v.
double sign(double v) {
  return (v > 0.0) - (v < 0.0);
}

/// @brief Evaluates the selected normal tail without cancellation.
/// @param z Signed normal statistic.
/// @param alternative Requested tail.
/// @return Probability in [0,1].
double z_p_value(double z, Alternative alternative) {
  return detail::normal_test_p(z, alternative);
}

/// @brief Uncorrected Wilson score interval for a binomial proportion.
/// @param x Success count.
/// @param n Positive trial count.
/// @param z Positive normal critical value.
/// @param lower Receives lower bound.
/// @param upper Receives upper bound.
void wilson_interval(double x, double n, double z, double& lower, double& upper) {
  const double p_hat  = x / n;
  const double denom  = 1.0 + z * z / n;
  const double center = (p_hat + z * z / (2.0 * n)) / denom;
  const double half   = z * std::sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n)) / denom;
  lower               = std::clamp(center - half, 0.0, 1.0);
  upper               = std::clamp(center + half, 0.0, 1.0);
}

/// @brief Inverts a beta CDF with 200 bisection steps.
/// @param target Probability in (0,1).
/// @param a Positive first beta shape.
/// @param b Positive second beta shape.
/// @return Quantile in [0,1].
double beta_quantile(double target, double a, double b) {
  double lo = 0.0, hi = 1.0;
  for (int i = 0; i < 200; ++i) {
    const double mid = 0.5 * (lo + hi);
    if (random::detail::regularized_incomplete_beta(a, b, mid) < target)
      lo = mid;
    else
      hi = mid;
  }
  return 0.5 * (lo + hi);
}

} // namespace

HypothesisTestResult proportion_test_one_sample(
    std::size_t successes,
    std::size_t n,
    double      p,
    Alternative alternative,
    bool        correct,
    double      conf_level) {
  detail::check_alternative(alternative);
  detail::check_confidence(conf_level);
  if (!std::isfinite(p) || p < 0 || p > 1)
    throw std::invalid_argument("proportion test: invalid null probability");
  if (n == 0)
    throw std::invalid_argument("proportion_test_one_sample: n must be > 0");
  if (successes > n)
    throw std::invalid_argument("proportion_test_one_sample: successes must be <= n");

  const double x     = static_cast<double>(successes);
  const double nd    = static_cast<double>(n);
  const double p_hat = x / nd;
  const double diff  = p_hat - p;
  const double yates = correct ? std::min(0.5 / nd, std::fabs(diff)) : 0.0;
  if (p == 0 || p == 1)
    throw std::invalid_argument(
        "proportion_test: require 0 < null p < 1; use binomial_test for endpoints");
  const double se = std::sqrt(p * (1.0 - p) / nd);
  const double z  = (diff - sign(diff) * yates) / se;

  HypothesisTestResult result;
  result.statistic   = z * z;
  result.parameter1  = 1.0;
  result.p_value     = z_p_value(z, alternative);
  result.estimate1   = p_hat;
  result.alternative = alternative;
  result.method      = correct ? "1-sample proportions test with continuity correction"
                               : "1-sample proportions test";

  const double alpha  = 1.0 - conf_level;
  result.has_conf_int = true;
  if (alternative == Alternative::TwoSided) {
    wilson_interval(
        x,
        nd,
        random::normal_quantile(1.0 - 0.5 * alpha),
        result.conf_int_lower,
        result.conf_int_upper);
  } else if (alternative == Alternative::Less) {
    wilson_interval(
        x, nd, random::normal_quantile(conf_level), result.conf_int_lower, result.conf_int_upper);
    result.conf_int_lower = 0.0;
  } else {
    wilson_interval(
        x, nd, random::normal_quantile(conf_level), result.conf_int_lower, result.conf_int_upper);
    result.conf_int_upper = 1.0;
  }
  return result;
}

HypothesisTestResult proportion_test_two_sample(
    std::size_t successes1,
    std::size_t n1,
    std::size_t successes2,
    std::size_t n2,
    Alternative alternative,
    bool        correct,
    double      conf_level) {
  detail::check_alternative(alternative);
  detail::check_confidence(conf_level);
  if (n1 == 0 || n2 == 0)
    throw std::invalid_argument("proportion_test_two_sample: n1 and n2 must be > 0");
  if (successes1 > n1 || successes2 > n2)
    throw std::invalid_argument("proportion_test_two_sample: successes must be <= n");

  const double x1 = static_cast<double>(successes1), n1d = static_cast<double>(n1);
  const double x2 = static_cast<double>(successes2), n2d = static_cast<double>(n2);
  const double p1 = x1 / n1d, p2 = x2 / n2d;
  const double p_pool = (x1 + x2) / (n1d + n2d);

  const double diff      = p1 - p2;
  const double inv_n_sum = 1.0 / n1d + 1.0 / n2d;
  const double yates     = correct ? std::min(0.5 * inv_n_sum, std::fabs(diff)) : 0.0;
  const double se_pooled = std::sqrt(p_pool * (1.0 - p_pool) * inv_n_sum);
  const double z         = se_pooled > 0 ? (diff - sign(diff) * yates) / se_pooled : 0;

  HypothesisTestResult result;
  result.statistic   = z * z;
  result.parameter1  = 1.0;
  result.p_value     = z_p_value(z, alternative);
  result.estimate1   = p1;
  result.estimate2   = p2;
  result.alternative = alternative;
  result.method = correct ? "2-sample test for equality of proportions with continuity correction"
                          : "2-sample test for equality of proportions";

  const double alpha       = 1.0 - conf_level;
  const double se_unpooled = std::sqrt(p1 * (1.0 - p1) / n1d + p2 * (1.0 - p2) / n2d);
  const double correction  = correct ? 0.5 * inv_n_sum : 0.0;
  result.has_conf_int      = true;
  if (alternative == Alternative::TwoSided) {
    const double crit     = random::normal_quantile(1.0 - 0.5 * alpha);
    const double half     = crit * se_unpooled + correction;
    result.conf_int_lower = std::clamp(diff - half, -1.0, 1.0);
    result.conf_int_upper = std::clamp(diff + half, -1.0, 1.0);
  } else if (alternative == Alternative::Less) {
    const double crit     = random::normal_quantile(conf_level);
    result.conf_int_lower = -1.0;
    result.conf_int_upper = std::clamp(diff + crit * se_unpooled + correction, -1.0, 1.0);
  } else {
    const double crit     = random::normal_quantile(conf_level);
    result.conf_int_lower = std::clamp(diff - crit * se_unpooled - correction, -1.0, 1.0);
    result.conf_int_upper = 1.0;
  }
  return result;
}

HypothesisTestResult binomial_test(
    std::size_t successes,
    std::size_t n,
    double      p,
    Alternative alternative,
    double      conf_level) {
  detail::check_alternative(alternative);
  detail::check_confidence(conf_level);
  if (!std::isfinite(p) || p < 0 || p > 1)
    throw std::invalid_argument("proportion test: invalid null probability");
  if (n == 0)
    throw std::invalid_argument("binomial_test: n must be > 0");
  if (successes > n)
    throw std::invalid_argument("binomial_test: successes must be <= n");

  HypothesisTestResult result;
  result.statistic   = static_cast<double>(successes);
  result.parameter1  = static_cast<double>(n);
  result.estimate1   = static_cast<double>(successes) / static_cast<double>(n);
  result.alternative = alternative;
  result.method      = "Exact binomial test";

  result.p_value = detail::binomial_test_p(successes, n, p, alternative);

  const double alpha  = 1.0 - conf_level;
  result.has_conf_int = true;
  const double x = static_cast<double>(successes), nd = static_cast<double>(n);
  if (alternative == Alternative::TwoSided) {
    result.conf_int_lower = (successes == 0) ? 0.0 : beta_quantile(0.5 * alpha, x, nd - x + 1.0);
    result.conf_int_upper =
        (successes == n) ? 1.0 : beta_quantile(1.0 - 0.5 * alpha, x + 1.0, nd - x);
  } else if (alternative == Alternative::Less) {
    result.conf_int_lower = 0.0;
    result.conf_int_upper = (successes == n) ? 1.0 : beta_quantile(conf_level, x + 1.0, nd - x);
  } else {
    result.conf_int_lower =
        (successes == 0) ? 0.0 : beta_quantile(1.0 - conf_level, x, nd - x + 1.0);
    result.conf_int_upper = 1.0;
  }
  return result;
}

} // namespace datamunge::stats
