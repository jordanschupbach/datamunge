#pragma once

/// @file test_common.hpp
/// @brief Finite-domain checks and cancellation-free tails shared by classical tests.

#include <algorithm>
#include <cmath>
#include <datamunge/random/distributions.hpp>
#include <datamunge/stats/hypothesis_test_result.hpp>
#include <limits>
#include <stdexcept>
#include <vector>

namespace datamunge::stats::detail {
/// @brief Rejects unknown alternatives instead of silently selecting a tail.
/// @param a Requested alternative.
/// @throws std::invalid_argument Unknown enumerator.
inline void check_alternative(Alternative a) {
  if (a != Alternative::TwoSided && a != Alternative::Less && a != Alternative::Greater)
    throw std::invalid_argument("hypothesis test: invalid alternative");
}
/// @brief Requires a finite confidence level strictly between zero and one.
/// @param level Requested confidence level.
/// @throws std::invalid_argument Invalid level.
inline void check_confidence(double level) {
  if (!std::isfinite(level) || !(level > 0 && level < 1))
    throw std::invalid_argument("hypothesis test: require 0 < conf_level < 1");
}
/// @brief Requires finite observations and a minimum count; no values are omitted.
/// @param x Sample.
/// @param minimum Minimum allowed length.
/// @throws std::invalid_argument Too few values or a nonfinite value.
inline void check_sample(const std::vector<double>& x, std::size_t minimum = 1) {
  if (x.size() < minimum)
    throw std::invalid_argument("hypothesis test: too few observations");
  for (double v : x)
    if (!std::isfinite(v))
      throw std::invalid_argument("hypothesis test: observations must be finite");
}
/// @brief Extended-precision sample moments for means and variances.
struct TestMoments {
  long double mean;     ///< Arithmetic mean.
  long double variance; ///< Sample variance with n-1 denominator.
};
/// @brief Computes finite-sample moments without raw second-moment subtraction.
/// @param x At least two finite observations (validated here).
/// @return Extended-precision mean and variance; zero variance is allowed.
inline TestMoments test_moments(const std::vector<double>& x) {
  check_sample(x, 2);
  long double mean = 0, ss = 0;
  for (double v : x)
    mean += static_cast<long double>(v) / x.size();
  for (double v : x) {
    const long double d = v - mean;
    ss += d * d;
  }
  return {mean, ss / (x.size() - 1)};
}
/// @brief Standard-normal p-value computed directly with erfc in the selected tail.
/// @param z Signed standardized statistic.
/// @param a Valid alternative.
/// @return Probability in [0,1], including infinite-statistic limits.
inline double normal_test_p(double z, Alternative a) {
  check_alternative(a);
  if (a == Alternative::Less)
    return .5 * std::erfc(-z / std::sqrt(2.0));
  if (a == Alternative::Greater)
    return .5 * std::erfc(z / std::sqrt(2.0));
  return std::erfc(std::abs(z) / std::sqrt(2.0));
}
/// @brief Student p-value using the small lower tail by symmetry, avoiding 1-CDF.
/// @param t Signed t statistic.
/// @param df Positive degrees of freedom.
/// @param a Valid alternative.
/// @return Probability in [0,1].
inline double student_test_p(double t, double df, Alternative a) {
  check_alternative(a);
  if (a == Alternative::Less)
    return random::student_t_cdf(t, df);
  if (a == Alternative::Greater)
    return random::student_t_cdf(-t, df);
  return std::min(1.0, 2 * random::student_t_cdf(-std::abs(t), df));
}
/// @brief Upper F tail via the complementary incomplete-beta identity.
/// @param x Nonnegative F statistic (possibly infinity).
/// @param df1 Positive numerator degrees of freedom.
/// @param df2 Positive denominator degrees of freedom.
/// @return P(F>=x), evaluated without subtracting an almost-one CDF.
inline double f_test_survival(double x, double df1, double df2) {
  if (std::isinf(x))
    return 0;
  const double z = static_cast<double>(
      df2 / (static_cast<long double>(df2) + static_cast<long double>(df1) * x));
  return random::detail::regularized_incomplete_beta(.5 * df2, .5 * df1, z);
}
/// @brief Upper chi-square tail using the upper incomplete-gamma continued fraction.
/// @param x Nonnegative chi-square statistic (possibly infinity).
/// @param df Positive degrees of freedom.
/// @return P(chi-square>=x), retaining small representable probabilities.
inline double chi_squared_test_survival(double x, double df) {
  if (std::isinf(x))
    return 0;
  const double a = .5 * df, z = .5 * x;
  if (z == 0)
    return 1;
  return std::clamp(
      z < a + 1 ? 1 - random::detail::incomplete_gamma_series(a, z)
                : random::detail::incomplete_gamma_cf(a, z),
      0.0,
      1.0);
}
/// @brief Logarithm of a binomial coefficient in extended precision.
/// @param n Nonnegative total.
/// @param k Count in [0,n].
/// @return log(n choose k).
inline long double test_log_choose(std::size_t n, std::size_t k) {
  return std::lgamma(static_cast<long double>(n) + 1) -
         std::lgamma(static_cast<long double>(k) + 1) -
         std::lgamma(static_cast<long double>(n - k) + 1);
}
/// @brief Exact binomial tail or probability-ordered two-sided sum.
/// @param x Observed count in [0,n].
/// @param n Positive trial count.
/// @param p Null probability in [0,1].
/// @param a Valid alternative.
/// @return Finite-support probability; two-sided ordering uses a relative 1e-7 tolerance.
inline double binomial_test_p(std::size_t x, std::size_t n, double p, Alternative a) {
  if (p == 0)
    return a == Alternative::Less || x == 0 ? 1 : 0;
  if (p == 1)
    return a == Alternative::Greater || x == n ? 1 : 0;
  if (a == Alternative::Less)
    return x == n ? 1 : random::detail::regularized_incomplete_beta(n - x, x + 1, 1 - p);
  if (a == Alternative::Greater)
    return x == 0 ? 1 : random::detail::regularized_incomplete_beta(x, n - x + 1, p);
  /// @brief Log PMF keeps two-sided probability ordering meaningful in tiny tails.
  const auto log_pmf = [&](std::size_t k) {
    return test_log_choose(n, k) + k * std::log(static_cast<long double>(p)) +
           (n - k) * std::log1p(-static_cast<long double>(p));
  };
  const long double threshold = log_pmf(x) + std::log1p(1e-7L);
  long double       sum       = 0;
  for (std::size_t k = 0;; ++k) {
    const auto lp = log_pmf(k);
    if (lp <= threshold)
      sum += std::exp(lp);
    if (k == n)
      break;
  }
  return std::clamp(static_cast<double>(sum), 0.0, 1.0);
}
} // namespace datamunge::stats::detail
