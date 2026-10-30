/// @file ks_test.cpp
/// @brief One- and two-sample Kolmogorov-Smirnov tests with asymptotic p-values.

#include <algorithm>
#include <cmath>
#include <datamunge/random/distributions.hpp>
#include <datamunge/stats/detail/test_common.hpp>
#include <datamunge/stats/ks_test.hpp>
#include <stdexcept>
#include <string>

namespace datamunge::stats {

/// @brief The Kolmogorov distribution and the shared one-sample test.
namespace {

/// @brief Asymptotic two-sided Kolmogorov distribution p-value, P(K > lambda), via the alternating
///        series representation (Abramowitz & Stegun 26.2.16 / the standard "Q_KS" function).
/// @param lambda Nonnegative scaled discrepancy.
/// @return Approximate survival probability.
double kolmogorov_p_value(double lambda) {
  if (lambda < 0.2)
    return 1.0;
  double sum = 0.0;
  for (int k = 1; k <= 100; ++k) {
    const double term =
        ((k % 2 == 1) ? 1.0 : -1.0) * std::exp(-2.0 * static_cast<double>(k * k) * lambda * lambda);
    sum += term;
    if (std::fabs(term) < 1e-12)
      break;
  }
  return std::clamp(2.0 * sum, 0.0, 1.0);
}

/// @brief One-sample KS statistic and p-value of @p x against the continuous CDF @p cdf.
/// @tparam Cdf Continuous fixed-null CDF callable.
/// @param x At least two finite observations.
/// @param cdf Cumulative distribution with fixed parameters.
/// @param alternative Requested CDF-order alternative.
/// @param who Caller label.
/// @return Owned asymptotic KS test result.
template<class Cdf>
HypothesisTestResult
ks_one_sample(const std::vector<double>& x, Cdf cdf, Alternative alternative, const char* who) {
  detail::check_sample(x, 2);
  detail::check_alternative(alternative);
  if (x.size() < 2)
    throw std::invalid_argument(std::string(who) + ": need at least 2 observations");
  std::vector<double> sorted = x;
  std::sort(sorted.begin(), sorted.end());
  const auto n = static_cast<double>(sorted.size());

  double d_plus = 0.0, d_minus = 0.0;
  for (std::size_t idx = 0; idx < sorted.size(); ++idx) {
    const double f = cdf(sorted[idx]);
    const double i = static_cast<double>(idx + 1);
    d_plus         = std::max(d_plus, i / n - f);
    d_minus        = std::max(d_minus, f - (i - 1.0) / n);
  }

  HypothesisTestResult result;
  result.alternative = alternative;
  result.parameter1  = n;
  result.method      = "One-sample Kolmogorov-Smirnov test";
  if (alternative == Alternative::Greater) {
    result.statistic = d_plus;
    result.p_value   = std::exp(-2.0 * n * d_plus * d_plus);
  } else if (alternative == Alternative::Less) {
    result.statistic = d_minus;
    result.p_value   = std::exp(-2.0 * n * d_minus * d_minus);
  } else {
    const double d      = std::max(d_plus, d_minus);
    result.statistic    = d;
    const double lambda = (std::sqrt(n) + 0.12 + 0.11 / std::sqrt(n)) * d;
    result.p_value      = kolmogorov_p_value(lambda);
  }
  return result;
}

} // namespace

HypothesisTestResult ks_test_one_sample_normal(
    const std::vector<double>& x,
    double                     mean,
    double                     sd,
    Alternative                alternative) {
  if (!std::isfinite(mean) || !std::isfinite(sd) || sd <= 0)
    throw std::invalid_argument("KS: finite mean and positive finite sd required");
  return ks_one_sample(
      x,
      [&](double v) { return .5 * std::erfc(-(v - mean) / (sd * std::sqrt(2.0))); },
      alternative,
      "ks_test_one_sample_normal");
}

HypothesisTestResult ks_test_one_sample_uniform(
    const std::vector<double>& x,
    double                     lower,
    double                     upper,
    Alternative                alternative) {
  if (!std::isfinite(lower) || !std::isfinite(upper) || !(upper > lower))
    throw std::invalid_argument("ks_test_one_sample_uniform: upper must exceed lower");
  return ks_one_sample(
      x,
      [&](double v) { return std::clamp((v - lower) / (upper - lower), 0.0, 1.0); },
      alternative,
      "ks_test_one_sample_uniform");
}

HypothesisTestResult ks_test_two_sample(
    const std::vector<double>& x,
    const std::vector<double>& y,
    Alternative                alternative) {
  detail::check_sample(x);
  detail::check_sample(y);
  detail::check_alternative(alternative);
  if (x.empty() || y.empty())
    throw std::invalid_argument("ks_test_two_sample: both samples must be non-empty");
  std::vector<double> sx = x, sy = y;
  std::sort(sx.begin(), sx.end());
  std::sort(sy.begin(), sy.end());
  const auto n1 = static_cast<double>(sx.size());
  const auto n2 = static_cast<double>(sy.size());

  std::vector<double> combined = sx;
  combined.insert(combined.end(), sy.begin(), sy.end());
  std::sort(combined.begin(), combined.end());

  double d_plus = 0.0, d_minus = 0.0;
  for (const double v : combined) {
    const double f1 =
        static_cast<double>(std::upper_bound(sx.begin(), sx.end(), v) - sx.begin()) / n1;
    const double f2 =
        static_cast<double>(std::upper_bound(sy.begin(), sy.end(), v) - sy.begin()) / n2;
    d_plus  = std::max(d_plus, f1 - f2);
    d_minus = std::max(d_minus, f2 - f1);
  }

  const double n_e = n1 * n2 / (n1 + n2);

  HypothesisTestResult result;
  result.alternative = alternative;
  result.parameter1  = n1;
  result.parameter2  = n2;
  result.method      = "Two-sample Kolmogorov-Smirnov test";
  if (alternative == Alternative::Greater) {
    result.statistic = d_plus;
    result.p_value   = std::exp(-2.0 * n_e * d_plus * d_plus);
  } else if (alternative == Alternative::Less) {
    result.statistic = d_minus;
    result.p_value   = std::exp(-2.0 * n_e * d_minus * d_minus);
  } else {
    const double d      = std::max(d_plus, d_minus);
    result.statistic    = d;
    const double lambda = (std::sqrt(n_e) + 0.12 + 0.11 / std::sqrt(n_e)) * d;
    result.p_value      = kolmogorov_p_value(lambda);
  }
  return result;
}

} // namespace datamunge::stats
