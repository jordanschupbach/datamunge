/// @file correlation_test.cpp
/// @brief Pearson and Spearman correlation tests and Kendall's tau by Knight's algorithm.

#include <algorithm>
#include <cmath>
#include <datamunge/random/distributions.hpp>
#include <datamunge/stats/correlation_test.hpp>
#include <datamunge/stats/detail/ranking.hpp>
#include <datamunge/stats/detail/test_common.hpp>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Shared pieces of the correlation tests.
namespace {

/// @brief Sample Pearson correlation of @p x and @p y.
/// @param x Finite nonconstant first sample.
/// @param y Same-length finite nonconstant second sample.
/// @return Coefficient in [-1,1].
double pearson_r(const std::vector<double>& x, const std::vector<double>& y) {
  detail::check_sample(x, 3);
  detail::check_sample(y, 3);
  const auto  mx = detail::test_moments(x).mean, my = detail::test_moments(y).mean;
  long double xx = 0, yy = 0, xy = 0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    const long double a = x[i] - mx, b = y[i] - my;
    xx += a * a;
    yy += b * b;
    xy += a * b;
  }
  if (!(xx > 0) || !(yy > 0))
    throw std::invalid_argument("correlation_test: constant sample");
  return std::clamp(static_cast<double>(xy / std::sqrt(xx) / std::sqrt(yy)), -1.0, 1.0);
}

/// @brief p-value of Student's t statistic @p t with @p df degrees of freedom for @p alternative.
/// @param t Signed statistic.
/// @param df Positive degrees of freedom.
/// @param alternative Requested tail.
/// @return Probability in [0,1].
double t_p_value(double t, double df, Alternative alternative) {
  return detail::student_test_p(t, df, alternative);
}

/// @brief Test result for correlation @p r from @p n pairs: the t statistic
/// \f$r\sqrt{(n-2)/(1-r^2)}\f$ and its p-value.
/// @param r Correlation in [-1,1].
/// @param n Number of observed pairs.
/// @param alternative Requested tail.
/// @return Owned t-reference result without confidence bounds.
HypothesisTestResult correlation_result_from_r(double r, double n, Alternative alternative) {
  detail::check_alternative(alternative);
  const double df = n - 2.0;
  const double t  = r * std::sqrt(df / (1.0 - r * r));

  HypothesisTestResult result;
  result.statistic   = t;
  result.parameter1  = df;
  result.p_value     = t_p_value(t, df, alternative);
  result.estimate1   = r;
  result.alternative = alternative;
  return result;
}

} // namespace

HypothesisTestResult pearson_correlation_test(
    const std::vector<double>& x,
    const std::vector<double>& y,
    Alternative                alternative,
    double                     conf_level) {
  if (x.size() != y.size() || x.size() < 3)
    throw std::invalid_argument("pearson_correlation_test: x and y must have equal length >= 3");
  detail::check_confidence(conf_level);
  const auto   n = static_cast<double>(x.size());
  const double r = pearson_r(x, y);

  auto result   = correlation_result_from_r(r, n, alternative);
  result.method = "Pearson's product-moment correlation";

  if (n <= 3)
    return result;
  const double z      = std::atanh(r);
  const double se_z   = 1.0 / std::sqrt(n - 3.0);
  result.has_conf_int = true;
  if (alternative == Alternative::TwoSided) {
    const double crit     = random::normal_quantile(1.0 - 0.5 * (1.0 - conf_level));
    result.conf_int_lower = std::tanh(z - crit * se_z);
    result.conf_int_upper = std::tanh(z + crit * se_z);
  } else if (alternative == Alternative::Less) {
    const double crit     = random::normal_quantile(conf_level);
    result.conf_int_lower = -1.0;
    result.conf_int_upper = std::tanh(z + crit * se_z);
  } else {
    const double crit     = random::normal_quantile(conf_level);
    result.conf_int_lower = std::tanh(z - crit * se_z);
    result.conf_int_upper = 1.0;
  }
  return result;
}

HypothesisTestResult spearman_correlation_test(
    const std::vector<double>& x,
    const std::vector<double>& y,
    Alternative                alternative) {
  if (x.size() != y.size() || x.size() < 3)
    throw std::invalid_argument("spearman_correlation_test: x and y must have equal length >= 3");
  const auto   rank_x = detail::rank_with_ties(x);
  const auto   rank_y = detail::rank_with_ties(y);
  const double rho    = pearson_r(rank_x, rank_y);

  auto result   = correlation_result_from_r(rho, static_cast<double>(x.size()), alternative);
  result.method = "Spearman's rank correlation rho";
  return result;
}

KendallTau kendall_tau(const std::vector<double>& x, const std::vector<double>& y) {
  if (x.size() != y.size() || x.size() < 2)
    throw std::invalid_argument("kendall_tau: need two equal-length vectors of length >= 2");
  detail::check_sample(x, 2);
  detail::check_sample(y, 2);
  const std::size_t        n = x.size();
  std::vector<std::size_t> order(n);
  for (std::size_t i = 0; i < n; ++i)
    order[i] = i;
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    return x[a] != x[b] ? x[a] < x[b] : y[a] < y[b];
  });
  /// @brief Counts tied pairs for a caller-provided equality predicate.
  // Runs of equal values among adjacent elements of a sorted sequence give tied pairs.
  const auto tied_pairs = [&](auto&& same) {
    long long total = 0, run = 1;
    for (std::size_t k = 1; k < n; ++k) {
      if (same(order[k - 1], order[k]))
        ++run;
      else {
        total += run * (run - 1) / 2;
        run = 1;
      }
    }
    return total + run * (run - 1) / 2;
  };
  const long long n_x = tied_pairs([&](std::size_t a, std::size_t b) { return x[a] == x[b]; });
  const long long n_xy =
      tied_pairs([&](std::size_t a, std::size_t b) { return x[a] == x[b] && y[a] == y[b]; });
  // Merge sort on y counts the swaps needed: each is a pair with x_i < x_j (or tied x) and y_i >
  // y_j.
  std::vector<double> v(n), buffer(n);
  for (std::size_t k = 0; k < n; ++k)
    v[k] = y[order[k]];
  long long swaps = 0;
  for (std::size_t width = 1; width < n; width *= 2) {
    for (std::size_t lo = 0; lo < n; lo += 2 * width) {
      const std::size_t mid = std::min(lo + width, n), hi = std::min(lo + 2 * width, n);
      std::size_t       i = lo, j = mid, k = lo;
      while (i < mid && j < hi) {
        if (v[j] < v[i]) {
          swaps += static_cast<long long>(mid - i);
          buffer[k++] = v[j++];
        } else {
          buffer[k++] = v[i++];
        }
      }
      while (i < mid)
        buffer[k++] = v[i++];
      while (j < hi)
        buffer[k++] = v[j++];
    }
    v.swap(buffer);
  }
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return y[a] < y[b]; });
  const long long n_y = tied_pairs([&](std::size_t a, std::size_t b) { return y[a] == y[b]; });

  KendallTau      r;
  const long long n0 = static_cast<long long>(n) * static_cast<long long>(n - 1) / 2;
  r.ties_xy          = n_xy;
  r.ties_x           = n_x - n_xy;
  r.ties_y           = n_y - n_xy;
  r.discordant       = swaps; // pairs tied in x are sorted by y, so they never swap
  r.concordant       = n0 - n_x - n_y + n_xy - r.discordant;
  r.tau_a            = static_cast<double>(r.concordant - r.discordant) / static_cast<double>(n0);
  const double denom = std::sqrt(static_cast<double>(n0 - n_x) * static_cast<double>(n0 - n_y));
  r.tau_b            = denom > 0 ? static_cast<double>(r.concordant - r.discordant) / denom : 0.0;
  return r;
}

} // namespace datamunge::stats
