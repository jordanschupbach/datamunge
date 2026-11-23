/// @file count_distributions.cpp
/// @brief Tail calculation and integer inversion for two overdispersed count laws.

#include <datamunge/random/count_distributions.hpp>
#include <datamunge/random/distributions.hpp>
#include <datamunge/random/simplex.hpp>
#include "beta_kernels.hpp"
#include "count_likelihood.hpp"
#include "simplex_detail.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace datamunge::random {

/// @brief Validation and widened probability calculations.
namespace {

/// @brief Checks the negative binomial parameter domain.
/// @param mean Count mean.
/// @param precision Inverse overdispersion.
/// @throws std::invalid_argument Nonfinite or inadmissible parameters.
void check_negative_binomial(double mean, double precision) {
  if (!(mean >= 0 && precision > 0) || !std::isfinite(mean) || !std::isfinite(precision))
    throw std::invalid_argument("negative binomial: invalid mean or precision");
}

/// @brief Checks the beta-binomial shape parameters.
/// @param alpha First shape.
/// @param beta Second shape.
/// @throws std::invalid_argument Nonpositive or nonfinite shape.
void check_beta_binomial(double alpha, double beta) {
  if (!(alpha > 0 && beta > 0) || !std::isfinite(alpha) || !std::isfinite(beta))
    throw std::invalid_argument("beta-binomial: shapes must be positive and finite");
}

using detail::beta_deviance;
using detail::beta_kernel_log;
using detail::count_log_beta;
using detail::gamma_increment;
using detail::large_beta_tail;
using detail::negative_binomial_log_mass;

/// @brief Computes the logarithm of 1/(a B(a,b)) at a small first shape.
/// @param a First beta shape, at most one.
/// @param b Integer second beta shape.
/// @return Log normalization with its small-a scale preserved.
long double small_beta_log_normalization(long double a, long double b) {
  // log(1/(a B(a,b))) tends to zero with a. Keep that term at its natural scale.
  long double normalization = 0;
  if (b < 16) {
    for (int j = 1; j < b; ++j)
      normalization += std::log1p(a / j);
  } else {
    const auto log_gamma_one_plus_a = a < 1e-5L
        ? a * (-0.5772156649015328606L + a * (0.8224670334241132182L - a * 0.4006856343865314285L))
        : std::lgamma(1 + a);
    normalization = gamma_increment(b, a) - log_gamma_one_plus_a;
  }
  return normalization;
}

/// @brief Evaluates the log of I_p(a,b) by its hypergeometric series.
/// @param a First beta shape, at most one.
/// @param b Integer second beta shape.
/// @param p Success probability with b*p <= 1/2.
/// @param odds Failure odds, (1-p)/p.
/// @return Logarithm of the inclusive lower-tail probability.
long double small_beta_log_cdf(long double a, long double b, long double p, long double odds) {
  long double sum = 0, factor = 1;
  for (int j = 1; j < 40 && j < b; ++j) {
    factor *= ((j - b) * p) / j;
    const auto next = sum + factor * (a / (a + j));
    if (next == sum)
      break;
    sum = next;
  }
  return -a * std::log1p(odds) + small_beta_log_normalization(a, b) + std::log1p(sum);
}

/// @brief Integrates the upper beta tail in a scaled coordinate when a and p are small.
/// @param a First beta shape, at most one.
/// @param b Integer second beta shape.
/// @param p Success probability below 0.01 with b*p > 1/2.
/// @return Probability above p, preserving the scale as a tends to zero.
long double small_beta_upper_tail(long double a, long double b, long double p) {
  const auto start = b * p;
  if (start > 750)
    return 0;
  // For v = b*x, the integrand is proportional to (1-v/b)^(b-1)*v^(a-1).
  // On this branch b > 50, so the integrand decreases at least exponentially
  // at rate (b-1)/b. Factor its value at the start out of the integral.
  constexpr long double nodes[]{.183434642495649805L, .525532409916328986L,
                                .796666477413626740L, .960289856497536232L};
  constexpr long double weights[]{.362683783378361983L, .313706645877887287L,
                                  .222381034453374471L, .101228536290376259L};
  const auto end = std::min(48.0L, b - start);
  const auto integrand = [&](long double offset) {
    return std::exp((b - 1) * std::log1p(-offset / (b - start)) +
                    (a - 1) * std::log1p(offset / start));
  };
  long double integral = 0;
  for (long double lo = 0; lo < end;) {
    const auto hi = std::min(end, lo + (lo < 4 ? .125L : .5L));
    const auto mid = (lo + hi) / 2;
    const auto half_width = (hi - lo) / 2;
    for (int j = 0; j < 4; ++j) {
      const auto offset = half_width * nodes[j];
      integral += half_width * weights[j] * (integrand(mid - offset) + integrand(mid + offset));
    }
    lo = hi;
  }
  return (a / start) * std::exp((b - 1) * std::log1p(-p) + a * std::log(p) +
                               small_beta_log_normalization(a, b)) * integral;
}

/// @brief Checks whether Poisson corrections are negligible around the requested count.
/// @param k Event count.
/// @param mean Positive finite count mean.
/// @param precision Positive finite inverse overdispersion.
/// @return True when variance and log-mass corrections are below double precision.
bool use_poisson_limit(long double k, long double mean, long double precision) {
  const auto tolerance = std::numeric_limits<double>::epsilon() * precision / 8;
  return std::max(k, mean) < tolerance && std::abs(k - mean) < std::sqrt(tolerance);
}

/// @brief Computes a negative binomial tail from checked parameters with direct tail orientation.
/// @param k Event count.
/// @param mean Positive finite mean.
/// @param precision Positive finite precision.
/// @param lower_tail Whether to return the inclusive lower tail.
/// @return Requested probability.
double negative_binomial_tail(std::size_t k, double mean, double precision, bool lower_tail) {
  const long double a = precision;
  const long double b = static_cast<long double>(k) + 1;
  const long double ratio = static_cast<long double>(mean) / a;
  if (k == 0) {
    const auto log_mass = -a * std::log1p(ratio);
    return static_cast<double>(lower_tail ? std::exp(log_mass) : -std::expm1(log_mass));
  }
  if (use_poisson_limit(k, mean, a))
    return gamma_cdf(mean, static_cast<double>(b), 1, !lower_tail);
  const long double p = 1 / (1 + ratio), q = 1 / (1 + 1 / ratio);
  if (a <= 1 && b * p <= .5L) {
    const auto log_cdf = small_beta_log_cdf(a, b, p, ratio);
    return static_cast<double>(lower_tail ? std::exp(log_cdf) : -std::expm1(log_cdf));
  }
  if (a <= 1 && p < .01L) {
    const auto upper = small_beta_upper_tail(a, b, p);
    return static_cast<double>(lower_tail ? 1 - upper : upper);
  }
  const bool direct_lower = p < .5L ? p < (a + 1) / (a + b + 2)
                                    : q > (b + 1) / (a + b + 2);
  if (k < 16 && direct_lower && p > .9L) {
    long double sum = 0;
    for (std::size_t j = 0; j <= k; ++j)
      sum += std::exp(negative_binomial_log_mass(j, mean, a, ratio));
    return static_cast<double>(lower_tail ? sum : 1 - sum);
  }
  long double log_front;
  if (a >= 16 && b >= 16) {
    const auto crossing = p * (b - mean);
    const auto deviance = beta_deviance(a, b, ratio, crossing);
    if (std::min(a, b) >= 1e11L)
      return static_cast<double>(large_beta_tail(a, b, crossing, deviance, lower_tail));
    log_front = beta_kernel_log(a, b, deviance);
  } else {
    // The kernel divided by B(a,b) equals (a+b-1)*q times mass(a,b-1).
    log_front = std::log(a + k) - std::log1p(1 / ratio) +
                negative_binomial_log_mass(k, mean, a, ratio);
  }
  const auto front = std::exp(log_front);
  if (front == 0)
    return static_cast<double>(lower_tail != direct_lower);
  const auto fraction = direct_lower ? detail::incomplete_beta_cf(a, b, p, 100000) / a
                                     : detail::incomplete_beta_cf(b, a, q, 100000) / b;
  const auto log_tail = std::min(0.0L, log_front + std::log(fraction));
  return static_cast<double>(lower_tail == direct_lower ? std::exp(log_tail) : -std::expm1(log_tail));
}

/// @brief Computes log P(X=0) for a beta-binomial count.
/// @param n Number of trials.
/// @param a First beta shape.
/// @param b Second beta shape.
/// @return Logarithm of the mass at zero.
long double beta_binomial_zero(std::size_t n, long double a, long double b) {
  if (n <= 128 || a + b == b) {
    if (a + b == b && n > 128)
      return -a * std::log1p(static_cast<long double>(n) / b);
    long double sum = 0;
    for (std::size_t i = 0; i < n; ++i)
      sum -= std::log1p(a / (b + i));
    return sum;
  }
  return gamma_increment(b, n) - gamma_increment(a + b, n);
}

/// @brief Ratio between adjacent beta-binomial counts on the log scale.
/// @param k Current count, strictly less than n.
/// @param n Number of trials.
/// @param a First beta shape.
/// @param b Second beta shape.
/// @return Natural logarithm of mass(k+1)/mass(k).
long double beta_binomial_log_step(std::size_t k, std::size_t n, long double a, long double b) {
  return std::log((static_cast<long double>(n - k)) / (b + (n - k - 1))) +
         std::log((a + k) / (static_cast<long double>(k) + 1));
}

} // namespace

double dirichlet_log_pdf(const std::vector<double>& point, const std::vector<double>& concentration) {
  simplex_detail::check_concentration(concentration);
  if (point.size() != concentration.size())
    throw std::invalid_argument("Dirichlet: point and concentration sizes differ");
  for (const double value : point)
    if (!std::isfinite(value))
      throw std::invalid_argument("Dirichlet: point must be finite");

  const auto total = std::accumulate(point.begin(), point.end(), 0.L);
  constexpr auto outside = -std::numeric_limits<double>::infinity();
  if (std::abs(total - 1) > 64 * std::numeric_limits<double>::epsilon() ||
      std::ranges::any_of(point, [](double value) { return value <= 0; }))
    return outside;

  // Conditional beta densities factor the simplex density. Traverse
  // backwards to preserve even tiny remaining coordinate/shape sums.
  long double remainder = point.back(), shape_sum = concentration.back(), log_density = 0;
  for (std::size_t i = point.size() - 1; i > 0; --i) {
    const long double x = point[i - 1], a = concentration[i - 1], b = shape_sum;
    const auto log_kernel = a >= 16 && b >= 16
        ? beta_kernel_log(a, b, beta_deviance(a, b, remainder / x, (x * b - remainder * a) / (x + remainder)))
        : -count_log_beta(a, b) - a * std::log1p(remainder / x) - b * std::log1p(x / remainder);
    log_density += log_kernel + std::log(x + remainder) - std::log(x) -
                   std::log(remainder) + std::log(total);
    remainder += x;
    shape_sum += a;
  }
  return static_cast<double>(log_density);
}

double dirichlet_multinomial_log_pmf(const std::vector<std::size_t>& counts,
                                    const std::vector<double>& concentration) {
  simplex_detail::check_concentration(concentration);
  if (counts.size() != concentration.size())
    throw std::invalid_argument("Dirichlet–multinomial: counts and concentration sizes differ");

  std::size_t total = 0;
  long double shape_sum = 0;
  for (std::size_t i = 0; i < counts.size(); ++i) {
    if (counts[i] > std::numeric_limits<std::size_t>::max() - total)
      throw std::overflow_error("Dirichlet–multinomial: total count is too large");
    total += counts[i];
    shape_sum += concentration[i];
  }
  if (total == 0 || counts.size() == 1)
    return 0;

  // Independent negative binomials with common odds give the compound
  // law conditional on their total. Make that total its own mean.
  const auto odds = static_cast<long double>(total) / shape_sum;
  auto result = -negative_binomial_log_mass(total, total, shape_sum, odds);
  for (std::size_t i = 0; i < counts.size(); ++i) {
    const long double shape = concentration[i];
    result += negative_binomial_log_mass(counts[i], shape * odds, shape, odds);
  }
  return static_cast<double>(std::min(0.L, result));
}

double negative_binomial_pmf(std::size_t k, double mean, double precision) {
  check_negative_binomial(mean, precision);
  if (mean == 0)
    return k == 0 ? 1 : 0;
  const long double a = precision, x = k;
  const long double ratio = static_cast<long double>(mean) / a;
  if (use_poisson_limit(x, mean, a))
    return gamma_pdf(mean, static_cast<double>(x + 1));
  return static_cast<double>(std::exp(negative_binomial_log_mass(k, mean, a, ratio)));
}

double negative_binomial_cdf(std::size_t k, double mean, double precision, bool lower_tail) {
  check_negative_binomial(mean, precision);
  return mean == 0 ? static_cast<double>(lower_tail) : negative_binomial_tail(k, mean, precision, lower_tail);
}

std::size_t negative_binomial_quantile(double probability, double mean, double precision) {
  check_negative_binomial(mean, precision);
  detail::require_probability(probability, "negative_binomial_quantile");
  if (mean == 0 || probability == 0)
    return 0;
  constexpr auto maximum = std::numeric_limits<std::size_t>::max();
  if (probability == 1)
    return maximum;
  /// @brief Tests a CDF crossing using the small target tail.
  const auto reaches = [&](std::size_t k) {
    return probability <= .5 ? negative_binomial_tail(k, mean, precision, true) >= probability :
                               negative_binomial_tail(k, mean, precision, false) <= 1 - probability;
  };
  std::size_t lo = 0, hi = 1;
  while (hi < maximum && !reaches(hi)) {
    lo = hi + 1;
    hi = hi > maximum / 2 ? maximum : hi * 2;
  }
  while (lo < hi) {
    const auto k = std::midpoint(lo, hi);
    if (reaches(k))
      hi = k;
    else
      lo = k + 1;
  }
  return lo;
}

double beta_binomial_pmf(std::size_t k, std::size_t trials, double alpha, double beta) {
  check_beta_binomial(alpha, beta);
  if (k > trials)
    return 0;
  if (k > trials - k) {
    k = trials - k;
    std::swap(alpha, beta);
  }
  if (k <= 128) {
    auto mass = beta_binomial_zero(trials, alpha, beta);
    for (std::size_t i = 0; i < k; ++i)
      mass += beta_binomial_log_step(i, trials, alpha, beta);
    return static_cast<double>(std::exp(mass));
  }
  const long double x = k, y = trials - k;
  const auto result = -count_log_beta(x + 1, y + 1) - std::log(x + y + 1) +
                      gamma_increment(alpha, x) + gamma_increment(beta, y) -
                      gamma_increment(static_cast<long double>(alpha) + beta, x + y);
  return static_cast<double>(std::clamp(std::exp(result), 0.0L, 1.0L));
}

double beta_binomial_cdf(std::size_t k, std::size_t trials, double alpha, double beta, bool lower_tail) {
  check_beta_binomial(alpha, beta);
  if (k >= trials)
    return static_cast<double>(lower_tail);
  if (!lower_tail) {
    k = trials - k - 1;
    std::swap(alpha, beta);
  }
  long double mass = beta_binomial_zero(trials, alpha, beta);
  long double sum = std::exp(mass);
  for (std::size_t i = 0; i < k; ++i) {
    mass += beta_binomial_log_step(i, trials, alpha, beta);
    sum += std::exp(mass);
  }
  return static_cast<double>(std::min(1.0L, sum));
}

std::size_t beta_binomial_quantile(double probability, std::size_t trials, double alpha, double beta) {
  check_beta_binomial(alpha, beta);
  detail::require_probability(probability, "beta_binomial_quantile");
  if (probability == 0)
    return 0;
  if (probability == 1)
    return trials;
  const bool reverse = probability > .5;
  if (reverse)
    std::swap(alpha, beta);
  long double mass = beta_binomial_zero(trials, alpha, beta);
  long double sum = 0;
  for (std::size_t i = 0; i < trials; ++i) {
    sum += std::exp(mass);
    if (reverse ? static_cast<double>(1 - sum) < probability : static_cast<double>(sum) >= probability)
      return reverse ? trials - i : i;
    mass += beta_binomial_log_step(i, trials, alpha, beta);
  }
  return reverse ? 0 : trials;
}

} // namespace datamunge::random
