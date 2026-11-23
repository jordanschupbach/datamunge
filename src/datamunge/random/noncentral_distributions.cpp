/// @file noncentral_distributions.cpp
/// @brief Poisson mixtures and stable inversions for two noncentral distributions.

#include <datamunge/random/noncentral_distributions.hpp>
#include "beta_kernels.hpp"
#include "noncentral_t_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <numeric>
#include <stdexcept>

namespace datamunge::random {

/// @brief Internal tail evaluators and inverse-tail solvers.
namespace {

/// @brief Largest noncentrality evaluated by the full Poisson mixture.
constexpr long double mixture_limit = 2'000'000;

/// @brief Checks the nonnegative finite degrees of freedom and noncentrality of the chi-square law.
/// @param df Degrees of freedom.
/// @param noncentrality Distribution noncentrality.
/// @throws std::invalid_argument Negative or nonfinite parameters.
void check_chi_square(double df, double noncentrality) {
  if (!(df >= 0 && noncentrality >= 0) || !std::isfinite(df) || !std::isfinite(noncentrality))
    throw std::invalid_argument("noncentral chi-square: degrees of freedom and noncentrality must be finite and nonnegative");
}

/// @brief Checks the positive finite degrees of freedom and finite noncentrality of the F law.
/// @param first Numerator degrees of freedom.
/// @param second Denominator degrees of freedom.
/// @param noncentrality Numerator noncentrality.
/// @throws std::invalid_argument Invalid degrees of freedom or noncentrality.
void check_f(double first, double second, double noncentrality) {
  check_chi_square(first, noncentrality);
  if (!(first > 0 && second > 0) || !std::isfinite(second))
    throw std::invalid_argument("noncentral F: degrees of freedom must be positive and finite");
}

/// @brief Evaluates a beta tail without losing the smaller transformed coordinate.
/// @param a First beta shape.
/// @param b Second beta shape.
/// @param ratio Success odds of the transformed cutoff.
/// @param crossing Difference between the cutoff times the sum of shapes and the first shape.
/// @param lower Whether to return the lower tail.
/// @return Selected beta tail.
long double beta_tail(long double a, long double b, long double ratio,
                      long double crossing, bool lower) {
  if (ratio == 0 || std::isinf(ratio))
    return (ratio > 0) == lower ? 1 : 0;
  const auto p = 1 / (1 + 1 / ratio);
  const auto q = 1 / (1 + ratio);
  const auto deviance = a >= 16 && b >= 16 ? detail::beta_deviance(a, b, 1 / ratio, crossing) : 0;
  if (std::min(a, b) >= 1e8L)
    return detail::large_beta_tail(a, b, crossing, deviance, lower);
  const auto log_front = a >= 16 && b >= 16 ? detail::beta_kernel_log(a, b, deviance)
      : -detail::count_log_beta(a, b) - a * std::log1p(1 / ratio) - b * std::log1p(ratio);
  const bool direct_lower = p * (b + 1) < q * (a + 1);
  const auto fraction = direct_lower ? detail::incomplete_beta_cf(a, b, p, 10000) / a
                                     : detail::incomplete_beta_cf(b, a, q, 10000) / b;
  const auto log_tail = std::min(0.L, log_front + std::log(fraction));
  return direct_lower == lower ? std::exp(log_tail) : -std::expm1(log_tail);
}

/// @brief Sums a Poisson mixture from the mode, bounding both omitted tails.
/// @tparam Component Callable returning the selected conditional tail at a Poisson index.
/// @param mean Poisson mean, below half the mixture limit.
/// @param lower Whether component probabilities are decreasing in the index.
/// @param component Conditional probability function.
/// @return Tail probability before rounding to double.
template <class Component>
long double poisson_mixture(long double mean, bool lower, const Component& component) {
  if (mean == 0)
    return component(0);
  auto first = static_cast<std::size_t>(mean), last = first;
  auto left = first == 0 ? std::exp(-mean)
                         : std::exp(first * std::log(mean) - mean - std::lgamma(first + 1.L));
  auto right = left;
  auto left_term = component(first), right_term = left_term;
  auto sum = left * left_term;
  constexpr long double resolution = std::numeric_limits<double>::denorm_min();
  for (;;) {
    const auto left_next = first == 0 ? 0.L : left * (first / mean);
    const auto right_next = right * (mean / (last + 1.L));
    const auto left_bound = first == 0 ? 0.L : left_next / (1 - (first - 1.L) / mean);
    const auto right_bound = right_next / (1 - mean / (last + 2.L));
    const auto left_error = left_bound * (lower ? 1 : left_term);
    const auto right_error = right_bound * (lower ? right_term : 1);
    if (left_error + right_error <=
        std::max(resolution / 4, 8 * std::numeric_limits<long double>::epsilon() * sum))
      break;
    if (left_error >= right_error && first != 0) {
      left = left_next;
      left_term = component(--first);
      sum += left * left_term;
    } else {
      right = right_next;
      right_term = component(++last);
      sum += right * right_term;
    }
  }
  return std::clamp(sum, 0.L, 1.L);
}

/// @brief Forms a Lugannani–Rice probability from signed saddlepoint quantities.
/// @param deviance Nonnegative saddlepoint deviance.
/// @param u Signed saddlepoint times the tilted standard deviation.
/// @param near_mean Correction at the mean, equal to skewness divided by six.
/// @param lower Whether to return the lower tail.
/// @return Saddlepoint approximation on the selected tail.
long double saddle_probability(long double deviance, long double u, long double near_mean, bool lower) {
  const auto w = std::copysign(std::sqrt(2 * deviance), u);
  const auto correction = std::abs(w) < .001L ? near_mean : 1 / w - 1 / u;
  const auto direction = lower ? 1.L : -1.L;
  const auto density = std::exp(-deviance) / std::sqrt(2 * std::numbers::pi_v<long double>);
  return std::clamp(std::erfc(-direction * w / std::numbers::sqrt2_v<long double>) / 2 +
                    direction * density * correction, 0.L, 1.L);
}

/// @brief Applies the large-noncentrality saddlepoint approximation to a chi-square tail.
/// @param x Positive cutoff.
/// @param df Nonnegative degrees of freedom.
/// @param lambda Large positive noncentrality.
/// @param lower Whether to return the lower tail.
/// @return Saddlepoint tail approximation.
long double chi_square_saddle(long double x, long double df, long double lambda, bool lower) {
  const auto h = 2 * x / (df + std::hypot(df, 2 * std::sqrt(x) * std::sqrt(lambda)));
  const auto d = (x - (df + lambda)) / (df + lambda * (h + 1));
  const auto deviance = .5L * (df * (d < -.5L ? d - std::log(h) : detail::gamma_divergence(d)) + lambda * d * d);
  const auto u = d * std::sqrt(df / 2 + lambda * h);
  const auto skew = (df + 3 * lambda) / (3 * (df + 2 * lambda) * std::sqrt((df + 2 * lambda) / 2));
  return saddle_probability(deviance, u, skew, lower);
}

/// @brief Applies a saddlepoint approximation to the difference of two scaled chi-squares.
/// @param x Positive F cutoff.
/// @param df1 Numerator degrees of freedom.
/// @param df2 Denominator degrees of freedom.
/// @param lambda Large positive numerator noncentrality.
/// @param lower Whether to return the lower tail.
/// @return Selected noncentral F tail.
long double f_saddle(long double x, long double df1, long double df2, long double lambda, bool lower) {
  // At the cutoff the F event is a sign comparison of independent
  // scaled chi-squares. Solve the saddle equation using the ratio r
  // of their two tilted rates.
  const auto c = (df1 / df2) * x;
  const auto u = 1 / (1 + c), v = 1 / (1 + 1 / c);
  const auto linear = df1 + v * lambda;
  const auto r = (linear + std::hypot(linear, 2 * std::sqrt(v * df2) * std::sqrt(lambda))) /
                 (2 * df1 * x);
  const auto difference = u * std::fma(-df1, x - 1, lambda) / (v * df2 + u * (u / r) * lambda);
  const auto ha = u / r + v, hb = u + v * r;
  const auto da = -u * difference / r, db = v * difference;
  const auto dev_a = da < -.5L ? da - std::log(ha) : detail::gamma_divergence(da);
  const auto dev_b = db < -.5L ? db - std::log(hb) : detail::gamma_divergence(db);
  const auto deviance = (df1 * dev_a + lambda * da * da + df2 * dev_b) / 2;
  const auto tilted = std::copysign(std::sqrt(da * da * (df1 / 2 + lambda * ha) + db * db * df2 / 2), -difference);
  const auto variance_half = u * u * (df1 + 2 * lambda) + v * v * df2;
  const auto skew = (u * u * u * (df1 + 3 * lambda) - v * v * v * df2) /
                    (3 * variance_half * std::sqrt(variance_half / 2));
  return saddle_probability(deviance, tilted, skew, lower);
}

/// @brief Evaluates the large-noncentrality F approximation with a skew-aware transition.
/// @param x Positive F cutoff.
/// @param a Half the numerator degrees of freedom.
/// @param b Half the denominator degrees of freedom.
/// @param mean Half the numerator noncentrality.
/// @param lower Whether to return the lower tail.
/// @return Selected approximate tail probability.
long double large_f_tail(long double x, long double a, long double b, long double mean, bool lower) {
  const auto expected = a + mean;
  const auto saddle_weight = std::clamp((16 * (b / expected) - 1) / 3, 0.L, 1.L);
  if (saddle_weight == 1)
    return f_saddle(x, 2 * a, 2 * b, 2 * mean, lower);
  const auto matched = expected * (expected / (a + 2 * mean));
  const auto ratio = (matched / b) * (a / expected) * x;
  const auto crossing = (matched / expected) * std::fma(a, x - 1, -mean) / (1 + ratio);
  const auto moment = beta_tail(matched, b, ratio, crossing, lower);
  return saddle_weight == 0 ? moment : std::lerp(moment, f_saddle(x, 2 * a, 2 * b, 2 * mean, lower), saddle_weight);
}

/// @brief Evaluates a checked noncentral chi-square tail.
/// @param x Evaluation point.
/// @param df Valid degrees of freedom.
/// @param noncentrality Valid noncentrality.
/// @param lower Whether to return the lower tail.
/// @return Selected tail.
long double chi_tail(double x, double df, double noncentrality, bool lower) {
  if (x < 0 || (x == 0 && df != 0))
    return lower ? 0 : 1;
  if (std::isinf(x))
    return lower ? 1 : 0;
  const long double mean = noncentrality / 2.L;
  if (x == 0)
    return lower ? std::exp(-mean) : -std::expm1(-mean);
  if (noncentrality > mixture_limit)
    return chi_square_saddle(x, df, noncentrality, lower);
  const long double shape = df / 2.L, scaled = x / 2.L;
  const auto exact = poisson_mixture(mean, lower, [=](std::size_t k) -> long double {
    if (shape == 0 && k == 0)
      return lower ? 1 : 0;
    const auto a = std::max(static_cast<double>(shape + k), std::numeric_limits<double>::denorm_min());
    return detail::gamma_probability(scaled, a, lower);
  });
  const auto weight = std::clamp((noncentrality / mixture_limit - .9L) * 10, 0.L, 1.L);
  return weight == 0 ? exact : std::lerp(exact, chi_square_saddle(x, df, noncentrality, lower), weight);
}

/// @brief Evaluates a checked noncentral F tail.
/// @param x Evaluation point.
/// @param first Numerator degrees of freedom.
/// @param second Denominator degrees of freedom.
/// @param noncentrality Numerator noncentrality.
/// @param lower Whether to return the lower tail.
/// @return Selected tail.
long double f_tail(double x, double first, double second, double noncentrality, bool lower) {
  if (x <= 0 || std::isinf(x))
    return (x > 0) == lower ? 1 : 0;
  const long double a = first / 2.L, b = second / 2.L, mean = noncentrality / 2.L;
  if (noncentrality > mixture_limit)
    return large_f_tail(x, a, b, mean, lower);
  const auto ratio = (a / b) * x;
  const auto exact = poisson_mixture(mean, lower, [=](std::size_t k) {
    const auto crossing = std::fma(a, static_cast<long double>(x) - 1, -static_cast<long double>(k)) / (1 + ratio);
    return beta_tail(a + k, b, ratio, crossing, lower);
  });
  const auto weight = std::clamp((noncentrality / mixture_limit - .9L) * 10, 0.L, 1.L);
  return weight == 0 ? exact : std::lerp(exact, large_f_tail(x, a, b, mean, lower), weight);
}

/// @brief Converts a log-coordinate back to a finite positive double without overflowing.
/// @param log_value Logarithm in the finite positive double range.
/// @return Positive double coordinate.
double positive_at(long double log_value) {
  return static_cast<double>(std::min(std::exp(log_value), static_cast<long double>(std::numeric_limits<double>::max())));
}

/// @brief Brackets from one and bisects a positive parameter in log coordinates.
/// @tparam Reaches Callable with a false-to-true transition.
/// @param reaches Function indicating whether the target has been reached.
/// @return Positive parameter, or zero or infinity if outside the representable range.
template <class Reaches>
double invert_positive(const Reaches& reaches) {
  constexpr auto minimum = std::numeric_limits<double>::denorm_min();
  constexpr auto maximum = std::numeric_limits<double>::max();
  const auto log_min = std::log(static_cast<long double>(minimum));
  const auto log_max = std::log(static_cast<long double>(maximum));
  long double lo = 0, hi = 0;
  if (reaches(1)) {
    lo = -1;
    while (reaches(positive_at(lo))) {
      if (lo == log_min)
        return 0;
      hi = lo;
      lo = std::max(log_min, 2 * lo);
    }
  } else {
    hi = 1;
    while (!reaches(positive_at(hi))) {
      if (hi == log_max)
        return std::numeric_limits<double>::infinity();
      lo = hi;
      hi = std::min(log_max, 2 * hi);
    }
  }
  for (int i = 0; i < 70; ++i) {
    const auto mid = std::midpoint(lo, hi);
    if (mid == lo || mid == hi)
      break;
    (reaches(positive_at(mid)) ? hi : lo) = mid;
  }
  return positive_at(hi);
}

/// @brief Inverts the selected tail for the evaluation point using the smaller tail.
/// @tparam Tail Callable computing a tail at a cutoff.
/// @param p Target probability.
/// @param lower Whether the input probability represents the lower tail.
/// @param tail Evaluation function taking a cutoff and a tail indicator.
/// @return Quantile on the nonnegative real line.
template <class Tail>
double quantile(double p, bool lower, const Tail& tail) {
  detail::require_probability(p, "noncentral quantile");
  if (p == 0 || p == 1)
    return (p == 1) == lower ? std::numeric_limits<double>::infinity() : 0;
  const auto endpoint = static_cast<double>(tail(0, lower));
  if (lower ? p <= endpoint : p >= endpoint)
    return 0;
  if (p > .5) {
    p = 1 - p;
    lower = !lower;
  }
  /// @brief Checks attainment without subtracting the selected tail from one.
  const auto reached = [&](double x) { return lower ? tail(x, lower) >= p : tail(x, lower) <= p; };
  if (reached(0))
    return 0;
  const auto smallest = std::numeric_limits<double>::denorm_min();
  if (static_cast<double>(tail(smallest, lower)) == p)
    return smallest;
  return invert_positive(reached);
}

/// @brief Inverts the selected tail for the noncentrality using its monotonicity.
/// @tparam Tail Callable computing a tail at a noncentrality.
/// @param p Target probability.
/// @param lower Whether the probability represents the lower tail.
/// @param tail Evaluation function taking a noncentrality and a tail indicator.
/// @return Noncentrality consistent with the target.
/// @throws std::invalid_argument Probability outside the attainable range.
template <class Tail>
double invert_noncentrality(double p, bool lower, const Tail& tail) {
  detail::require_probability(p, "noncentrality");
  if ((lower && p == 0) || (!lower && p == 1))
    return std::numeric_limits<double>::infinity();
  if (static_cast<double>(tail(0, lower)) == p)
    return 0;
  if (p > .5) {
    p = 1 - p;
    lower = !lower;
  }
  const auto central = static_cast<double>(tail(0, lower));
  if (p == central)
    return 0;
  if ((lower && p > central) || (!lower && p < central))
    throw std::invalid_argument("noncentrality: requested probability is outside the attainable range");
  return invert_positive([&](double ncp) { return lower ? tail(ncp, lower) <= p : tail(ncp, lower) >= p; });
}

/// @brief Evaluates a checked noncentral t tail using reflection and direct integration.
/// @param x Cutoff, possibly infinite.
/// @param df Positive finite degrees of freedom.
/// @param noncentrality Signed normal displacement.
/// @param lower Whether to return the lower tail.
/// @return Requested probability.
long double t_tail(double x, double df, double noncentrality, bool lower) {
  if (std::isinf(x))
    return (x > 0) == lower ? 1 : 0;
  if (x == 0)
    return std::exp(detail::noncentral_t_log_normal(lower ? -noncentrality : noncentrality));
  if (x < 0)
    return t_tail(-x, df, -noncentrality, !lower);
  const long double d = df, t = x;
  if (noncentrality == 0) {
    const auto ratio = (d / t) / t;
    const auto crossing = (ratio - d) / (2 * (1 + ratio));
    const auto tail = beta_tail(d / 2, .5L, ratio, crossing, true) / 2;
    return lower ? 1 - tail : tail;
  }
  return detail::noncentral_t_integrated_tail(t, d, noncentrality, lower);
}

/// @brief Inverts a signed parameter using the smaller probability tail.
/// @tparam Tail Callable computing a tail from the signed parameter.
/// @param p Input probability.
/// @param lower Whether the input represents the lower tail.
/// @param increasing_lower Whether the lower tail increases with the parameter.
/// @param tail Evaluation function taking a signed parameter and a tail indicator.
/// @return Signed parameter at the requested probability.
template <class Tail>
double invert_signed_tail(double p, bool lower, bool increasing_lower, const Tail& tail) {
  detail::require_probability(p, "noncentral t inverse probability");
  if (p == 0 || p == 1)
    return std::copysign(std::numeric_limits<double>::infinity(), ((p == 1) == (lower == increasing_lower)) ? 1 : -1);
  if (static_cast<double>(tail(0, lower)) == p)
    return 0;
  if (p > .5) {
    p = 1 - p;
    lower = !lower;
  }
  const auto at_zero = static_cast<double>(tail(0, lower));
  if (p == at_zero)
    return 0;
  const bool above = p > at_zero;
  const double sign = above == (increasing_lower == lower) ? 1 : -1;
  return sign * invert_positive([&](double magnitude) {
    const auto value = tail(sign * magnitude, lower);
    return above ? value >= p : value <= p;
  });
}

/// @brief Checks the noncentral t degrees of freedom and numerator displacement.
/// @param df Positive finite degrees of freedom.
/// @param noncentrality Finite signed displacement.
/// @throws std::invalid_argument Invalid parameters.
void check_t(double df, double noncentrality) {
  if (!(df > 0) || !std::isfinite(df) || !std::isfinite(noncentrality))
    throw std::invalid_argument("noncentral t: degrees of freedom must be positive and parameters finite");
}

} // namespace

double noncentral_t_cdf(double x, double df, double noncentrality, bool lower_tail) {
  check_t(df, noncentrality);
  if (std::isnan(x))
    throw std::invalid_argument("noncentral t: evaluation point is NaN");
  return static_cast<double>(t_tail(x, df, noncentrality, lower_tail));
}

double noncentral_t_quantile(double probability, double df, double noncentrality, bool lower_tail) {
  check_t(df, noncentrality);
  return invert_signed_tail(probability, lower_tail, true, [=](double x, bool lower) {
    return t_tail(x, df, noncentrality, lower);
  });
}

double noncentral_t_noncentrality(double probability, double x, double df, bool lower_tail) {
  check_t(df, 0);
  if (!std::isfinite(x))
    throw std::invalid_argument("noncentral t: noncentrality inversion requires a finite cutoff");
  return invert_signed_tail(probability, lower_tail, false, [=](double noncentrality, bool lower) {
    return t_tail(x, df, noncentrality, lower);
  });
}

double noncentral_chi_square_cdf(double x, double df, double noncentrality, bool lower_tail) {
  check_chi_square(df, noncentrality);
  if (std::isnan(x))
    throw std::invalid_argument("noncentral chi-square: evaluation point is NaN");
  return static_cast<double>(chi_tail(x, df, noncentrality, lower_tail));
}

double noncentral_chi_square_quantile(double probability, double df, double noncentrality, bool lower_tail) {
  check_chi_square(df, noncentrality);
  if (df == 0 && noncentrality == 0) {
    detail::require_probability(probability, "noncentral quantile");
    return 0;
  }
  return quantile(probability, lower_tail, [=](double x, bool lower) { return chi_tail(x, df, noncentrality, lower); });
}

double noncentral_chi_square_noncentrality(double probability, double x, double df, bool lower_tail) {
  check_chi_square(df, 0);
  if (!std::isfinite(x) || !(x > 0 || (x == 0 && df == 0)))
    throw std::invalid_argument("noncentral chi-square: require a positive finite cutoff, or zero with zero degrees of freedom");
  return invert_noncentrality(probability, lower_tail, [=](double ncp, bool lower) { return chi_tail(x, df, ncp, lower); });
}

double noncentral_f_cdf(double x, double numerator_df, double denominator_df, double noncentrality, bool lower_tail) {
  check_f(numerator_df, denominator_df, noncentrality);
  if (std::isnan(x))
    throw std::invalid_argument("noncentral F: evaluation point is NaN");
  return static_cast<double>(f_tail(x, numerator_df, denominator_df, noncentrality, lower_tail));
}

double noncentral_f_quantile(double probability, double numerator_df, double denominator_df,
                             double noncentrality, bool lower_tail) {
  check_f(numerator_df, denominator_df, noncentrality);
  return quantile(probability, lower_tail, [=](double x, bool lower) {
    return f_tail(x, numerator_df, denominator_df, noncentrality, lower);
  });
}

double noncentral_f_noncentrality(double probability, double x, double numerator_df, double denominator_df,
                                  bool lower_tail) {
  check_f(numerator_df, denominator_df, 0);
  if (!(x > 0) || !std::isfinite(x))
    throw std::invalid_argument("noncentral F: require a positive finite cutoff");
  return invert_noncentrality(probability, lower_tail, [=](double ncp, bool lower) {
    return f_tail(x, numerator_df, denominator_df, ncp, lower);
  });
}

} // namespace datamunge::random
