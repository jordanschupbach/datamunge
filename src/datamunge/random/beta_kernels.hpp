#pragma once
/// @file beta_kernels.hpp
/// @brief Internal stable beta normalizers, deviances and large-shape tails.

#include <datamunge/random/distributions.hpp>
#include <cmath>

namespace datamunge::random::detail {

/// @brief Stirling log-gamma remainder at a large positive argument.
/// @param x Argument at least 16.
/// @return Correction to the leading Stirling expression.
inline long double stirling_remainder(long double x) {
  const auto u = 1 / x;
  const auto square = u * u;
  return u * (1.0L / 12 + square * (-1.0L / 360 + square *
      (1.0L / 1260 + square * (-1.0L / 1680 + square / 1188))));
}

/// @brief Computes a positive log-gamma increment without subtracting two large log-gamma values.
/// @param x Positive starting argument.
/// @param delta Nonnegative increment.
/// @return log Gamma(x+delta) minus log Gamma(x).
inline long double gamma_increment(long double x, long double delta) {
  if (x < 16)
    return std::lgamma(x + delta) - std::lgamma(x);
  const auto log_ratio = std::log1p(delta / x);
  auto correction = stirling_remainder(x + delta) - stirling_remainder(x);
  if (delta < .001L * x) {
    const auto u = 1 / x, u2 = u * u;
    correction = u * (std::expm1(-log_ratio) / 12 + u2 *
        (-std::expm1(-3 * log_ratio) / 360 + u2 * (std::expm1(-5 * log_ratio) / 1260 + u2 *
        (-std::expm1(-7 * log_ratio) / 1680 + u2 * std::expm1(-9 * log_ratio) / 1188))));
  }
  return delta * std::log(x) + (x + delta - .5L) * log_ratio - delta + correction;
}

/// @brief Evaluates log beta using a widened, cancellation-resistant gamma ratio.
/// @param a Positive first shape.
/// @param b Positive second shape.
/// @return Natural logarithm of B(a,b).
inline long double count_log_beta(long double a, long double b) {
  return std::lgamma(std::min(a, b)) - gamma_increment(std::max(a, b), std::min(a, b));
}

/// @brief Computes the beta deviance without subtracting large log terms near the mean.
/// @param a First shape.
/// @param b Second shape.
/// @param odds Failure odds.
/// @param crossing Difference between (a+b)/(1+odds) and a.
/// @return Decrease in log beta kernel from its value at a/(a+b).
inline long double beta_deviance(long double a, long double b, long double odds, long double crossing) {
  const auto left = crossing / a;
  const auto right = -crossing / b;
  // Evaluate the log ratios directly near zero probability, where 1+left or
  // 1+right can round to zero despite a finite log ratio.
  const auto left_log = std::log1p(b / a) - std::log1p(odds);
  const auto right_log = std::log1p(a / b) - std::log1p(1 / odds);
  return a * (left < -.5L ? left - left_log : gamma_divergence(left)) +
         b * (right < -.5L ? right - right_log : gamma_divergence(right));
}

/// @brief Evaluates the logarithm of the beta kernel divided by B(a,b).
/// @param a First shape, at least 16.
/// @param b Second shape, at least 16.
/// @param deviance Decrease in log beta kernel from its value at a/(a+b).
/// @return Logarithm of the normalized kernel.
inline long double beta_kernel_log(long double a, long double b, long double deviance) {
  return .5L * std::log((a / (a + b)) * b / (2 * pi)) - deviance +
         stirling_remainder(a + b) - stirling_remainder(a) - stirling_remainder(b);
}

/// @brief Evaluates the leading correction to the quadratic beta deviance.
/// @param d Relative displacement close to zero.
/// @return Twice (d - log(1+d)) divided by d squared, minus one.
inline long double relative_deviance_correction(long double d) {
  if (std::abs(d) < .001L)
    return d * (-2.0L / 3 + d * (.5L + d * (-.4L + d * (1.0L / 3 - d * (2.0L / 7)))));
  return 2 * gamma_divergence(d) / (d * d) - 1;
}

/// @brief Applies the Lugannani–Rice approximation to the difference of two scaled gamma variables.
/// @param a Large first beta shape.
/// @param b Large second beta shape.
/// @param crossing Difference between (a+b) times the evaluation point and a.
/// @param deviance Decrease in log beta kernel from its value at a/(a+b).
/// @param lower_tail Whether to return the lower tail.
/// @return Approximate beta tail probability.
inline long double large_beta_tail(long double a, long double b, long double crossing,
                            long double deviance, bool lower_tail) {
  if (deviance > 750)
    return static_cast<long double>((crossing > 0) == lower_tail);

  const auto inverse_scale = std::sqrt(1 / a + 1 / b);
  const auto u = crossing * inverse_scale;
  const auto w = std::copysign(std::sqrt(2 * deviance), crossing);
  auto correction = (1 / a - 1 / b) / (3 * inverse_scale);
  if (u != 0) {
    // Write 1/w - 1/u using w/u to retain precision near the mean.
    const auto delta = (b / (a + b)) * relative_deviance_correction(crossing / a) +
                       (a / (a + b)) * relative_deviance_correction(-crossing / b);
    const auto w_over_u = std::sqrt(1 + delta);
    correction = -delta / (u * w_over_u * (1 + w_over_u));
  }
  const auto normal_correction = std::exp(-deviance) * correction / std::sqrt(2 * pi);
  const auto sign = lower_tail ? 1 : -1;
  return std::clamp(.5L * std::erfc(-sign * w / std::sqrt(2.0L)) + sign * normal_correction,
                    0.0L, 1.0L);
}

} // namespace datamunge::random::detail
