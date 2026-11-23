#pragma once

/// @file count_likelihood.hpp
/// @brief Internal log masses shared by count laws and likelihood fits.

#include "beta_kernels.hpp"

#include <cmath>
#include <cstddef>
#include <numbers>

namespace datamunge::random::detail {

/// @brief Computes log mass using already validated negative binomial parameters.
/// @param k Event count.
/// @param mean Positive finite count mean.
/// @param a Positive finite precision.
/// @param odds Failure odds.
/// @return Natural logarithm of the probability mass.
inline long double negative_binomial_log_mass(std::size_t k, long double mean, long double a, long double odds) {
  const long double x = k;
  if (a >= 16 && x >= 16) {
    const auto deviance = beta_deviance(a, x, odds, (x - mean) / (1 + odds));
    return beta_kernel_log(a, x, deviance) - std::log(x);
  }
  if (k < 16) {
    long double result = -a * std::log1p(odds);
    for (std::size_t j = 0; j < k; ++j)
      result += std::log(mean) - std::log1p(odds) + std::log1p(j / a) -
                std::log(static_cast<long double>(j) + 1);
    return result;
  }
  return gamma_increment(x, a) - std::lgamma(a) - std::log(x) -
         a * std::log1p(odds) - x * std::log1p(1 / odds);
}

/// @brief Stable log probability for the Poisson limit.
inline long double poisson_log_mass(long double count, long double eta, long double mean) {
    if (count == 0)
        return -mean;
    if (count < 16)
        return count * eta - mean - std::lgamma(count + 1);
    const auto change = (mean - count) / count;
    const auto loss = change < -.5L ? mean - count + count * (std::log(count) - eta)
                                   : count * gamma_divergence(change);
    return -loss - .5L * std::log(2 * std::numbers::pi_v<long double> * count) -
           stirling_remainder(count);
}

} // namespace datamunge::random::detail
