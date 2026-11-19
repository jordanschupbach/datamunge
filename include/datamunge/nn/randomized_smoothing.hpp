/// @file
/// @brief Pointwise L2 certificates from independent Gaussian classifier votes.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>

#include "datamunge/random/distributions.hpp"

namespace datamunge::nn {

/// @brief Pointwise certificate for a Gaussian-smoothed classifier.
///
/// A zero radius indicates abstention. The candidate is retained for diagnostics.
///
/// @par Example
/// @code
/// const auto result = dm::nn::gaussian_smoothing_certificate(
///     std::array<std::size_t, 2>{8, 2}, std::array<std::size_t, 2>{100, 0}, 0.16);
/// std::print("{}\n", result.radius > 0.0); // true
/// @endcode
/// @see gaussian_smoothing_certificate()
struct GaussianSmoothingCertificate {
  std::size_t candidate; ///< Zero-based class selected by the first sample.
  double probability_lower; ///< One-sided Clopper-Pearson lower confidence bound.
  double radius; ///< Certified open-ball L2 radius in input units, or zero on abstention.
};

/// @brief Certify a class and an L2 radius from two independent Gaussian vote histograms.
///
/// Each histogram uses the same class order. Votes must come from repeated predictions
/// on the same input with independent, unclipped isotropic Gaussian noise of standard
/// deviation sigma. The first histogram selects the plurality class, breaking ties
/// by smallest index. The second estimates its probability. Coverage is pointwise.
/// @param selection Class vote counts from the selection draws, with at least one vote.
/// @param estimation Class vote counts from the independent estimation draws.
/// @param sigma Positive, finite standard deviation of noise per coordinate.
/// @param alpha Pointwise failure probability in (0,1), default 0.001.
/// @return Candidate label, lower probability bound, and sigma times the standard
/// normal quantile of the bound when it exceeds 0.5, otherwise radius zero.
/// @throws std::invalid_argument Invalid histogram sizes, empty draws, overflow of
/// estimation count, invalid sigma, or alpha outside (0,1).
///
/// @par Example
/// @code
/// const auto result = dm::nn::gaussian_smoothing_certificate(
///     std::array<std::size_t, 2>{5, 5}, std::array<std::size_t, 2>{50, 50}, 0.16);
/// std::print("{} {:.1f}\n", result.candidate, result.radius); // 0 0.0
/// @endcode
/// @par References
/// - Cohen, Rosenfeld and Kolter (2019), Certified Adversarial Robustness via Randomized Smoothing.
///   https://proceedings.mlr.press/v97/cohen19c.html
/// @see GaussianSmoothingCertificate, datamunge::random::beta_quantile()
inline GaussianSmoothingCertificate gaussian_smoothing_certificate(
    std::span<const std::size_t> selection, std::span<const std::size_t> estimation,
    double sigma, double alpha = 0.001) {
  if (selection.size() < 2 || selection.size() != estimation.size() ||
      !(sigma > 0.0 && std::isfinite(sigma)) ||
      !(alpha > 0.0 && alpha < 1.0)) {
    throw std::invalid_argument("Gaussian smoothing requires matching class counts, positive sigma and valid confidence");
  }
  const auto winner = std::max_element(selection.begin(), selection.end());
  std::size_t draws = 0;
  for (const auto count : estimation) {
    if (count > std::numeric_limits<std::size_t>::max() - draws) {
      throw std::invalid_argument("Gaussian smoothing estimation count overflow");
    }
    draws += count;
  }
  if (*winner == 0 || draws == 0) {
    throw std::invalid_argument("Gaussian smoothing requires votes in both samples");
  }
  const auto candidate = static_cast<std::size_t>(winner - selection.begin());
  const auto successes = estimation[candidate];
  const double lower = successes == 0 ? 0.0 : random::beta_quantile(
      alpha, static_cast<double>(successes), static_cast<double>(draws - successes) + 1.0);
  if (lower <= 0.5) {
    return {candidate, lower, 0.0};
  }
  double left = 0.0, right = 10.0;
  // The upper tail avoids subtracting a CDF close to one from one.
  for (int i = 0; i < 52; ++i) {
    const double midpoint = (left + right) / 2.0;
    if (random::normal_cdf(-midpoint) >= 1.0 - lower) {
      left = midpoint;
    } else {
      right = midpoint;
    }
  }
  return {candidate, lower, sigma * left};
}

} // namespace datamunge::nn
