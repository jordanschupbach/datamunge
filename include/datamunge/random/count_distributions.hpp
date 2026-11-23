#pragma once

/// @file count_distributions.hpp
/// @brief Negative binomial and beta-binomial probability laws for overdispersed counts.

#include <cstddef>

namespace datamunge::random {

/// @brief Evaluates the mass of a negative binomial count with variance mean + mean squared / precision.
/// @param k Event count.
/// @param mean Finite nonnegative count mean.
/// @param precision Positive finite inverse overdispersion.
/// @return Probability of k events. A zero mean gives a point mass at zero.
/// @throws std::invalid_argument Invalid mean or precision.
/// @par Example
/// @code
/// std::print("{:.3f}\n", dm::random::negative_binomial_pmf(2, 1, 1)); // 0.125
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/NegBinomial.html
/// @see negative_binomial_cdf()
double negative_binomial_pmf(std::size_t k, double mean, double precision);

/// @brief Evaluates a negative binomial tail by the incomplete-beta identity.
///
/// Uses the Poisson limit when the relative variance correction is below
/// floating-point precision. Uses a saddlepoint approximation when both
/// incomplete-beta shapes are at least 1e11.
/// @param k Event count.
/// @param mean Finite nonnegative count mean.
/// @param precision Positive finite inverse overdispersion.
/// @param lower_tail True for P(X <= k), false for P(X > k), default true.
/// @return Requested probability.
/// @throws std::invalid_argument Invalid mean or precision.
/// @par Example
/// @code
/// std::print("{:.3f}\n", dm::random::negative_binomial_cdf(2, 1, 1, false)); // 0.125
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/NegBinomial.html
/// - https://doi.org/10.1017/S0001867800050278
/// @see negative_binomial_quantile()
double negative_binomial_cdf(std::size_t k, double mean, double precision, bool lower_tail = true);

/// @brief Finds the smallest negative binomial count whose CDF reaches a probability.
/// @param probability Cumulative probability in [0,1].
/// @param mean Finite nonnegative count mean.
/// @param precision Positive finite inverse overdispersion.
/// @return Integer quantile. Returns maximum size_t for an unbounded or out-of-range quantile.
/// @throws std::invalid_argument Invalid probability, mean or precision.
/// @par Example
/// @code
/// std::print("{}\n", dm::random::negative_binomial_quantile(.75, 1, 1)); // 1
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/NegBinomial.html
/// @see negative_binomial_cdf()
std::size_t negative_binomial_quantile(double probability, double mean, double precision);

/// @brief Evaluates binomial count mass with a beta-distributed success probability.
/// @param k Success count.
/// @param trials Number of trials.
/// @param alpha Positive finite first beta shape.
/// @param beta Positive finite second beta shape.
/// @return Probability of k successes, zero outside [0,trials].
/// @throws std::invalid_argument Invalid shape.
/// @par Example
/// @code
/// std::print("{:.2f}\n", dm::random::beta_binomial_pmf(2, 3, 1, 1)); // 0.25
/// @endcode
/// @par References
/// - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.betabinom.html
/// @see beta_binomial_cdf()
double beta_binomial_pmf(std::size_t k, std::size_t trials, double alpha, double beta);

/// @brief Computes a beta-binomial tail by summing log-recursive probabilities.
/// @param k Success count.
/// @param trials Number of trials.
/// @param alpha Positive finite first beta shape.
/// @param beta Positive finite second beta shape.
/// @param lower_tail True for P(X <= k), false for P(X > k), default true.
/// @return Requested probability.
/// @throws std::invalid_argument Invalid shape.
/// @par Example
/// @code
/// std::print("{:.2f}\n", dm::random::beta_binomial_cdf(2, 3, 1, 1, false)); // 0.25
/// @endcode
/// @par References
/// - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.betabinom.html
/// @see beta_binomial_quantile()
double beta_binomial_cdf(std::size_t k, std::size_t trials, double alpha, double beta,
                         bool lower_tail = true);

/// @brief Finds the smallest beta-binomial count whose CDF reaches a probability.
/// @param probability Cumulative probability in [0,1].
/// @param trials Number of trials.
/// @param alpha Positive finite first beta shape.
/// @param beta Positive finite second beta shape.
/// @return Integer quantile in [0,trials], including the support endpoints for zero and one.
/// @throws std::invalid_argument Invalid probability or shape.
/// @par Example
/// @code
/// std::print("{}\n", dm::random::beta_binomial_quantile(.5, 3, 1, 1)); // 1
/// @endcode
/// @par References
/// - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.betabinom.html
/// @see beta_binomial_cdf()
std::size_t beta_binomial_quantile(double probability, std::size_t trials, double alpha, double beta);

} // namespace datamunge::random
