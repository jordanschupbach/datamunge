#pragma once
/// @file simplex.hpp
/// @brief Dirichlet and Dirichlet–multinomial probabilities and sampling.

#include <datamunge/random/random.hpp>
#include <cstddef>
#include <vector>

namespace datamunge::random {

/// @brief Evaluates the normalized Dirichlet log density on the open simplex.
///
/// Coordinates must be positive and sum to one within 64 double epsilons.
/// Points within that tolerance are normalized before evaluation. Zero
/// coordinates and finite points outside the simplex have log density
/// negative infinity. A single category has log density zero at one.
/// @param point Simplex coordinates in category order.
/// @param concentration Positive finite concentration for each category.
/// @return Log density with respect to the first K-1 coordinates.
/// @throws std::invalid_argument Invalid concentrations, mismatched dimensions, or nonfinite coordinates.
/// @par Example
/// @code
/// std::print("{:.1f}\n", dm::random::dirichlet_log_pdf({.5, .5}, {1, 1})); // 0.0
/// @endcode
/// @par References
/// - https://mc-stan.org/docs/functions-reference/simplex_distributions.html
/// @see dirichlet_sample(), dirichlet_multinomial_log_pmf()
double dirichlet_log_pdf(const std::vector<double>& point, const std::vector<double>& concentration);

/// @brief Evaluates normalized Dirichlet–multinomial log probability for a count vector.
///
/// The fixed number of trials is the sum of the supplied category counts.
/// The zero vector has log probability zero.
/// @param counts Nonnegative counts in category order.
/// @param concentration Positive finite concentration for each category.
/// @return Natural logarithm of the compound count probability.
/// @throws std::invalid_argument Empty or invalid concentrations or mismatched dimensions.
/// @throws std::overflow_error The total count exceeds the range of std::size_t.
/// @par Example
/// @code
/// const auto logp = dm::random::dirichlet_multinomial_log_pmf({2, 1}, {1, 1});
/// std::print("{:.2f}\n", std::exp(logp)); // 0.25
/// @endcode
/// @par References
/// - https://mc-stan.org/docs/functions-reference/multivariate_discrete_distributions.html#dirichlet-multinomial-distribution
/// @see dirichlet_log_pdf(), dirichlet_multinomial_sample()
double dirichlet_multinomial_log_pmf(const std::vector<std::size_t>& counts,
                                    const std::vector<double>& concentration);

/// @brief Draws a Dirichlet vector using independent gamma variates and normalization.
///
/// Uses the supplied engine and its normal cache. Equal initial engine states
/// reproduce results in the same environment. Very small coordinates can
/// round to zero in the returned double vector.
/// @param rng Caller-owned random generator, advanced by sampling.
/// @param concentration Positive finite concentration for each category.
/// @return One simplex point in category order. A single category returns {1}.
/// @throws std::invalid_argument Empty, nonpositive, or nonfinite concentrations.
/// @par Example
/// @code
/// dm::random::SplitMix64 rng(7);
/// std::print("{}\n", dm::random::dirichlet_sample(rng, {2}).front()); // 1
/// @endcode
/// @par References
/// - G. Marsaglia and W. W. Tsang (2000), https://doi.org/10.1145/358407.358414
/// @see dirichlet_log_pdf(), dirichlet_multinomial_sample()
std::vector<double> dirichlet_sample(RandomGenerator& rng, const std::vector<double>& concentration);

/// @brief Draws counts from a multinomial with a fresh Dirichlet probability vector.
///
/// Conditional binomials allocate the remaining trials to successive
/// categories. Equal initial engine states reproduce results under
/// the same libm and standard-library implementation.
/// @param rng Caller-owned random generator, advanced by sampling.
/// @param trials Total observations to allocate, including zero.
/// @param concentration Positive finite concentration for each category.
/// @return Counts in category order, summing to trials.
/// @throws std::invalid_argument Empty, nonpositive, or nonfinite concentrations.
/// @par Example
/// @code
/// dm::random::SplitMix64 rng(7);
/// std::print("{}\n", dm::random::dirichlet_multinomial_sample(rng, 12, {2}).front()); // 12
/// @endcode
/// @par References
/// - https://mc-stan.org/docs/functions-reference/multivariate_discrete_distributions.html#dirichlet-multinomial-distribution
/// @see dirichlet_sample(), dirichlet_multinomial_log_pmf()
std::vector<std::size_t> dirichlet_multinomial_sample(RandomGenerator& rng, std::size_t trials,
                                                       const std::vector<double>& concentration);

} // namespace datamunge::random
