#pragma once
/// @file energy_test.hpp
/// @brief Two-sample energy inference for multivariate observations.

#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/random/random.hpp>
#include <datamunge/stats/hypothesis_test_result.hpp>

namespace datamunge::stats {

/// @brief Compares two multivariate distributions by a simulated pooled-label energy test.
///
/// Uses Euclidean distances and the empirical energy distance with the
/// original sample sizes n and m. Each draw assigns the pooled rows uniformly
/// to groups of the same sizes. The upper tail includes ties, allowing for
/// floating-point roundoff. Its p-value is (1 + tail count) / (1 + draws).
/// The pooled distance cache takes quadratic storage in the total sample size.
/// @param first Nonempty observation-by-feature matrix of finite coordinates.
/// @param second Second nonempty matrix with the same positive number of features.
/// @param draws Positive number of independent assignments to sample.
/// @param rng Caller-owned generator, advanced by the permutations.
/// @return Statistic nm/(n+m) times empirical energy distance, estimate1 the unscaled distance,
/// parameter1 assignment count, parameter2 upper-tail count, and p-value.
/// @throws std::invalid_argument Empty samples, differing dimensions, nonfinite coordinates, or zero draws.
/// @throws std::length_error The pooled distance cache exceeds its container's capacity.
/// @throws std::overflow_error A distance or the observed result exceeds the representable range.
/// @par Example
/// @code
/// dm::random::SplitMix64 rng(42);
/// const dm::linalg::DenseMatrix<double> x(1, 1, {0}), y(1, 1, {3});
/// std::print("{}\n", dm::stats::energy_distance_test(x, y, 99, rng).statistic); // 3
/// @endcode
/// @par References
/// - G. J. Székely and M. L. Rizzo (2013), https://doi.org/10.1016/j.jspi.2013.03.018
/// @see energy_distance_test_exact()
HypothesisTestResult energy_distance_test(const linalg::DenseMatrix<double>& first,
                                           const linalg::DenseMatrix<double>& second,
                                           std::size_t draws, random::RandomGenerator& rng);

/// @brief Enumerates the exact pooled-label reference for the two-sample energy test.
///
/// Enumerates each subset of the first-group size once, including the
/// observed assignment. Distinct rows with identical coordinates remain
/// distinct observations. The p-value is the fraction at least as extreme
/// as observed, with roundoff-aware ties included.
/// @param first Nonempty observation-by-feature matrix of finite coordinates.
/// @param second Second nonempty matrix with the same positive number of features.
/// @param max_assignments Positive cap on exact assignments, default 100000.
/// @return Statistic, estimates and assignment counts as in energy_distance_test(), and the exact p-value.
/// @throws std::invalid_argument Empty samples, differing dimensions, nonfinite coordinates, or zero cap.
/// @throws std::length_error The assignment cap or the distance-cache container's capacity is exceeded.
/// @throws std::overflow_error A distance or the observed result exceeds the representable range.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(2, 1, {0, 1}), y(2, 1, {4, 5});
/// const auto fit = dm::stats::energy_distance_test_exact(x, y);
/// std::print("{:.3f}\n", fit.p_value); // 0.333
/// @endcode
/// @par References
/// - G. J. Székely and M. L. Rizzo (2013), https://doi.org/10.1016/j.jspi.2013.03.018
/// @see energy_distance_test()
HypothesisTestResult energy_distance_test_exact(const linalg::DenseMatrix<double>& first,
                                                 const linalg::DenseMatrix<double>& second,
                                                 std::size_t max_assignments = 100'000);

} // namespace datamunge::stats
