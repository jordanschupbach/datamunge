#pragma once
/// @file freedman_lane.hpp
/// @brief Residual permutation test for terms in a nested linear model.

#include <cstddef>
#include <vector>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/random/random.hpp>
#include <datamunge/stats/hypothesis_test_result.hpp>

namespace datamunge::stats {

/// @brief Tests added linear-model terms by the Freedman–Lane residual permutation.
///
/// The full design appends the tested columns to the nuisance columns.
/// The algorithm permutes residuals from the nuisance fit, retaining its
/// fitted component, and recomputes the partial F statistic. Fixed
/// design bases are reused for each fit. The upper tail includes ties and
/// uses the plus-one Monte Carlo correction. Use with exchangeable errors.
/// @param response Finite observations, one per design row.
/// @param nuisance Reduced-model design with any intercept supplied explicitly. May have zero columns.
/// @param tested One or more columns to test jointly after adjusting for nuisance.
/// @param draws Positive Monte Carlo permutation count.
/// @param rng Caller-owned generator used by the uniform permutations.
/// @return Partial F, numerator and denominator degrees of freedom in parameter1
/// and parameter2, and the permutation p-value.
/// @throws std::invalid_argument Invalid inputs, rank-deficient full design or no reduced-model residual variation.
/// @throws std::runtime_error Failure of the design factorization.
/// @par Example
/// @code
/// dm::random::SplitMix64 rng(42);
/// const dm::linalg::DenseMatrix<double> intercept(5, 1, 1.0);
/// const dm::linalg::DenseMatrix<double> trend(5, 1, {0, 1, 2, 3, 4});
/// const auto test = dm::stats::freedman_lane_test({1, 1, 3, 4, 5}, intercept, trend, 99, rng);
/// std::print("{}\n", test.parameter2); // 3
/// @endcode
/// @par References
/// - D. A. Freedman and D. Lane (1983), https://doi.org/10.1080/07350015.1983.10509354
/// - A. M. Winkler et al. (2014), https://pmc.ncbi.nlm.nih.gov/articles/PMC4010955/
/// @see parametric_bootstrap_test()
HypothesisTestResult freedman_lane_test(const std::vector<double>& response,
                                      const linalg::DenseMatrix<double>& nuisance,
                                      const linalg::DenseMatrix<double>& tested,
                                      std::size_t draws, random::RandomGenerator& rng);

} // namespace datamunge::stats
