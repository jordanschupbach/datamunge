#pragma once

/// @file canonical_correlation.hpp
/// @brief Paired canonical variates, loadings and sequential Wilks tests.

#include <datamunge/linalg/dense_matrix.hpp>

#include <vector>

namespace datamunge::stats {

/// @brief Bartlett chi-square approximation for a tail of the canonical correlations.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(5,1,{1,2,3,4,6}), y(5,1,{2,1,4,3,7});
/// const auto fit = dm::stats::canonical_correlation(x,y);
/// std::println("{}", fit.sequential_tests.front().df); // 1
/// @endcode
struct CanonicalCorrelationTest {
    std::size_t first_component; ///< First remaining canonical pair, numbered from one.
    double wilks_lambda;          ///< Product of one minus squared correlations in the remaining pairs.
    double chi_squared;           ///< Bartlett's asymptotic chi-square statistic.
    std::size_t df;               ///< Product of the remaining dimensions of the two views.
    double p_value;               ///< Upper chi-square tail probability.
};

/// @brief Canonical correlations and paired unit-variance linear combinations.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(5,1,{1,2,3,4,6}), y(5,1,{2,1,4,3,7});
/// const auto fit = dm::stats::canonical_correlation(x,y);
/// std::println("{}", fit.x_scores.rows()); // 5
/// @endcode
struct CanonicalCorrelationFit {
    std::vector<double> correlations; ///< Nonnegative correlations in decreasing order.
    std::vector<double> x_means;      ///< Means to subtract from the columns of the first view.
    std::vector<double> y_means;      ///< Means to subtract from the columns of the second view.
    linalg::DenseMatrix<double> x_weights;  ///< Original-scale coefficients for the first view, one pair per column.
    linalg::DenseMatrix<double> y_weights;  ///< Original-scale coefficients for the second view, one pair per column.
    linalg::DenseMatrix<double> x_scores;   ///< First-view canonical scores for each input row.
    linalg::DenseMatrix<double> y_scores;   ///< Second-view canonical scores for each input row.
    linalg::DenseMatrix<double> x_loadings; ///< Correlations of the first-view variables with their own scores. Zero for constant variables.
    linalg::DenseMatrix<double> y_loadings; ///< Correlations of the second-view variables with their own scores. Zero for constant variables.
    std::size_t x_rank;                    ///< Numerical rank of the centered first view.
    std::size_t y_rank;                    ///< Numerical rank of the centered second view.
    std::vector<CanonicalCorrelationTest> sequential_tests; ///< Wilks tests in pair order, empty when the joint sample covariance is singular.
};

/// @brief Fits paired canonical linear combinations to two observation matrices.
///
/// Each pair has unit sample variance and nonnegative correlation. Pairs
/// are orthogonal within each view, including pairs with zero correlation.
/// For a deficient view, the fit uses its estimable column space and
/// minimum-norm weights in the standardized variables. The first-view
/// loading with largest absolute magnitude fixes each pair's sign.
/// Bartlett's sequential Wilks tests assume independent, jointly Gaussian
/// observations and require a nonsingular joint sample covariance.
/// @param x Finite first-view observations, one row per pair and one column per variable.
/// @param y Finite second-view observations in the same row order.
/// @param relative_tolerance Relative cutoff for within-view singular values. The minimum and default are the largest input dimension times machine epsilon.
/// @return Correlations, coefficients, scores, within-view loadings and available sequential tests.
/// @throws std::invalid_argument Invalid shapes, fewer than two paired rows, invalid tolerance or nonfinite input.
/// @throws std::domain_error Either view has zero centered rank.
/// @throws std::length_error A dense workspace exceeds its storage capacity.
/// @throws std::runtime_error A required decomposition failed to converge.
/// @throws std::overflow_error A score weight exceeds finite double range.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(5,1,{1,2,3,4,6}), y(5,1,{2,1,4,3,7});
/// const auto fit = dm::stats::canonical_correlation(x,y);
/// std::println("{}", fit.correlations.size()); // 1
/// @endcode
/// @par References
/// - Hotelling (1936), https://doi.org/10.1093/biomet/28.3-4.321
/// - Bartlett (1941), https://doi.org/10.1093/biomet/32.1.29
[[nodiscard]] CanonicalCorrelationFit canonical_correlation(const linalg::DenseMatrix<double>& x,
    const linalg::DenseMatrix<double>& y, double relative_tolerance = 0);

} // namespace datamunge::stats
