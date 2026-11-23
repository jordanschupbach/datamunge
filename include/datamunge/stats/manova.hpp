#pragma once

/// @file manova.hpp
/// @brief Multivariate analysis of variance for nested linear models.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstdint>
#include <optional>
#include <vector>

namespace datamunge::stats {

/// @brief F reference distribution for a multivariate criterion.
/// @par Example
/// @code
/// const dm::stats::MANOVAFReference reference{3.0, 2, 12, 0.08779};
/// std::println("{:.1f}", reference.statistic); // 3.0
/// @endcode
struct MANOVAFReference {
    double statistic;      ///< F statistic for this criterion.
    double df_numerator;   ///< Numerator degrees of freedom, potentially fractional.
    double df_denominator; ///< Denominator degrees of freedom, potentially fractional.
    double p_value;        ///< Upper F tail for this criterion.
};

/// @brief Multivariate criterion and its F reference when defined.
/// @par Example
/// @code
/// const dm::stats::MANOVACriterion criterion{0.25, std::nullopt};
/// std::println("{:.2f}", criterion.statistic); // 0.25
/// @endcode
struct MANOVACriterion {
    double statistic;                         ///< Criterion on its own scale.
    std::optional<MANOVAFReference> reference; ///< F reference, absent when its degrees of freedom are nonpositive.
};

/// @brief Hypothesis and error cross-products, generalized eigenvalues and four multivariate tests.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> y(6,2,{1,2,2,1,3,2,4,5,5,4,6,5});
/// const auto fit = dm::stats::one_way_manova(y,{3,3,3,8,8,8});
/// std::println("{}", fit.error_df); // 4
/// @endcode
struct MANOVAResult {
    linalg::DenseMatrix<double> hypothesis; ///< Hypothesis sums of squares and cross-products.
    linalg::DenseMatrix<double> error;      ///< Full-model residual sums of squares and cross-products.
    std::vector<double> roots;               ///< Largest generalized eigenvalues of hypothesis relative to error, in decreasing order.
    std::size_t hypothesis_df;              ///< Rank of the full design minus rank of the reduced design.
    std::size_t error_df;                   ///< Number of rows minus rank of the full design.
    MANOVACriterion pillai;                  ///< Pillai's trace.
    MANOVACriterion wilks;                   ///< Wilks' lambda.
    MANOVACriterion hotelling_lawley;        ///< Hotelling–Lawley trace.
    MANOVACriterion roy;                     ///< Roy's largest root, with an upper-bound F statistic for multi-root effects.
};

/// @brief Tests a nested linear-model effect across several outcomes simultaneously.
///
/// Uses the residual covariance from the full model. The reference laws
/// assume independent multivariate Gaussian errors with a common covariance.
/// The usual F transformations are exact when the effect or outcome space is
/// one-dimensional, and otherwise approximate. Roy's F is an upper bound
/// for multi-root effects.
/// @param response Finite observations as rows and outcomes as columns.
/// @param full_design Finite full-rank design for the full model with any intercept included explicitly.
/// @param reduced_design Finite full-rank nested design omitting the tested effect. A zero-column design represents a zero fitted mean.
/// @return The raw cross-products, generalized roots, and the four criteria.
/// @throws std::invalid_argument Incompatible or nonfinite input, aliased or non-nested designs, or insufficient residual degrees of freedom.
/// @throws std::domain_error The full-model residual covariance is singular.
/// @throws std::length_error A working matrix exceeds its storage capacity.
/// @throws std::runtime_error A required factorization failed to converge.
/// @throws std::overflow_error An error or hypothesis cross-product exceeds finite double range.
/// @par Example
/// @code
/// using dm::linalg::DenseMatrix;
/// const DenseMatrix<double> y(6,2,{1,2,2,1,3,2,4,5,5,4,6,5});
/// const DenseMatrix<double> full(6,2,{1,0,1,0,1,0,1,1,1,1,1,1});
/// const auto fit = dm::stats::manova(y,full,DenseMatrix<double>(6,1,1));
/// std::println("{}", fit.hypothesis_df); // 1
/// @endcode
/// @par References
/// - Pillai (1955), https://doi.org/10.1214/aoms/1177728599
[[nodiscard]] MANOVAResult manova(const linalg::DenseMatrix<double>& response,
    const linalg::DenseMatrix<double>& full_design, const linalg::DenseMatrix<double>& reduced_design);

/// @brief Tests equality of the multivariate means in two or more independent groups.
/// @param response Finite observations as rows and outcomes as columns.
/// @param group One group label per observation, with at least two distinct labels.
/// @return The raw cross-products, roots and criteria, as in manova().
/// @throws std::invalid_argument Incompatible or nonfinite input, fewer than two groups, or insufficient residual degrees of freedom.
/// @throws std::domain_error The pooled within-group covariance is singular.
/// @throws std::length_error A working matrix exceeds its storage capacity.
/// @throws std::runtime_error A required factorization failed to converge.
/// @throws std::overflow_error An error or hypothesis cross-product exceeds finite double range.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> y(6,2,{1,2,2,1,3,2,4,5,5,4,6,5});
/// const auto fit = dm::stats::one_way_manova(y,{-2,-2,-2,7,7,7});
/// std::println("{:.1f}", fit.error(0,0)); // 4.0
/// @endcode
[[nodiscard]] MANOVAResult one_way_manova(const linalg::DenseMatrix<double>& response,
    const std::vector<std::int64_t>& group);

} // namespace datamunge::stats
