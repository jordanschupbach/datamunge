#pragma once

/// @file hausman_test.hpp
/// @brief Hausman specification test for shared panel regression coefficients.

#include <datamunge/linalg/dense_matrix.hpp>

#include <string>
#include <vector>

namespace datamunge::stats {

/// @brief Named regression coefficients and their estimated sampling covariance.
/// @par Example
/// @code
/// const dm::stats::HausmanEstimates fit{{"slope"}, {2}, dm::linalg::DenseMatrix<double>(1,1,{.5})};
/// std::println("{}", fit.terms.front()); // slope
/// @endcode
struct HausmanEstimates {
    std::vector<std::string> terms;         ///< Unique labels for the estimable quantities supplied by the model.
    std::vector<double> coefficients;       ///< Point estimates in term order.
    linalg::DenseMatrix<double> covariance; ///< Sampling covariance in the same term order.
};

/// @brief Hausman statistic, effective rank, and differences by common term.
/// @par Example
/// @code
/// using dm::linalg::DenseMatrix;
/// const auto test = dm::stats::hausman_test({{"slope"}, {2}, DenseMatrix<double>(1,1,{.5})},
///     {{"slope"}, {1}, DenseMatrix<double>(1,1,{.25})});
/// std::println("{:.1f}", test.statistic); // 4.0
/// @endcode
struct HausmanTestResult {
    double statistic;                        ///< Quadratic form on the retained covariance directions.
    std::size_t df;                          ///< Numerical rank of the covariance of the difference.
    double p_value;                          ///< Upper-tail chi-square probability.
    std::vector<std::string> terms;          ///< Shared coefficient labels in fixed-effect estimate order.
    std::vector<double> difference;          ///< Fixed-effect minus random-effect estimates.
    linalg::DenseMatrix<double> covariance_difference; ///< Fixed-effect minus random-effect sampling covariance.
    std::vector<double> untested_difference; ///< Projection of the estimate difference onto discarded directions after scaling, expressed back in coefficient units.
};

/// @brief Compares fixed- and random-effects coefficients on their common estimable subspace.
///
/// Match quantities by label and use fits of the same sample and parameterization.
/// Under the null, the random-effects estimate must be efficient and the fixed-effects
/// estimate consistent. This gives the covariance of the difference as fixed minus
/// random sampling covariance. To determine its numerical rank, standardize each
/// shared quantity by the larger of the two supplied standard errors, or by one
/// if both are zero. Retain the positive eigenvalues above the cutoff, and return
/// the discarded component of the estimate difference as a diagnostic.
/// @param fixed_effects Consistent named coefficient estimates and sampling covariance.
/// @param random_effects Named coefficient estimates and sampling covariance from the estimator efficient under the null.
/// @param relative_tolerance Positive relative eigenvalue cutoff below one, using the larger standardized model eigenvalue or one, with a machine-precision floor.
/// @return Chi-square test and estimate and covariance differences in the matched order.
/// @throws std::invalid_argument Incompatible, nonfinite, nonsymmetric or unnamed inputs, nonpositive tolerance, or no shared terms.
/// @throws std::domain_error A covariance on the shared terms is indefinite, or the covariance difference has zero numerical rank.
/// @throws std::length_error The square working matrix exceeds its storage capacity.
/// @throws std::runtime_error The symmetric eigensolver failed to converge.
/// @throws std::overflow_error A returned coefficient or covariance difference exceeds the finite reporting range.
/// @par Example
/// @code
/// using dm::linalg::DenseMatrix;
/// const auto test = dm::stats::hausman_test({{"slope"}, {2}, DenseMatrix<double>(1,1,{.5})},
///     {{"intercept","slope"}, {3,1}, DenseMatrix<double>(2,2,{1,0,0,.25})});
/// std::println("{} {:.4f}", test.df, test.p_value); // 1 0.0455
/// @endcode
/// @par References
/// - Hausman (1978), https://doi.org/10.2307/1913827
[[nodiscard]] HausmanTestResult hausman_test(const HausmanEstimates& fixed_effects,
    const HausmanEstimates& random_effects, double relative_tolerance = 1e-10);

} // namespace datamunge::stats
