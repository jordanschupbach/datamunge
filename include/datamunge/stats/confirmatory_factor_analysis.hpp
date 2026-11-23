#pragma once

/// @file confirmatory_factor_analysis.hpp
/// @brief Gaussian covariance fitting for explicitly constrained factor models.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstdint>

namespace datamunge::stats {

/// @brief Fixed values and starting values for one model matrix.
/// @par Example
/// @code
/// const dm::stats::CFAParameters block{dm::linalg::DenseMatrix<double>(1,1,1), {}};
/// std::println("{:.0f}", block.values(0,0)); // 1
/// @endcode
struct CFAParameters {
    linalg::DenseMatrix<double> values;            ///< Fixed entries or starting values, in original measurement units.
    linalg::DenseMatrix<std::uint8_t> free_entries;  ///< One to estimate an entry, zero to fix it. An empty mask fixes every entry.
};

/// @brief Explicit loading, factor covariance and residual covariance structures.
/// @par Example
/// @code
/// const dm::stats::CFASpecification model{
///     {dm::linalg::DenseMatrix<double>(2,1,{.7,.6}),{}},
///     {dm::linalg::DenseMatrix<double>(1,1,1),{}},
///     {dm::linalg::DenseMatrix<double>::identity(2),{}}};
/// std::println("{}", model.loadings.values.rows()); // 2
/// @endcode
struct CFASpecification {
    CFAParameters loadings;             ///< Observed variables by factors. Fixed zeros prohibit cross-loadings.
    CFAParameters factor_covariance;    ///< Symmetric positive-definite latent covariance and symmetric mask.
    CFAParameters residual_covariance;  ///< Symmetric positive-definite residual covariance and symmetric mask.
};

/// @brief Outcome of Gaussian covariance optimization.
/// @par Example
/// @code
/// const auto status = dm::stats::CFAStatus::Converged;
/// std::println("{}", status == dm::stats::CFAStatus::Converged); // true
/// @endcode
enum class CFAStatus {
    Converged,        ///< The score met tolerance or every parameter was fixed.
    IterationLimit,  ///< Accepted updates reached the configured limit.
    NumericalFailure ///< The best available fit could not take another numerically useful step.
};

/// @brief Optimizer controls in scaled covariance coordinates.
/// @par Example
/// @code
/// const dm::stats::CFAOptions options{.max_iterations=100};
/// std::println("{}", options.max_iterations); // 100
/// @endcode
struct CFAOptions {
    std::size_t max_iterations{600}; ///< Positive accepted-update limit per starting point.
    double tolerance{1e-9};          ///< Positive absolute score tolerance below one.
};

/// @brief Fitted factor model and covariance discrepancy.
/// @par Example
/// @code
/// const auto status = dm::stats::CFAFit{}.degrees_of_freedom;
/// std::println("{}", status); // 0
/// @endcode
struct CFAFit {
    linalg::DenseMatrix<double> loadings;              ///< Fitted loading coefficients in original units.
    linalg::DenseMatrix<double> factor_covariance;     ///< Fitted latent covariance.
    linalg::DenseMatrix<double> residual_covariance;   ///< Fitted residual covariance.
    linalg::DenseMatrix<double> implied_covariance;    ///< Loading–factor–loading cross-product plus residual covariance.
    linalg::DenseMatrix<double> covariance_residuals;   ///< Empirical minus implied covariance.
    double discrepancy{};                            ///< Gaussian likelihood-ratio discrepancy per observation.
    std::size_t degrees_of_freedom{};                  ///< Distinct observed covariance entries minus estimated parameters.
    CFAStatus status{CFAStatus::NumericalFailure};      ///< Optimizer stopping reason.
};

/// @brief Fits a confirmatory factor model by minimizing Gaussian covariance discrepancy.
///
/// Fixed parameters identify factor scales and orientations. For example,
/// fix each factor variance to one and fix enough cross-loadings to zero
/// to prevent rotations. A Jacobian rank check diagnoses local identification.
/// Free variances are fitted on the log scale. Both latent and residual
/// covariance matrices remain positive definite.
/// @param empirical_covariance Finite symmetric positive-definite observed covariance.
/// @param specification Fixed values and starting values for every model matrix.
/// @param options Optimizer controls.
/// @return Fitted covariance components, residuals, discrepancy, and status.
/// @throws std::invalid_argument Invalid shapes, masks, starting covariances, or controls.
/// @throws std::domain_error The covariance is singular or the specification is locally unidentified.
/// @throws std::length_error A dense workspace exceeds storage capacity.
/// @throws std::runtime_error No usable model was found or an identification factorization failed.
/// @throws std::overflow_error A fitted value exceeds the finite reporting range.
/// @par Example
/// @code
/// using dm::linalg::DenseMatrix;
/// const dm::stats::CFASpecification model{
///     {DenseMatrix<double>(2,1,{.7,.6}),{}},
///     {DenseMatrix<double>(1,1,1),{}},
///     {DenseMatrix<double>(2,2,{.5,0,0,.5}),{}}};
/// const auto fit = dm::stats::fit_confirmatory_factor_analysis(DenseMatrix<double>::identity(2),model);
/// std::println("{:.2f}", fit.implied_covariance(0,1)); // 0.42
/// @endcode
/// @par References
/// - Jöreskog (1969), https://doi.org/10.1007/BF02289343
[[nodiscard]] CFAFit fit_confirmatory_factor_analysis(const linalg::DenseMatrix<double>& empirical_covariance,
    const CFASpecification& specification, const CFAOptions& options = {});

} // namespace datamunge::stats
