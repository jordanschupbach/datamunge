#pragma once

/// @file variance_components.hpp
/// @brief Gaussian mixed models with multiple independent random-effect blocks.

#include <datamunge/linalg/dense_matrix.hpp>

#include <vector>

namespace datamunge::stats {

/// @brief Outcome of the constrained variance-component likelihood search.
/// @par Example
/// @code
/// const auto status = dm::stats::VarianceComponentsStatus::Boundary;
/// std::println("{}", status == dm::stats::VarianceComponentsStatus::Boundary); // true
/// @endcode
enum class VarianceComponentsStatus {
    Converged,       ///< The projected score met tolerance at an interior fit.
    Boundary,        ///< A converged fit has at least one zero component variance.
    IterationLimit,  ///< The best candidate exhausted its accepted updates.
    NumericalFailure ///< The best candidate failed the line search or a conditioning check.
};

/// @brief Selects ML or REML and the controls for variance-component optimization.
/// @par Example
/// @code
/// const dm::stats::VarianceComponentsOptions options{.reml = false};
/// std::println("{}", options.reml); // false
/// @endcode
struct VarianceComponentsOptions {
    bool reml{true};                ///< Restricted maximum likelihood. Use false for full ML.
    std::size_t max_iterations{200}; ///< Maximum accepted updates from each deterministic start.
    double tolerance{1e-9};         ///< Projected score tolerance per observation in scaled log-ratio coordinates.
};

/// @brief Fixed effects, variance estimates, and conditional random-effect predictions.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(6, 1, 1.);
/// const dm::linalg::DenseMatrix<double> z(6, 3, {1,0,0,1,0,0, 0,1,0,0,1,0, 0,0,1,0,0,1});
/// const auto fit = dm::stats::fit_variance_components(x, {0,1,3,4,6,7}, {z});
/// std::println("{:.1f}", fit.coefficients[0]); // 3.5
/// @endcode
struct VarianceComponentsFit {
    std::vector<double> coefficients;        ///< Fixed effects in the original design order.
    linalg::DenseMatrix<double> covariance;  ///< Fixed-effect covariance conditional on the fitted variances.
    std::vector<double> variances;           ///< Random coefficient variances in block order.
    double residual_variance;                ///< Gaussian residual variance.
    std::vector<std::vector<double>> effects; ///< Conditional random-effect modes in block and column order.
    std::vector<double> fitted;              ///< Training means including random effects.
    double log_likelihood;                    ///< Maximized log likelihood or restricted log likelihood.
    VarianceComponentsStatus status;         ///< Optimization outcome.
    bool reml{true};                         ///< True for a restricted maximum likelihood fit.

    /// @brief Predicts means with fixed effects and, optionally, fitted random effects.
    /// @param fixed_design Finite new fixed predictors in training column order.
    /// @param random_designs Optional new random designs in training block and column order.
    /// Give a row of zeros in a block for a new group, or omit all blocks for marginal means.
    /// @return Predictions in query row order.
    /// @throws std::invalid_argument Invalid dimensions or nonfinite inputs or coefficients.
    /// @throws std::overflow_error A prediction exceeds the finite reporting range.
    /// @par Example
    /// @code
    /// const dm::linalg::DenseMatrix<double> x(6, 1, 1.);
    /// const dm::linalg::DenseMatrix<double> z(6, 3, {1,0,0,1,0,0, 0,1,0,0,1,0, 0,0,1,0,0,1});
    /// const auto fit = dm::stats::fit_variance_components(x, {0,1,3,4,6,7}, {z});
    /// std::println("{:.1f}", fit.predict(x)[0]); // 3.5
    /// @endcode
    [[nodiscard]] std::vector<double> predict(const linalg::DenseMatrix<double>& fixed_design,
        const std::vector<linalg::DenseMatrix<double>>& random_designs = {}) const;
};

/// @brief Fits Gaussian crossed, nested, or other independent random-effect blocks.
///
/// Block j contributes variance[j] times Z[j] Z[j] transpose to the response
/// covariance. Random coefficients within a block are independent with the
/// same variance. Use a group indicator matrix for each random intercept, or
/// multiply the indicators by a numeric covariate for a random slope. For a
/// nested block, use distinct indicator columns for each parent-child pair.
/// The solver forms the dense marginal covariance and profiles the positive
/// residual variance. Nonnegative component ratios can reach zero exactly.
/// @param fixed_design Finite full-rank fixed design. Supply any intercept explicitly.
/// @param response Finite Gaussian responses with positive residual variation.
/// @param random_designs One or more finite, nonzero random design matrices with the same rows.
/// @param options ML or REML and convergence controls.
/// @return Fitted coefficients, variances, conditional modes and diagnostics.
/// @throws std::invalid_argument Invalid inputs, aliased fixed columns or nonidentifiable variance blocks.
/// @throws std::length_error A dense covariance or information matrix exceeds storage capacity.
/// @throws std::runtime_error No numerically usable profile fit was found.
/// @throws std::overflow_error A fitted value exceeds the reporting range.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(6, 1, 1.);
/// const dm::linalg::DenseMatrix<double> z(6, 3, {1,0,0,1,0,0, 0,1,0,0,1,0, 0,0,1,0,0,1});
/// const auto fit = dm::stats::fit_variance_components(x, {0,1,3,4,6,7}, {z});
/// std::println("{:.1f}", fit.residual_variance); // 0.5
/// @endcode
/// @par References
/// - Patterson and Thompson (1971), https://doi.org/10.1093/biomet/58.3.545
/// - Bates, Mächler, Bolker and Walker (2015), https://doi.org/10.18637/jss.v067.i01
[[nodiscard]] VarianceComponentsFit fit_variance_components(const linalg::DenseMatrix<double>& fixed_design,
    const std::vector<double>& response, const std::vector<linalg::DenseMatrix<double>>& random_designs,
    const VarianceComponentsOptions& options = {});

} // namespace datamunge::stats
