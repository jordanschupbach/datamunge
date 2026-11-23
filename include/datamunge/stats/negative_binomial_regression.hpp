#pragma once

/// @file negative_binomial_regression.hpp
/// @brief NB2 regression with estimated dispersion and log-exposure offsets.

#include <datamunge/linalg/dense_matrix.hpp>

#include <vector>

namespace datamunge::stats {

/// @brief Outcome of the negative binomial likelihood search.
/// @par Example
/// @code
/// const auto status = dm::stats::NegativeBinomialRegressionStatus::PoissonBoundary;
/// std::println("{}", status == dm::stats::NegativeBinomialRegressionStatus::PoissonBoundary); // true
/// @endcode
enum class NegativeBinomialRegressionStatus {
    Converged,       ///< Interior maximum with joint information.
    PoissonBoundary, ///< The maximum is at zero overdispersion.
    IterationLimit,  ///< The coefficient or dispersion search used its iteration budget.
    Separated,       ///< A coefficient recession direction prevents a finite fit.
    NumericalFailure ///< Rank or curvature was lost during the likelihood search.
};

/// @brief Exposure offsets, prior weights, and numerical controls.
/// @par Example
/// @code
/// const dm::stats::NegativeBinomialRegressionOptions options{.offset = {0, std::log(2.)}};
/// std::println("{}", options.offset.size()); // 2
/// @endcode
struct NegativeBinomialRegressionOptions {
    std::vector<double> offset{};  ///< Fixed additive log-scale offset per row. Empty means zero.
    std::vector<double> weights{}; ///< Positive finite case weights. Empty means unit weights.
    std::size_t max_iterations{100}; ///< Maximum updates per coefficient fit or dispersion bracket.
    double tolerance{1e-8};          ///< Tolerance in linear predictor and log-dispersion units, in (0, 1).
};

/// @brief NB2 coefficients, joint covariance, and model means.
/// @par Example
/// @code
/// const auto fit = dm::stats::negative_binomial_regression(dm::linalg::DenseMatrix<double>(4, 1, 1),
///     {0, 0, 1, 11});
/// std::println("{:.1f}", fit.fitted[0]); // 3.0
/// @endcode
struct NegativeBinomialRegressionResult {
    std::vector<double> coefficients;       ///< Log-mean coefficients in design-column order.
    double dispersion;                     ///< Alpha in variance = mean + alpha * mean squared.
    linalg::DenseMatrix<double> covariance; ///< Coefficients followed by alpha. At the Poisson boundary only the coefficient block is present.
    std::vector<double> fitted;             ///< Mean counts including the training offsets.
    double log_likelihood;                 ///< Full weighted log likelihood, including count constants.
    NegativeBinomialRegressionStatus status; ///< Convergence and boundary diagnostic.

    /// @brief Predicts mean counts at new predictors and offsets.
    /// @param design Finite design with the fitted columns, including any supplied intercept.
    /// @param offset Log-scale offsets, or an empty vector for unit exposure.
    /// @return Expected counts for the specified exposure.
    /// @throws std::invalid_argument Incompatible shapes or nonfinite inputs.
    /// @throws std::overflow_error A predicted mean exceeds finite double range.
    /// @par Example
    /// @code
    /// const auto fit = dm::stats::negative_binomial_regression(dm::linalg::DenseMatrix<double>(4, 1, 1),
    ///     {0, 0, 1, 11});
    /// std::println("{:.1f}", fit.predict(dm::linalg::DenseMatrix<double>(1, 1, 1), {std::log(2.)})[0]); // 6.0
    /// @endcode
    [[nodiscard]] std::vector<double> predict(const linalg::DenseMatrix<double>& design,
                                              const std::vector<double>& offset = {}) const;
};

/// @brief Fits log-mean coefficients and overdispersion by maximum likelihood.
///
/// The mean is exp(design times coefficients plus offset). For exposure
/// times or amounts, supply their logarithms as offsets. The NB2 likelihood
/// profiles out the regression coefficients while optimizing overdispersion
/// and includes the limiting Poisson model. At an interior solution the
/// covariance uses observed information for the coefficients and dispersion.
/// A nonconverged result returns the best finite iterate with empty covariance.
/// @param design Full-rank finite design. Include any desired intercept column.
/// @param counts Nonnegative integer counts, with one value per design row.
/// @param options Offsets, case weights, and numerical controls.
/// @return Coefficients, overdispersion, fitted means, likelihood, and fit diagnostic.
/// @throws std::invalid_argument Invalid shapes, inputs, aliased design, or zero counts throughout.
/// @throws std::length_error Coefficient covariance dimensions exceed capacity.
/// @throws std::overflow_error A result exceeds the finite reporting range.
/// @par Example
/// @code
/// const auto fit = dm::stats::negative_binomial_regression(dm::linalg::DenseMatrix<double>(4, 1, 1),
///     {0, 0, 1, 11});
/// std::println("{}", fit.dispersion > 0); // true
/// @endcode
/// @par References
/// - Venables and Ripley (2002), Modern Applied Statistics with S.
/// - R MASS, https://stat.ethz.ch/R-manual/R-devel/library/MASS/html/glm.nb.html
[[nodiscard]] NegativeBinomialRegressionResult negative_binomial_regression(
    const linalg::DenseMatrix<double>& design, const std::vector<std::size_t>& counts,
    const NegativeBinomialRegressionOptions& options = {});

} // namespace datamunge::stats
