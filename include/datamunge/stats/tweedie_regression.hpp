#pragma once

/// @file tweedie_regression.hpp
/// @brief Maximum likelihood for log-link compound Poisson–gamma regression.

#include <datamunge/linalg/dense_matrix.hpp>

#include <vector>

namespace datamunge::stats {

/// @brief Outcome of the Tweedie likelihood profile and covariance computation.
/// @par Example
/// @code
/// const auto status = dm::stats::TweedieRegressionStatus::Converged;
/// std::println("{}", status == dm::stats::TweedieRegressionStatus::Converged); // true
/// @endcode
enum class TweedieRegressionStatus {
    Converged,       ///< The selected fit has a regular information matrix.
    PowerBoundary,   ///< The estimated power is at a search bound, with conditional covariance.
    NumericalFailure ///< A finite fit is available but the dispersion search or covariance is unresolved.
};

/// @brief Power search range, exposure offsets, and weights for Tweedie maximum likelihood.
/// @par Example
/// @code
/// const dm::stats::TweedieRegressionOptions options{.minimum_power = 1.5, .maximum_power = 1.5};
/// std::println("{:.1f}", options.minimum_power); // 1.5
/// @endcode
struct TweedieRegressionOptions {
    double minimum_power{1.01};    ///< Lower power bound, strictly greater than one.
    double maximum_power{1.99};    ///< Upper power bound, strictly less than two. Equal bounds fix the power.
    std::vector<double> offset{};  ///< Additive log-scale mean offsets, with zero as the default.
    std::vector<double> weights{}; ///< Positive finite case weights, with one as the default.
};

/// @brief An evaluated point of the likelihood profile with mean and dispersion refitted.
/// @par Example
/// @code
/// const dm::stats::TweedieProfilePoint point{1.5, .8, -20.};
/// std::println("{:.1f}", point.dispersion); // 0.8
/// @endcode
struct TweedieProfilePoint {
    double power;          ///< Tweedie variance power fixed during the conditional fit.
    double dispersion;     ///< Estimated multiplicative variance coefficient.
    double log_likelihood; ///< Weighted log likelihood, with a point mass for exact zeros.
};

/// @brief Estimated mean, dispersion, variance power, and likelihood profile.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(5, 1, 1.);
/// const auto fit = dm::stats::tweedie_regression(x, {0,.5,1,2,4},
///     {.minimum_power = 1.5, .maximum_power = 1.5});
/// std::println("{:.1f}", fit.fitted_mean[0]); // 1.5
/// @endcode
struct TweedieRegressionResult {
    std::vector<double> coefficients;       ///< Log-mean coefficients in design-column order.
    double dispersion;                      ///< Dispersion phi, with variance phi times mean to the power.
    double power;                           ///< Variance power p.
    std::vector<double> fitted_mean;        ///< Mean at each training row and offset.
    double log_likelihood;                  ///< Weighted log density and zero mass.
    linalg::DenseMatrix<double> covariance; ///< Observed information inverse, ordered coefficients, phi, then an interior estimated p.
    std::vector<TweedieProfilePoint> profile; ///< Finite grid points and the optimum, sorted by power.
    TweedieRegressionStatus status;         ///< Profile and information outcome.

    /// @brief Predicts response means using the fitted log link.
    /// @param design Finite predictors with one column per fitted coefficient.
    /// @param offset Additional log offsets, or empty for unit exposure.
    /// @return Response-scale means in row order.
    /// @throws std::invalid_argument Incompatible dimensions or nonfinite inputs.
    /// @throws std::overflow_error A predicted mean is outside the finite positive double range.
    /// @par Example
    /// @code
    /// const dm::linalg::DenseMatrix<double> x(5, 1, 1.);
    /// const auto fit = dm::stats::tweedie_regression(x, {0,.5,1,2,4},
    ///     {.minimum_power = 1.5, .maximum_power = 1.5});
    /// std::println("{:.1f}", fit.predict(dm::linalg::DenseMatrix<double>(1,1,1.), {std::log(2.)})[0]); // 3.0
    /// @endcode
    [[nodiscard]] std::vector<double> predict(const linalg::DenseMatrix<double>& design,
        const std::vector<double>& offset = {}) const;
};

/// @brief Fits a log-link compound Poisson–gamma GLM by profiled maximum likelihood.
///
/// The model has a point mass at zero and continuous density over positive
/// outcomes. The likelihood uses a positive series with tail control. Mean
/// coefficients are refitted at each power and dispersion is profiled on its
/// log scale. The deterministic grid search refines sampled local maxima.
/// Profile points may be absent when the density series exceeds its work limit.
/// An empty covariance denotes unresolved estimation or information.
/// @param design Finite full-rank mean design with more rows than columns. Supply any intercept explicitly.
/// @param response Finite nonnegative values, including at least one positive value.
/// @param options Power range, case weights, and additive log offsets.
/// @return Fitted model, with conditional covariance when power is fixed or on a bound.
/// @throws std::invalid_argument Invalid input or rank-deficient design.
/// @throws std::length_error Coefficient matrix exceeds representable capacity.
/// @throws std::overflow_error A fitted output is outside the reporting range.
/// @throws std::runtime_error No usable finite profile point was found.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(5, 1, 1.);
/// const auto fit = dm::stats::tweedie_regression(x, {0,.5,1,2,4},
///     {.minimum_power = 1.5, .maximum_power = 1.5});
/// std::println("{:.1f}", fit.power); // 1.5
/// @endcode
/// @par References
/// - Dunn and Smyth (2005), https://doi.org/10.1007/s11222-005-4070-y
/// - Smyth and Jørgensen (2002), https://doi.org/10.2143/AST.32.1.1020
[[nodiscard]] TweedieRegressionResult tweedie_regression(const linalg::DenseMatrix<double>& design,
    const std::vector<double>& response, const TweedieRegressionOptions& options = {});

} // namespace datamunge::stats
