#pragma once

/// @file robust_regression.hpp
/// @brief Huber and Tukey bisquare M-estimation for linear regression.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <optional>
#include <vector>

namespace datamunge::stats {

/// @brief Selects the residual weight function for robust regression.
/// @par Example
/// @code
/// const auto loss = dm::stats::RobustRegressionLoss::TukeyBisquare;
/// std::println("{}", loss == dm::stats::RobustRegressionLoss::TukeyBisquare); // true
/// @endcode
/// @par References
/// - Huber (1973), https://doi.org/10.1214/aos/1176342503
/// - Beaton and Tukey (1974), https://doi.org/10.1080/00401706.1974.10489171
/// @see robust_regression()
enum class RobustRegressionLoss {
    Huber,         ///< Clips the influence of large standardized residuals.
    TukeyBisquare, ///< Smoothly brings the influence to zero at the tuning cutoff.
};

/// @brief Describes why an iteratively reweighted fit stopped.
/// @par Example
/// @code
/// std::println("{}", dm::stats::RobustRegressionStatus::Converged ==
///                    dm::stats::RobustRegressionStatus::IterationLimit); // false
/// @endcode
/// @see robust_regression()
enum class RobustRegressionStatus {
    Converged,     ///< Maximum fitted-value change relative to scale reached tolerance.
    IterationLimit, ///< The reweighting limit was reached.
    ZeroScale,     ///< The estimated median absolute residual is zero.
    RankDeficient, ///< Reweighting lost design rank. The result retains the last coefficients.
    LinearSolveLimit, ///< The weighted design's SVD reached its sweep limit.
};

/// @brief Controls the weight function, scale and stopping rule.
/// @par Example
/// @code
/// const dm::stats::RobustRegressionOptions options{.fixed_scale = 1.0};
/// std::println("{}", *options.fixed_scale); // 1
/// @endcode
/// @see robust_regression()
struct RobustRegressionOptions {
    RobustRegressionLoss loss{RobustRegressionLoss::Huber}; ///< Residual weight function.
    std::optional<double> tuning_constant{}; ///< Positive cutoff. Defaults to 1.345 for Huber and 4.685 for Tukey.
    std::optional<double> fixed_scale{};   ///< Positive scale in response units. Defaults to median absolute residual scaling.
    std::vector<double> initial_coefficients{}; ///< Finite starting coefficients. An empty vector selects least squares.
    std::size_t max_iterations{100};   ///< Positive limit on weighted least-squares updates.
    double tolerance{1e-8};            ///< Positive tolerance on fitted-value change divided by the current scale.
};

/// @brief Robust coefficient estimates, residuals, weights and stopping diagnostics.
/// @par Example
/// @code
/// const auto fit = dm::stats::robust_regression(dm::linalg::DenseMatrix<double>(3, 1, 1.0), {2, 2, 2});
/// std::println("{}", fit.coefficients.size()); // 1
/// @endcode
/// @see robust_regression()
struct RobustRegressionResult {
    std::vector<double> coefficients; ///< Coefficients in design column order.
    std::vector<double> residuals;    ///< Original response minus fitted response.
    std::vector<double> weights;      ///< Last implied robust weights in design row order.
    double scale;                     ///< Final scale, in response units.
    double relative_change;           ///< Last completed change divided by the scale used for that update, or zero before any update.
    std::size_t iterations;           ///< Completed weighted least-squares updates.
    RobustRegressionStatus status;    ///< Stopping diagnostic.
};

/// @brief Fits a Huber or Tukey bisquare M-estimator by iteratively reweighted least squares.
///
/// Each iteration uses a fixed scale or the median absolute residual divided
/// by the standard-normal 75th percentile. A zero estimated scale stops the
/// fit and assigns weight one to zero residuals and zero to the remaining
/// residuals. Tukey bisquare estimation can depend on the starting coefficients.
/// @param design Finite full-column-rank design with more rows than columns.
/// Include a column of ones when fitting an intercept.
/// @param response Finite response in design row order.
/// @param options Loss, optional scale and starting coefficients, and iteration controls.
/// @return Owned coefficients, final residuals and weights, scale and stopping diagnostic.
/// @throws std::invalid_argument Invalid dimensions, inputs, controls or initial design rank.
/// @throws std::runtime_error The initial design's SVD reaches its sweep limit.
/// @throws std::overflow_error An estimate or diagnostic exceeds finite double range.
/// @par Example
/// @code
/// const auto fit = dm::stats::robust_regression(dm::linalg::DenseMatrix<double>(3, 1, 1.0), {2, 2, 2});
/// std::println("{:.1f}", fit.coefficients.front()); // 2.0
/// @endcode
/// @par References
/// - Huber (1973), https://doi.org/10.1214/aos/1176342503
/// - Beaton and Tukey (1974), https://doi.org/10.1080/00401706.1974.10489171
/// @see RobustRegressionOptions, RobustRegressionResult
RobustRegressionResult robust_regression(const linalg::DenseMatrix<double>& design,
                                         const std::vector<double>& response,
                                         const RobustRegressionOptions& options = {});

} // namespace datamunge::stats
