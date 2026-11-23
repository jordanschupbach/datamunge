#pragma once

/// @file fused_lasso.hpp
/// @brief Weighted regression with sparsity and adjacent-coefficient fusion.

#include <datamunge/linalg/dense_matrix.hpp>

#include <vector>

namespace datamunge::stats {

/// @brief Penalties, case weights, and solver controls for fused lasso.
/// @par Example
/// @code
/// const dm::stats::FusedLassoOptions options{.lambda_l1 = .2, .lambda_difference = .5};
/// std::println("{:.1f}", options.lambda_difference); // 0.5
/// @endcode
struct FusedLassoOptions {
    double lambda_l1{0};         ///< Nonnegative slope sparsity penalty.
    double lambda_difference{0}; ///< Nonnegative penalty on adjacent slope differences.
    std::vector<double> weights{}; ///< Positive finite case weights, or empty for unit weights.
    std::size_t max_iterations{10000}; ///< Maximum accepted proximal updates.
    double tolerance{1e-8};      ///< Stationarity tolerance relative to the largest null-model slope score.
};

/// @brief Consecutive design columns that share one fitted slope.
/// @par Example
/// @code
/// const dm::stats::FusedLassoSegment segment{1, 4, -.5};
/// std::println("{}", segment.end - segment.begin); // 3
/// @endcode
struct FusedLassoSegment {
    std::size_t begin; ///< First column index in the segment, inclusive and zero-based.
    std::size_t end;   ///< One past the last column index in the segment.
    double level;      ///< Common fitted slope over the segment.
};

/// @brief Fitted slopes, constant segments and convergence information.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(4, 2, {-1,-1, -1,1, 1,-1, 1,1});
/// const auto fit = dm::stats::fused_lasso(x, {-3,1,-1,3}, {.lambda_difference = 1});
/// std::println("{}", fit.segments.size()); // 1
/// @endcode
struct FusedLassoFit {
    double intercept;                       ///< Unpenalized intercept.
    std::vector<double> coefficients;       ///< Slopes in original design-column order and units.
    std::vector<FusedLassoSegment> segments; ///< Maximal runs of equal slopes, including zero runs.
    double objective;                       ///< Half the weighted mean squared error plus both penalties.
    double kkt_residual;                    ///< Smallest uniform slack needed in the slope stationarity equations.
    std::size_t iterations;                 ///< Number of accepted proximal updates.
    bool converged;                         ///< Whether the stationarity tolerance was satisfied.

    /// @brief Evaluates the fitted mean on new design rows.
    /// @param design Finite predictor rows with columns in training order and units.
    /// @return Predicted response in row order.
    /// @throws std::invalid_argument Invalid shape or nonfinite inputs or coefficients.
    /// @throws std::overflow_error A prediction exceeds the finite reporting range.
    /// @par Example
    /// @code
    /// const dm::linalg::DenseMatrix<double> x(4, 2, {-1,-1, -1,1, 1,-1, 1,1});
    /// const auto fit = dm::stats::fused_lasso(x, {-3,1,-1,3}, {.lambda_difference = 1});
    /// std::println("{:.1f}", fit.predict(x).back()); // 3.0
    /// @endcode
    [[nodiscard]] std::vector<double> predict(const linalg::DenseMatrix<double>& design) const;
};

/// @brief Fits Gaussian fused lasso using the supplied predictor order.
///
/// Minimizes weighted mean squared error divided by two plus
/// lambda_l1 times the sum of absolute slopes and lambda_difference times
/// the sum of absolute successive slope differences. Slopes use the original
/// predictor units. An unpenalized intercept is fitted automatically.
/// Rank-deficient designs can admit multiple slope solutions.
/// @param design Nonempty finite predictor matrix without an intercept column.
/// @param response Finite observed responses in row order.
/// @param options Penalties and numerical controls.
/// @return Coefficients and their maximal constant segments, with a convergence flag.
/// @throws std::invalid_argument Incompatible shapes or invalid inputs or options.
/// @throws std::overflow_error A preprocessed or fitted quantity exceeds its finite range.
/// @throws std::runtime_error Backtracking could not find a finite proximal step.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(4, 2, {-1,-1, -1,1, 1,-1, 1,1});
/// const auto fit = dm::stats::fused_lasso(x, {-3,1,-1,3}, {.lambda_l1 = .5, .lambda_difference = 1});
/// std::println("{:.1f}", fit.segments[0].level); // 1.0
/// @endcode
/// @par References
/// - Tibshirani et al. (2005), https://doi.org/10.1111/j.1467-9868.2005.00490.x
/// - Condat (2013), https://doi.org/10.1109/LSP.2013.2278339
/// - Beck and Teboulle (2009), https://doi.org/10.1137/080716542
[[nodiscard]] FusedLassoFit fused_lasso(const linalg::DenseMatrix<double>& design,
    const std::vector<double>& response, const FusedLassoOptions& options = {});

} // namespace datamunge::stats
