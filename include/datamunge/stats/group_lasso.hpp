#pragma once

/// @file group_lasso.hpp
/// @brief Gaussian group and sparse group lasso regularization paths.

#include <datamunge/linalg/dense_matrix.hpp>

#include <vector>

namespace datamunge::stats {

/// @brief Controls the group and within-group penalties, preprocessing, and path solver.
/// @par Example
/// @code
/// const dm::stats::GroupLassoOptions options{.alpha = .5, .lambdas = {2, 1, .5}};
/// std::println("{}", options.lambdas.size()); // 3
/// @endcode
struct GroupLassoOptions {
    double alpha{0};                ///< Fraction of the penalty on individual slopes in [0,1]. Zero is group lasso.
    std::vector<double> lambdas{}; ///< Finite, nonnegative, strictly decreasing values. Empty requests a geometric path.
    std::size_t path_length{40};   ///< Automatic path length. A null signal yields one zero-lambda fit.
    double lambda_min_ratio{.01};  ///< Smallest lambda divided by the largest in an automatic path.
    bool standardize{true};         ///< Scale centered columns to weighted population root mean square.
    std::vector<double> weights{}; ///< Positive finite case weights in row order, or empty for unit weights.
    std::vector<double> group_weights{}; ///< Positive penalty multipliers in ascending group label order. Defaults to square roots of group sizes.
    std::size_t max_iterations{10000}; ///< Accepted proximal steps per path point.
    double tolerance{1e-8};          ///< Relative stationarity tolerance, scaled by the largest null-model slope score.
};

/// @brief One fitted point on a Gaussian group-penalty path.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(4, 1, {-1,-1,1,1});
/// const auto path = dm::stats::group_lasso_path(x, {1,1,5,5}, {7}, {.lambdas = {1}});
/// std::println("{:.1f}", path.front().coefficients[0]); // 1.0
/// @endcode
struct GroupLassoFit {
    double lambda;                         ///< Penalty strength for this fit.
    double intercept;                      ///< Unpenalized intercept in original predictor units.
    std::vector<double> coefficients;      ///< Slopes in input column order and original units.
    std::vector<std::size_t> active_groups; ///< Ascending labels of groups with at least one nonzero slope.
    double objective;                      ///< Weighted mean squared residuals divided by two, plus the penalty.
    double kkt_residual;                   ///< Maximum coordinate or inactive-group stationarity residual on the fitted scale.
    std::size_t iterations;                ///< Accepted proximal steps at this fit.
    bool converged;                        ///< Whether its stationarity residual met the tolerance.

    /// @brief Predicts means at finite design rows in the original predictor units.
    /// @param design Query matrix with one column per slope, in training order.
    /// @return Prediction for each query row.
    /// @throws std::invalid_argument Invalid shape or nonfinite fit or query data.
    /// @throws std::overflow_error A prediction exceeds the finite reporting range.
    /// @par Example
    /// @code
    /// const dm::linalg::DenseMatrix<double> x(4, 1, {-1,-1,1,1});
    /// const auto path = dm::stats::group_lasso_path(x, {1,1,5,5}, {7}, {.lambdas = {1}});
    /// std::println("{:.1f}", path.front().predict(dm::linalg::DenseMatrix<double>(1,1,2))[0]); // 5.0
    /// @endcode
    [[nodiscard]] std::vector<double> predict(const linalg::DenseMatrix<double>& design) const;
};

/// @brief Fits a warm-started regularization path for group or sparse group lasso.
///
/// Minimizes weighted mean squared error divided by two plus
/// lambda times the sum of alpha times the L1 norm and (1-alpha) times
/// the weighted sum of group L2 norms. The penalty applies after optional
/// scaling, and the intercept is always unpenalized. Groups partition the
/// slope columns. Use alpha zero for group lasso, or a value strictly
/// between zero and one for sparse group lasso. Dependent designs can
/// admit multiple slope solutions. Inspect convergence at each path point.
/// @param design Nonempty finite predictor matrix, without an intercept column.
/// @param response Finite observed responses in row order.
/// @param groups One group label per predictor. Labels need not be consecutive.
/// @param options Case weights, penalty weights, path, and numerical controls.
/// @return Fitted points in descending penalty order with selected group labels.
/// @throws std::invalid_argument Invalid dimensions, options, or nonfinite data.
/// @throws std::overflow_error Preprocessed or fitted quantities exceed their finite range.
/// @throws std::runtime_error Backtracking failed to find a finite proximal step.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(4, 1, {-1,-1,1,1});
/// const auto path = dm::stats::group_lasso_path(x, {1,1,5,5}, {7}, {.lambdas = {3,1}});
/// std::println("{}", path.back().active_groups.front()); // 7
/// @endcode
/// @par References
/// - Yuan and Lin (2006), https://doi.org/10.1111/j.1467-9868.2005.00532.x
/// - Simon, Friedman, Hastie and Tibshirani (2013), https://doi.org/10.1080/10618600.2012.681250
[[nodiscard]] std::vector<GroupLassoFit> group_lasso_path(const linalg::DenseMatrix<double>& design,
    const std::vector<double>& response, const std::vector<std::size_t>& groups,
    const GroupLassoOptions& options = {});

} // namespace datamunge::stats
