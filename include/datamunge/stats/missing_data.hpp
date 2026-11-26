#pragma once

/// @file missing_data.hpp
/// @brief Missing-data patterns, maximum-likelihood EM for incomplete multivariate normal data, and Little's MCAR test.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <vector>

namespace datamunge::stats {

/// @brief Rows sharing one pattern of observed variables.
/// @par Example
/// @code
/// const dm::stats::MissingPattern p{.observed = {true, false}, .rows = {0, 3}};
/// std::println("{}", p.rows.size()); // 2
/// @endcode
struct MissingPattern {
    std::vector<bool> observed;     ///< Whether each variable is observed in this pattern.
    std::vector<std::size_t> rows;  ///< Rows with this pattern, in increasing order.
};

/// @brief Groups rows by their pattern of missing values (NaN marks a missing value).
///
/// Patterns are ordered by decreasing number of observed variables, and then by first appearance.
/// @param data Data matrix, one row per case.
/// @return The distinct missing-data patterns.
/// @throws std::invalid_argument For an empty matrix or infinite values.
/// @par Example
/// @code
/// const double na = std::nan("");
/// const dm::linalg::DenseMatrix<double> x(3, 2, {1, 2, 3, na, 5, 6});
/// const auto p = dm::stats::missing_patterns(x);
/// std::println("{} {}", p.size(), p[0].rows.size()); // 2 2
/// @endcode
/// @see em_multivariate_normal()
[[nodiscard]] std::vector<MissingPattern> missing_patterns(const linalg::DenseMatrix<double>& data);

/// @brief Options for em_multivariate_normal().
/// @par Example
/// @code
/// const dm::stats::MvnEmOptions o;
/// std::println("{}", o.max_iterations); // 10000
/// @endcode
struct MvnEmOptions {
    std::size_t max_iterations{10000}; ///< Iteration limit.
    double tolerance{1e-10};           ///< Stop when no mean or covariance entry changes by more than tolerance * (1 + |value|).
};

/// @brief Maximum-likelihood mean and covariance from incomplete multivariate normal data.
/// @par Example
/// @code
/// const dm::stats::MvnEmResult r;
/// std::println("{}", r.converged); // false
/// @endcode
struct MvnEmResult {
    std::vector<double> mean;                ///< ML mean vector.
    linalg::DenseMatrix<double> covariance;  ///< ML covariance (divisor n).
    double log_likelihood{};                 ///< Observed-data log-likelihood at the estimates.
    std::size_t iterations{};                ///< EM iterations.
    bool converged{false};                   ///< True if the tolerance was met.
    double convergence_rate{};               ///< Ratio of the last two parameter changes; estimates the largest fraction of missing information.
    std::size_t cases{};                     ///< Rows with at least one observed value (the rest carry no information).
    std::size_t patterns{};                  ///< Distinct missing-data patterns among those rows.
};

/// @brief Estimates the mean and covariance of multivariate normal data with missing values by EM.
///
/// The E-step replaces each case's sufficient statistics by their conditional expectations given
/// its observed values, \f$E[x_M\mid x_O]=\mu_M+\Sigma_{MO}\Sigma_{OO}^{-1}(x_O-\mu_O)\f$, adding
/// the conditional covariance \f$\Sigma_{MM}-\Sigma_{MO}\Sigma_{OO}^{-1}\Sigma_{OM}\f$ to the
/// cross-products. The M-step takes the complete-data ML estimates. Starting values are the
/// available-case means and variances with zero covariances. The estimates are the ML estimates
/// under missing at random (MAR), so they are consistent when missingness depends on observed
/// variables, unlike listwise deletion. Convergence is linear, at a rate equal to the largest
/// fraction of missing information.
/// @param data Data matrix with NaN for missing values.
/// @param options Iteration control.
/// @return ML mean, covariance, log-likelihood and convergence information.
/// @throws std::invalid_argument For infinite values, a variable with no observed values, or fewer
/// than two cases.
/// @throws std::runtime_error If an observed-variable covariance block becomes singular.
/// @par Example
/// @code
/// const double na = std::nan("");
/// const dm::linalg::DenseMatrix<double> x(5, 2, {1, 2, 2, 5, 3, 4, 4, na, 5, na});
/// const auto r = dm::stats::em_multivariate_normal(x);
/// std::println("{:.3f} {:.3f}", r.mean[0], r.mean[1]); // 3.000 4.667
/// @endcode
/// @par References
/// - A. P. Dempster, N. M. Laird and D. B. Rubin (1977). "Maximum likelihood from incomplete data via
///   the EM algorithm." JRSS B 39(1), 1-38. doi:10.1111/j.2517-6161.1977.tb01600.x
/// - R. J. A. Little and D. B. Rubin (2019). Statistical Analysis with Missing Data, 3rd ed. Wiley, chapter 11.
/// @see little_mcar_test(), missing_patterns()
[[nodiscard]] MvnEmResult em_multivariate_normal(const linalg::DenseMatrix<double>& data,
                                                 const MvnEmOptions& options = {});

/// @brief Result of Little's test of missing completely at random.
/// @par Example
/// @code
/// const dm::stats::LittleMcarTest t;
/// std::println("{}", t.patterns); // 0
/// @endcode
struct LittleMcarTest {
    double statistic{};    ///< \f$d^2=\sum_jn_j(\bar y_j-\hat\mu_{O_j})^\top\hat\Sigma_{O_j}^{-1}(\bar y_j-\hat\mu_{O_j})\f$.
    double df{};           ///< \f$\sum_jk_j-p\f$, with \f$k_j\f$ the observed variables in pattern j.
    double p_value{};      ///< Upper-tail chi-square p-value.
    std::size_t patterns{};///< Missing-data patterns with at least one observed variable.
    MvnEmResult fit;       ///< EM estimates used by the test.
};

/// @brief Little's chi-square test that data are missing completely at random (MCAR).
///
/// Under MCAR the observed-variable means of every missing-data pattern estimate the same
/// population means. The statistic compares each pattern's means with the EM estimates, weighted by
/// the pattern size and the ML covariance of the observed variables, as `naniar::mcar_test`. A small
/// p-value is evidence against MCAR. A large one does not establish MCAR or MAR: the test has
/// low power when patterns are small, and it cannot detect missingness that depends on the
/// unobserved values themselves when this leaves the observed means unchanged.
/// @param data Data matrix with NaN for missing values.
/// @param options EM iteration control.
/// @return Statistic, degrees of freedom, p-value and the EM fit.
/// @throws std::invalid_argument As for em_multivariate_normal(), or if there is only one pattern (df = 0).
/// @throws std::runtime_error If EM fails.
/// @par Example
/// @code
/// const double na = std::nan("");
/// const dm::linalg::DenseMatrix<double> x(6, 2, {1, 2, 2, 1, 3, 4, 4, 3, 5, na, 6, na});
/// const auto t = dm::stats::little_mcar_test(x);
/// std::println("{} {}", t.df, t.patterns); // 1 2
/// @endcode
/// @par References
/// - R. J. A. Little (1988). "A test of missing completely at random for multivariate data with
///   missing values." JASA 83(404), 1198-1202. doi:10.1080/01621459.1988.10478722
/// @see em_multivariate_normal()
[[nodiscard]] LittleMcarTest little_mcar_test(const linalg::DenseMatrix<double>& data, const MvnEmOptions& options = {});

/// @brief Linear regression implied by a multivariate normal mean and covariance.
/// @par Example
/// @code
/// const dm::stats::MvnRegression r;
/// std::println("{}", r.slopes.size()); // 0
/// @endcode
struct MvnRegression {
    double intercept{};          ///< \f$\mu_y-\beta^\top\mu_x\f$.
    std::vector<double> slopes;  ///< \f$\beta=\Sigma_{xx}^{-1}\Sigma_{xy}\f$, in the order of the predictors.
    double residual_variance{};  ///< \f$\sigma_{yy}-\Sigma_{yx}\beta\f$.
    double r_squared{};          ///< \f$1-\text{residual variance}/\sigma_{yy}\f$.
};

/// @brief Regression of one variable on others from a mean vector and covariance matrix.
///
/// Applied to em_multivariate_normal() estimates, this gives the ML regression under MAR when the
/// response and the predictors may both be incomplete, which is the sweep-operator approach of
/// Little and Rubin. With complete data and the ML covariance it reproduces ordinary least squares.
/// @param mean Mean vector.
/// @param covariance Covariance matrix.
/// @param response Index of the response variable.
/// @param predictors Indices of the predictors (distinct, not including the response).
/// @return Intercept, slopes, residual variance and R squared.
/// @throws std::invalid_argument For mismatched sizes or invalid indices.
/// @throws std::runtime_error If the predictor covariance is singular.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> s(2, 2, {1, 0.5, 0.5, 1});
/// const auto r = dm::stats::mvn_regression({0, 2}, s, 1, {0});
/// std::println("{} {} {}", r.intercept, r.slopes[0], r.residual_variance); // 2 0.5 0.75
/// @endcode
/// @see em_multivariate_normal()
[[nodiscard]] MvnRegression mvn_regression(const std::vector<double>& mean, const linalg::DenseMatrix<double>& covariance,
                                           std::size_t response, const std::vector<std::size_t>& predictors);

} // namespace datamunge::stats
