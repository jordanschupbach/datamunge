#pragma once

/// @file spatial_regression.hpp
/// @brief Maximum-likelihood spatial lag (SAR) and spatial error (SEM) regression.

#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/stats/spatial_autocorrelation.hpp>

#include <cstddef>
#include <limits>
#include <vector>

namespace datamunge::stats {

/// @brief Options for spatial_lag_regression() and spatial_error_regression().
/// @par Example
/// @code
/// const dm::stats::SpatialRegressionOptions o{.lower = -0.9, .upper = 0.99};
/// std::println("{}", o.upper); // 0.99
/// @endcode
struct SpatialRegressionOptions {
    double lower{std::numeric_limits<double>::quiet_NaN()}; ///< Lower search bound (NaN: 1 / smallest eigenvalue of W, or -1).
    double upper{std::numeric_limits<double>::quiet_NaN()}; ///< Upper search bound (NaN: 1 / largest eigenvalue of W, or 1).
};

/// @brief Direct, indirect and total effects of one regressor in a spatial lag model.
/// @par Example
/// @code
/// dm::stats::SpatialImpact i{.direct = -1.09, .indirect = -0.73, .total = -1.81};
/// std::println("{:.2f}", i.direct + i.indirect); // -1.82
/// @endcode
struct SpatialImpact {
    double direct{};   ///< Average own-location effect, beta_k tr((I - rho W)^{-1}) / n.
    double indirect{}; ///< Average spillover, total - direct.
    double total{};    ///< Average total effect, beta_k 1'(I - rho W)^{-1} 1 / n.
};

/// @brief Fit of a spatial lag or spatial error regression.
/// @par Example
/// @code
/// dm::stats::SpatialRegressionResult r;
/// std::println("{}", r.coefficients.size()); // 0
/// @endcode
struct SpatialRegressionResult {
    double spatial_parameter{};  ///< rho (lag model) or lambda (error model).
    double spatial_se{};         ///< Asymptotic standard error of the spatial parameter.
    double spatial_z{};          ///< Its Wald z statistic.
    double spatial_p_value{};    ///< Two-sided normal p-value of the Wald test.
    std::vector<double> coefficients;    ///< Regression coefficients beta.
    std::vector<double> standard_errors; ///< Asymptotic standard errors of beta.
    std::vector<double> z;               ///< Wald z statistics of beta.
    std::vector<double> p_values;        ///< Two-sided normal p-values of beta.
    double sigma2{};             ///< ML innovation variance.
    double log_likelihood{};     ///< Maximised log-likelihood.
    double ols_log_likelihood{}; ///< Log-likelihood of the nonspatial OLS fit.
    double lr_statistic{};       ///< 2 (log_likelihood - ols_log_likelihood).
    double lr_p_value{};         ///< Chi-squared(1) p-value of the likelihood-ratio test.
    double aic{};                ///< -2 log_likelihood + 2 (p + 2).
    double lower{};              ///< Lower bound of the search interval used.
    double upper{};              ///< Upper bound of the search interval used.
    std::vector<double> fitted;    ///< y minus the residuals: rho W y + X beta (lag), y - (I - lambda W)(y - X beta) (error).
    std::vector<double> residuals; ///< Estimated innovations epsilon, as in spatialreg.
    std::vector<SpatialImpact> impacts; ///< Lag model only: one per column of X (ignore the intercept's).
};

/// @brief Spatial lag (simultaneous autoregressive) regression \f$y=\rho Wy+X\beta+\varepsilon\f$.
///
/// The concentrated log-likelihood
/// \f$-\frac n2\log(2\pi\hat\sigma^2(\rho))-\frac n2+\log|I-\rho W|\f$ is maximised over rho by a grid and
/// golden-section search. The log-determinant uses the eigenvalues of W when W is symmetric or similar
/// to a symmetric matrix (for example, a row-standardised symmetric neighbour matrix) and an LU
/// factorisation otherwise. Standard errors come from the analytic information matrix of
/// (beta, rho, sigma^2) (Anselin 1988), and impacts follow LeSage and Pace (2009). Matches
/// `spatialreg::lagsarlm(method = "eigen")` and, for row-standardised W, `impacts()`; for other W the
/// total effect uses the mean row sum of \f$(I-\rho W)^{-1}\f$ rather than spatialreg's
/// \f$1/(1-\rho)\f$. Fitted values are \f$\rho Wy+X\beta\f$, so
/// the residuals are the estimated innovations, as in spatialreg.
/// @param y Response (length n).
/// @param x n-by-p design matrix (include a column of ones for an intercept).
/// @param weights n-by-n spatial weights with a zero diagonal.
/// @param options Search interval.
/// @return Estimates, standard errors, likelihoods and impacts.
/// @throws std::invalid_argument For mismatched sizes, nonfinite values, a nonzero diagonal, n <= p + 1,
/// or an empty interval.
/// @throws std::runtime_error If X is rank deficient or the information matrix is singular.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> xy(6, 1, std::vector<double>{0, 1, 2, 3, 4, 5});
/// const auto w = dm::stats::row_standardize(dm::stats::distance_band_weights(xy, 0, 1));
/// const dm::linalg::DenseMatrix<double> x(6, 1, std::vector<double>(6, 1.0));
/// const auto fit = dm::stats::spatial_lag_regression({1, 2, 2.5, 3, 2.2, 1.5}, x, w);
/// std::println("{}", fit.spatial_parameter > 0); // true
/// @endcode
SpatialRegressionResult spatial_lag_regression(const std::vector<double>& y, const linalg::DenseMatrix<double>& x,
                                               const linalg::DenseMatrix<double>& weights,
                                               const SpatialRegressionOptions& options = {});

/// @brief Spatial error regression \f$y=X\beta+u,\ u=\lambda Wu+\varepsilon\f$.
///
/// For each lambda, beta is the GLS estimate from the filtered data (I - lambda W)y and
/// (I - lambda W)X; the concentrated log-likelihood is maximised as in spatial_lag_regression().
/// The covariance of beta is \f$\hat\sigma^2[X^\top(I-\lambda W)^\top(I-\lambda W)X]^{-1}\f$ and lambda's
/// standard error comes from the (lambda, sigma^2) information block. Matches
/// `spatialreg::errorsarlm(method = "eigen")`. The residuals are the innovations
/// \f$(I-\lambda W)(y-X\beta)\f$ and the fitted values are y minus them; the trend alone is X beta.
/// @param y Response (length n).
/// @param x n-by-p design matrix.
/// @param weights n-by-n spatial weights with a zero diagonal.
/// @param options Search interval.
/// @return Estimates, standard errors and likelihoods (no impacts).
/// @throws std::invalid_argument As for spatial_lag_regression().
/// @throws std::runtime_error As for spatial_lag_regression().
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> xy(6, 1, std::vector<double>{0, 1, 2, 3, 4, 5});
/// const auto w = dm::stats::row_standardize(dm::stats::distance_band_weights(xy, 0, 1));
/// const dm::linalg::DenseMatrix<double> x(6, 1, std::vector<double>(6, 1.0));
/// const auto fit = dm::stats::spatial_error_regression({1, 2, 2.5, 3, 2.2, 1.5}, x, w);
/// std::println("{}", fit.impacts.empty()); // true
/// @endcode
SpatialRegressionResult spatial_error_regression(const std::vector<double>& y, const linalg::DenseMatrix<double>& x,
                                                 const linalg::DenseMatrix<double>& weights,
                                                 const SpatialRegressionOptions& options = {});

} // namespace datamunge::stats
