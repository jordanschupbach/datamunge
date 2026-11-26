#pragma once

/// @file cointegration.hpp
/// @brief Johansen cointegration rank tests and vector error-correction models (VECM).

#include <datamunge/linalg/dense_matrix.hpp>

#include <array>
#include <cstddef>
#include <vector>

namespace datamunge::stats {

/// @brief Deterministic terms of the Johansen test, as statsmodels' `det_order`.
/// @par Example
/// @code
/// const auto d = dm::stats::JohansenDeterministic::constant;
/// std::println("{}", d == dm::stats::JohansenDeterministic::none); // false
/// @endcode
enum class JohansenDeterministic {
    none,     ///< No deterministic terms (det_order = -1).
    constant, ///< Demean levels and differences (det_order = 0).
    linear,   ///< Remove a linear trend from levels, then demean (det_order = 1).
};

/// @brief Options for johansen_test().
/// @par Example
/// @code
/// const dm::stats::JohansenOptions options{.lagged_differences = 2};
/// std::println("{}", options.lagged_differences); // 2
/// @endcode
struct JohansenOptions {
    JohansenDeterministic deterministic{JohansenDeterministic::constant}; ///< Deterministic terms.
    std::size_t lagged_differences{1}; ///< Lagged differences \f$\Delta y_{t-1},\dots,\Delta y_{t-k}\f$ partialled out.
};

/// @brief Result of johansen_test().
/// @par Example
/// @code
/// const dm::stats::JohansenTest j;
/// std::println("{}", j.trace_rank); // 0
/// @endcode
struct JohansenTest {
    std::vector<double> eigenvalues; ///< Squared canonical correlations \f$\hat\lambda_1\ge\dots\ge\hat\lambda_K\f$.

    /// K by K eigenvectors (columns, matching `eigenvalues`), scaled so \f$V^\top S_{kk}V=I\f$; the
    /// first nonzero entry of each column is positive.
    linalg::DenseMatrix<double> eigenvectors;

    /// Trace statistics \f$-T\sum_{i>r}\ln(1-\hat\lambda_i)\f$ for H0: rank \f$\le r\f$, r = 0..K-1.
    std::vector<double> trace;

    /// Maximum-eigenvalue statistics \f$-T\ln(1-\hat\lambda_{r+1})\f$ for H0: rank r against r + 1.
    std::vector<double> max_eigenvalue;

    /// Osterwald-Lenum 90%, 95% and 99% critical values for `trace` (NaN beyond 12 free dimensions).
    std::vector<std::array<double,3>> trace_critical;
    std::vector<std::array<double,3>> max_eigenvalue_critical; ///< Critical values for `max_eigenvalue`.
    std::size_t trace_rank{};          ///< First r whose trace test is not rejected at 5% (K if all are).
    std::size_t max_eigenvalue_rank{}; ///< First r whose maximum-eigenvalue test is not rejected at 5%.
    std::size_t observations{};        ///< Effective sample size T.
};

/// @brief Johansen reduced-rank tests for the number of cointegrating relations.
///
/// Partials the lagged differences (and the deterministic terms) out of \f$\Delta y_t\f$ and
/// \f$y_{t-1}\f$, forms the moment matrices \f$S_{00},S_{0k},S_{kk}\f$ of the residuals and solves
/// \f$|\lambda S_{kk}-S_{k0}S_{00}^{-1}S_{0k}|=0\f$. Follows `statsmodels` `coint_johansen`, including its
/// detrending conventions and tables.
/// @param y n by K matrix of finite observations in levels.
/// @param options Deterministic terms and lagged differences.
/// @return Eigenvalues, eigenvectors, trace and maximum-eigenvalue statistics and critical values.
/// @throws std::invalid_argument For nonfinite data, fewer than two series, or too few observations.
/// @par Example
/// @code
/// std::mt19937 g(2);
/// dm::linalg::DenseMatrix<double> y(200, 2);
/// for (std::size_t t = 1; t < 200; ++t) {
///     y(t, 0) = y(t - 1, 0) + g() / 4294967296.0 - .5;  // random walk
///     y(t, 1) = y(t, 0) + g() / 4294967296.0 - .5;      // cointegrated with it
/// }
/// const auto j = dm::stats::johansen_test(y);
/// std::println("{}", j.trace_rank); // 1
/// @endcode
/// @par References
/// - S. Johansen (1991). "Estimation and hypothesis testing of cointegration vectors in Gaussian
///   vector autoregressive models". Econometrica 59(6), 1551-1580. doi:10.2307/2938278
/// - M. Osterwald-Lenum (1992). "A note with quantiles of the asymptotic distribution of the maximum
///   likelihood cointegration rank test statistics". OBES 54(3), 461-472.
///   doi:10.1111/j.1468-0084.1992.tb00013.x
/// @see fit_vecm()
[[nodiscard]] JohansenTest johansen_test(const linalg::DenseMatrix<double>& y, const JohansenOptions& options = {});

/// @brief Placement of a deterministic term in a VECM.
/// @par Example
/// @code
/// const auto p = dm::stats::VecmTerm::inside;
/// std::println("{}", p == dm::stats::VecmTerm::outside); // false
/// @endcode
enum class VecmTerm {
    none,    ///< Term absent.
    inside,  ///< Restricted to the cointegrating relations (statsmodels "ci" / "li").
    outside, ///< Unrestricted in the short-run equations (statsmodels "co" / "lo").
};

/// @brief Options for fit_vecm().
/// @par Example
/// @code
/// const dm::stats::VecmOptions options{.rank = 1, .lagged_differences = 2};
/// std::println("{}", options.rank); // 1
/// @endcode
struct VecmOptions {
    std::size_t rank{1};               ///< Cointegrating rank r, 1 to K - 1.
    std::size_t lagged_differences{1}; ///< Short-run lags k (the levels VAR has order k + 1).
    VecmTerm constant{VecmTerm::outside}; ///< Constant placement.
    VecmTerm trend{VecmTerm::none};       ///< Linear-trend placement.
};

/// @brief A fitted VECM \f$\Delta y_t=\alpha(\beta^\top y_{t-1}+\beta_d^\top d^{\,i}_t)+\sum_{i\le k}\Gamma_i\Delta y_{t-i}+C d^{\,o}_t+u_t\f$.
/// @par Example
/// @code
/// const dm::stats::VecmModel m;
/// std::println("{}", m.rank); // 0
/// @endcode
struct VecmModel {
    std::size_t rank{};               ///< Cointegrating rank r.
    std::size_t lagged_differences{}; ///< Short-run lags k.
    std::size_t variables{};          ///< Number of series K.
    std::size_t observations{};       ///< Effective sample size T = n - k - 1.
    VecmTerm constant{VecmTerm::none}; ///< Constant placement.
    VecmTerm trend{VecmTerm::none};    ///< Trend placement.
    linalg::DenseMatrix<double> alpha; ///< K by r adjustment (loading) coefficients.

    /// K by r cointegrating vectors, normalized so the first r rows form the identity.
    linalg::DenseMatrix<double> beta;

    /// Rows for the restricted constant then trend (as present), by r; empty without inside terms.
    linalg::DenseMatrix<double> beta_deterministic;
    linalg::DenseMatrix<double> gamma; ///< K by Kk short-run matrices \f$[\Gamma_1,\dots,\Gamma_k]\f$.

    /// K by (constant, trend) unrestricted deterministic coefficients C; empty without outside terms.
    linalg::DenseMatrix<double> deterministic;
    linalg::DenseMatrix<double> alpha_standard_errors;  ///< Standard errors of `alpha`, given beta.
    linalg::DenseMatrix<double> beta_standard_errors;   ///< Standard errors of `beta` (zero for the identity rows).
    linalg::DenseMatrix<double> beta_deterministic_standard_errors; ///< Standard errors of `beta_deterministic`.
    linalg::DenseMatrix<double> gamma_standard_errors;  ///< Standard errors of `gamma`, given beta.
    linalg::DenseMatrix<double> deterministic_standard_errors; ///< Standard errors of `deterministic`.
    linalg::DenseMatrix<double> sigma;     ///< Residual covariance with divisor T.
    linalg::DenseMatrix<double> residuals; ///< T by K residuals.
    std::vector<double> eigenvalues;       ///< All K canonical-correlation eigenvalues, decreasing.
    double log_likelihood{};               ///< Maximized Gaussian log likelihood.

    /// Levels-VAR matrices \f$A_1=I+\alpha\beta^\top+\Gamma_1,\ A_i=\Gamma_i-\Gamma_{i-1},\ A_{k+1}=-\Gamma_k\f$.
    std::vector<linalg::DenseMatrix<double>> var_coefficients;
    linalg::DenseMatrix<double> history; ///< Last k + 1 observations, oldest first.
};

/// @brief Fits a VECM of given cointegrating rank by Johansen's reduced-rank maximum likelihood.
///
/// Matches `statsmodels.tsa.vector_ar.vecm.VECM(y, k_ar_diff, coint_rank, deterministic).fit()`,
/// including its trend indexing and standard errors. Beta standard errors follow Lütkepohl (2005,
/// p. 297) with the cointegrating-vector elements laid out by row and column.
/// @param y n by K matrix of finite observations in levels.
/// @param options Rank, lags and deterministic terms.
/// @return Loadings, cointegrating vectors, short-run matrices, covariances and standard errors.
/// @throws std::invalid_argument For nonfinite data, rank outside 1..K-1, or too few observations.
/// @par Example
/// @code
/// std::mt19937 g(1);
/// dm::linalg::DenseMatrix<double> y(300, 2);
/// for (std::size_t t = 1; t < 300; ++t) {
///     y(t, 0) = y(t - 1, 0) + g() / 4294967296.0 - .5;
///     y(t, 1) = 2 * y(t, 0) + g() / 4294967296.0 - .5;
/// }
/// const auto m = dm::stats::fit_vecm(y, {.rank = 1});
/// std::println("{:.2f}", m.beta(1, 0)); // -0.51
/// @endcode
/// @par References
/// - S. Johansen (1995). Likelihood-Based Inference in Cointegrated Vector Autoregressive Models.
///   Oxford University Press. doi:10.1093/0198774508.001.0001
/// - H. Lütkepohl (2005). New Introduction to Multiple Time Series Analysis, Chapter 7. Springer.
/// @see johansen_test(), forecast_vecm()
[[nodiscard]] VecmModel fit_vecm(const linalg::DenseMatrix<double>& y, const VecmOptions& options = {});

/// @brief Point forecasts from the levels-VAR representation of a fitted VECM.
/// @param model Fitted model.
/// @param horizon Steps ahead (positive).
/// @return Horizon by K forecasts; matches `statsmodels` `VECMResults.predict`.
/// @throws std::invalid_argument For a zero horizon.
/// @par Example
/// @code
/// std::mt19937 g(2);
/// dm::linalg::DenseMatrix<double> y(100, 2);
/// for (std::size_t t = 1; t < 100; ++t) {
///     y(t, 0) = y(t - 1, 0) + g() / 4294967296.0 - .5;
///     y(t, 1) = y(t, 0) + g() / 4294967296.0 - .5;
/// }
/// const auto f = dm::stats::forecast_vecm(dm::stats::fit_vecm(y), 3);
/// std::println("{}", f.rows()); // 3
/// @endcode
/// @see fit_vecm()
[[nodiscard]] linalg::DenseMatrix<double> forecast_vecm(const VecmModel& model, std::size_t horizon);

} // namespace datamunge::stats
