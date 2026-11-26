#pragma once

/// @file var.hpp
/// @brief Vector autoregressions: least-squares fitting, lag-order selection, stability and forecasts.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <vector>

namespace datamunge::stats {

/// @brief Deterministic terms of a vector autoregression.
/// @par Example
/// @code
/// const auto t = dm::stats::VarTrend::constant;
/// std::println("{}", t == dm::stats::VarTrend::none); // false
/// @endcode
enum class VarTrend {
    none,     ///< No deterministic terms.
    constant, ///< One intercept per equation.
    linear,   ///< Intercept and linear time trend t, with t = 1 at the first row of the data (statsmodels).
};

/// @brief Options for fit_var().
/// @par Example
/// @code
/// const dm::stats::VarOptions options{.lags = 2};
/// std::println("{}", options.lags); // 2
/// @endcode
struct VarOptions {
    std::size_t lags{1};                  ///< Lag order p.
    VarTrend trend{VarTrend::constant};   ///< Deterministic terms.
};

/// @brief A fitted VAR(p) \f$y_t=\nu+\delta t+A_1y_{t-1}+\dots+A_py_{t-p}+u_t\f$, \f$u_t\sim(0,\Sigma_u)\f$.
/// @par Example
/// @code
/// const dm::stats::VarModel m;
/// std::println("{}", m.stable); // false
/// @endcode
struct VarModel {
    std::size_t lags{};         ///< Lag order p.
    std::size_t variables{};    ///< Number of series K.
    std::size_t observations{}; ///< Estimation sample size T (n - p).
    VarTrend trend{VarTrend::constant}; ///< Deterministic terms.

    /// Coefficients, one column per equation and one row per regressor: deterministic terms
    /// (intercept, then trend), then \f$y_{1,t-1},\dots,y_{K,t-1},y_{1,t-2},\dots\f$ (statsmodels `params`).
    linalg::DenseMatrix<double> parameters;
    linalg::DenseMatrix<double> standard_errors; ///< Standard errors in the layout of `parameters`.

    /// \f$(Z^\top Z)^{-1}\f$ for the regressor matrix Z; the coefficient covariance is
    /// \f$\operatorname{Cov}(\hat B_{ak},\hat B_{bl})=(Z^\top Z)^{-1}_{ab}\,\Sigma_{u,kl}\f$.
    linalg::DenseMatrix<double> design_inverse;

    /// Lag matrices \f$A_1,\dots,A_p\f$ with \f$(A_l)_{ij}\f$ the effect of \f$y_{j,t-l}\f$ on \f$y_{i,t}\f$.
    std::vector<linalg::DenseMatrix<double>> lag_coefficients;

    /// Residual covariance with divisor \f$T-Kp-d\f$ for d deterministic terms.
    linalg::DenseMatrix<double> sigma;
    linalg::DenseMatrix<double> sigma_mle;  ///< Residual covariance with divisor T.
    linalg::DenseMatrix<double> residuals;  ///< T by K residuals.
    double log_likelihood{}; ///< Gaussian log likelihood at \f$\hat\Sigma_{\text{MLE}}\f$.
    double aic{};  ///< \f$\ln|\hat\Sigma_{\text{MLE}}|+2N/T\f$, \f$N=pK^2+dK\f$ free parameters.
    double bic{};  ///< \f$\ln|\hat\Sigma_{\text{MLE}}|+N\ln T/T\f$.
    double hqic{}; ///< \f$\ln|\hat\Sigma_{\text{MLE}}|+2N\ln\ln T/T\f$.
    double fpe{};  ///< Final prediction error \f$((T+Kp+d)/(T-Kp-d))^K|\hat\Sigma_{\text{MLE}}|\f$.

    /// Moduli of the companion-matrix eigenvalues, largest first; the VAR is stable when all are below 1.
    std::vector<double> companion_moduli;
    bool stable{}; ///< Whether every companion eigenvalue lies strictly inside the unit circle.
    linalg::DenseMatrix<double> history; ///< Last p observations (p by K), oldest first, for forecasting.
};

/// @brief Fits a VAR(p) by equation-wise least squares (equivalently Gaussian conditional ML).
///
/// Matches `statsmodels.tsa.api.VAR(y).fit(p, trend)`.
/// @param y n by K matrix of finite observations, one row per time point.
/// @param options Lag order and deterministic terms.
/// @return Coefficients, standard errors, covariances, information criteria and stability.
/// @throws std::invalid_argument For nonfinite data, no variables, a singular design, or too few observations.
/// @par Example
/// @code
/// std::mt19937 g(1);
/// dm::linalg::DenseMatrix<double> y(200, 2);
/// for (std::size_t t = 1; t < 200; ++t) {
///     y(t, 0) = .5 * y(t - 1, 0) + g() / 4294967296.0;
///     y(t, 1) = .3 * y(t - 1, 0) + .2 * y(t - 1, 1) + g() / 4294967296.0;
/// }
/// const auto m = dm::stats::fit_var(y, {.lags = 1});
/// std::println("{:.2f} {}", m.lag_coefficients[0](1, 0), m.stable); // 0.28 true
/// @endcode
/// @par References
/// - H. Lütkepohl (2005). New Introduction to Multiple Time Series Analysis. Springer.
///   doi:10.1007/978-3-540-27752-1
/// - C. A. Sims (1980). "Macroeconomics and reality". Econometrica 48(1), 1-48. doi:10.2307/1912017
/// @see select_var_order(), forecast_var()
[[nodiscard]] VarModel fit_var(const linalg::DenseMatrix<double>& y, const VarOptions& options = {});

/// @brief Information criteria for VAR orders 0..max on a common sample.
/// @par Example
/// @code
/// const dm::stats::VarOrderSelection s;
/// std::println("{}", s.aic.size()); // 0
/// @endcode
struct VarOrderSelection {
    std::vector<double> aic;  ///< AIC for p = 0, 1, ..., max.
    std::vector<double> bic;  ///< BIC (Schwarz).
    std::vector<double> hqic; ///< Hannan–Quinn.
    std::vector<double> fpe;  ///< Final prediction error.
    std::size_t aic_order{};  ///< Order minimizing AIC.
    std::size_t bic_order{};  ///< Order minimizing BIC.
    std::size_t hqic_order{}; ///< Order minimizing HQIC.
    std::size_t fpe_order{};  ///< Order minimizing FPE.
};

/// @brief Chooses a VAR lag order by information criteria.
///
/// Every order is fitted to the same last \f$n-p_{\max}\f$ observations so the criteria are
/// comparable. Matches `statsmodels` `VAR(y).select_order(max_lags, trend)`.
/// @param y n by K observations.
/// @param max_lags Largest order considered.
/// @param trend Deterministic terms.
/// @return Criteria by order and the minimizing orders.
/// @throws std::invalid_argument As fit_var(), for the largest order.
/// @par Example
/// @code
/// std::mt19937 g(1);
/// dm::linalg::DenseMatrix<double> y(200, 1);
/// for (std::size_t t = 2; t < 200; ++t)
///     y(t, 0) = .5 * y(t - 1, 0) - .4 * y(t - 2, 0) + g() / 4294967296.0;
/// std::println("{}", dm::stats::select_var_order(y, 6).bic_order); // 2
/// @endcode
/// @see fit_var()
[[nodiscard]] VarOrderSelection select_var_order(const linalg::DenseMatrix<double>& y, std::size_t max_lags,
                                                 VarTrend trend = VarTrend::constant);

/// @brief Reference distribution for granger_causality().
/// @par Example
/// @code
/// const auto k = dm::stats::GrangerStatistic::wald;
/// std::println("{}", k == dm::stats::GrangerStatistic::f); // false
/// @endcode
enum class GrangerStatistic {
    f,    ///< Wald statistic divided by the restrictions, against \f$F(r,\,K(T-Kp-d))\f$.
    wald, ///< Wald statistic against \f$\chi^2_r\f$.
};

/// @brief Result of granger_causality().
/// @par Example
/// @code
/// const dm::stats::GrangerTest g;
/// std::println("{}", g.restrictions); // 0
/// @endcode
struct GrangerTest {
    double statistic{};          ///< F or Wald statistic.
    double p_value{};            ///< Upper-tail p-value.
    std::size_t restrictions{};  ///< Zero restrictions r = p × |caused| × |causing|.
    double denominator_df{};     ///< F denominator degrees of freedom \f$K(T-Kp-d)\f$ (0 for Wald).
};

/// @brief Tests whether lags of the `causing` series help predict the `caused` series in a VAR.
///
/// The null sets every coefficient of \f$y_{j,t-l}\f$ (j causing, l = 1..p) to zero in the
/// equations of the caused series, conditional on all other lags in the system. The Wald statistic
/// \f$(C\hat\beta)^\top[C(\,(Z^\top Z)^{-1}\otimes\hat\Sigma_u)C^\top]^{-1}C\hat\beta\f$ uses the full
/// cross-equation covariance, so blocks spanning several equations are tested jointly. Matches
/// `statsmodels` `VARResults.test_causality`.
/// @param model Fitted VAR with at least one lag.
/// @param caused Indices of the series whose equations are restricted.
/// @param causing Indices of the series whose lags are excluded; disjoint from `caused`.
/// @param kind F (default) or chi-squared reference.
/// @return Statistic, p-value and degrees of freedom.
/// @throws std::invalid_argument For a VAR(0), empty, overlapping, repeated or out-of-range indices.
/// @par Example
/// @code
/// std::mt19937 g(1);
/// dm::linalg::DenseMatrix<double> y(200, 2);
/// for (std::size_t t = 1; t < 200; ++t) {
///     y(t, 0) = .5 * y(t - 1, 0) + g() / 4294967296.0;
///     y(t, 1) = .3 * y(t - 1, 0) + g() / 4294967296.0; // series 0 drives series 1
/// }
/// const auto m = dm::stats::fit_var(y, {.lags = 1});
/// std::println("{} {}", dm::stats::granger_causality(m, {1}, {0}).p_value < .01,
///              dm::stats::granger_causality(m, {0}, {1}).p_value < .01); // true false
/// @endcode
/// @par References
/// - C. W. J. Granger (1969). "Investigating causal relations by econometric models and
///   cross-spectral methods". Econometrica 37(3), 424-438. doi:10.2307/1912791
/// - H. Lütkepohl (2005). New Introduction to Multiple Time Series Analysis, Section 3.6. Springer.
/// @see fit_var()
[[nodiscard]] GrangerTest granger_causality(const VarModel& model, const std::vector<std::size_t>& caused,
                                            const std::vector<std::size_t>& causing,
                                            GrangerStatistic kind = GrangerStatistic::f);

/// @brief Point forecasts and forecast-error covariances of a VAR.
/// @par Example
/// @code
/// const dm::stats::VarForecast f;
/// std::println("{}", f.covariance.size()); // 0
/// @endcode
struct VarForecast {
    linalg::DenseMatrix<double> mean; ///< Horizon by K point forecasts.

    /// MSE matrices \f$\sum_{i<h}\Phi_i\hat\Sigma_u\Phi_i^\top\f$ for h = 1, ..., horizon, with
    /// \f$\Phi_i\f$ the moving-average coefficients (parameter uncertainty ignored).
    std::vector<linalg::DenseMatrix<double>> covariance;
};

/// @brief Iterates the fitted VAR forward from the end of the estimation sample.
/// @param model Fitted model.
/// @param horizon Steps ahead (positive).
/// @return Point forecasts and MSE matrices; matches `statsmodels` `forecast` and `forecast_cov`.
/// @throws std::invalid_argument For a zero horizon.
/// @par Example
/// @code
/// dm::linalg::DenseMatrix<double> y(50, 1);
/// for (std::size_t t = 1; t < 50; ++t)
///     y(t, 0) = 2 + .5 * y(t - 1, 0) + std::sin(3.0 * t);
/// const auto f = dm::stats::forecast_var(dm::stats::fit_var(y), 30);
/// std::println("{:.1f}", f.mean(29, 0)); // 3.9
/// @endcode
/// @see fit_var()
[[nodiscard]] VarForecast forecast_var(const VarModel& model, std::size_t horizon);

} // namespace datamunge::stats
