#pragma once

/// @file arima_exact.hpp
/// @brief Exact Gaussian likelihood of seasonal ARIMA models by Kalman filtering, maximum likelihood fits with
/// missing observations, and forecasts.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace datamunge::stats {

/// @brief Orders of a seasonal ARIMA\f$(p,d,q)(P,D,Q)_s\f$ model.
/// @par Example
/// @code
/// const dm::stats::ArimaOrder airline{.d = 1, .q = 1, .seasonal_d = 1, .seasonal_q = 1, .period = 12};
/// std::println("{}", airline.period); // 12
/// @endcode
struct ArimaOrder {
    std::size_t p{};          ///< Regular AR order.
    std::size_t d{};          ///< Regular differences.
    std::size_t q{};          ///< Regular MA order.
    std::size_t seasonal_p{}; ///< Seasonal AR order P.
    std::size_t seasonal_d{}; ///< Seasonal differences D.
    std::size_t seasonal_q{}; ///< Seasonal MA order Q.
    std::size_t period{};     ///< Seasonal period s; required (at least 2) when P, D or Q is positive.
};

/// @brief ARMA coefficients with R's sign conventions.
///
/// The model is \f$\phi(B)\Phi(B^s)(1-B)^d(1-B^s)^D(y_t-\mu)=\theta(B)\Theta(B^s)\varepsilon_t\f$
/// with \f$\phi(B)=1-\sum\phi_iB^i\f$ and \f$\theta(B)=1+\sum\theta_jB^j\f$ (and likewise seasonally).
/// @par Example
/// @code
/// const dm::stats::ArimaCoefficients c{.ar = {0.5}};
/// std::println("{} {}", c.ar.size(), c.ma.size()); // 1 0
/// @endcode
struct ArimaCoefficients {
    std::vector<double> ar;          ///< \f$\phi_1,\dots,\phi_p\f$.
    std::vector<double> ma;          ///< \f$\theta_1,\dots,\theta_q\f$.
    std::vector<double> seasonal_ar; ///< \f$\Phi_1,\dots,\Phi_P\f$.
    std::vector<double> seasonal_ma; ///< \f$\Theta_1,\dots,\Theta_Q\f$.
};

/// @brief Exact Gaussian log likelihood and innovations at given parameters.
/// @par Example
/// @code
/// const dm::stats::ArimaLikelihood l;
/// std::println("{}", l.observations_used); // 0
/// @endcode
struct ArimaLikelihood {
    double log_likelihood{};  ///< Log likelihood with \f$\sigma^2\f$ concentrated out, as `logLik(arima(...))`.
    double sigma2{};          ///< \f$\hat\sigma^2=\sum_t v_t^2/F_t\f$ over the observations that enter the likelihood.

    /// Nonmissing observations minus the \f$d+sD\f$ consumed by differencing.
    std::size_t observations_used{};

    /// Innovations divided by \f$\sqrt{F_t}\f$ (each has variance \f$\sigma^2\f$), as
    /// `residuals(arima(...))`; NaN at missing observations.
    std::vector<double> residuals;
};

/// @brief Evaluates the exact likelihood of a seasonal ARIMA model by the Kalman filter.
///
/// Uses the state space form of `stats::makeARIMA` in R: a Harvey ARMA block of dimension
/// \f$r=\max(p+sP,\,q+sQ+1)\f$ started from its stationary covariance (solved by the
/// doubling algorithm), extended by \f$d+sD\f$ differencing states with a diffuse prior
/// variance \f$\kappa=10^6\f$. Observations whose prediction variance \f$F_t\f$ is \f$10^4\f$ or
/// more are still dominated by the diffuse prior and are left out of the likelihood, as in R.
/// Missing observations (NaN) are skipped by the filter, which integrates them out of the
/// likelihood exactly. With \f$\sigma^2\f$ profiled out,
/// \f$\ell=-\tfrac n2\{\log(2\pi\hat\sigma^2)+1\}-\tfrac n{2\nu}\sum\log F_t\f$, where \f$n\f$ is
/// `observations_used` and \f$\nu\f$ the number of terms in the sums.
/// @param y Series in time order; NaN marks a missing value.
/// @param order Model orders.
/// @param coefficients Coefficients whose lengths match the orders.
/// @param mean Mean \f$\mu\f$ subtracted before filtering (use 0 when differencing).
/// @return Log likelihood, \f$\hat\sigma^2\f$, observation count and standardized residuals.
/// @throws std::invalid_argument For mismatched coefficient lengths, an invalid seasonal period, an infinite
/// value or a nonfinite mean, fewer usable observations than one, or a nonstationary AR part.
/// @par Example
/// @code
/// const std::vector<double> y{.3, -.2, .5, .9, .4, -.1, -.6, .2};
/// const auto l = dm::stats::arima_likelihood(y, {.p = 1}, {.ar = {.4}});
/// std::println("{:.4f} {:.4f}", l.log_likelihood, l.sigma2); // -4.9526 0.1976
/// @endcode
/// @par References
/// - G. Gardner, A. C. Harvey and G. D. A. Phillips (1980). "Algorithm AS 154: An algorithm for exact
///   maximum likelihood estimation of autoregressive-moving average models by means of Kalman filtering".
///   Applied Statistics 29(3), 311-322. doi:10.2307/2346910
/// - R. H. Jones (1980). "Maximum likelihood fitting of ARMA models to time series with missing
///   observations". Technometrics 22(3), 389-395. doi:10.1080/00401706.1980.10486171
/// - J. Durbin and S. J. Koopman (2012). Time Series Analysis by State Space Methods, 2nd ed.
///   Oxford University Press, ch. 4-5. doi:10.1093/acprof:oso/9780199641178.001.0001
/// @see fit_arima_exact()
[[nodiscard]] ArimaLikelihood arima_likelihood(const std::vector<double>& y, const ArimaOrder& order,
                                               const ArimaCoefficients& coefficients, double mean = 0);

/// @brief Options for fit_arima_exact().
/// @par Example
/// @code
/// const dm::stats::ExactArimaOptions options{.order = {.p = 2}};
/// std::println("{}", options.include_mean); // true
/// @endcode
struct ExactArimaOptions {
    ArimaOrder order;            ///< Model orders.
    bool include_mean{true};     ///< Estimate \f$\mu\f$; ignored (no mean) when d or D is positive, as in R.

    /// Optional n by k matrix of finite exogenous regressors \f$x_t\f$; the model becomes
    /// \f$y_t=\mu+x_t^\top\beta+u_t\f$ with ARIMA errors \f$u_t\f$ (empty for none).
    linalg::DenseMatrix<double> regressors;
    std::vector<std::string> regressor_names; ///< Names of the regressor columns; empty gives "x1", "x2", ....
    std::size_t max_iterations{500}; ///< L-BFGS iteration limit.
};

/// @brief A maximum likelihood seasonal ARIMA fit and the filter state needed to forecast.
/// @par Example
/// @code
/// const dm::stats::ExactArimaFit fit;
/// std::println("{}", fit.estimates.size()); // 0
/// @endcode
struct ExactArimaFit {
    ArimaOrder order;                   ///< Fitted orders.
    ArimaCoefficients coefficients;     ///< Estimated ARMA coefficients.
    double mean{};                      ///< Estimated \f$\mu\f$ (0 when not estimated).
    std::vector<double> regression;     ///< Estimated \f$\beta\f$, one per regressor column.

    /// All estimates in R's order: ar, ma, sar, sma, intercept, then regression coefficients.
    std::vector<double> estimates;
    std::vector<std::string> names;     ///< Names "ar1", ..., "ma1", ..., "sar1", ..., "sma1", ..., "intercept", regressor names.

    /// Inverse of the numerically differentiated Hessian of \f$-\ell\f$ (profile over \f$\sigma^2\f$),
    /// as `vcov(arima(...))`.
    linalg::DenseMatrix<double> covariance;
    std::vector<double> standard_errors; ///< Square roots of the covariance diagonal.
    double sigma2{};                    ///< Innovation variance estimate.
    double log_likelihood{};            ///< Maximized log likelihood.
    double aic{};                       ///< \f$-2\ell+2(k+1)\f$ with k estimates.
    std::size_t observations_used{};    ///< As in ArimaLikelihood.
    std::vector<double> residuals;      ///< Standardized innovations at the estimate (NaN where missing).
    bool converged{};                   ///< Whether the largest gradient of \f$-\ell/n\f$ in the search coordinates fell below \f$10^{-4}\f$.

    std::vector<double> final_state;    ///< Filtered state at the last time point (Kalman \f$a_{n|n}\f$).
    linalg::DenseMatrix<double> final_state_covariance; ///< Its covariance divided by \f$\sigma^2\f$.
};

/// @brief Fits a seasonal ARIMA model, optionally with regressors, by exact maximum likelihood.
///
/// Maximizes the profile likelihood of arima_likelihood() over the ARMA coefficients, the mean
/// and any regression coefficients; the errors \f$y_t-\mu-x_t^\top\beta\f$ follow the ARIMA
/// model, so differencing applies to the errors (equivalently to both y and x), as in R's
/// `arima(xreg = ...)`.
/// Each AR and MA polynomial is reparameterized through partial autocorrelations
/// (\f$\tanh\f$ and the Durbin–Levinson recursion), so the search stays in the stationary and
/// invertible region. Regression coefficients start at least squares on the differenced
/// series and are searched in units of their standard errors times \f$\sqrt n\f$. L-BFGS with
/// central-difference gradients starts from zero ARMA coefficients. Missing values are
/// allowed anywhere in y.
/// @param y Series; NaN marks a missing value.
/// @param options Orders, mean term and iteration limit.
/// @return Estimates with standard errors, \f$\sigma^2\f$, log likelihood, AIC, residuals and final state.
/// @throws std::invalid_argument For invalid orders, regressors with the wrong number of rows, nonfinite
/// regressors, a wrong number of names, or too few nonmissing observations for the parameters.
/// @par Example
/// @code
/// std::vector<double> y{0};
/// for (int t = 1; t < 200; ++t)
///     y.push_back(.7 * y.back() + std::sin(37.0 * t)); // deterministic, AR(1)-like input
/// const auto fit = dm::stats::fit_arima_exact(y, {.order = {.p = 1}, .include_mean = false});
/// std::println("{:.3f}", fit.coefficients.ar[0]); // 0.764
/// @endcode
/// @par References
/// - G. E. P. Box, G. M. Jenkins, G. C. Reinsel and G. M. Ljung (2015). Time Series Analysis:
///   Forecasting and Control, 5th ed. Wiley, ch. 7. ISBN 978-1-118-67502-1
/// - O. E. Barndorff-Nielsen and G. Schou (1973). "On the parametrization of autoregressive models
///   by partial autocorrelations". Journal of Multivariate Analysis 3(4), 408-419. doi:10.1016/0047-259X(73)90030-4
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/arima.html
/// @see arima_likelihood(), forecast_arima_exact()
[[nodiscard]] ExactArimaFit fit_arima_exact(const std::vector<double>& y, const ExactArimaOptions& options);

/// @brief Point forecasts and standard errors from a fitted model.
/// @par Example
/// @code
/// const dm::stats::ArimaForecast f;
/// std::println("{}", f.mean.size()); // 0
/// @endcode
struct ArimaForecast {
    std::vector<double> mean;            ///< \f$\hat y_{n+h}\f$, h = 1, ..., horizon.
    std::vector<double> standard_error;  ///< Prediction standard errors \f$\sqrt{\sigma^2 ZP_{n+h}Z^\top}\f$.
};

/// @brief Forecasts by running the Kalman prediction step from the end of the sample.
///
/// Matches `predict(arima_fit, n.ahead, newxreg)` in R, including the effect of missing values
/// at the end of the series. The standard errors treat the coefficients as known.
/// @param fit Model from fit_arima_exact().
/// @param horizon Number of steps ahead.
/// @param future_regressors horizon by k values of the regressors, required when the model has k > 0.
/// @return Forecast means and standard errors.
/// @throws std::invalid_argument When the fit lacks a filter state of the model's dimension, or the future
/// regressors have the wrong shape.
/// @par Example
/// @code
/// std::vector<double> y{0};
/// for (int t = 1; t < 200; ++t)
///     y.push_back(.7 * y.back() + std::sin(37.0 * t));
/// const auto fit = dm::stats::fit_arima_exact(y, {.order = {.p = 1}, .include_mean = false});
/// const auto f = dm::stats::forecast_arima_exact(fit, 2);
/// std::println("{:.3f} {:.3f}", f.mean[1] / f.mean[0], f.standard_error[0]); // 0.764 0.701
/// @endcode
/// @see fit_arima_exact()
[[nodiscard]] ArimaForecast forecast_arima_exact(const ExactArimaFit& fit, std::size_t horizon,
                                                 const linalg::DenseMatrix<double>& future_regressors = {});

} // namespace datamunge::stats
