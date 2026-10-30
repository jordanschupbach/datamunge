#pragma once

/// @file arima.hpp
/// @brief Seasonal ARIMA(p, d, q)(P, D, Q)_s model fit by conditional sum of squares.
///
/// Declares ARIMAOptions (model orders and optimizer settings) and the ARIMA class,
/// which differences a series, fits its AR/MA coefficients with differential evolution,
/// and produces forecasts with psi-weight standard errors.

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace datamunge::stats {

/// @brief Model orders and optimizer settings for an ARIMA fit.
///
/// The regular orders are (p, d, q) and the seasonal orders are (P, D, Q) with period s.
/// Leave every seasonal field at zero for a plain ARIMA(p, d, q) model.
struct ARIMAOptions {
    std::size_t p{0}; ///< Non-seasonal AR order (p).
    std::size_t d{0}; ///< Non-seasonal differencing order (d).
    std::size_t q{0}; ///< Non-seasonal MA order (q).

    std::size_t seasonal_p{0};      ///< Seasonal AR order (P).
    std::size_t seasonal_d{0};      ///< Seasonal differencing order (D).
    std::size_t seasonal_q{0};      ///< Seasonal MA order (Q).
    std::size_t seasonal_period{0}; ///< Seasonal period (s); must be nonzero when P, D, or Q is nonzero.

    /// @brief Fit an intercept (mean of the fully-differenced series). Only meaningful when
    ///        d == 0 && seasonal_d == 0 -- matching R's arima(), a mean term is not fit on a
    ///        series that has already been differenced to stationarity.
    bool include_mean{true};

    std::size_t de_population_size{60};  ///< Population size of the differential-evolution optimizer.
    std::size_t de_max_generations{400}; ///< Maximum number of differential-evolution generations.
    double coefficient_bound{1.5}; ///< Box constraint [-bound, bound] applied to every AR/MA coefficient during optimization.
    std::uint64_t seed{42}; ///< Random seed for the differential-evolution optimizer.
};

/// @brief A Box-Jenkins seasonal ARIMA(p, d, q)(P, D, Q)_s model, fit to a single numeric
///        series by conditional sum of squares (CSS): the series is differenced (regular
///        d times, then seasonal D times with period s) to stationarity, the combined
///        AR/MA polynomials (regular composed with seasonal, via polynomial multiplication)
///        are estimated by minimizing the conditional residual sum of squares with
///        DifferentialEvolution, and forecasts are produced by recursing the fitted
///        difference equation forward and re-integrating through the differencing steps.
///
///        Setting seasonal_period == 0 (the default) reduces this to a plain, non-seasonal
///        ARIMA(p, d, q) model -- i.e. SARIMA is the general case, ARIMA the special case,
///        exactly as in R's arima()/statsmodels' SARIMAX().
///
///        Note: CSS estimation approximates the exact Gaussian likelihood by conditioning
///        on the first max(p + P*s, q + Q*s) observations rather than filtering the full
///        series (e.g. via a Kalman filter); it is a standard, textbook ARIMA fitting
///        method (R's arima(method = "CSS")) but will differ slightly from exact-ML fits.
class ARIMA {
 public:
    /// @brief Fit the model to a series.
    /// @param y The observed series, in time order.
    /// @param options Model orders and optimizer settings.
    /// @throws std::invalid_argument If a seasonal order is set while seasonal_period is zero,
    ///         or if the series is too short for the requested differencing or AR order.
    ARIMA(const std::vector<double>& y, ARIMAOptions options = {});

    /// @brief The options the model was fit with.
    /// @return The stored ARIMAOptions.
    [[nodiscard]] const ARIMAOptions& options() const { return options_; }

    /// @brief Fitted non-seasonal AR coefficients \f$\phi_1,\dots,\phi_p\f$.
    /// @return A vector of length p; the AR polynomial is \f$1-\sum_i\phi_i B^i\f$.
    [[nodiscard]] const std::vector<double>& ar_coefficients() const { return phi_; }
    /// @brief Fitted non-seasonal MA coefficients \f$\theta_1,\dots,\theta_q\f$.
    /// @return A vector of length q; the MA polynomial is \f$1+\sum_j\theta_j B^j\f$.
    [[nodiscard]] const std::vector<double>& ma_coefficients() const { return theta_; }
    /// @brief Fitted seasonal AR coefficients \f$\Phi_1,\dots,\Phi_P\f$.
    /// @return A vector of length P; the polynomial is \f$1-\sum_i\Phi_i B^{is}\f$.
    [[nodiscard]] const std::vector<double>& seasonal_ar_coefficients() const { return seasonal_phi_; }
    /// @brief Fitted seasonal MA coefficients \f$\Theta_1,\dots,\Theta_Q\f$.
    /// @return A vector of length Q; the polynomial is \f$1+\sum_j\Theta_j B^{js}\f$.
    [[nodiscard]] const std::vector<double>& seasonal_ma_coefficients() const { return seasonal_theta_; }
    /// @brief Fitted mean of the differenced series.
    /// @return The mean, or 0 when no mean term was fit (include_mean is false, or d or D is nonzero).
    [[nodiscard]] double mean() const { return mean_; }

    /// @brief Length of the input series.
    /// @return The number of observations passed to the constructor.
    [[nodiscard]] std::size_t observations() const { return n_; }
    /// @brief Number of residuals that enter the sum of squares.
    /// @return The differenced series length minus \f$p+sP\f$ (the conditioned-on presample).
    [[nodiscard]] std::size_t n_used() const { return n_used_; }
    /// @brief Innovation variance estimate.
    /// @return The CSS residual sum of squares divided by n_used().
    [[nodiscard]] double sigma2() const { return sigma2_; }
    /// @brief Gaussian log-likelihood at the CSS estimate, over n_used() residuals.
    /// @return \f$-\tfrac{n}{2}(\log 2\pi + \log\sigma^2 + 1)\f$ with \f$n\f$ = n_used().
    [[nodiscard]] double log_likelihood() const { return log_likelihood_; }
    /// @brief Akaike information criterion.
    /// @return \f$-2\ell + 2k\f$, where \f$k\f$ counts the coefficients, the mean (if fit) and \f$\sigma^2\f$.
    [[nodiscard]] double aic() const { return aic_; }
    /// @brief Bayesian information criterion.
    /// @return \f$-2\ell + k\log n\f$, with \f$k\f$ as in aic() and \f$n\f$ = n_used().
    [[nodiscard]] double bic() const { return bic_; }

    /// @brief One-step-ahead fitted values on the original scale, aligned to the input
    ///        series (the first `observations() - n_used()` entries -- the presample points
    ///        conditioned on by CSS -- are copies of the observed value, matching a
    ///        residual of exactly zero there rather than an undefined value).
    [[nodiscard]] const std::vector<double>& fitted_values() const { return fitted_; }
    /// @brief One-step-ahead residuals on the original scale (zero for the presample points).
    [[nodiscard]] const std::vector<double>& residuals() const { return residuals_; }

    /// @brief Point forecasts for the next @p horizon periods, on the original scale.
    /// @param horizon Number of steps ahead to forecast.
    /// @return A vector of length @p horizon (empty when @p horizon is 0).
    [[nodiscard]] std::vector<double> forecast(std::size_t horizon) const;

    /// @brief AR coefficients of the stationary (differenced) model after multiplying the regular
    ///        and seasonal polynomials, \f$w_t=\sum_i a_i w_{t-i}+\varepsilon_t+\sum_j b_j\varepsilon_{t-j}\f$.
    /// @return The \f$a_i\f$, \f$i=1,\dots,p+sP\f$ (zeros where no lag enters).
    [[nodiscard]] const std::vector<double>& expanded_ar_coefficients() const { return combined_ar_; }

    /// @brief MA coefficients \f$b_j\f$ of the stationary model, after multiplying the regular and
    ///        seasonal MA polynomials; for the airline model \f$(\theta,0,\dots,0,\Theta,\theta\Theta)\f$.
    /// @return The \f$b_j\f$, \f$j=1,\dots,q+sQ\f$.
    [[nodiscard]] const std::vector<double>& expanded_ma_coefficients() const { return combined_ma_; }

    /// @brief The first @p count \f$\psi\f$-weights of the full model, differencing included:
    ///        \f$y_t=\sum_{j\ge0}\psi_j\varepsilon_{t-j}\f$ formally, with \f$\psi_0=1\f$.
    ///
    /// The h-step forecast error is \f$\sum_{j<h}\psi_j\varepsilon_{t+h-j}\f$, so its variance is
    /// \f$\sigma^2\sum_{j<h}\psi_j^2\f$; forecast_with_intervals() uses exactly these weights.
    /// @param count Number of weights.
    /// @return \f$\psi_0,\dots,\psi_{count-1}\f$.
    [[nodiscard]] std::vector<double> psi_weights(std::size_t count) const;

    /// @brief Point forecasts and their standard errors (from the psi-weight / MA(infinity)
    ///        expansion of the full, non-stationary AR/MA representation of y), on the
    ///        original scale.
    /// @param horizon Number of steps ahead to forecast.
    /// @return A pair (point forecasts, standard errors), each of length @p horizon.
    [[nodiscard]] std::pair<std::vector<double>, std::vector<double>> forecast_with_intervals(
        std::size_t horizon) const;

 private:
    /// @brief Difference the series, estimate all coefficients, and store the fit state.
    /// @param y The observed series.
    /// @throws std::invalid_argument On a missing seasonal period or a series too short for
    ///         the requested differencing or AR order.
    void fit(const std::vector<double>& y);

    ARIMAOptions options_; ///< Options used for the fit.

    std::vector<double> phi_;            ///< Regular AR coefficients, length p.
    std::vector<double> theta_;          ///< Regular MA coefficients, length q.
    std::vector<double> seasonal_phi_;   ///< Seasonal AR coefficients, length P.
    std::vector<double> seasonal_theta_; ///< Seasonal MA coefficients, length Q.
    double              mean_{0.0};      ///< Fitted mean of the differenced series (0 if not fit).

    std::size_t n_{0};                ///< Length of the input series.
    std::size_t n_used_{0};           ///< Residuals in the CSS sum: differenced length minus p + sP.
    double      sigma2_{0.0};         ///< Innovation variance: CSS sum of squares over n_used_.
    double      log_likelihood_{0.0}; ///< Gaussian log-likelihood at the CSS estimate.
    double      aic_{0.0};            ///< Akaike information criterion.
    double      bic_{0.0};            ///< Bayesian information criterion.

    std::vector<double> fitted_;    ///< One-step-ahead fitted values on the original scale.
    std::vector<double> residuals_; ///< One-step-ahead residuals on the original scale (zero for the presample).

    // State needed to forecast: the fully-differenced (stationary) residual series and its
    // recent history, plus enough trailing raw values to invert each differencing step.
    std::vector<double> combined_ar_;  ///< Regression coefficients of the expanded regular*seasonal AR polynomial (w-space): \f$w_t=\sum_i a_i w_{t-i}+\dots\f$.
    std::vector<double> combined_ma_;  ///< Coefficients of the expanded regular*seasonal MA polynomial (w-space), without the leading 1.
    std::vector<double> w_;            ///< Fully-differenced series (not demeaned).
    std::vector<double> w_residuals_;  ///< CSS residuals in w-space, aligned to w_ (zero for the first p + sP entries).

    // Full (non-stationary, y-space) AR/MA regression coefficients -- the w-space
    // polynomials further composed with the (1-B)^d and (1-B^s)^D differencing operators --
    // used only to compute psi-weights for forecast standard errors.
    std::vector<double> full_ar_regression_; ///< y-space AR regression coefficients: AR polynomial times both differencing operators.
    std::vector<double> full_ma_regression_; ///< y-space MA coefficients (same as combined_ma_; differencing does not change them).

    // Tails needed to invert each differencing step when re-integrating forecasts:
    // regular_diff_tail_[k] is the single trailing value before regular-diff step k (lag 1);
    // seasonal_diff_tail_[k] is the trailing seasonal_period values before seasonal-diff
    // step k (lag seasonal_period). Both indexed in the order the forward differencing was
    // applied (step 0 first), and inverted in reverse.
    std::vector<std::vector<double>> regular_diff_tail_;  ///< Per regular-diff step, the last value before that step (one value each).
    std::vector<std::vector<double>> seasonal_diff_tail_; ///< Per seasonal-diff step, the last s values before that step.
};

} // namespace datamunge::stats
