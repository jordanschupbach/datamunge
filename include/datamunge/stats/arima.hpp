#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace datamunge::stats {

struct ARIMAOptions {
    std::size_t p{0}; // non-seasonal AR order
    std::size_t d{0}; // non-seasonal differencing order
    std::size_t q{0}; // non-seasonal MA order

    std::size_t seasonal_p{0};      // seasonal AR order (P)
    std::size_t seasonal_d{0};      // seasonal differencing order (D)
    std::size_t seasonal_q{0};      // seasonal MA order (Q)
    std::size_t seasonal_period{0}; // seasonal period (s); required when P, D, or Q is nonzero

    /// @brief Fit an intercept (mean of the fully-differenced series). Only meaningful when
    ///        d == 0 && seasonal_d == 0 -- matching R's arima(), a mean term is not fit on a
    ///        series that has already been differenced to stationarity.
    bool include_mean{true};

    std::size_t de_population_size{60};
    std::size_t de_max_generations{400};
    /// @brief Box constraint applied to every AR/MA coefficient during optimization.
    double coefficient_bound{1.5};
    std::uint64_t seed{42};
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
    ARIMA(const std::vector<double>& y, ARIMAOptions options = {});

    [[nodiscard]] const ARIMAOptions& options() const { return options_; }

    [[nodiscard]] const std::vector<double>& ar_coefficients() const { return phi_; }
    [[nodiscard]] const std::vector<double>& ma_coefficients() const { return theta_; }
    [[nodiscard]] const std::vector<double>& seasonal_ar_coefficients() const { return seasonal_phi_; }
    [[nodiscard]] const std::vector<double>& seasonal_ma_coefficients() const { return seasonal_theta_; }
    [[nodiscard]] double mean() const { return mean_; }

    [[nodiscard]] std::size_t observations() const { return n_; }
    [[nodiscard]] std::size_t n_used() const { return n_used_; }
    [[nodiscard]] double sigma2() const { return sigma2_; }
    [[nodiscard]] double log_likelihood() const { return log_likelihood_; }
    [[nodiscard]] double aic() const { return aic_; }
    [[nodiscard]] double bic() const { return bic_; }

    /// @brief One-step-ahead fitted values on the original scale, aligned to the input
    ///        series (the first `observations() - n_used()` entries -- the presample points
    ///        conditioned on by CSS -- are copies of the observed value, matching a
    ///        residual of exactly zero there rather than an undefined value).
    [[nodiscard]] const std::vector<double>& fitted_values() const { return fitted_; }
    /// @brief One-step-ahead residuals on the original scale (zero for the presample points).
    [[nodiscard]] const std::vector<double>& residuals() const { return residuals_; }

    /// @brief Point forecasts for the next @p horizon periods, on the original scale.
    [[nodiscard]] std::vector<double> forecast(std::size_t horizon) const;

    /// @brief Point forecasts and their standard errors (from the psi-weight / MA(infinity)
    ///        expansion of the full, non-stationary AR/MA representation of y), on the
    ///        original scale.
    [[nodiscard]] std::pair<std::vector<double>, std::vector<double>> forecast_with_intervals(
        std::size_t horizon) const;

 private:
    void fit(const std::vector<double>& y);

    ARIMAOptions options_;

    std::vector<double> phi_;            // regular AR coefficients, length p
    std::vector<double> theta_;          // regular MA coefficients, length q
    std::vector<double> seasonal_phi_;   // seasonal AR coefficients, length P
    std::vector<double> seasonal_theta_; // seasonal MA coefficients, length Q
    double              mean_{0.0};

    std::size_t n_{0};
    std::size_t n_used_{0};
    double      sigma2_{0.0};
    double      log_likelihood_{0.0};
    double      aic_{0.0};
    double      bic_{0.0};

    std::vector<double> fitted_;
    std::vector<double> residuals_;

    // State needed to forecast: the fully-differenced (stationary) residual series and its
    // recent history, plus enough trailing raw values to invert each differencing step.
    std::vector<double> combined_ar_;  // expanded regular*seasonal AR polynomial (w-space)
    std::vector<double> combined_ma_;  // expanded regular*seasonal MA polynomial (w-space)
    std::vector<double> w_;            // fully-differenced series (not demeaned)
    std::vector<double> w_residuals_;  // CSS residuals in w-space, aligned to w_

    // Full (non-stationary, y-space) AR/MA regression coefficients -- the w-space
    // polynomials further composed with the (1-B)^d and (1-B^s)^D differencing operators --
    // used only to compute psi-weights for forecast standard errors.
    std::vector<double> full_ar_regression_;
    std::vector<double> full_ma_regression_;

    // Tails needed to invert each differencing step when re-integrating forecasts:
    // regular_diff_tail_[k] is the single trailing value before regular-diff step k (lag 1);
    // seasonal_diff_tail_[k] is the trailing seasonal_period values before seasonal-diff
    // step k (lag seasonal_period). Both indexed in the order the forward differencing was
    // applied (step 0 first), and inverted in reverse.
    std::vector<std::vector<double>> regular_diff_tail_;
    std::vector<std::vector<double>> seasonal_diff_tail_;
};

} // namespace datamunge::stats
