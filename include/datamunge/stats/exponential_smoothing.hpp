#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::stats {

enum class TrendType { None, Additive, AdditiveDamped };
enum class SeasonalType { None, Additive, Multiplicative };

struct ExponentialSmoothingOptions {
    TrendType trend{TrendType::None};
    SeasonalType seasonal{SeasonalType::None};
    std::size_t seasonal_period{0}; // required when seasonal != SeasonalType::None

    // Smoothing parameters. Leave at the sentinel -1.0 (the default) to fit by minimizing
    // one-step-ahead SSE via DifferentialEvolution (the usual case); set to any value in the
    // valid range to pin a parameter there instead. (A plain sentinel, rather than
    // std::optional, is used here so this struct stays usable from every SWIG-bound
    // language.)
    double alpha{-1.0}; // level smoothing, in (0, 1)
    double beta{-1.0};  // trend smoothing, in (0, 1); ignored when trend == None
    double gamma{-1.0}; // seasonal smoothing, in (0, 1); ignored when seasonal == None
    double phi{-1.0};   // damping factor, in (0, 1]; ignored unless trend == AdditiveDamped

    std::size_t de_population_size{40};
    std::size_t de_max_generations{250};
    std::uint64_t seed{42};
};

/// @brief Exponential smoothing (Holt-Winters family): simple exponential smoothing (no
///        trend, no seasonality), Holt linear trend (with optional damping), and seasonal
///        Holt-Winters (additive or multiplicative), unified under one class the way R's
///        HoltWinters() and statsmodels' ExponentialSmoothing() are -- which components are
///        active is controlled entirely by ExponentialSmoothingOptions::trend/seasonal.
///
///        Initial level/trend/seasonal-index states are set once from a simple classical
///        decomposition of the first one or two seasonal cycles and then held fixed; only
///        the smoothing parameters (alpha, beta, gamma, phi) are fit, by minimizing
///        one-step-ahead sum of squared errors with DifferentialEvolution.
class ExponentialSmoothing {
 public:
    ExponentialSmoothing(const std::vector<double>& y, ExponentialSmoothingOptions options = {});

    [[nodiscard]] const ExponentialSmoothingOptions& options() const { return options_; }

    [[nodiscard]] double alpha() const { return alpha_; }
    [[nodiscard]] double beta() const { return beta_; }
    [[nodiscard]] double gamma() const { return gamma_; }
    [[nodiscard]] double phi() const { return phi_; }

    [[nodiscard]] std::size_t observations() const { return n_; }
    [[nodiscard]] double sse() const { return sse_; }
    [[nodiscard]] double sigma2() const { return sigma2_; }
    [[nodiscard]] double log_likelihood() const { return log_likelihood_; }
    [[nodiscard]] double aic() const { return aic_; }
    [[nodiscard]] double bic() const { return bic_; }

    /// @brief In-sample one-step-ahead fitted values, aligned to the input series. The
    ///        warm-up prefix used to set initial states (see class docs) is copied from the
    ///        observed values (residual exactly zero there).
    [[nodiscard]] const std::vector<double>& fitted_values() const { return fitted_; }
    [[nodiscard]] const std::vector<double>& residuals() const { return residuals_; }

    /// @brief Point forecasts for the next @p horizon periods.
    [[nodiscard]] std::vector<double> forecast(std::size_t horizon) const;

 private:
    void fit(const std::vector<double>& y);

    ExponentialSmoothingOptions options_;
    std::size_t warmup_{0};

    double alpha_{0.0};
    double beta_{0.0};
    double gamma_{0.0};
    double phi_{1.0};

    std::size_t n_{0};
    double sse_{0.0};
    double sigma2_{0.0};
    double log_likelihood_{0.0};
    double aic_{0.0};
    double bic_{0.0};

    std::vector<double> fitted_;
    std::vector<double> residuals_;

    // Final smoothed state (as of the last observation), used to seed forecast().
    double final_level_{0.0};
    double final_trend_{0.0};
    std::vector<double> final_season_; // length seasonal_period
};

} // namespace datamunge::stats
