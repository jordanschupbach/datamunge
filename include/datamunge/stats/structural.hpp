#pragma once

/// @file structural.hpp
/// @brief Structural (unobserved-components) time-series models fitted by Kalman-filter maximum likelihood.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <vector>

namespace datamunge::stats {

/// @brief Structural model type, as R's `StructTS` `type`.
/// @par Example
/// @code
/// const auto m = dm::stats::StructuralModel::local_level;
/// std::println("{}", m == dm::stats::StructuralModel::basic_seasonal); // false
/// @endcode
enum class StructuralModel {
    local_level,        ///< \f$y_t=\mu_t+\varepsilon_t,\ \mu_{t+1}=\mu_t+\xi_t\f$ ("level").
    local_linear_trend, ///< Adds a stochastic slope: \f$\mu_{t+1}=\mu_t+\nu_t+\xi_t,\ \nu_{t+1}=\nu_t+\zeta_t\f$ ("trend").
    basic_seasonal,     ///< Trend plus dummy seasonal \f$\gamma_{t+1}=-\sum_{j=1}^{s-1}\gamma_{t+1-j}+\omega_t\f$ ("BSM").
};

/// @brief Options for fit_structural().
/// @par Example
/// @code
/// const dm::stats::StructuralOptions options{.model = dm::stats::StructuralModel::basic_seasonal, .period = 4};
/// std::println("{}", options.period); // 4
/// @endcode
struct StructuralOptions {
    StructuralModel model{StructuralModel::local_level}; ///< Model type.
    std::size_t period{0};             ///< Seasonal period s (at least 2) for the basic structural model.
    std::size_t max_iterations{1000};  ///< L-BFGS iteration limit.
};

/// @brief A fitted structural model.
/// @par Example
/// @code
/// const dm::stats::StructuralFit f;
/// std::println("{}", f.irregular_variance); // 0
/// @endcode
struct StructuralFit {
    StructuralModel model{StructuralModel::local_level}; ///< Model type.
    std::size_t period{};          ///< Seasonal period (0 without seasonality).
    double level_variance{};       ///< \f$\sigma_\xi^2\f$.
    double slope_variance{};       ///< \f$\sigma_\zeta^2\f$ (0 for the local level).
    double seasonal_variance{};    ///< \f$\sigma_\omega^2\f$ (0 without seasonality).
    double irregular_variance{};   ///< \f$\sigma_\varepsilon^2\f$.

    /// Gaussian log likelihood of the observed values, including (as R) the first observation's
    /// prediction error under the large-variance prior \f$P_0=10^6\operatorname{var}(y)/100\f$. With
    /// missing values R's `StructTS` reports this rescaled by \f$n/n_{\text{obs}}\f$; this value is not.
    double log_likelihood{};
    std::vector<double> standardized_residuals; ///< \f$v_t/\sqrt{F_t}\f$ (NaN where y is missing).

    /// n by c filtered components \f$a_{t|t}\f$: level, then slope, then current seasonal (as present).
    linalg::DenseMatrix<double> filtered;
    linalg::DenseMatrix<double> smoothed; ///< n by c fixed-interval smoothed components.
    std::vector<double> final_state;      ///< Full filtered state at the last time point.
    linalg::DenseMatrix<double> final_covariance; ///< Its covariance.
    bool converged{};                     ///< Whether the optimizer's gradient condition was met.
};

/// @brief Fits a local level, local linear trend or basic structural model by maximum likelihood.
///
/// Runs the Kalman filter from \f$a_0=(y_1,0,\dots)\f$ with every entry of \f$P_0\f$ equal to
/// \f$10^6\operatorname{var}(y)/100\f$ and maximizes the Gaussian likelihood over the
/// disturbance variances (each \f$\ge0\f$), then smooths the states. Missing values (NaN) after
/// the first observation are skipped by the filter. Matches R's `StructTS`, `tsSmooth` and `predict`.
/// @param y Series with a finite first value; later NaN values are treated as missing.
/// @param options Model type, period and iteration limit.
/// @return Variances, log likelihood, residuals and filtered and smoothed components.
/// @throws std::invalid_argument For a missing first value, infinite values, fewer than 3 observed values,
/// a constant series, or a seasonal model with period below 2.
/// @par Example
/// @code
/// std::mt19937 g(4); // raw mt19937 output is portable; uniform(-0.5, 0.5) noise has variance 1/12
/// std::vector<double> y;
/// double level = 0;
/// for (int t = 0; t < 300; ++t) {
///     level += g() / 4294967296.0 - .5;                 // level variance 0.083
///     y.push_back(level + 4 * (g() / 4294967296.0 - .5)); // irregular variance 1.33
/// }
/// const auto f = dm::stats::fit_structural(y);
/// std::println("{:.2f} {:.2f}", f.level_variance, f.irregular_variance); // 0.09 1.50
/// @endcode
/// @par References
/// - A. C. Harvey (1989). Forecasting, Structural Time Series Models and the Kalman Filter.
///   Cambridge University Press. doi:10.1017/CBO9781107049994
/// - J. Durbin and S. J. Koopman (2012). Time Series Analysis by State Space Methods, 2nd ed.
///   Oxford University Press. doi:10.1093/acprof:oso/9780199641178.001.0001
/// @see forecast_structural()
[[nodiscard]] StructuralFit fit_structural(const std::vector<double>& y, const StructuralOptions& options = {});

/// @brief Forecasts from a fitted structural model.
/// @par Example
/// @code
/// const dm::stats::StructuralForecast f;
/// std::println("{}", f.mean.size()); // 0
/// @endcode
struct StructuralForecast {
    std::vector<double> mean;           ///< Point forecasts \f$Z a_{T+h|T}\f$.
    std::vector<double> standard_error; ///< \f$\sqrt{Z P_{T+h|T}Z^\top+\sigma_\varepsilon^2}\f$.
};

/// @brief h-step forecasts from the end of the sample, as R's `predict.StructTS`.
/// @param fit Fitted model.
/// @param horizon Steps ahead (positive).
/// @return Forecasts and standard errors.
/// @throws std::invalid_argument For a zero horizon.
/// @par Example
/// @code
/// const auto f = dm::stats::fit_structural({1, 2, 1.5, 2.5, 2, 3, 2.5, 3.5});
/// std::println("{}", dm::stats::forecast_structural(f, 3).mean.size()); // 3
/// @endcode
/// @see fit_structural()
[[nodiscard]] StructuralForecast forecast_structural(const StructuralFit& fit, std::size_t horizon);

} // namespace datamunge::stats
