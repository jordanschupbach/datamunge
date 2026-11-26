#pragma once

/// @file garch.hpp
/// @brief ARCH-LM test for conditional heteroskedasticity and Gaussian GARCH(p, q) estimation with forecasts.

#include <cstddef>
#include <vector>

namespace datamunge::stats {

/// @brief Result of arch_lm_test().
/// @par Example
/// @code
/// const dm::stats::ArchLmTest t;
/// std::println("{}", t.lags); // 0
/// @endcode
struct ArchLmTest {
    double lm{};             ///< Engle's LM statistic \f$(m-\text{ddof})R^2\f$.
    double lm_p_value{};     ///< Upper \f$\chi^2_L\f$ p-value of `lm`.
    double f{};              ///< F statistic of the auxiliary regression.
    double f_p_value{};      ///< Upper \f$F(L,\,m-L-1)\f$ p-value of `f`.
    std::size_t lags{};      ///< Lags L of squared residuals.
    std::size_t observations{}; ///< Auxiliary-regression sample size m = n - L.
};

/// @brief Engle's Lagrange-multiplier test for ARCH effects.
///
/// Regresses \f$e_t^2\f$ on a constant and \f$e_{t-1}^2,\dots,e_{t-L}^2\f$; under no ARCH, \f$mR^2\f$ is
/// asymptotically \f$\chi^2_L\f$. Matches `statsmodels.stats.diagnostic.het_arch`.
/// @param residuals Residuals (or demeaned returns), finite.
/// @param lags Number of lags L (positive).
/// @param ddof Degrees of freedom subtracted from m in the LM statistic (statsmodels `ddof`).
/// @return LM and F statistics with p-values.
/// @throws std::invalid_argument For nonfinite values, zero lags, too few observations or constant squares.
/// @par Example
/// @code
/// std::mt19937 g(1);
/// std::vector<double> e;
/// for (int t = 0; t < 500; ++t)
///     e.push_back(g() / 4294967296.0 - .5);
/// std::println("{}", dm::stats::arch_lm_test(e, 4).lm_p_value > .05); // true
/// @endcode
/// @par References
/// - R. F. Engle (1982). "Autoregressive conditional heteroscedasticity with estimates of the variance
///   of United Kingdom inflation". Econometrica 50(4), 987-1007. doi:10.2307/1912773
/// @see fit_garch()
[[nodiscard]] ArchLmTest arch_lm_test(const std::vector<double>& residuals, std::size_t lags, std::size_t ddof = 0);

/// @brief Mean specification for fit_garch().
/// @par Example
/// @code
/// const auto m = dm::stats::GarchMean::zero;
/// std::println("{}", m == dm::stats::GarchMean::constant); // false
/// @endcode
enum class GarchMean {
    zero,     ///< \f$y_t=\varepsilon_t\f$.
    constant, ///< \f$y_t=\mu+\varepsilon_t\f$, with \f$\mu\f$ estimated jointly.
};

/// @brief Options for fit_garch().
/// @par Example
/// @code
/// const dm::stats::GarchOptions options{.arch_order = 1, .garch_order = 1};
/// std::println("{}", options.garch_order); // 1
/// @endcode
struct GarchOptions {
    std::size_t arch_order{1};  ///< p: lags of \f$\varepsilon^2\f$ (positive).
    std::size_t garch_order{1}; ///< q: lags of \f$\sigma^2\f$ (0 gives an ARCH(p) model).
    GarchMean mean{GarchMean::constant}; ///< Mean model.
    std::size_t max_iterations{1000};    ///< L-BFGS iteration limit.
};

/// @brief A fitted GARCH(p, q): \f$\sigma_t^2=\omega+\sum_i\alpha_i\varepsilon_{t-i}^2+\sum_j\beta_j\sigma_{t-j}^2\f$.
/// @par Example
/// @code
/// const dm::stats::GarchFit f;
/// std::println("{}", f.alpha.size()); // 0
/// @endcode
struct GarchFit {
    double mu{};                 ///< Mean (0 for the zero-mean model).
    double omega{};              ///< Variance intercept \f$\omega>0\f$.
    std::vector<double> alpha;   ///< ARCH coefficients \f$\alpha_1,\dots,\alpha_p\ge0\f$.
    std::vector<double> beta;    ///< GARCH coefficients \f$\beta_1,\dots,\beta_q\ge0\f$.

    /// All estimates in the order mu (if estimated), omega, alpha, beta (as `arch` `params`).
    std::vector<double> parameters;

    /// Bollerslev–Wooldridge robust (sandwich) standard errors, in the order of `parameters`.
    std::vector<double> standard_errors;
    std::vector<double> classical_standard_errors; ///< Inverse-Hessian standard errors.
    double log_likelihood{};     ///< Gaussian log likelihood.
    double aic{};                ///< \f$-2\ell+2k\f$.
    double bic{};                ///< \f$-2\ell+k\ln n\f$.
    double persistence{};        ///< \f$\sum\alpha_i+\sum\beta_j\f$ (below 1).
    double unconditional_variance{}; ///< \f$\omega/(1-\text{persistence})\f$.

    /// Presample value for \f$\varepsilon^2\f$ and \f$\sigma^2\f$: exponentially weighted (0.94) mean of the
    /// first \f$\min(75,n)\f$ squared demeaned observations, as `arch`.
    double backcast{};
    std::vector<double> residuals;            ///< \f$\varepsilon_t=y_t-\mu\f$.
    std::vector<double> conditional_variance; ///< \f$\sigma_t^2\f$.
    std::vector<double> standardized_residuals; ///< \f$\varepsilon_t/\sigma_t\f$.
    bool converged{};            ///< Whether the optimizer's gradient condition was met.
};

/// @brief Fits a Gaussian GARCH(p, q) model by maximum likelihood.
///
/// Optimizes the likelihood with L-BFGS in coordinates that keep \f$\omega>0\f$, all coefficients
/// positive and the persistence below one, then refines with Newton steps on the natural
/// parameters. Matches `arch.arch_model(y, vol="GARCH", p, q, dist="normal")` at interior optima.
/// @param y Returns, finite.
/// @param options Orders and mean model.
/// @return Estimates, robust and classical standard errors, likelihood and conditional variances.
/// @throws std::invalid_argument For nonfinite data, zero ARCH order, or fewer than 10 + parameters observations.
/// @par Example
/// @code
/// std::mt19937 g(7); // raw mt19937 output is portable; uniform shocks scaled to unit variance
/// std::vector<double> y;
/// double s2 = 1, e = 0;
/// for (int t = 0; t < 2000; ++t) {
///     s2 = .1 + .1 * e * e + .8 * s2;
///     e = std::sqrt(s2) * std::sqrt(12.0) * (g() / 4294967296.0 - .5);
///     y.push_back(e);
/// }
/// const auto f = dm::stats::fit_garch(y, {.mean = dm::stats::GarchMean::zero});
/// std::println("{:.1f}", f.persistence); // 0.9
/// @endcode
/// @par References
/// - T. Bollerslev (1986). "Generalized autoregressive conditional heteroskedasticity". Journal of
///   Econometrics 31(3), 307-327. doi:10.1016/0304-4076(86)90063-1
/// - T. Bollerslev and J. M. Wooldridge (1992). "Quasi-maximum likelihood estimation and inference in
///   dynamic models with time-varying covariances". Econometric Reviews 11(2), 143-172.
///   doi:10.1080/07474939208800229
/// @see arch_lm_test(), forecast_garch_variance()
[[nodiscard]] GarchFit fit_garch(const std::vector<double>& y, const GarchOptions& options = {});

/// @brief Multi-step conditional variance forecasts \f$E_T[\sigma_{T+h}^2]\f$, h = 1..horizon.
/// @param fit Fitted model.
/// @param horizon Steps (positive).
/// @return Forecast variances; they converge to `unconditional_variance`.
/// @throws std::invalid_argument For a zero horizon.
/// @par Example
/// @code
/// dm::stats::GarchFit f;
/// f.omega = .1; f.alpha = {.1}; f.beta = {.8};
/// f.residuals = {1}; f.conditional_variance = {1};
/// std::println("{:.2f}", dm::stats::forecast_garch_variance(f, 2)[1]); // 1.00
/// @endcode
/// @see fit_garch()
[[nodiscard]] std::vector<double> forecast_garch_variance(const GarchFit& fit, std::size_t horizon);

} // namespace datamunge::stats
