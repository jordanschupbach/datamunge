#pragma once

/// @file gpd.hpp
/// @brief Generalized Pareto distribution (GPD) for threshold exceedances: distribution functions, maximum-
/// likelihood fitting, threshold diagnostics (mean residual life and parameter stability) and return levels at a
/// specified rate of observations per period.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <vector>

namespace datamunge::stats {

/// @brief GPD distribution function of an excess y >= 0: \f$1-(1+\xi y/\sigma)_+^{-1/\xi}\f$ (exponential when
/// xi = 0).
/// @param y Excess over the threshold.
/// @param scale \f$\sigma>0\f$.
/// @param shape \f$\xi\f$.
/// @return Probability (0 for y < 0).
/// @throws std::invalid_argument If scale <= 0.
/// @par Example
/// @code
/// std::println("{:.4f}", dm::stats::gpd_cdf(1, 1, 0)); // 0.6321
/// @endcode
double gpd_cdf(double y, double scale, double shape);

/// @brief GPD quantile \f$\sigma[(1-p)^{-\xi}-1]/\xi\f$.
/// @param p Probability in [0, 1).
/// @param scale \f$\sigma>0\f$.
/// @param shape \f$\xi\f$.
/// @return Excess quantile.
/// @throws std::invalid_argument If p is outside [0, 1) or scale <= 0.
/// @par Example
/// @code
/// std::println("{:.4f}", dm::stats::gpd_quantile(0.5, 1, 0.5)); // 0.8284
/// @endcode
double gpd_quantile(double p, double scale, double shape);

/// @brief GPD log density of an excess (\f$-\infty\f$ outside the support).
/// @param y Excess.
/// @param scale \f$\sigma>0\f$.
/// @param shape \f$\xi\f$.
/// @return Log density.
/// @throws std::invalid_argument If scale <= 0.
/// @par Example
/// @code
/// std::println("{:.4f}", dm::stats::gpd_log_density(1, 1, 0)); // -1.0000
/// @endcode
double gpd_log_density(double y, double scale, double shape);

/// @brief Options for fit_gpd().
/// @par Example
/// @code
/// dm::stats::GpdOptions o{.exponential = true};
/// std::println("{}", o.exponential); // true
/// @endcode
struct GpdOptions {
    bool exponential{false};         ///< Fix the shape at 0 (exponential excesses).
    std::size_t max_iterations{200}; ///< Newton iterations.
    double tolerance{1e-10};         ///< Newton step tolerance.
};

/// @brief A fitted GPD for the exceedances of a threshold.
/// @par Example
/// @code
/// dm::stats::GpdFit f{.threshold = 30};
/// std::println("{}", f.threshold); // 30
/// @endcode
struct GpdFit {
    double threshold{};                     ///< u.
    double scale{};                         ///< \f$\hat\sigma\f$.
    double shape{};                         ///< \f$\hat\xi\f$ (0 when exponential).
    double scale_se{};                      ///< Standard error.
    double shape_se{};                      ///< Standard error (NaN when exponential).
    linalg::DenseMatrix<double> covariance; ///< Inverse observed information of (sigma[, xi]).
    double log_likelihood{};                ///< Log-likelihood of the excesses.
    std::size_t exceedances{};              ///< k: observations above u.
    std::size_t observations{};             ///< n: all observations.
    double exceedance_rate{};               ///< \f$\hat\zeta_u=k/n\f$.
    bool exponential{};                     ///< Whether the shape was fixed at 0.
    std::size_t iterations{};               ///< Newton iterations.
    bool converged{};                       ///< Whether Newton converged.
};

/// @brief Fit a GPD to the excesses of `data` over `threshold` by maximum likelihood.
/// @param data All observations (finite).
/// @param threshold u; at least 3 observations must exceed it.
/// @param options Options.
/// @return The fit.
/// @throws std::invalid_argument With fewer than 3 exceedances or nonfinite data.
/// @par Example
/// @code
/// const std::vector<double> x{1, 5, 2, 8, 3, 12, 4, 6, 2, 9, 15, 7};
/// std::println("{}", dm::stats::fit_gpd(x, 4).exceedances); // 7
/// @endcode
GpdFit fit_gpd(const std::vector<double>& data, double threshold, const GpdOptions& options = {});

/// @brief A return level from a GPD fit with intervals.
/// @par Example
/// @code
/// dm::stats::GpdReturnLevel r{.period = 100};
/// std::println("{}", r.period); // 100
/// @endcode
struct GpdReturnLevel {
    double period{};         ///< Return period in periods (e.g. years).
    double observations{};   ///< m = period x observations per period.
    double level{};          ///< \f$x_m=u+\sigma[(m\zeta\theta)^\xi-1]/\xi\f$.
    double standard_error{}; ///< Delta method including the binomial uncertainty of \f$\zeta\f$ (Coles 2001, 4.3.3).
    double wald_lower{};     ///< Wald interval.
    double wald_upper{};     ///< Upper Wald limit.
    double profile_lower{};  ///< Profile-likelihood interval (zeta fixed at its estimate).
    double profile_upper{};  ///< Upper profile limit (NaN if not found in the search range).
};

/// @brief Level exceeded on average once per `period` periods, with `per_period` observations per period.
/// @param fit Fit of `data`.
/// @param data The observations used in the fit.
/// @param period Return period (> 0, with m * zeta > 1).
/// @param per_period Observations per period (e.g. 365.25 for daily data and years).
/// @param confidence Interval level.
/// @param extremal_index \f$\theta\in(0,1]\f$ when `fit` is to cluster maxima of a dependent series (Coles 2001,
/// 5.3.2): the level becomes \f$u+\sigma[(m\zeta\theta)^\xi-1]/\xi\f$ with \f$\zeta\f$ the rate of all
/// exceedances. Leave at 1 for independent data or a fit to all exceedances. Treated as known.
/// @return Level and intervals.
/// @throws std::invalid_argument If the period is too short (m zeta theta <= 1), per_period <= 0, confidence is
/// outside (0, 1) or extremal_index is outside (0, 1].
/// @par Example
/// @code
/// const std::vector<double> x{1, 5, 2, 8, 3, 12, 4, 6, 2, 9, 15, 7};
/// const auto f = dm::stats::fit_gpd(x, 4, {.exponential = true});
/// std::println("{:.1f}", dm::stats::gpd_return_level(f, x, 10, 12).level); // 24.6
/// @endcode
GpdReturnLevel gpd_return_level(const GpdFit& fit, const std::vector<double>& data, double period,
                                double per_period, double confidence = 0.95, double extremal_index = 1.0);

/// @brief One row of the threshold diagnostics.
/// @par Example
/// @code
/// dm::stats::ThresholdDiagnostic t{.threshold = 5};
/// std::println("{}", t.threshold); // 5
/// @endcode
struct ThresholdDiagnostic {
    double threshold{};           ///< u.
    std::size_t exceedances{};    ///< k.
    double mean_excess{};         ///< Mean residual life \f$\bar e(u)\f$: mean of the excesses.
    double mean_excess_lower{};   ///< Normal-theory interval for the mean excess.
    double mean_excess_upper{};   ///< Upper limit.
    double shape{};               ///< \f$\hat\xi(u)\f$ (NaN if fewer than 10 exceedances).
    double shape_lower{};         ///< Wald interval for the shape.
    double shape_upper{};         ///< Upper limit.
    double modified_scale{};      ///< \f$\sigma^*=\hat\sigma(u)-\hat\xi(u)u\f$, constant above a valid threshold.
    double modified_scale_lower{}; ///< Delta-method interval for \f$\sigma^*\f$.
    double modified_scale_upper{}; ///< Upper limit.
};

/// @brief Mean-residual-life and parameter-stability diagnostics over a grid of thresholds.
///
/// Above a threshold where the GPD holds, the mean excess is linear in u (slope \f$\xi/(1-\xi)\f$) and \f$\hat\xi\f$
/// and \f$\sigma^*\f$ are constant up to sampling error.
/// @param data Observations.
/// @param thresholds Candidate thresholds.
/// @param confidence Interval level.
/// @return One row per threshold with at least 2 exceedances.
/// @throws std::invalid_argument On nonfinite data or confidence outside (0, 1).
/// @par Example
/// @code
/// const std::vector<double> x{1, 5, 2, 8, 3, 12, 4, 6, 2, 9, 15, 7};
/// std::println("{:.2f}", dm::stats::threshold_diagnostics(x, {4}).front().mean_excess); // 4.86
/// @endcode
std::vector<ThresholdDiagnostic> threshold_diagnostics(const std::vector<double>& data,
                                                       const std::vector<double>& thresholds,
                                                       double confidence = 0.95);

} // namespace datamunge::stats
