#pragma once

/// @file rdd.hpp
/// @brief Regression discontinuity designs: local polynomial estimates with robust bias-corrected
/// inference (Calonico, Cattaneo and Titiunik) for sharp and fuzzy designs.

#include <cstddef>
#include <limits>
#include <vector>

namespace datamunge::causal {

/// @brief Kernel used to weight observations by their distance from the cutoff.
enum class RdKernel {
    triangular,   ///< \f$K(u) = 1 - |u|\f$ on \f$|u| \le 1\f$.
    epanechnikov, ///< \f$K(u) = \tfrac34 (1 - u^2)\f$ on \f$|u| \le 1\f$.
    uniform,      ///< \f$K(u) = \tfrac12\f$ on \f$|u| \le 1\f$.
};

/// @brief Residuals used in the variance estimators.
enum class RdVariance {
    nearest_neighbor, ///< Nearest-neighbour residuals from `matches` closest running-variable values.
    hc0,              ///< Fitted-value residuals, no correction.
    hc1,              ///< Fitted-value residuals scaled by \f$\sqrt{n / (n - k)}\f$.
    hc2,              ///< Fitted-value residuals scaled by \f$1 / \sqrt{1 - h_{ii}}\f$.
    hc3,              ///< Fitted-value residuals scaled by \f$1 / (1 - h_{ii})\f$.
};

/// @brief Options for regression_discontinuity() and rd_bandwidths().
///
/// The defaults match R's `rdrobust`: local linear fits (`order` 1) with a local quadratic bias
/// correction, triangular kernel, nearest-neighbour variance with three matches, and the common
/// MSE-optimal bandwidth (`mserd`) with mass-point adjustment.
/// @par Example
/// @code
/// const dm::causal::RdOptions o{.cutoff = 0.5, .order = 2};
/// std::println("{} {}", o.cutoff, o.order); // 0.5 2
/// @endcode
struct RdOptions {
    double cutoff{0.0}; ///< Treatment is assigned when the running variable is at or above this value.
    int order{1};       ///< Polynomial order p of the point estimate.
    int bias_order{-1}; ///< Order q of the bias-correction fit; negative means p + 1.
    int derivative{0};  ///< Derivative estimated at the cutoff (0 for a level jump, 1 for a kink).
    RdKernel kernel{RdKernel::triangular};             ///< Kernel.
    RdVariance variance{RdVariance::nearest_neighbor}; ///< Variance estimator.
    int matches{3};                                    ///< Nearest-neighbour matches (minimum 1).
    double bandwidth{std::numeric_limits<double>::quiet_NaN()}; ///< Main bandwidth h; NaN selects it.

    /// Bias bandwidth b; NaN uses h when h is given and the MSE-optimal b otherwise.
    double bias_bandwidth{std::numeric_limits<double>::quiet_NaN()};
    double confidence{0.95};        ///< Level of the normal confidence intervals.
    bool mass_point_adjust{true};   ///< Count distinct values and enforce 10 per side when ties are heavy.
    bool sharp_bandwidth{false};    ///< In fuzzy designs, select bandwidths for the outcome alone.
};

/// @brief Bandwidths chosen by rd_bandwidths().
/// @par Example
/// @code
/// const dm::causal::RdBandwidths b{.h = 0.2, .b = 0.35};
/// std::println("{}", b.h / b.b < 1); // true
/// @endcode
struct RdBandwidths {
    double h{};            ///< MSE-optimal main bandwidth.
    double b{};            ///< MSE-optimal bias bandwidth.
    double pilot{};        ///< Rule-of-thumb pilot bandwidth used for the variance and bias constants.
    double d{};            ///< Pilot bandwidth of the derivative estimate that enters b.
    bool mass_points{};    ///< True when at least 20 percent of one side's values are repeats.
    std::size_t distinct_left{};  ///< Distinct running-variable values below the cutoff.
    std::size_t distinct_right{}; ///< Distinct running-variable values at or above the cutoff.
};

/// @brief One estimate with its normal-theory inference.
/// @par Example
/// @code
/// const dm::causal::RdEstimate e{.estimate = 7.4, .standard_error = 1.5};
/// std::println("{:.1f}", e.estimate / e.standard_error); // 4.9
/// @endcode
struct RdEstimate {
    double estimate{};       ///< Point estimate.
    double standard_error{}; ///< Standard error.
    double z{};              ///< estimate / standard_error.
    double p_value{};        ///< Two-sided normal p-value.
    double lower{};          ///< Lower confidence limit.
    double upper{};          ///< Upper confidence limit.
};

/// @brief Result of regression_discontinuity().
///
/// For a fuzzy design `conventional`, `bias_corrected` and `robust` refer to the ratio of the outcome
/// jump to the treatment jump, and `first_stage_*` describe the treatment jump alone.
/// @par Example
/// @code
/// dm::causal::RdResult r;
/// r.robust.estimate = 7.5;
/// std::println("{}", r.fuzzy); // false
/// @endcode
struct RdResult {
    RdEstimate conventional;   ///< Local polynomial estimate with its conventional standard error.
    RdEstimate bias_corrected; ///< Bias-corrected estimate with the conventional standard error.
    RdEstimate robust;         ///< Bias-corrected estimate with the robust standard error.
    RdEstimate first_stage_conventional; ///< Fuzzy designs: treatment jump, conventional inference.
    RdEstimate first_stage_robust;       ///< Fuzzy designs: bias-corrected treatment jump, robust inference.
    bool fuzzy{false};          ///< True when a treatment variable was given.
    double h_left{};            ///< Main bandwidth below the cutoff.
    double h_right{};           ///< Main bandwidth above the cutoff.
    double b_left{};            ///< Bias bandwidth below the cutoff.
    double b_right{};           ///< Bias bandwidth above the cutoff.
    double left_limit{};        ///< Fitted outcome (or its derivative) approaching the cutoff from below.
    double right_limit{};       ///< Fitted outcome (or its derivative) at the cutoff from above.
    std::size_t n_left{};       ///< Observations below the cutoff.
    std::size_t n_right{};      ///< Observations at or above the cutoff.
    std::size_t effective_left{};  ///< Observations below the cutoff with positive kernel weight at h.
    std::size_t effective_right{}; ///< Observations above the cutoff with positive kernel weight at h.
};

/// @brief MSE-optimal common bandwidths (`rdrobust`'s `mserd`) for a sharp or fuzzy design.
///
/// A rule-of-thumb pilot \f$2.576\,\min(s_x, \mathrm{IQR}_x/1.349)\,M^{-1/5}\f$ (kernel-specific
/// constant, M distinct values) estimates the variance constants. A higher-order derivative fit gives
/// the bias bandwidth b, and b in turn gives the bias constant of h. Each bandwidth is capped at the
/// largest distance from the cutoff to the data.
/// @param outcome Outcome values.
/// @param running Running-variable values.
/// @param options Polynomial orders, kernel, variance estimator and cutoff (bandwidth fields ignored).
/// @param treatment Treatment received (fuzzy design) or empty (sharp design).
/// @return The selected bandwidths.
/// @throws std::invalid_argument For mismatched sizes, nonfinite values, bad orders, or too few
/// distinct values on a side.
/// @throws std::runtime_error When a local fit is singular.
/// @par Example
/// @code
/// std::vector<double> x, y;
/// for (int i = 0; i < 400; ++i) {
///     x.push_back(-1 + 2 * (i + 0.5) / 400);
///     y.push_back(x.back() + (x.back() >= 0) + 0.3 * std::sin(37.0 * i));
/// }
/// const auto bw = dm::causal::rd_bandwidths(y, x, {});
/// std::println("{}", bw.h > 0 && bw.h <= bw.b); // true
/// @endcode
RdBandwidths rd_bandwidths(const std::vector<double>& outcome, const std::vector<double>& running,
                           const RdOptions& options, const std::vector<double>& treatment = {});

/// @brief Regression discontinuity estimate with conventional and robust bias-corrected inference.
///
/// Weighted polynomials of order p are fitted on each side within bandwidth h. The difference of
/// their intercepts (or derivatives) is the conventional estimate. A fit of order q within the bias
/// bandwidth b estimates the leading bias, and its subtraction gives the bias-corrected estimate. The
/// robust standard error accounts for the variability of that correction (Calonico, Cattaneo and
/// Titiunik 2014), so robust intervals stay valid at the MSE-optimal h. With a treatment variable the
/// design is fuzzy and the estimate is the ratio of outcome and treatment jumps, with delta-method
/// variances. Results match R's `rdrobust` without covariates or clustering.
/// @param outcome Outcome values.
/// @param running Running-variable values.
/// @param options Orders, kernel, variance estimator, bandwidths and confidence level.
/// @param treatment Treatment received for a fuzzy design, or empty for a sharp design.
/// @return Estimates, bandwidths and sample counts.
/// @throws std::invalid_argument For mismatched sizes, nonfinite values, bad options, or too few
/// observations on a side.
/// @throws std::runtime_error When a local fit is singular or the fuzzy first stage is zero.
/// @par Example
/// @code
/// std::vector<double> x, y;
/// for (int i = 0; i < 400; ++i) {
///     x.push_back(-1 + 2 * (i + 0.5) / 400);
///     y.push_back(x.back() + (x.back() >= 0) + 0.3 * std::sin(37.0 * i));
/// }
/// const auto rd = dm::causal::regression_discontinuity(y, x, {});
/// std::println("{}", rd.robust.lower < 1 && 1 < rd.robust.upper); // true
/// @endcode
RdResult regression_discontinuity(const std::vector<double>& outcome, const std::vector<double>& running,
                                  const RdOptions& options = {},
                                  const std::vector<double>& treatment = {});

} // namespace datamunge::causal
