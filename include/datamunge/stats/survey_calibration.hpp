#pragma once

/// @file survey_calibration.hpp
/// @brief Calibration of survey weights to known totals (linear, raking, logit distances) and raking to margins.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <limits>
#include <vector>

namespace datamunge::stats {

/// @brief Distance between design and calibrated weights, as `survey::calibrate(calfun =)`.
/// @par Example
/// @code
/// const auto d = dm::stats::CalibrationDistance::raking;
/// std::println("{}", d == dm::stats::CalibrationDistance::linear); // false
/// @endcode
enum class CalibrationDistance {
    linear, ///< Chi-square distance: \f$g=1+x^\top\lambda\f$ (GREG weights), truncated to the bounds if given.
    raking, ///< Multiplicative: \f$g=\exp(x^\top\lambda)\f$, truncated to the bounds if given; positive weights.
    logit   ///< Logit: \f$g\f$ stays strictly inside finite bounds \f$(L,U)\f$ with \f$L<1<U\f$.
};

/// @brief Options for calibrate_weights().
/// @par Example
/// @code
/// const dm::stats::CalibrationOptions o{.distance = dm::stats::CalibrationDistance::logit, .lower = 0.5, .upper = 2};
/// std::println("{}", o.max_iterations); // 50
/// @endcode
struct CalibrationOptions {
    CalibrationDistance distance{CalibrationDistance::linear};  ///< Distance function.
    double lower{-std::numeric_limits<double>::infinity()};     ///< Lower bound on g (required finite for logit).
    double upper{std::numeric_limits<double>::infinity()};      ///< Upper bound on g (required finite for logit).
    double tolerance{1e-9};                                     ///< Stop when every |achieved - target| / (1 + |target|) is below this.
    std::size_t max_iterations{50};                             ///< Newton iteration limit.
};

/// @brief Calibrated weights and the remaining imbalance.
/// @par Example
/// @code
/// const dm::stats::CalibrationResult r;
/// std::println("{}", r.converged); // false
/// @endcode
struct CalibrationResult {
    std::vector<double> weights;      ///< Calibrated weights \f$w_ig_i\f$.
    std::vector<double> g;            ///< Adjustment factors \f$g_i\f$.
    std::vector<double> achieved;     ///< Weighted totals of each constraint after calibration.
    std::vector<double> residual;     ///< Achieved minus target, per constraint.
    double max_relative_residual{};   ///< \f$\max_j|r_j|/(1+|T_j|)\f$.
    std::size_t iterations{};         ///< Iterations used.
    bool converged{false};            ///< True if the tolerance was met.
};

/// @brief Calibrates weights to known population totals of auxiliary variables (Deville–Särndal).
///
/// Finds weights \f$w_ig_i\f$, as close as possible to the design weights in the chosen distance,
/// that satisfy \f$\sum_iw_ig_ix_i=T\f$ exactly. With \f$g_i=F(x_i^\top\lambda)\f$, Newton's method
/// solves for \f$\lambda\f$ from \f$\lambda=0\f$, as in `survey::calibrate`, but with an Armijo
/// line search on the convex dual \f$\sum_iw_i\Phi(x_i^\top\lambda)-\lambda^\top T\f$
/// (\f$\Phi'=F\f$). The damped iteration converges whenever a solution exists, including bounded
/// logit problems where undamped Newton (and `survey`) stalls near the bounds. The linear distance gives the GREG estimator in one step and can give
/// negative weights. Raking keeps weights positive. The logit distance keeps every \f$g_i\f$ strictly
/// inside \f$(L,U)\f$. If the bounds are too tight for the targets, the iteration stops without
/// convergence and the residuals show the imbalance. Newton systems are solved after Jacobi
/// scaling, so auxiliary variables on very different scales are handled.
/// @param weights Design weights, positive.
/// @param auxiliary Auxiliary values, one row per unit and one column per constraint (include a column of ones to fix the population size).
/// @param targets Known population total of each column.
/// @param options Distance, bounds and iteration control.
/// @return Calibrated weights, adjustment factors, achieved totals and residuals.
/// @throws std::invalid_argument For mismatched sizes, non-finite inputs, nonpositive weights, or
/// bounds that do not satisfy lower < 1 < upper (finite for logit).
/// @throws std::runtime_error If the constraint matrix is singular at the start (linearly dependent
/// columns). A later loss of rank, when every movable weight sits at a bound, ends the iteration
/// without convergence instead.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(4, 2, {1, 0, 1, 0, 1, 1, 1, 1}); // rows: (1, group)
/// const auto r = dm::stats::calibrate_weights({10, 10, 10, 10}, x, {50, 30});
/// std::println("{:.1f} {:.1f} {}", r.weights[0], r.weights[2], r.converged); // 10.0 15.0 true
/// @endcode
/// @par References
/// - J.-C. Deville and C.-E. Särndal (1992). "Calibration estimators in survey sampling." JASA 87(418),
///   376-382. doi:10.1080/01621459.1992.10475217
/// @see rake_weights()
[[nodiscard]] CalibrationResult calibrate_weights(const std::vector<double>& weights,
                                                  const linalg::DenseMatrix<double>& auxiliary,
                                                  const std::vector<double>& targets,
                                                  const CalibrationOptions& options = {});

/// @brief One categorical margin for rake_weights(): a category per unit and the population count of each category.
/// @par Example
/// @code
/// const dm::stats::RakingMargin m{.categories = {0, 1, 1}, .totals = {40, 60}};
/// std::println("{}", m.totals.size()); // 2
/// @endcode
struct RakingMargin {
    std::vector<std::size_t> categories; ///< Category index of each unit, in [0, totals.size()).
    std::vector<double> totals;          ///< Population count of each category.
};

/// @brief Options for rake_weights().
/// @par Example
/// @code
/// const dm::stats::RakingOptions o;
/// std::println("{}", o.max_iterations); // 1000
/// @endcode
struct RakingOptions {
    double tolerance{1e-10};          ///< Stop when the largest relative margin error falls below this.
    std::size_t max_iterations{1000}; ///< Limit on full cycles through the margins.
};

/// @brief Rakes weights to several categorical margins by iterative proportional fitting, as `survey::rake`.
///
/// Each cycle post-stratifies to every margin in turn, scaling the weights in each category to its
/// population count. At convergence all margins hold simultaneously. The weights then equal raking
/// calibration (calibrate_weights() with the raking distance) on the category indicators, which are
/// collinear across margins and so cannot be passed to it directly. The margins' grand totals must
/// agree.
/// @param weights Design weights, positive.
/// @param margins Categorical margins with their population counts.
/// @param options Tolerance and cycle limit.
/// @return Raked weights, with achieved and residual counts listed margin by margin, category by category.
/// @throws std::invalid_argument For mismatched sizes, out-of-range categories, a category with
/// a positive target but no sampled units, nonpositive weights, or margins with different grand
/// totals.
/// @par Example
/// @code
/// const auto r = dm::stats::rake_weights({1, 1, 1, 1}, {{.categories = {0, 0, 1, 1}, .totals = {30, 70}},
///                                                      {.categories = {0, 1, 0, 1}, .totals = {50, 50}}});
/// std::println("{:.1f} {:.1f} {}", r.weights[0], r.weights[3], r.converged); // 15.0 35.0 true
/// @endcode
/// @par References
/// - W. E. Deming and F. F. Stephan (1940). "On a least squares adjustment of a sampled frequency
///   table when the expected marginal totals are known." Annals of Mathematical Statistics 11(4),
///   427-444. doi:10.1214/aoms/1177731829
/// @see calibrate_weights()
[[nodiscard]] CalibrationResult rake_weights(const std::vector<double>& weights,
                                             const std::vector<RakingMargin>& margins,
                                             const RakingOptions& options = {});

} // namespace datamunge::stats
