#pragma once
/// @file coordinate_descent.hpp
/// @brief CoordinateDescent controls and native optimization contract.
#include <datamunge/optim/function_types.hpp>
#include <cstddef>
namespace datamunge::optim {
/// @brief Configuration for CoordinateDescent; fields retain their documented native defaults.
/// Supply finite values in the supported ranges. Controls are copied at construction;
/// validation, stopping and loss scaling belong to optimize().
/// @par Example
/// @code
/// dm::optim::CoordinateDescentOptions o;
/// std::print("{}\n",o.max_iterations); // 1000
/// @endcode
/// @see CoordinateDescent
struct CoordinateDescentOptions {
  double step_size{1.0}; ///< Fixed learning-rate/initial search step; use the objective scaling stated by optimize(). Default 1.0.
  std::size_t max_iterations{1000}; ///< Maximum native iterations/epochs as described by optimize(); zero performs no updates. Default 1000.
  double tolerance{1e-8}; ///< Stopping threshold; its actual gradient/change/displacement/scale meaning is method-specific. Default 1e-8.
  double armijo_c1{1e-4}; ///< Sufficient-decrease fraction in (0,1). Default 1e-4.
  double backtracking_factor{0.5}; ///< Trial-step multiplier in (0,1). Default 0.5.
  std::size_t max_line_search_trials{30}; ///< Maximum candidate line-search trials per update. Default 30.
};
/// @brief CoordinateDescent on the DifferentiableFunction callback interface.
/// Each epoch visits coordinates in index order. Every partial update requests a full gradient and an Armijo-backtracked value sequence; this is not exact one-dimensional minimisation. An exhausted search restores that coordinate. Stops on the full end-of-epoch gradient norm.
/// The caller supplies finite, dimensionally consistent callbacks. Inputs are borrowed;
/// coordinates mutate in place and can retain a partial state if a callback throws.
/// Calls use local optimizer state and propagate callback exceptions; sharing a mutable
/// objective across threads requires external synchronisation. Normal return alone
/// is not a uniform convergence certificate.
/// @par Example
/// @code
/// const dm::optim::CoordinateDescent optimizer;
/// std::print("{}\n",optimizer.options().max_iterations); // 1000
/// @endcode
/// @par References
/// - S. J. Wright (2015), Coordinate descent algorithms, https://doi.org/10.1007/s10107-015-0892-3. This implementation uses the stated full-gradient coordinate/block steps.
/// @see CoordinateDescentOptions, trace_optimize()
class CoordinateDescent {
 public:
  /// @brief Store a copy of the controls without running the objective.
  /// @param options Method-specific controls; defaults are listed by CoordinateDescentOptions.
  /// @par Example
  /// @code
  /// const dm::optim::CoordinateDescent optimizer;
  /// std::print("{}\n",optimizer.options().max_iterations); // 1000
  /// @endcode
  /// @see optimize()
  explicit CoordinateDescent(CoordinateDescentOptions options = {});
  /// @brief Run CoordinateDescent from the supplied point using the configured finite budget.
  /// Each epoch visits coordinates in index order. Every partial update requests a full gradient and an Armijo-backtracked value sequence; this is not exact one-dimensional minimisation. An exhausted search restores that coordinate. Stops on the full end-of-epoch gradient norm.
  /// @param function Borrowed DifferentiableFunction with matching finite output shapes.
  /// @param coordinates Owned-by-caller finite start; overwritten with the native returned point.
  /// @return Objective at the native returned coordinates; not a stop-status code.
  /// @throws std::invalid_argument CoordinateDescent: step_size must be positive; CoordinateDescent: tolerance must be non-negative; CoordinateDescent: armijo_c1 must be in (0, 1); CoordinateDescent: backtracking_factor must be in (0, 1).
  /// @throws std::exception Exceptions from supplied callbacks propagate. Raw solvers do not uniformly reject nonfinite inputs.
  /// @par Example
  /// @code
  /// dm::optim::QuadraticObjective f({2,8},{1,-2});
  /// dm::optim::CoordinateDescentOptions controls;controls.max_iterations=5;
  /// std::vector<double> x{0,0};
  /// const auto value=dm::optim::CoordinateDescent(controls).optimize(f,x);
  /// std::print("{}\n",std::isfinite(value)); // true
  /// @endcode
  /// @see trace_optimize(), CoordinateDescentOptions
  double optimize(DifferentiableFunction& function, std::vector<double>& coordinates) const;
  /// @brief Borrow this optimizer's stored configuration.
  /// @return Const option reference, valid during this object's lifetime; no mutation.
  [[nodiscard]] const CoordinateDescentOptions& options() const noexcept {return options_;}
 private:
  CoordinateDescentOptions options_; ///< Owned method configuration, copied at construction.
};
} // namespace datamunge::optim
