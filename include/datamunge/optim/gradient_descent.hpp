#pragma once
/// @file gradient_descent.hpp
/// @brief GradientDescent controls and native optimization contract.
#include <datamunge/optim/function_types.hpp>
#include <cstddef>
namespace datamunge::optim {
/// @brief Configuration for GradientDescent; fields retain their documented native defaults.
/// Supply finite values in the supported ranges. Controls are copied at construction;
/// validation, stopping and loss scaling belong to optimize().
/// @par Example
/// @code
/// dm::optim::GradientDescentOptions o;
/// std::print("{}\n",o.max_iterations); // 10000
/// @endcode
/// @see GradientDescent
struct GradientDescentOptions {
  double step_size{0.01}; ///< Fixed learning-rate/initial search step; use the objective scaling stated by optimize(). Default 0.01.
  double momentum{0.0}; ///< Fixed momentum coefficient; zero removes velocity carryover. Default 0.0.
  std::size_t max_iterations{10000}; ///< Maximum native iterations/epochs as described by optimize(); zero performs no updates. Default 10000.
  double tolerance{1e-8}; ///< Stopping threshold; its actual gradient/change/displacement/scale meaning is method-specific. Default 1e-8.
};
/// @brief GradientDescent on the DifferentiableFunction callback interface.
/// Velocity is momentum*velocity-step_size*gradient. Updates before testing the gradient just used or absolute objective change. Equal objective values can stop an oscillation away from stationarity. O(p) reset velocity state; no line search.
/// The caller supplies finite, dimensionally consistent callbacks. Inputs are borrowed;
/// coordinates mutate in place and can retain a partial state if a callback throws.
/// Calls use local optimizer state and propagate callback exceptions; sharing a mutable
/// objective across threads requires external synchronisation. Normal return alone
/// is not a uniform convergence certificate.
/// @par Example
/// @code
/// const dm::optim::GradientDescent optimizer;
/// std::print("{}\n",optimizer.options().max_iterations); // 10000
/// @endcode
/// @par References
/// - J. Nocedal and S. J. Wright (2006), Numerical Optimization, second edition, https://doi.org/10.1007/978-0-387-40065-5. The implemented variant and stopping rules are specified above.
/// @see GradientDescentOptions, trace_optimize()
class GradientDescent {
 public:
  /// @brief Store a copy of the controls without running the objective.
  /// @param options Method-specific controls; defaults are listed by GradientDescentOptions.
  /// @par Example
  /// @code
  /// const dm::optim::GradientDescent optimizer;
  /// std::print("{}\n",optimizer.options().max_iterations); // 10000
  /// @endcode
  /// @see optimize()
  explicit GradientDescent(GradientDescentOptions options = {});
  /// @brief Run GradientDescent from the supplied point using the configured finite budget.
  /// Velocity is momentum*velocity-step_size*gradient. Updates before testing the gradient just used or absolute objective change. Equal objective values can stop an oscillation away from stationarity. O(p) reset velocity state; no line search.
  /// @param function Borrowed DifferentiableFunction with matching finite output shapes.
  /// @param coordinates Owned-by-caller finite start; overwritten with the native returned point.
  /// @return Objective at the native returned coordinates; not a stop-status code.
  /// @throws std::exception Exceptions from supplied callbacks propagate. Raw solvers do not uniformly reject nonfinite inputs.
  /// @par Example
  /// @code
  /// dm::optim::QuadraticObjective f({2,8},{1,-2});
  /// dm::optim::GradientDescentOptions controls;controls.max_iterations=5;
  /// std::vector<double> x{0,0};
  /// const auto value=dm::optim::GradientDescent(controls).optimize(f,x);
  /// std::print("{}\n",std::isfinite(value)); // true
  /// @endcode
  /// @see trace_optimize(), GradientDescentOptions
  double optimize(DifferentiableFunction& function, std::vector<double>& coordinates) const;
  /// @brief Borrow this optimizer's stored configuration.
  /// @return Const option reference, valid during this object's lifetime; no mutation.
  [[nodiscard]] const GradientDescentOptions& options() const noexcept {return options_;}
 private:
  GradientDescentOptions options_; ///< Owned method configuration, copied at construction.
};
} // namespace datamunge::optim
