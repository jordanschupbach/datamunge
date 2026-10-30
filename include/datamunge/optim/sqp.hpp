#pragma once
/// @file sqp.hpp
/// @brief SQP controls and native optimization contract.
#include <datamunge/optim/function_types.hpp>
namespace datamunge::optim {
/// @brief Configuration for SQP; fields retain their documented native defaults.
/// Supply finite values in the supported ranges. Controls are copied at construction;
/// validation, stopping and loss scaling belong to optimize().
/// @par Example
/// @code
/// dm::optim::SQPOptions o;
/// std::print("{}\n",o.max_iterations); // 200
/// @endcode
/// @see SQP
struct SQPOptions {
  double step_size{1.0}; ///< Fixed learning-rate/initial search step; use the objective scaling stated by optimize(). Default 1.0.
  double regularization{1e-6}; ///< Constraint row squared-norm denominator offset. Default 1e-6.
  std::size_t max_iterations{200}; ///< Maximum native iterations/epochs as described by optimize(); zero performs no updates. Default 200.
  double tolerance{1e-8}; ///< Stopping threshold; its actual gradient/change/displacement/scale meaning is method-specific. Default 1e-8.
};
/// @brief SQP on the EqualityConstrainedFunction callback interface.
/// Restricted projection-like implementation, not a general SQP QP solver: takes an objective-gradient step, then sequential linearised row corrections using squared Jacobian-row norms plus regularization. Its stopping condition compares constraint norm and the constant absolute step_size to tolerance, not the actual displacement or Lagrangian gradient.
/// The caller supplies finite, dimensionally consistent callbacks. Inputs are borrowed;
/// coordinates mutate in place and can retain a partial state if a callback throws.
/// Calls use local optimizer state and propagate callback exceptions; sharing a mutable
/// objective across threads requires external synchronisation. Normal return alone
/// is not a uniform convergence certificate.
/// @par Example
/// @code
/// const dm::optim::SQP optimizer;
/// std::print("{}\n",optimizer.options().max_iterations); // 200
/// @endcode
/// @par References
/// - J. Nocedal and S. J. Wright (2006), Numerical Optimization, second edition, https://doi.org/10.1007/978-0-387-40065-5. The implemented variant and stopping rules are specified above.
/// @see SQPOptions, trace_optimize()
class SQP {
 public:
  /// @brief Store a copy of the controls without running the objective.
  /// @param options Method-specific controls; defaults are listed by SQPOptions.
  /// @par Example
  /// @code
  /// const dm::optim::SQP optimizer;
  /// std::print("{}\n",optimizer.options().max_iterations); // 200
  /// @endcode
  /// @see optimize()
  explicit SQP(SQPOptions options = {});
  /// @brief Run SQP from the supplied point using the configured finite budget.
  /// Restricted projection-like implementation, not a general SQP QP solver: takes an objective-gradient step, then sequential linearised row corrections using squared Jacobian-row norms plus regularization. Its stopping condition compares constraint norm and the constant absolute step_size to tolerance, not the actual displacement or Lagrangian gradient.
  /// @param function Borrowed EqualityConstrainedFunction with matching finite output shapes.
  /// @param coordinates Owned-by-caller finite start; overwritten with the native returned point.
  /// @return Objective at the native returned coordinates; not a stop-status code.
  /// @throws std::exception Exceptions from supplied callbacks propagate. Raw solvers do not uniformly reject nonfinite inputs.
  /// @par Example
  /// @code
  /// dm::optim::EqualityQuadratic f({2,0},1);
  /// dm::optim::SQPOptions controls;controls.max_iterations=5;
  /// std::vector<double> x{0,0};
  /// const auto value=dm::optim::SQP(controls).optimize(f,x);
  /// std::print("{}\n",std::isfinite(value)); // true
  /// @endcode
  /// @see trace_optimize(), SQPOptions
  double optimize(EqualityConstrainedFunction& function,std::vector<double>& coordinates) const;
  /// @brief Borrow this optimizer's stored configuration.
  /// @return Const option reference, valid during this object's lifetime; no mutation.
  [[nodiscard]] const SQPOptions& options() const noexcept {return options_;}
 private:
  SQPOptions options_; ///< Owned method configuration, copied at construction.
};
} // namespace datamunge::optim
