#pragma once
/// @file augmented_lagrangian.hpp
/// @brief AugmentedLagrangian controls and native optimization contract.
#include <datamunge/optim/function_types.hpp>
namespace datamunge::optim {
/// @brief Configuration for AugmentedLagrangian; fields retain their documented native defaults.
/// Supply finite values in the supported ranges. Controls are copied at construction;
/// validation, stopping and loss scaling belong to optimize().
/// @par Example
/// @code
/// dm::optim::AugmentedLagrangianOptions o;
/// std::print("{}\n",o.max_outer_iterations); // 50
/// @endcode
/// @see AugmentedLagrangian
struct AugmentedLagrangianOptions {
  double step_size{0.01}; ///< Fixed learning-rate/initial search step; use the objective scaling stated by optimize(). Default 0.01.
  double initial_penalty{1.0}; ///< Positive initial equality penalty rho. Default 1.0.
  std::size_t max_outer_iterations{50}; ///< Maximum multiplier/barrier outer stages. Default 50.
  std::size_t inner_iterations{200}; ///< Inner updates per outer stage; for SVRG zero uses component count. Default 200.
  double tolerance{1e-8}; ///< Stopping threshold; its actual gradient/change/displacement/scale meaning is method-specific. Default 1e-8.
};
/// @brief AugmentedLagrangian on the EqualityConstrainedFunction callback interface.
/// Uses fixed inner gradient steps on f+lambda*c+rho*||c||^2/2, capped so each displacement norm is at most one. Updates multipliers after each inner budget and doubles rho until feasibility tolerance or outer budget. No dual-stationarity test is exposed. Objective return alone is not a KKT certificate.
/// The caller supplies finite, dimensionally consistent callbacks. Inputs are borrowed;
/// coordinates mutate in place and can retain a partial state if a callback throws.
/// Calls use local optimizer state and propagate callback exceptions; sharing a mutable
/// objective across threads requires external synchronisation. Normal return alone
/// is not a uniform convergence certificate.
/// @par Example
/// @code
/// const dm::optim::AugmentedLagrangian optimizer;
/// std::print("{}\n",optimizer.options().max_outer_iterations); // 50
/// @endcode
/// @par References
/// - J. Nocedal and S. J. Wright (2006), Numerical Optimization, second edition, https://doi.org/10.1007/978-0-387-40065-5. The implemented variant and stopping rules are specified above.
/// @see AugmentedLagrangianOptions, trace_optimize()
class AugmentedLagrangian {
 public:
  /// @brief Store a copy of the controls without running the objective.
  /// @param options Method-specific controls; defaults are listed by AugmentedLagrangianOptions.
  /// @par Example
  /// @code
  /// const dm::optim::AugmentedLagrangian optimizer;
  /// std::print("{}\n",optimizer.options().max_outer_iterations); // 50
  /// @endcode
  /// @see optimize()
  explicit AugmentedLagrangian(AugmentedLagrangianOptions options = {});
  /// @brief Run AugmentedLagrangian from the supplied point using the configured finite budget.
  /// Uses fixed inner gradient steps on f+lambda*c+rho*||c||^2/2, capped so each displacement norm is at most one. Updates multipliers after each inner budget and doubles rho until feasibility tolerance or outer budget. No dual-stationarity test is exposed. Objective return alone is not a KKT certificate.
  /// @param function Borrowed EqualityConstrainedFunction with matching finite output shapes.
  /// @param coordinates Owned-by-caller finite start; overwritten with the native returned point.
  /// @return Objective at the native returned coordinates; not a stop-status code.
  /// @throws std::invalid_argument AugmentedLagrangian: positive step_size and penalty required.
  /// @throws std::exception Exceptions from supplied callbacks propagate. Raw solvers do not uniformly reject nonfinite inputs.
  /// @par Example
  /// @code
  /// dm::optim::EqualityQuadratic f({2,0},1);
  /// dm::optim::AugmentedLagrangianOptions controls;controls.max_outer_iterations=5;
  /// std::vector<double> x{0,0};
  /// const auto value=dm::optim::AugmentedLagrangian(controls).optimize(f,x);
  /// std::print("{}\n",std::isfinite(value)); // true
  /// @endcode
  /// @see trace_optimize(), AugmentedLagrangianOptions
  double optimize(EqualityConstrainedFunction& function,std::vector<double>& coordinates) const;
  /// @brief Borrow this optimizer's stored configuration.
  /// @return Const option reference, valid during this object's lifetime; no mutation.
  [[nodiscard]] const AugmentedLagrangianOptions& options() const noexcept {return options_;}
 private:
  AugmentedLagrangianOptions options_; ///< Owned method configuration, copied at construction.
};
} // namespace datamunge::optim
