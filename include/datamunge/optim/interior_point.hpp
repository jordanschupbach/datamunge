#pragma once
/// @file interior_point.hpp
/// @brief InteriorPoint controls and native optimization contract.
#include <datamunge/optim/function_types.hpp>
namespace datamunge::optim {
/// @brief Configuration for InteriorPoint; fields retain their documented native defaults.
/// Supply finite values in the supported ranges. Controls are copied at construction;
/// validation, stopping and loss scaling belong to optimize().
/// @par Example
/// @code
/// dm::optim::InteriorPointOptions o;
/// std::print("{}\n",o.max_outer_iterations); // 20
/// @endcode
/// @see InteriorPoint
struct InteriorPointOptions {
  double step_size{0.01}; ///< Fixed learning-rate/initial search step; use the objective scaling stated by optimize(). Default 0.01.
  double initial_barrier{1.0}; ///< Positive starting log-barrier weight mu. Default 1.0.
  double barrier_decay{0.1}; ///< Barrier multiplier after each inner budget. Default 0.1.
  std::size_t max_outer_iterations{20}; ///< Maximum multiplier/barrier outer stages. Default 20.
  std::size_t inner_iterations{200}; ///< Inner updates per outer stage; for SVRG zero uses component count. Default 200.
};
/// @brief InteriorPoint on the InequalityConstrainedFunction callback interface.
/// Gradient steps on f-mu*sum(log(-c)) with c<0. Backtracks only for strict feasibility, not barrier-objective decrease, then decays mu after each fixed inner budget. Requires a strictly feasible start and can fail later when feasible steps underflow. No KKT or barrier-stationarity stop test; no phase-I feasibility search.
/// The caller supplies finite, dimensionally consistent callbacks. Inputs are borrowed;
/// coordinates mutate in place and can retain a partial state if a callback throws.
/// Calls use local optimizer state and propagate callback exceptions; sharing a mutable
/// objective across threads requires external synchronisation. Normal return alone
/// is not a uniform convergence certificate.
/// @par Example
/// @code
/// const dm::optim::InteriorPoint optimizer;
/// std::print("{}\n",optimizer.options().max_outer_iterations); // 20
/// @endcode
/// @par References
/// - J. Nocedal and S. J. Wright (2006), Numerical Optimization, second edition, https://doi.org/10.1007/978-0-387-40065-5. The implemented variant and stopping rules are specified above.
/// @see InteriorPointOptions, trace_optimize()
class InteriorPoint {
 public:
  /// @brief Store a copy of the controls without running the objective.
  /// @param options Method-specific controls; defaults are listed by InteriorPointOptions.
  /// @par Example
  /// @code
  /// const dm::optim::InteriorPoint optimizer;
  /// std::print("{}\n",optimizer.options().max_outer_iterations); // 20
  /// @endcode
  /// @see optimize()
  explicit InteriorPoint(InteriorPointOptions options = {});
  /// @brief Run InteriorPoint from the supplied point using the configured finite budget.
  /// Gradient steps on f-mu*sum(log(-c)) with c<0. Backtracks only for strict feasibility, not barrier-objective decrease, then decays mu after each fixed inner budget. Requires a strictly feasible start and can fail later when feasible steps underflow. No KKT or barrier-stationarity stop test; no phase-I feasibility search.
  /// @param function Borrowed InequalityConstrainedFunction with matching finite output shapes.
  /// @param coordinates Owned-by-caller finite start; overwritten with the native returned point.
  /// @return Objective at the native returned coordinates; not a stop-status code.
  /// @throws std::invalid_argument InteriorPoint: starting point must be strictly feasible.
  /// @throws std::runtime_error InteriorPoint: unable to find feasible step.
  /// @throws std::exception Exceptions from supplied callbacks propagate. Raw solvers do not uniformly reject nonfinite inputs.
  /// @par Example
  /// @code
  /// dm::optim::NonnegativeQuadratic f({-1,2});
  /// dm::optim::InteriorPointOptions controls;controls.max_outer_iterations=1;
  /// std::vector<double> x{.5,.5};
  /// const auto value=dm::optim::InteriorPoint(controls).optimize(f,x);
  /// std::print("{}\n",std::isfinite(value)); // true
  /// @endcode
  /// @see trace_optimize(), InteriorPointOptions
  double optimize(InequalityConstrainedFunction& function,std::vector<double>& coordinates) const;
  /// @brief Borrow this optimizer's stored configuration.
  /// @return Const option reference, valid during this object's lifetime; no mutation.
  [[nodiscard]] const InteriorPointOptions& options() const noexcept {return options_;}
 private:
  InteriorPointOptions options_; ///< Owned method configuration, copied at construction.
};
} // namespace datamunge::optim
