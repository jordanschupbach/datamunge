#pragma once
/// @file lbfgs.hpp
/// @brief LBFGS controls and native optimization contract.
#include <datamunge/optim/function_types.hpp>
#include <cstddef>
namespace datamunge::optim {
/// @brief Configuration for LBFGS; fields retain their documented native defaults.
/// Supply finite values in the supported ranges. Controls are copied at construction;
/// validation, stopping and loss scaling belong to optimize().
/// @par Example
/// @code
/// dm::optim::LBFGSOptions o;
/// std::print("{}\n",o.max_iterations); // 500
/// @endcode
/// @see LBFGS
struct LBFGSOptions {
  std::size_t max_iterations{500}; ///< Maximum native iterations/epochs as described by optimize(); zero performs no updates. Default 500.
  double tolerance{1e-8}; ///< Stopping threshold; its actual gradient/change/displacement/scale meaning is method-specific. Default 1e-8.
  std::size_t history_size{10}; ///< Positive maximum stored curvature-pair count. Default 10.
  double armijo_c1{1e-4}; ///< Sufficient-decrease fraction in (0,1). Default 1e-4.
  double wolfe_c2{0.9}; ///< Weak-Wolfe curvature fraction. Default 0.9.
  std::size_t max_line_search_trials{50}; ///< Maximum candidate line-search trials per update. Default 50.
};
/// @brief LBFGS on the DifferentiableFunction callback interface.
/// Limited-memory BFGS with weak-Wolfe bracket/bisection search, descent fallback and positive-curvature pair retention. If the search budget ends, a previous Armijo step can be used; if none was found, returns. Tests gradient norm or absolute objective change. O(history_size*p) state; history_size must be positive.
/// The caller supplies finite, dimensionally consistent callbacks. Inputs are borrowed;
/// coordinates mutate in place and can retain a partial state if a callback throws.
/// Calls use local optimizer state and propagate callback exceptions; sharing a mutable
/// objective across threads requires external synchronisation. Normal return alone
/// is not a uniform convergence certificate.
/// @par Example
/// @code
/// const dm::optim::LBFGS optimizer;
/// std::print("{}\n",optimizer.options().max_iterations); // 500
/// @endcode
/// @par References
/// - J. Nocedal and S. J. Wright (2006), Numerical Optimization, second edition, https://doi.org/10.1007/978-0-387-40065-5. The implemented variant and stopping rules are specified above.
/// @see LBFGSOptions, trace_optimize()
class LBFGS {
 public:
  /// @brief Store a copy of the controls without running the objective.
  /// @param options Method-specific controls; defaults are listed by LBFGSOptions.
  /// @par Example
  /// @code
  /// const dm::optim::LBFGS optimizer;
  /// std::print("{}\n",optimizer.options().max_iterations); // 500
  /// @endcode
  /// @see optimize()
  explicit LBFGS(LBFGSOptions options = {});
  /// @brief Run LBFGS from the supplied point using the configured finite budget.
  /// Limited-memory BFGS with weak-Wolfe bracket/bisection search, descent fallback and positive-curvature pair retention. If the search budget ends, a previous Armijo step can be used; if none was found, returns. Tests gradient norm or absolute objective change. O(history_size*p) state; history_size must be positive.
  /// @param function Borrowed DifferentiableFunction with matching finite output shapes.
  /// @param coordinates Owned-by-caller finite start; overwritten with the native returned point.
  /// @return Objective at the native returned coordinates; not a stop-status code.
  /// @throws std::exception Exceptions from supplied callbacks propagate. Raw solvers do not uniformly reject nonfinite inputs.
  /// @par Example
  /// @code
  /// dm::optim::QuadraticObjective f({2,8},{1,-2});
  /// dm::optim::LBFGSOptions controls;controls.max_iterations=5;
  /// std::vector<double> x{0,0};
  /// const auto value=dm::optim::LBFGS(controls).optimize(f,x);
  /// std::print("{}\n",std::isfinite(value)); // true
  /// @endcode
  /// @see trace_optimize(), LBFGSOptions
  double optimize(DifferentiableFunction& function, std::vector<double>& coordinates) const;
  /// @brief Borrow this optimizer's stored configuration.
  /// @return Const option reference, valid during this object's lifetime; no mutation.
  [[nodiscard]] const LBFGSOptions& options() const noexcept {return options_;}
 private:
  LBFGSOptions options_; ///< Owned method configuration, copied at construction.
};
} // namespace datamunge::optim
