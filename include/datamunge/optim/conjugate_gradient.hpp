#pragma once
/// @file conjugate_gradient.hpp
/// @brief ConjugateGradient controls and native optimization contract.
#include <datamunge/optim/function_types.hpp>
#include <cstddef>
namespace datamunge::optim {
/// @brief Configuration for ConjugateGradient; fields retain their documented native defaults.
/// Supply finite values in the supported ranges. Controls are copied at construction;
/// validation, stopping and loss scaling belong to optimize().
/// @par Example
/// @code
/// dm::optim::ConjugateGradientOptions o;
/// std::print("{}\n",o.max_iterations); // 1000
/// @endcode
/// @see ConjugateGradient
struct ConjugateGradientOptions {
  std::size_t max_iterations{1000}; ///< Maximum native iterations/epochs as described by optimize(); zero performs no updates. Default 1000.
  double tolerance{1e-8}; ///< Stopping threshold; its actual gradient/change/displacement/scale meaning is method-specific. Default 1e-8.
  double armijo_c1{1e-4}; ///< Sufficient-decrease fraction in (0,1). Default 1e-4.
  double backtracking_factor{0.5}; ///< Trial-step multiplier in (0,1). Default 0.5.
  std::size_t max_line_search_trials{30}; ///< Maximum candidate line-search trials per update. Default 30.
};
/// @brief ConjugateGradient on the DifferentiableFunction callback interface.
/// Nonlinear Polak-Ribiere-plus directions with negative beta truncated to zero, periodic/descent restarts and an Armijo line search. It is not linear CG with exact line minimisation and does not imply termination in p steps. Returns when gradient/change/search conditions stop progress.
/// The caller supplies finite, dimensionally consistent callbacks. Inputs are borrowed;
/// coordinates mutate in place and can retain a partial state if a callback throws.
/// Calls use local optimizer state and propagate callback exceptions; sharing a mutable
/// objective across threads requires external synchronisation. Normal return alone
/// is not a uniform convergence certificate.
/// @par Example
/// @code
/// const dm::optim::ConjugateGradient optimizer;
/// std::print("{}\n",optimizer.options().max_iterations); // 1000
/// @endcode
/// @par References
/// - J. Nocedal and S. J. Wright (2006), Numerical Optimization, second edition, https://doi.org/10.1007/978-0-387-40065-5. The implemented variant and stopping rules are specified above.
/// @see ConjugateGradientOptions, trace_optimize()
class ConjugateGradient {
 public:
  /// @brief Store a copy of the controls without running the objective.
  /// @param options Method-specific controls; defaults are listed by ConjugateGradientOptions.
  /// @par Example
  /// @code
  /// const dm::optim::ConjugateGradient optimizer;
  /// std::print("{}\n",optimizer.options().max_iterations); // 1000
  /// @endcode
  /// @see optimize()
  explicit ConjugateGradient(ConjugateGradientOptions options = {});
  /// @brief Run ConjugateGradient from the supplied point using the configured finite budget.
  /// Nonlinear Polak-Ribiere-plus directions with negative beta truncated to zero, periodic/descent restarts and an Armijo line search. It is not linear CG with exact line minimisation and does not imply termination in p steps. Returns when gradient/change/search conditions stop progress.
  /// @param function Borrowed DifferentiableFunction with matching finite output shapes.
  /// @param coordinates Owned-by-caller finite start; overwritten with the native returned point.
  /// @return Objective at the native returned coordinates; not a stop-status code.
  /// @throws std::invalid_argument ConjugateGradient: tolerance must be non-negative; ConjugateGradient: armijo_c1 must be in (0, 1); ConjugateGradient: backtracking_factor must be in (0, 1).
  /// @throws std::exception Exceptions from supplied callbacks propagate. Raw solvers do not uniformly reject nonfinite inputs.
  /// @par Example
  /// @code
  /// dm::optim::QuadraticObjective f({2,8},{1,-2});
  /// dm::optim::ConjugateGradientOptions controls;controls.max_iterations=5;
  /// std::vector<double> x{0,0};
  /// const auto value=dm::optim::ConjugateGradient(controls).optimize(f,x);
  /// std::print("{}\n",std::isfinite(value)); // true
  /// @endcode
  /// @see trace_optimize(), ConjugateGradientOptions
  double optimize(DifferentiableFunction& function, std::vector<double>& coordinates) const;
  /// @brief Borrow this optimizer's stored configuration.
  /// @return Const option reference, valid during this object's lifetime; no mutation.
  [[nodiscard]] const ConjugateGradientOptions& options() const noexcept {return options_;}
 private:
  ConjugateGradientOptions options_; ///< Owned method configuration, copied at construction.
};
} // namespace datamunge::optim
