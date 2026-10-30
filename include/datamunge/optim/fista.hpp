#pragma once
/// @file fista.hpp
/// @brief FISTA controls and native optimization contract.
#include <datamunge/optim/function_types.hpp>
namespace datamunge::optim {
/// @brief Configuration for FISTA; fields retain their documented native defaults.
/// Supply finite values in the supported ranges. Controls are copied at construction;
/// validation, stopping and loss scaling belong to optimize().
/// @par Example
/// @code
/// dm::optim::FISTAOptions o;
/// std::print("{}\n",o.max_iterations); // 10000
/// @endcode
/// @see FISTA
struct FISTAOptions {
  double step_size{0.01}; ///< Fixed learning-rate/initial search step; use the objective scaling stated by optimize(). Default 0.01.
  std::size_t max_iterations{10000}; ///< Maximum native iterations/epochs as described by optimize(); zero performs no updates. Default 10000.
  double tolerance{1e-8}; ///< Stopping threshold; its actual gradient/change/displacement/scale meaning is method-specific. Default 1e-8.
};
/// @brief FISTA on the ProximalFunction callback interface.
/// Classical t recurrence (1+sqrt(1+4*t^2))/2 and extrapolated gradient points, followed by the supplied proximal map. Stops on successive accepted-coordinate displacement. No adaptive restart, backtracking or monotonicity safeguard. Supply the correct proximal dimensions and a rate at most 1/L for the standard convex bound.
/// The caller supplies finite, dimensionally consistent callbacks. Inputs are borrowed;
/// coordinates mutate in place and can retain a partial state if a callback throws.
/// Calls use local optimizer state and propagate callback exceptions; sharing a mutable
/// objective across threads requires external synchronisation. Normal return alone
/// is not a uniform convergence certificate.
/// @par Example
/// @code
/// const dm::optim::FISTA optimizer;
/// std::print("{}\n",optimizer.options().max_iterations); // 10000
/// @endcode
/// @par References
/// - A. Beck and M. Teboulle (2009), A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems, https://doi.org/10.1137/080716542
/// @see FISTAOptions, trace_optimize()
class FISTA {
 public:
  /// @brief Store a copy of the controls without running the objective.
  /// @param options Method-specific controls; defaults are listed by FISTAOptions.
  /// @par Example
  /// @code
  /// const dm::optim::FISTA optimizer;
  /// std::print("{}\n",optimizer.options().max_iterations); // 10000
  /// @endcode
  /// @see optimize()
  explicit FISTA(FISTAOptions options = {});
  /// @brief Run FISTA from the supplied point using the configured finite budget.
  /// Classical t recurrence (1+sqrt(1+4*t^2))/2 and extrapolated gradient points, followed by the supplied proximal map. Stops on successive accepted-coordinate displacement. No adaptive restart, backtracking or monotonicity safeguard. Supply the correct proximal dimensions and a rate at most 1/L for the standard convex bound.
  /// @param function Borrowed ProximalFunction with matching finite output shapes.
  /// @param coordinates Owned-by-caller finite start; overwritten with the native returned point.
  /// @return Objective at the native returned coordinates; not a stop-status code.
  /// @throws std::invalid_argument FISTA: step_size must be positive.
  /// @throws std::exception Exceptions from supplied callbacks propagate. Raw solvers do not uniformly reject nonfinite inputs.
  /// @par Example
  /// @code
  /// dm::optim::L1LeastSquaresObjective f({{1}},{3},1);
  /// dm::optim::FISTAOptions controls;controls.max_iterations=5;
  /// std::vector<double> x{0};
  /// const auto value=dm::optim::FISTA(controls).optimize(f,x);
  /// std::print("{}\n",std::isfinite(value)); // true
  /// @endcode
  /// @see trace_optimize(), FISTAOptions
  double optimize(ProximalFunction& function, std::vector<double>& coordinates) const;
  /// @brief Borrow this optimizer's stored configuration.
  /// @return Const option reference, valid during this object's lifetime; no mutation.
  [[nodiscard]] const FISTAOptions& options() const noexcept {return options_;}
 private:
  FISTAOptions options_; ///< Owned method configuration, copied at construction.
};
} // namespace datamunge::optim
