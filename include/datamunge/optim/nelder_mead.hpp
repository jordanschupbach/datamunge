#pragma once
/// @file nelder_mead.hpp
/// @brief NelderMead controls and native optimization contract.
#include <datamunge/optim/function_types.hpp>
#include <cstddef>
namespace datamunge::optim {
/// @brief Configuration for NelderMead; fields retain their documented native defaults.
/// Supply finite values in the supported ranges. Controls are copied at construction;
/// validation, stopping and loss scaling belong to optimize().
/// @par Example
/// @code
/// dm::optim::NelderMeadOptions o;
/// std::print("{}\n",o.max_iterations); // 2000
/// @endcode
/// @see NelderMead
struct NelderMeadOptions {
  double initial_simplex_scale{0.05}; ///< Per-coordinate displacement forming the initial simplex. Default 0.05.
  double reflection{1.0}; ///< Simplex reflection coefficient. Default 1.0.
  double expansion{2.0}; ///< Simplex expansion coefficient. Default 2.0.
  double contraction{0.5}; ///< Simplex contraction coefficient. Default 0.5.
  double shrink{0.5}; ///< Simplex shrink fraction. Default 0.5.
  std::size_t max_iterations{2000}; ///< Maximum native iterations/epochs as described by optimize(); zero performs no updates. Default 2000.
  double tolerance{1e-8}; ///< Stopping threshold; its actual gradient/change/displacement/scale meaning is method-specific. Default 1e-8.
};
/// @brief NelderMead on the ArbitraryFunction callback interface.
/// Local simplex reflection, expansion, contraction and shrink search. Starts with per-coordinate displacements initial_simplex_scale and stops when simplex diameter and value spread are small. Those conditions do not prove stationarity or a global minimum. O(p^2) simplex storage.
/// The caller supplies finite, dimensionally consistent callbacks. Inputs are borrowed;
/// coordinates mutate in place and can retain a partial state if a callback throws.
/// Calls use local optimizer state and propagate callback exceptions; sharing a mutable
/// objective across threads requires external synchronisation. Normal return alone
/// is not a uniform convergence certificate.
/// @par Example
/// @code
/// const dm::optim::NelderMead optimizer;
/// std::print("{}\n",optimizer.options().max_iterations); // 2000
/// @endcode
/// @par References
/// - J. A. Nelder and R. Mead (1965), A Simplex Method for Function Minimization, https://doi.org/10.1093/comjnl/7.4.308
/// @see NelderMeadOptions, trace_optimize()
class NelderMead {
 public:
  /// @brief Store a copy of the controls without running the objective.
  /// @param options Method-specific controls; defaults are listed by NelderMeadOptions.
  /// @par Example
  /// @code
  /// const dm::optim::NelderMead optimizer;
  /// std::print("{}\n",optimizer.options().max_iterations); // 2000
  /// @endcode
  /// @see optimize()
  explicit NelderMead(NelderMeadOptions options = {});
  /// @brief Run NelderMead from the supplied point using the configured finite budget.
  /// Local simplex reflection, expansion, contraction and shrink search. Starts with per-coordinate displacements initial_simplex_scale and stops when simplex diameter and value spread are small. Those conditions do not prove stationarity or a global minimum. O(p^2) simplex storage.
  /// @param function Borrowed ArbitraryFunction with matching finite output shapes.
  /// @param coordinates Owned-by-caller finite start; overwritten with the native returned point.
  /// @return Objective at the native returned coordinates; not a stop-status code.
  /// @throws std::invalid_argument NelderMead: initial_simplex_scale must be positive; NelderMead: reflection must be positive; NelderMead: expansion must be greater than 1; NelderMead: contraction must be in (0, 1); NelderMead: shrink must be in (0, 1); NelderMead: tolerance must be non-negative.
  /// @throws std::exception Exceptions from supplied callbacks propagate. Raw solvers do not uniformly reject nonfinite inputs.
  /// @par Example
  /// @code
  /// dm::optim::QuadraticObjective f({2,8},{1,-2});
  /// dm::optim::NelderMeadOptions controls;controls.max_iterations=5;
  /// std::vector<double> x{0,0};
  /// const auto value=dm::optim::NelderMead(controls).optimize(f,x);
  /// std::print("{}\n",std::isfinite(value)); // true
  /// @endcode
  /// @see trace_optimize(), NelderMeadOptions
  double optimize(ArbitraryFunction& function, std::vector<double>& coordinates) const;
  /// @brief Borrow this optimizer's stored configuration.
  /// @return Const option reference, valid during this object's lifetime; no mutation.
  [[nodiscard]] const NelderMeadOptions& options() const noexcept {return options_;}
 private:
  NelderMeadOptions options_; ///< Owned method configuration, copied at construction.
};
} // namespace datamunge::optim
