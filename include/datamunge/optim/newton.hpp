#pragma once
/// @file newton.hpp
/// @brief Newton controls and native optimization contract.
#include <datamunge/optim/function_types.hpp>
namespace datamunge::optim {
/// @brief Configuration for Newton; fields retain their documented native defaults.
/// Supply finite values in the supported ranges. Controls are copied at construction;
/// validation, stopping and loss scaling belong to optimize().
/// @par Example
/// @code
/// dm::optim::NewtonOptions o;
/// std::print("{}\n",o.max_iterations); // 100
/// @endcode
/// @see Newton
struct NewtonOptions {
  std::size_t max_iterations{100}; ///< Maximum native iterations/epochs as described by optimize(); zero performs no updates. Default 100.
  double tolerance{1e-8}; ///< Stopping threshold; its actual gradient/change/displacement/scale meaning is method-specific. Default 1e-8.
  double damping{1e-10}; ///< Fixed diagonal Hessian shift; not adaptive damping. Default 1e-10.
};
/// @brief Newton on the HessianFunction callback interface.
/// Adds fixed damping to the supplied Hessian diagonal, solves with partial-pivot Gaussian elimination, and takes a full step. This is not a globalised line-search method; indefinite or nearly singular Hessians can increase the loss or produce huge steps. Stops on gradient norm. Dense O(p^3) solve/O(p^2) storage.
/// The caller supplies finite, dimensionally consistent callbacks. Inputs are borrowed;
/// coordinates mutate in place and can retain a partial state if a callback throws.
/// Calls use local optimizer state and propagate callback exceptions; sharing a mutable
/// objective across threads requires external synchronisation. Normal return alone
/// is not a uniform convergence certificate.
/// @par Example
/// @code
/// const dm::optim::Newton optimizer;
/// std::print("{}\n",optimizer.options().max_iterations); // 100
/// @endcode
/// @par References
/// - J. Nocedal and S. J. Wright (2006), Numerical Optimization, second edition, https://doi.org/10.1007/978-0-387-40065-5. The implemented variant and stopping rules are specified above.
/// @see NewtonOptions, trace_optimize()
class Newton {
 public:
  /// @brief Store a copy of the controls without running the objective.
  /// @param options Method-specific controls; defaults are listed by NewtonOptions.
  /// @par Example
  /// @code
  /// const dm::optim::Newton optimizer;
  /// std::print("{}\n",optimizer.options().max_iterations); // 100
  /// @endcode
  /// @see optimize()
  explicit Newton(NewtonOptions options = {});
  /// @brief Run Newton from the supplied point using the configured finite budget.
  /// Adds fixed damping to the supplied Hessian diagonal, solves with partial-pivot Gaussian elimination, and takes a full step. This is not a globalised line-search method; indefinite or nearly singular Hessians can increase the loss or produce huge steps. Stops on gradient norm. Dense O(p^3) solve/O(p^2) storage.
  /// @param function Borrowed HessianFunction with matching finite output shapes.
  /// @param coordinates Owned-by-caller finite start; overwritten with the native returned point.
  /// @return Objective at the native returned coordinates; not a stop-status code.
  /// @throws std::invalid_argument Newton: Hessian has wrong dimension; Newton: singular Hessian.
  /// @throws std::exception Exceptions from supplied callbacks propagate. Raw solvers do not uniformly reject nonfinite inputs.
  /// @par Example
  /// @code
  /// dm::optim::QuadraticObjective f({2,8},{1,-2});
  /// dm::optim::NewtonOptions controls;controls.max_iterations=5;
  /// std::vector<double> x{0,0};
  /// const auto value=dm::optim::Newton(controls).optimize(f,x);
  /// std::print("{}\n",std::isfinite(value)); // true
  /// @endcode
  /// @see trace_optimize(), NewtonOptions
  double optimize(HessianFunction& function, std::vector<double>& coordinates) const;
  /// @brief Borrow this optimizer's stored configuration.
  /// @return Const option reference, valid during this object's lifetime; no mutation.
  [[nodiscard]] const NewtonOptions& options() const noexcept {return options_;}
 private:
  NewtonOptions options_; ///< Owned method configuration, copied at construction.
};
} // namespace datamunge::optim
