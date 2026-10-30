#pragma once
/// @file nadam.hpp
/// @brief Nadam controls and native optimization contract.
#include <datamunge/optim/function_types.hpp>
#include <cstddef>
namespace datamunge::optim {
/// @brief Configuration for Nadam; fields retain their documented native defaults.
/// Supply finite values in the supported ranges. Controls are copied at construction;
/// validation, stopping and loss scaling belong to optimize().
/// @par Example
/// @code
/// dm::optim::NadamOptions o;
/// std::print("{}\n",o.max_iterations); // 10000
/// @endcode
/// @see Nadam
struct NadamOptions {
  double step_size{0.001}; ///< Fixed learning-rate/initial search step; use the objective scaling stated by optimize(). Default 0.001.
  double beta1{0.9}; ///< First-moment decay in [0,1). Default 0.9.
  double beta2{0.999}; ///< Second-moment decay in [0,1). Default 0.999.
  double epsilon{1e-8}; ///< Positive numerical offset; its root placement is specified by the method. Default 1e-8.
  std::size_t max_iterations{10000}; ///< Maximum native iterations/epochs as described by optimize(); zero performs no updates. Default 10000.
  double tolerance{1e-8}; ///< Stopping threshold; its actual gradient/change/displacement/scale meaning is method-specific. Default 1e-8.
};
/// @brief Nadam on the DifferentiableFunction callback interface.
/// Fixed-beta Nadam variant: direction beta1*m/(1-beta1^t)+(1-beta1)*g/(1-beta1^t), divided by sqrt(v/(1-beta2^t))+epsilon. Does not implement scheduled momentum products. Tests the gradient used after updating; O(p) reset state.
/// The caller supplies finite, dimensionally consistent callbacks. Inputs are borrowed;
/// coordinates mutate in place and can retain a partial state if a callback throws.
/// Calls use local optimizer state and propagate callback exceptions; sharing a mutable
/// objective across threads requires external synchronisation. Normal return alone
/// is not a uniform convergence certificate.
/// @par Example
/// @code
/// const dm::optim::Nadam optimizer;
/// std::print("{}\n",optimizer.options().max_iterations); // 10000
/// @endcode
/// @par References
/// - T. Dozat (2016), Incorporating Nesterov Momentum into Adam, https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ
/// @see NadamOptions, trace_optimize()
class Nadam {
 public:
  /// @brief Store a copy of the controls without running the objective.
  /// @param options Method-specific controls; defaults are listed by NadamOptions.
  /// @par Example
  /// @code
  /// const dm::optim::Nadam optimizer;
  /// std::print("{}\n",optimizer.options().max_iterations); // 10000
  /// @endcode
  /// @see optimize()
  explicit Nadam(NadamOptions options = {});
  /// @brief Run Nadam from the supplied point using the configured finite budget.
  /// Fixed-beta Nadam variant: direction beta1*m/(1-beta1^t)+(1-beta1)*g/(1-beta1^t), divided by sqrt(v/(1-beta2^t))+epsilon. Does not implement scheduled momentum products. Tests the gradient used after updating; O(p) reset state.
  /// @param function Borrowed DifferentiableFunction with matching finite output shapes.
  /// @param coordinates Owned-by-caller finite start; overwritten with the native returned point.
  /// @return Objective at the native returned coordinates; not a stop-status code.
  /// @throws std::invalid_argument Nadam: step_size must be positive; Nadam: beta1 must be in [0, 1); Nadam: beta2 must be in [0, 1); Nadam: epsilon must be positive; Nadam: tolerance must be non-negative.
  /// @throws std::exception Exceptions from supplied callbacks propagate. Raw solvers do not uniformly reject nonfinite inputs.
  /// @par Example
  /// @code
  /// dm::optim::QuadraticObjective f({2,8},{1,-2});
  /// dm::optim::NadamOptions controls;controls.max_iterations=5;
  /// std::vector<double> x{0,0};
  /// const auto value=dm::optim::Nadam(controls).optimize(f,x);
  /// std::print("{}\n",std::isfinite(value)); // true
  /// @endcode
  /// @see trace_optimize(), NadamOptions
  double optimize(DifferentiableFunction& function, std::vector<double>& coordinates) const;
  /// @brief Borrow this optimizer's stored configuration.
  /// @return Const option reference, valid during this object's lifetime; no mutation.
  [[nodiscard]] const NadamOptions& options() const noexcept {return options_;}
 private:
  NadamOptions options_; ///< Owned method configuration, copied at construction.
};
} // namespace datamunge::optim
