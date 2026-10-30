#pragma once
/// @file rmsprop.hpp
/// @brief RMSProp controls and native optimization contract.
#include <datamunge/optim/function_types.hpp>
#include <cstddef>
namespace datamunge::optim {
/// @brief Configuration for RMSProp; fields retain their documented native defaults.
/// Supply finite values in the supported ranges. Controls are copied at construction;
/// validation, stopping and loss scaling belong to optimize().
/// @par Example
/// @code
/// dm::optim::RMSPropOptions o;
/// std::print("{}\n",o.max_iterations); // 10000
/// @endcode
/// @see RMSProp
struct RMSPropOptions {
  double step_size{0.001}; ///< Fixed learning-rate/initial search step; use the objective scaling stated by optimize(). Default 0.001.
  double decay_rate{0.9}; ///< Squared-history exponential decay in [0,1). Default 0.9.
  double epsilon{1e-8}; ///< Positive numerical offset; its root placement is specified by the method. Default 1e-8.
  std::size_t max_iterations{10000}; ///< Maximum native iterations/epochs as described by optimize(); zero performs no updates. Default 10000.
  double tolerance{1e-8}; ///< Stopping threshold; its actual gradient/change/displacement/scale meaning is method-specific. Default 1e-8.
};
/// @brief RMSProp on the DifferentiableFunction callback interface.
/// Uses an exponential squared-gradient average with decay_rate, dividing by sqrt(average)+epsilon. Tests the pre-update gradient norm after updating. Fixed rates can oscillate. O(p) local state, reset on every call.
/// The caller supplies finite, dimensionally consistent callbacks. Inputs are borrowed;
/// coordinates mutate in place and can retain a partial state if a callback throws.
/// Calls use local optimizer state and propagate callback exceptions; sharing a mutable
/// objective across threads requires external synchronisation. Normal return alone
/// is not a uniform convergence certificate.
/// @par Example
/// @code
/// const dm::optim::RMSProp optimizer;
/// std::print("{}\n",optimizer.options().max_iterations); // 10000
/// @endcode
/// @par References
/// - G. Hinton (2012), lecture 6e, https://www.cs.utoronto.ca/~hinton/coursera_lectures.html
/// @see RMSPropOptions, trace_optimize()
class RMSProp {
 public:
  /// @brief Store a copy of the controls without running the objective.
  /// @param options Method-specific controls; defaults are listed by RMSPropOptions.
  /// @par Example
  /// @code
  /// const dm::optim::RMSProp optimizer;
  /// std::print("{}\n",optimizer.options().max_iterations); // 10000
  /// @endcode
  /// @see optimize()
  explicit RMSProp(RMSPropOptions options = {});
  /// @brief Run RMSProp from the supplied point using the configured finite budget.
  /// Uses an exponential squared-gradient average with decay_rate, dividing by sqrt(average)+epsilon. Tests the pre-update gradient norm after updating. Fixed rates can oscillate. O(p) local state, reset on every call.
  /// @param function Borrowed DifferentiableFunction with matching finite output shapes.
  /// @param coordinates Owned-by-caller finite start; overwritten with the native returned point.
  /// @return Objective at the native returned coordinates; not a stop-status code.
  /// @throws std::invalid_argument RMSProp: step_size must be positive; RMSProp: decay_rate must be in [0, 1); RMSProp: epsilon must be positive; RMSProp: tolerance must be non-negative.
  /// @throws std::exception Exceptions from supplied callbacks propagate. Raw solvers do not uniformly reject nonfinite inputs.
  /// @par Example
  /// @code
  /// dm::optim::QuadraticObjective f({2,8},{1,-2});
  /// dm::optim::RMSPropOptions controls;controls.max_iterations=5;
  /// std::vector<double> x{0,0};
  /// const auto value=dm::optim::RMSProp(controls).optimize(f,x);
  /// std::print("{}\n",std::isfinite(value)); // true
  /// @endcode
  /// @see trace_optimize(), RMSPropOptions
  double optimize(DifferentiableFunction& function, std::vector<double>& coordinates) const;
  /// @brief Borrow this optimizer's stored configuration.
  /// @return Const option reference, valid during this object's lifetime; no mutation.
  [[nodiscard]] const RMSPropOptions& options() const noexcept {return options_;}
 private:
  RMSPropOptions options_; ///< Owned method configuration, copied at construction.
};
} // namespace datamunge::optim
