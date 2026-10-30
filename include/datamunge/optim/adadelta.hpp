#pragma once
/// @file adadelta.hpp
/// @brief AdaDelta controls and native optimization contract.
#include <datamunge/optim/function_types.hpp>
#include <cstddef>
namespace datamunge::optim {
/// @brief Configuration for AdaDelta; fields retain their documented native defaults.
/// Supply finite values in the supported ranges. Controls are copied at construction;
/// validation, stopping and loss scaling belong to optimize().
/// @par Example
/// @code
/// dm::optim::AdaDeltaOptions o;
/// std::print("{}\n",o.max_iterations); // 10000
/// @endcode
/// @see AdaDelta
struct AdaDeltaOptions {
  double decay_rate{0.95}; ///< Squared-history exponential decay in [0,1). Default 0.95.
  double epsilon{1e-6}; ///< Positive numerical offset; its root placement is specified by the method. Default 1e-6.
  std::size_t max_iterations{10000}; ///< Maximum native iterations/epochs as described by optimize(); zero performs no updates. Default 10000.
  double tolerance{1e-8}; ///< Stopping threshold; its actual gradient/change/displacement/scale meaning is method-specific. Default 1e-8.
};
/// @brief AdaDelta on the DifferentiableFunction callback interface.
/// Uses squared-gradient and squared-update exponential averages. The update multiplies the gradient by sqrt(previous_update_average+epsilon)/sqrt(current_gradient_average+epsilon). No separate learning-rate option. Tests the gradient used after applying its update; O(p) reset local state.
/// The caller supplies finite, dimensionally consistent callbacks. Inputs are borrowed;
/// coordinates mutate in place and can retain a partial state if a callback throws.
/// Calls use local optimizer state and propagate callback exceptions; sharing a mutable
/// objective across threads requires external synchronisation. Normal return alone
/// is not a uniform convergence certificate.
/// @par Example
/// @code
/// const dm::optim::AdaDelta optimizer;
/// std::print("{}\n",optimizer.options().max_iterations); // 10000
/// @endcode
/// @par References
/// - M. D. Zeiler (2012), ADADELTA: An Adaptive Learning Rate Method, https://arxiv.org/abs/1212.5701
/// @see AdaDeltaOptions, trace_optimize()
class AdaDelta {
 public:
  /// @brief Store a copy of the controls without running the objective.
  /// @param options Method-specific controls; defaults are listed by AdaDeltaOptions.
  /// @par Example
  /// @code
  /// const dm::optim::AdaDelta optimizer;
  /// std::print("{}\n",optimizer.options().max_iterations); // 10000
  /// @endcode
  /// @see optimize()
  explicit AdaDelta(AdaDeltaOptions options = {});
  /// @brief Run AdaDelta from the supplied point using the configured finite budget.
  /// Uses squared-gradient and squared-update exponential averages. The update multiplies the gradient by sqrt(previous_update_average+epsilon)/sqrt(current_gradient_average+epsilon). No separate learning-rate option. Tests the gradient used after applying its update; O(p) reset local state.
  /// @param function Borrowed DifferentiableFunction with matching finite output shapes.
  /// @param coordinates Owned-by-caller finite start; overwritten with the native returned point.
  /// @return Objective at the native returned coordinates; not a stop-status code.
  /// @throws std::invalid_argument AdaDelta: decay_rate must be in [0, 1); AdaDelta: epsilon must be positive; AdaDelta: tolerance must be non-negative.
  /// @throws std::exception Exceptions from supplied callbacks propagate. Raw solvers do not uniformly reject nonfinite inputs.
  /// @par Example
  /// @code
  /// dm::optim::QuadraticObjective f({2,8},{1,-2});
  /// dm::optim::AdaDeltaOptions controls;controls.max_iterations=5;
  /// std::vector<double> x{0,0};
  /// const auto value=dm::optim::AdaDelta(controls).optimize(f,x);
  /// std::print("{}\n",std::isfinite(value)); // true
  /// @endcode
  /// @see trace_optimize(), AdaDeltaOptions
  double optimize(DifferentiableFunction& function, std::vector<double>& coordinates) const;
  /// @brief Borrow this optimizer's stored configuration.
  /// @return Const option reference, valid during this object's lifetime; no mutation.
  [[nodiscard]] const AdaDeltaOptions& options() const noexcept {return options_;}
 private:
  AdaDeltaOptions options_; ///< Owned method configuration, copied at construction.
};
} // namespace datamunge::optim
