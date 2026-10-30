#pragma once
/// @file adagrad.hpp
/// @brief AdaGrad controls and native optimization contract.
#include <datamunge/optim/function_types.hpp>
#include <cstddef>
namespace datamunge::optim {
/// @brief Configuration for AdaGrad; fields retain their documented native defaults.
/// Supply finite values in the supported ranges. Controls are copied at construction;
/// validation, stopping and loss scaling belong to optimize().
/// @par Example
/// @code
/// dm::optim::AdaGradOptions o;
/// std::print("{}\n",o.max_iterations); // 10000
/// @endcode
/// @see AdaGrad
struct AdaGradOptions {
  double step_size{0.01}; ///< Fixed learning-rate/initial search step; use the objective scaling stated by optimize(). Default 0.01.
  double epsilon{1e-8}; ///< Positive numerical offset; its root placement is specified by the method. Default 1e-8.
  std::size_t max_iterations{10000}; ///< Maximum native iterations/epochs as described by optimize(); zero performs no updates. Default 10000.
  double tolerance{1e-8}; ///< Stopping threshold; its actual gradient/change/displacement/scale meaning is method-specific. Default 1e-8.
};
/// @brief AdaGrad on the DifferentiableFunction callback interface.
/// Accumulates squared full gradients; divides each update by sqrt(sum+epsilon). Tests the pre-update gradient norm after applying the update. Every optimize call resets the accumulator. O(p) local storage.
/// The caller supplies finite, dimensionally consistent callbacks. Inputs are borrowed;
/// coordinates mutate in place and can retain a partial state if a callback throws.
/// Calls use local optimizer state and propagate callback exceptions; sharing a mutable
/// objective across threads requires external synchronisation. Normal return alone
/// is not a uniform convergence certificate.
/// @par Example
/// @code
/// const dm::optim::AdaGrad optimizer;
/// std::print("{}\n",optimizer.options().max_iterations); // 10000
/// @endcode
/// @par References
/// - J. Duchi, E. Hazan and Y. Singer (2011), Adaptive Subgradient Methods for Online Learning and Stochastic Optimization, https://jmlr.org/papers/v12/duchi11a.html
/// @see AdaGradOptions, trace_optimize()
class AdaGrad {
 public:
  /// @brief Store a copy of the controls without running the objective.
  /// @param options Method-specific controls; defaults are listed by AdaGradOptions.
  /// @par Example
  /// @code
  /// const dm::optim::AdaGrad optimizer;
  /// std::print("{}\n",optimizer.options().max_iterations); // 10000
  /// @endcode
  /// @see optimize()
  explicit AdaGrad(AdaGradOptions options = {});
  /// @brief Run AdaGrad from the supplied point using the configured finite budget.
  /// Accumulates squared full gradients; divides each update by sqrt(sum+epsilon). Tests the pre-update gradient norm after applying the update. Every optimize call resets the accumulator. O(p) local storage.
  /// @param function Borrowed DifferentiableFunction with matching finite output shapes.
  /// @param coordinates Owned-by-caller finite start; overwritten with the native returned point.
  /// @return Objective at the native returned coordinates; not a stop-status code.
  /// @throws std::invalid_argument AdaGrad: step_size must be positive; AdaGrad: epsilon must be positive; AdaGrad: tolerance must be non-negative.
  /// @throws std::exception Exceptions from supplied callbacks propagate. Raw solvers do not uniformly reject nonfinite inputs.
  /// @par Example
  /// @code
  /// dm::optim::QuadraticObjective f({2,8},{1,-2});
  /// dm::optim::AdaGradOptions controls;controls.max_iterations=5;
  /// std::vector<double> x{0,0};
  /// const auto value=dm::optim::AdaGrad(controls).optimize(f,x);
  /// std::print("{}\n",std::isfinite(value)); // true
  /// @endcode
  /// @see trace_optimize(), AdaGradOptions
  double optimize(DifferentiableFunction& function, std::vector<double>& coordinates) const;
  /// @brief Borrow this optimizer's stored configuration.
  /// @return Const option reference, valid during this object's lifetime; no mutation.
  [[nodiscard]] const AdaGradOptions& options() const noexcept {return options_;}
 private:
  AdaGradOptions options_; ///< Owned method configuration, copied at construction.
};
} // namespace datamunge::optim
