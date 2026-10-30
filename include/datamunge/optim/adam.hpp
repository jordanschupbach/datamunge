#pragma once
/// @file adam.hpp
/// @brief Adam controls and native optimization contract.
#include <datamunge/optim/function_types.hpp>
#include <cstddef>
namespace datamunge::optim {
/// @brief Configuration for Adam; fields retain their documented native defaults.
/// Supply finite values in the supported ranges. Controls are copied at construction;
/// validation, stopping and loss scaling belong to optimize().
/// @par Example
/// @code
/// dm::optim::AdamOptions o;
/// std::print("{}\n",o.max_iterations); // 10000
/// @endcode
/// @see Adam
struct AdamOptions {
  double step_size{0.001}; ///< Fixed learning-rate/initial search step; use the objective scaling stated by optimize(). Default 0.001.
  double beta1{0.9}; ///< First-moment decay in [0,1). Default 0.9.
  double beta2{0.999}; ///< Second-moment decay in [0,1). Default 0.999.
  double epsilon{1e-8}; ///< Positive numerical offset; its root placement is specified by the method. Default 1e-8.
  std::size_t max_iterations{10000}; ///< Maximum native iterations/epochs as described by optimize(); zero performs no updates. Default 10000.
  double tolerance{1e-8}; ///< Stopping threshold; its actual gradient/change/displacement/scale meaning is method-specific. Default 1e-8.
};
/// @brief Adam on the DifferentiableFunction callback interface.
/// Bias-corrected full-batch first and second moments, denominator sqrt(vhat)+epsilon. Tests the pre-update gradient norm after applying its update. No line search or general nonconvex convergence guarantee; O(p) reset local state.
/// The caller supplies finite, dimensionally consistent callbacks. Inputs are borrowed;
/// coordinates mutate in place and can retain a partial state if a callback throws.
/// Calls use local optimizer state and propagate callback exceptions; sharing a mutable
/// objective across threads requires external synchronisation. Normal return alone
/// is not a uniform convergence certificate.
/// @par Example
/// @code
/// const dm::optim::Adam optimizer;
/// std::print("{}\n",optimizer.options().max_iterations); // 10000
/// @endcode
/// @par References
/// - D. P. Kingma and J. Ba (2015), Adam: A Method for Stochastic Optimization, https://arxiv.org/abs/1412.6980
/// @see AdamOptions, trace_optimize()
class Adam {
 public:
  /// @brief Store a copy of the controls without running the objective.
  /// @param options Method-specific controls; defaults are listed by AdamOptions.
  /// @par Example
  /// @code
  /// const dm::optim::Adam optimizer;
  /// std::print("{}\n",optimizer.options().max_iterations); // 10000
  /// @endcode
  /// @see optimize()
  explicit Adam(AdamOptions options = {});
  /// @brief Run Adam from the supplied point using the configured finite budget.
  /// Bias-corrected full-batch first and second moments, denominator sqrt(vhat)+epsilon. Tests the pre-update gradient norm after applying its update. No line search or general nonconvex convergence guarantee; O(p) reset local state.
  /// @param function Borrowed DifferentiableFunction with matching finite output shapes.
  /// @param coordinates Owned-by-caller finite start; overwritten with the native returned point.
  /// @return Objective at the native returned coordinates; not a stop-status code.
  /// @throws std::exception Exceptions from supplied callbacks propagate. Raw solvers do not uniformly reject nonfinite inputs.
  /// @par Example
  /// @code
  /// dm::optim::QuadraticObjective f({2,8},{1,-2});
  /// dm::optim::AdamOptions controls;controls.max_iterations=5;
  /// std::vector<double> x{0,0};
  /// const auto value=dm::optim::Adam(controls).optimize(f,x);
  /// std::print("{}\n",std::isfinite(value)); // true
  /// @endcode
  /// @see trace_optimize(), AdamOptions
  double optimize(DifferentiableFunction& function, std::vector<double>& coordinates) const;
  /// @brief Borrow this optimizer's stored configuration.
  /// @return Const option reference, valid during this object's lifetime; no mutation.
  [[nodiscard]] const AdamOptions& options() const noexcept {return options_;}
 private:
  AdamOptions options_; ///< Owned method configuration, copied at construction.
};
} // namespace datamunge::optim
