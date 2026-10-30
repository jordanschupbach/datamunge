#pragma once
/// @file amsgrad.hpp
/// @brief AMSGrad controls and native optimization contract.
#include <datamunge/optim/function_types.hpp>
#include <cstddef>
namespace datamunge::optim {
/// @brief Configuration for AMSGrad; fields retain their documented native defaults.
/// Supply finite values in the supported ranges. Controls are copied at construction;
/// validation, stopping and loss scaling belong to optimize().
/// @par Example
/// @code
/// dm::optim::AMSGradOptions o;
/// std::print("{}\n",o.max_iterations); // 10000
/// @endcode
/// @see AMSGrad
struct AMSGradOptions {
  double step_size{0.001}; ///< Fixed learning-rate/initial search step; use the objective scaling stated by optimize(). Default 0.001.
  double beta1{0.9}; ///< First-moment decay in [0,1). Default 0.9.
  double beta2{0.999}; ///< Second-moment decay in [0,1). Default 0.999.
  double epsilon{1e-8}; ///< Positive numerical offset; its root placement is specified by the method. Default 1e-8.
  std::size_t max_iterations{10000}; ///< Maximum native iterations/epochs as described by optimize(); zero performs no updates. Default 10000.
  double tolerance{1e-8}; ///< Stopping threshold; its actual gradient/change/displacement/scale meaning is method-specific. Default 1e-8.
};
/// @brief AMSGrad on the DifferentiableFunction callback interface.
/// Bias-corrected AMSGrad variant: maximises uncorrected second moments, then divides that maximum by 1-beta2^t. The corrected denominator need not be monotone, so the original uncorrected theorem is not automatically inherited. Tests the gradient used after updating; O(p) reset state.
/// The caller supplies finite, dimensionally consistent callbacks. Inputs are borrowed;
/// coordinates mutate in place and can retain a partial state if a callback throws.
/// Calls use local optimizer state and propagate callback exceptions; sharing a mutable
/// objective across threads requires external synchronisation. Normal return alone
/// is not a uniform convergence certificate.
/// @par Example
/// @code
/// const dm::optim::AMSGrad optimizer;
/// std::print("{}\n",optimizer.options().max_iterations); // 10000
/// @endcode
/// @par References
/// - S. Reddi, S. Kale and S. Kumar (2018), On the Convergence of Adam and Beyond, https://openreview.net/forum?id=ryQu7f-RZ
/// @see AMSGradOptions, trace_optimize()
class AMSGrad {
 public:
  /// @brief Store a copy of the controls without running the objective.
  /// @param options Method-specific controls; defaults are listed by AMSGradOptions.
  /// @par Example
  /// @code
  /// const dm::optim::AMSGrad optimizer;
  /// std::print("{}\n",optimizer.options().max_iterations); // 10000
  /// @endcode
  /// @see optimize()
  explicit AMSGrad(AMSGradOptions options = {});
  /// @brief Run AMSGrad from the supplied point using the configured finite budget.
  /// Bias-corrected AMSGrad variant: maximises uncorrected second moments, then divides that maximum by 1-beta2^t. The corrected denominator need not be monotone, so the original uncorrected theorem is not automatically inherited. Tests the gradient used after updating; O(p) reset state.
  /// @param function Borrowed DifferentiableFunction with matching finite output shapes.
  /// @param coordinates Owned-by-caller finite start; overwritten with the native returned point.
  /// @return Objective at the native returned coordinates; not a stop-status code.
  /// @throws std::invalid_argument AMSGrad: step_size must be positive; AMSGrad: beta1 must be in [0, 1); AMSGrad: beta2 must be in [0, 1); AMSGrad: epsilon must be positive; AMSGrad: tolerance must be non-negative.
  /// @throws std::exception Exceptions from supplied callbacks propagate. Raw solvers do not uniformly reject nonfinite inputs.
  /// @par Example
  /// @code
  /// dm::optim::QuadraticObjective f({2,8},{1,-2});
  /// dm::optim::AMSGradOptions controls;controls.max_iterations=5;
  /// std::vector<double> x{0,0};
  /// const auto value=dm::optim::AMSGrad(controls).optimize(f,x);
  /// std::print("{}\n",std::isfinite(value)); // true
  /// @endcode
  /// @see trace_optimize(), AMSGradOptions
  double optimize(DifferentiableFunction& function, std::vector<double>& coordinates) const;
  /// @brief Borrow this optimizer's stored configuration.
  /// @return Const option reference, valid during this object's lifetime; no mutation.
  [[nodiscard]] const AMSGradOptions& options() const noexcept {return options_;}
 private:
  AMSGradOptions options_; ///< Owned method configuration, copied at construction.
};
} // namespace datamunge::optim
