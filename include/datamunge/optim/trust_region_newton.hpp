#pragma once
/// @file trust_region_newton.hpp
/// @brief TrustRegionNewton controls and native optimization contract.
#include <datamunge/optim/function_types.hpp>
namespace datamunge::optim {
/// @brief Configuration for TrustRegionNewton; fields retain their documented native defaults.
/// Supply finite values in the supported ranges. Controls are copied at construction;
/// validation, stopping and loss scaling belong to optimize().
/// @par Example
/// @code
/// dm::optim::TrustRegionNewtonOptions o;
/// std::print("{}\n",o.max_iterations); // 200
/// @endcode
/// @see TrustRegionNewton
struct TrustRegionNewtonOptions {
  double initial_radius{1.0}; ///< Positive starting Cauchy-step radius. Default 1.0.
  double max_radius{1000.0}; ///< Maximum radius after successful Cauchy steps. Default 1000.0.
  std::size_t max_iterations{200}; ///< Maximum native iterations/epochs as described by optimize(); zero performs no updates. Default 200.
  double tolerance{1e-8}; ///< Stopping threshold; its actual gradient/change/displacement/scale meaning is method-specific. Default 1e-8.
};
/// @brief TrustRegionNewton on the HessianFunction callback interface.
/// Cauchy-point variant: clips a steepest-descent ray using gHg and the current radius. Accepts any strict objective decrease and doubles the radius, otherwise halves it. It does not compute a model-agreement ratio, dogleg step or exact trust-region subproblem. Stops on gradient norm or tiny radius; O(p^2) Hessian work.
/// The caller supplies finite, dimensionally consistent callbacks. Inputs are borrowed;
/// coordinates mutate in place and can retain a partial state if a callback throws.
/// Calls use local optimizer state and propagate callback exceptions; sharing a mutable
/// objective across threads requires external synchronisation. Normal return alone
/// is not a uniform convergence certificate.
/// @par Example
/// @code
/// const dm::optim::TrustRegionNewton optimizer;
/// std::print("{}\n",optimizer.options().max_iterations); // 200
/// @endcode
/// @par References
/// - J. Nocedal and S. J. Wright (2006), Numerical Optimization, second edition, https://doi.org/10.1007/978-0-387-40065-5. The implemented variant and stopping rules are specified above.
/// @see TrustRegionNewtonOptions, trace_optimize()
class TrustRegionNewton {
 public:
  /// @brief Store a copy of the controls without running the objective.
  /// @param options Method-specific controls; defaults are listed by TrustRegionNewtonOptions.
  /// @par Example
  /// @code
  /// const dm::optim::TrustRegionNewton optimizer;
  /// std::print("{}\n",optimizer.options().max_iterations); // 200
  /// @endcode
  /// @see optimize()
  explicit TrustRegionNewton(TrustRegionNewtonOptions options = {});
  /// @brief Run TrustRegionNewton from the supplied point using the configured finite budget.
  /// Cauchy-point variant: clips a steepest-descent ray using gHg and the current radius. Accepts any strict objective decrease and doubles the radius, otherwise halves it. It does not compute a model-agreement ratio, dogleg step or exact trust-region subproblem. Stops on gradient norm or tiny radius; O(p^2) Hessian work.
  /// @param function Borrowed HessianFunction with matching finite output shapes.
  /// @param coordinates Owned-by-caller finite start; overwritten with the native returned point.
  /// @return Objective at the native returned coordinates; not a stop-status code.
  /// @throws std::invalid_argument TrustRegionNewton: initial_radius must be positive.
  /// @throws std::exception Exceptions from supplied callbacks propagate. Raw solvers do not uniformly reject nonfinite inputs.
  /// @par Example
  /// @code
  /// dm::optim::QuadraticObjective f({2,8},{1,-2});
  /// dm::optim::TrustRegionNewtonOptions controls;controls.max_iterations=5;
  /// std::vector<double> x{0,0};
  /// const auto value=dm::optim::TrustRegionNewton(controls).optimize(f,x);
  /// std::print("{}\n",std::isfinite(value)); // true
  /// @endcode
  /// @see trace_optimize(), TrustRegionNewtonOptions
  double optimize(HessianFunction& function,std::vector<double>& coordinates) const;
  /// @brief Borrow this optimizer's stored configuration.
  /// @return Const option reference, valid during this object's lifetime; no mutation.
  [[nodiscard]] const TrustRegionNewtonOptions& options() const noexcept {return options_;}
 private:
  TrustRegionNewtonOptions options_; ///< Owned method configuration, copied at construction.
};
} // namespace datamunge::optim
