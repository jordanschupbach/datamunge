#pragma once
/// @file levenberg_marquardt.hpp
/// @brief LevenbergMarquardt controls and native optimization contract.
#include <datamunge/optim/function_types.hpp>
namespace datamunge::optim {
/// @brief Configuration for LevenbergMarquardt; fields retain their documented native defaults.
/// Supply finite values in the supported ranges. Controls are copied at construction;
/// validation, stopping and loss scaling belong to optimize().
/// @par Example
/// @code
/// dm::optim::LevenbergMarquardtOptions o;
/// std::print("{}\n",o.max_iterations); // 200
/// @endcode
/// @see LevenbergMarquardt
struct LevenbergMarquardtOptions {
  double initial_damping{1e-3}; ///< Positive initial LM diagonal shift. Default 1e-3.
  std::size_t max_iterations{200}; ///< Maximum native iterations/epochs as described by optimize(); zero performs no updates. Default 200.
  double tolerance{1e-8}; ///< Stopping threshold; its actual gradient/change/displacement/scale meaning is method-specific. Default 1e-8.
};
/// @brief LevenbergMarquardt on the ResidualFunction callback interface.
/// Forms J transpose J plus scalar damping*I and solves by unpivoted elimination. Accepts only strict reductions of half residual SSE, multiplying damping by .3 on acceptance and 10 on rejection. Tests proposed step norm even after rejection. A small rejected step is not a gradient certificate; no parameter constraints. O(np^2+p^3) work.
/// The caller supplies finite, dimensionally consistent callbacks. Inputs are borrowed;
/// coordinates mutate in place and can retain a partial state if a callback throws.
/// Calls use local optimizer state and propagate callback exceptions; sharing a mutable
/// objective across threads requires external synchronisation. Normal return alone
/// is not a uniform convergence certificate.
/// @par Example
/// @code
/// const dm::optim::LevenbergMarquardt optimizer;
/// std::print("{}\n",optimizer.options().max_iterations); // 200
/// @endcode
/// @par References
/// - K. Levenberg (1944), A Method for the Solution of Certain Non-Linear Problems in Least Squares, https://doi.org/10.1090/qam/10666; D. W. Marquardt (1963), An Algorithm for Least-Squares Estimation of Nonlinear Parameters, https://doi.org/10.1137/0111030
/// @see LevenbergMarquardtOptions, trace_optimize()
class LevenbergMarquardt {
 public:
  /// @brief Store a copy of the controls without running the objective.
  /// @param options Method-specific controls; defaults are listed by LevenbergMarquardtOptions.
  /// @par Example
  /// @code
  /// const dm::optim::LevenbergMarquardt optimizer;
  /// std::print("{}\n",optimizer.options().max_iterations); // 200
  /// @endcode
  /// @see optimize()
  explicit LevenbergMarquardt(LevenbergMarquardtOptions options = {});
  /// @brief Run LevenbergMarquardt from the supplied point using the configured finite budget.
  /// Forms J transpose J plus scalar damping*I and solves by unpivoted elimination. Accepts only strict reductions of half residual SSE, multiplying damping by .3 on acceptance and 10 on rejection. Tests proposed step norm even after rejection. A small rejected step is not a gradient certificate; no parameter constraints. O(np^2+p^3) work.
  /// @param function Borrowed ResidualFunction with matching finite output shapes.
  /// @param coordinates Owned-by-caller finite start; overwritten with the native returned point.
  /// @return Objective at the native returned coordinates (half residual SSE); not a stop-status code.
  /// @throws std::invalid_argument LevenbergMarquardt: initial_damping must be positive.
  /// @throws std::exception Exceptions from supplied callbacks propagate. Raw solvers do not uniformly reject nonfinite inputs.
  /// @par Example
  /// @code
  /// const auto data=dm::optim::optimization_curve_data();dm::optim::CurveResidual f(data);
  /// dm::optim::LevenbergMarquardtOptions controls;controls.max_iterations=5;
  /// std::vector<double> x{2,.5};
  /// const auto value=dm::optim::LevenbergMarquardt(controls).optimize(f,x);
  /// std::print("{}\n",std::isfinite(value)); // true
  /// @endcode
  /// @see trace_optimize(), LevenbergMarquardtOptions
  double optimize(ResidualFunction& function,std::vector<double>& coordinates) const;
  /// @brief Borrow this optimizer's stored configuration.
  /// @return Const option reference, valid during this object's lifetime; no mutation.
  [[nodiscard]] const LevenbergMarquardtOptions& options() const noexcept {return options_;}
 private:
  LevenbergMarquardtOptions options_; ///< Owned method configuration, copied at construction.
};
} // namespace datamunge::optim
