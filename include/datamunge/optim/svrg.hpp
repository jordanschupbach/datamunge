#pragma once
/// @file svrg.hpp
/// @brief SVRG controls and native optimization contract.
#include <datamunge/optim/function_types.hpp>
#include <cstddef>
#include <cstdint>
namespace datamunge::optim {
/// @brief Configuration for SVRG; fields retain their documented native defaults.
/// Supply finite values in the supported ranges. Controls are copied at construction;
/// validation, stopping and loss scaling belong to optimize().
/// @par Example
/// @code
/// dm::optim::SVRGOptions o;
/// std::print("{}\n",o.max_epochs); // 100
/// @endcode
/// @see SVRG
struct SVRGOptions {
  double step_size{0.01}; ///< Fixed learning-rate/initial search step; use the objective scaling stated by optimize(). Default 0.01.
  std::size_t max_epochs{100}; ///< Maximum passes/outer epochs; zero performs no epoch updates. Default 100.
  std::size_t inner_iterations{0}; ///< Inner updates per outer stage; for SVRG zero uses component count. Default 0.
  double tolerance{1e-8}; ///< Stopping threshold; its actual gradient/change/displacement/scale meaning is method-specific. Default 1e-8.
  std::uint64_t seed{42}; ///< Local mt19937_64 seed; same-build reproducibility, not guaranteed independent streams. Default 42.
};
/// @brief SVRG on the DifferentiableSeparableFunction callback interface.
/// Uses mean component gradients: current_i-snapshot_i+full_snapshot_gradient/n. The division by n is essential for an unbiased mean-loss estimator and matches SGD/SAGA scale. Earlier code omitted it. Stops on absolute full-sum objective change per epoch. O(p) snapshot state, two component-gradient calls per inner update plus one full gradient per epoch.
/// The caller supplies finite, dimensionally consistent callbacks. Inputs are borrowed;
/// coordinates mutate in place and can retain a partial state if a callback throws.
/// Calls use local optimizer state and propagate callback exceptions; sharing a mutable
/// objective across threads requires external synchronisation. Normal return alone
/// is not a uniform convergence certificate.
/// @par Example
/// @code
/// const dm::optim::SVRG optimizer;
/// std::print("{}\n",optimizer.options().max_epochs); // 100
/// @endcode
/// @par References
/// - R. Johnson and T. Zhang (2013), Accelerating Stochastic Gradient Descent using Predictive Variance Reduction, https://proceedings.neurips.cc/paper/2013/hash/ac1dd209cbcc5e5d1c6e28598e8cbbe8-Abstract.html
/// @see SVRGOptions, trace_optimize()
class SVRG {
 public:
  /// @brief Store a copy of the controls without running the objective.
  /// @param options Method-specific controls; defaults are listed by SVRGOptions.
  /// @par Example
  /// @code
  /// const dm::optim::SVRG optimizer;
  /// std::print("{}\n",optimizer.options().max_epochs); // 100
  /// @endcode
  /// @see optimize()
  explicit SVRG(SVRGOptions options = {});
  /// @brief Run SVRG from the supplied point using the configured finite budget.
  /// Uses mean component gradients: current_i-snapshot_i+full_snapshot_gradient/n. The division by n is essential for an unbiased mean-loss estimator and matches SGD/SAGA scale. Earlier code omitted it. Stops on absolute full-sum objective change per epoch. O(p) snapshot state, two component-gradient calls per inner update plus one full gradient per epoch.
  /// @param function Borrowed DifferentiableSeparableFunction with matching finite output shapes.
  /// @param coordinates Owned-by-caller finite start; overwritten with the native returned point.
  /// @return Objective at the native returned coordinates; not a stop-status code.
  /// @throws std::invalid_argument SVRG: step_size must be positive; SVRG: tolerance must be non-negative; SVRG: function has zero terms.
  /// @throws std::exception Exceptions from supplied callbacks propagate. Raw solvers do not uniformly reject nonfinite inputs.
  /// @par Example
  /// @code
  /// dm::optim::LeastSquaresObjective f({{1},{1}},{1,3});
  /// dm::optim::SVRGOptions controls;controls.max_epochs=5;
  /// std::vector<double> x{0};
  /// const auto value=dm::optim::SVRG(controls).optimize(f,x);
  /// std::print("{}\n",std::isfinite(value)); // true
  /// @endcode
  /// @see trace_optimize(), SVRGOptions
  double optimize(DifferentiableSeparableFunction& function, std::vector<double>& coordinates) const;
  /// @brief Borrow this optimizer's stored configuration.
  /// @return Const option reference, valid during this object's lifetime; no mutation.
  [[nodiscard]] const SVRGOptions& options() const noexcept {return options_;}
 private:
  SVRGOptions options_; ///< Owned method configuration, copied at construction.
};
} // namespace datamunge::optim
