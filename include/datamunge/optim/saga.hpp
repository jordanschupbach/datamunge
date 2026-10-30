#pragma once
/// @file saga.hpp
/// @brief SAGA controls and native optimization contract.
#include <datamunge/optim/function_types.hpp>
#include <cstddef>
#include <cstdint>
namespace datamunge::optim {
/// @brief Configuration for SAGA; fields retain their documented native defaults.
/// Supply finite values in the supported ranges. Controls are copied at construction;
/// validation, stopping and loss scaling belong to optimize().
/// @par Example
/// @code
/// dm::optim::SAGAOptions o;
/// std::print("{}\n",o.max_iterations); // 10000
/// @endcode
/// @see SAGA
struct SAGAOptions {
  double step_size{0.01}; ///< Fixed learning-rate/initial search step; use the objective scaling stated by optimize(). Default 0.01.
  std::size_t max_iterations{10000}; ///< Maximum native iterations/epochs as described by optimize(); zero performs no updates. Default 10000.
  double tolerance{1e-8}; ///< Stopping threshold; its actual gradient/change/displacement/scale meaning is method-specific. Default 1e-8.
  std::uint64_t seed{42}; ///< Local mt19937_64 seed; same-build reproducibility, not guaranteed independent streams. Default 42.
};
/// @brief SAGA on the DifferentiableSeparableFunction callback interface.
/// Uses current_i-table_i+mean(table), then refreshes one stored component gradient and its running mean. The estimator targets mean component loss while evaluate() returns their sum. Stops on epoch-boundary objective change or the last estimator norm; the latter is not the full gradient norm. O(n*p) gradient table.
/// The caller supplies finite, dimensionally consistent callbacks. Inputs are borrowed;
/// coordinates mutate in place and can retain a partial state if a callback throws.
/// Calls use local optimizer state and propagate callback exceptions; sharing a mutable
/// objective across threads requires external synchronisation. Normal return alone
/// is not a uniform convergence certificate.
/// @par Example
/// @code
/// const dm::optim::SAGA optimizer;
/// std::print("{}\n",optimizer.options().max_iterations); // 10000
/// @endcode
/// @par References
/// - A. Defazio, F. Bach and S. Lacoste-Julien (2014), SAGA: A Fast Incremental Gradient Method With Support for Non-Strongly Convex Composite Objectives, https://arxiv.org/abs/1407.0202
/// @see SAGAOptions, trace_optimize()
class SAGA {
 public:
  /// @brief Store a copy of the controls without running the objective.
  /// @param options Method-specific controls; defaults are listed by SAGAOptions.
  /// @par Example
  /// @code
  /// const dm::optim::SAGA optimizer;
  /// std::print("{}\n",optimizer.options().max_iterations); // 10000
  /// @endcode
  /// @see optimize()
  explicit SAGA(SAGAOptions options = {});
  /// @brief Run SAGA from the supplied point using the configured finite budget.
  /// Uses current_i-table_i+mean(table), then refreshes one stored component gradient and its running mean. The estimator targets mean component loss while evaluate() returns their sum. Stops on epoch-boundary objective change or the last estimator norm; the latter is not the full gradient norm. O(n*p) gradient table.
  /// @param function Borrowed DifferentiableSeparableFunction with matching finite output shapes.
  /// @param coordinates Owned-by-caller finite start; overwritten with the native returned point.
  /// @return Objective at the native returned coordinates; not a stop-status code.
  /// @throws std::invalid_argument SAGA: step_size must be positive; SAGA: tolerance must be non-negative; SAGA: function has zero terms.
  /// @throws std::exception Exceptions from supplied callbacks propagate. Raw solvers do not uniformly reject nonfinite inputs.
  /// @par Example
  /// @code
  /// dm::optim::LeastSquaresObjective f({{1},{1}},{1,3});
  /// dm::optim::SAGAOptions controls;controls.max_iterations=5;
  /// std::vector<double> x{0};
  /// const auto value=dm::optim::SAGA(controls).optimize(f,x);
  /// std::print("{}\n",std::isfinite(value)); // true
  /// @endcode
  /// @see trace_optimize(), SAGAOptions
  double optimize(DifferentiableSeparableFunction& function, std::vector<double>& coordinates) const;
  /// @brief Borrow this optimizer's stored configuration.
  /// @return Const option reference, valid during this object's lifetime; no mutation.
  [[nodiscard]] const SAGAOptions& options() const noexcept {return options_;}
 private:
  SAGAOptions options_; ///< Owned method configuration, copied at construction.
};
} // namespace datamunge::optim
