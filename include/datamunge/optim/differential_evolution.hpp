#pragma once
/// @file differential_evolution.hpp
/// @brief DifferentialEvolution controls and native optimization contract.
#include <datamunge/optim/function_types.hpp>
#include <cstddef>
#include <cstdint>
#include <string>
namespace datamunge::optim {
/// @brief Configuration for DifferentialEvolution; fields retain their documented native defaults.
/// Supply finite values in the supported ranges. Controls are copied at construction;
/// validation, stopping and loss scaling belong to optimize().
/// @par Example
/// @code
/// dm::optim::DEOptions o;
/// std::print("{}\n",o.max_generations); // 500
/// @endcode
/// @see DifferentialEvolution
struct DEOptions {
  std::size_t population_size{50}; ///< Population size; CMA-ES zero selects its dimension-based default. Default 50.
  std::size_t max_generations{500}; ///< Maximum population generations. Default 500.
  double differential_weight{0.8}; ///< Differential mutation scale F. Default 0.8.
  double crossover_rate{0.9}; ///< Crossover probability CR/per-gene inclusion as defined by the method. Default 0.9.
  std::string mutation_strategy{"rand1"}; ///< rand1, best1, current_to_best1 or rand2. Default "rand1".
  std::string crossover_strategy{"binomial"}; ///< DE binomial/exponential or GA single_point/uniform/blend. Default "binomial".
  double tolerance{1e-10}; ///< Stopping threshold; its actual gradient/change/displacement/scale meaning is method-specific. Default 1e-10.
  std::uint64_t seed{42}; ///< Local mt19937_64 seed; same-build reproducibility, not guaranteed independent streams. Default 42.
};
/// @brief DifferentialEvolution on the ArbitraryFunction callback interface.
/// Box-constrained differential mutation/crossover with greedy trial replacement. Supports the declared rand1/best1/current_to_best1/rand2 and binomial/exponential strategies. Coordinates seed one clamped member; more than twenty small-best-improvement generations can stop the run. O(population_size*p) storage; no global certificate.
/// The caller supplies finite, dimensionally consistent callbacks. Inputs are borrowed;
/// coordinates mutate in place and can retain a partial state if a callback throws.
/// Calls use local optimizer state and propagate callback exceptions; sharing a mutable
/// objective across threads requires external synchronisation. Normal return alone
/// is not a uniform convergence certificate.
/// @par Example
/// @code
/// const dm::optim::DifferentialEvolution optimizer;
/// std::print("{}\n",optimizer.options().max_generations); // 500
/// @endcode
/// @par References
/// - R. Storn and K. Price (1997), Differential Evolution - A Simple and Efficient Heuristic for global Optimization over Continuous Spaces, https://doi.org/10.1023/A:1008202821328
/// @see DEOptions, trace_optimize()
class DifferentialEvolution {
 public:
  /// @brief Store a copy of the controls without running the objective.
  /// @param options Method-specific controls; defaults are listed by DEOptions.
  /// @par Example
  /// @code
  /// const dm::optim::DifferentialEvolution optimizer;
  /// std::print("{}\n",optimizer.options().max_generations); // 500
  /// @endcode
  /// @see optimize()
  explicit DifferentialEvolution(DEOptions options = {});
  /// @brief Run DifferentialEvolution from the supplied point using the configured finite budget.
  /// Box-constrained differential mutation/crossover with greedy trial replacement. Supports the declared rand1/best1/current_to_best1/rand2 and binomial/exponential strategies. Coordinates seed one clamped member; more than twenty small-best-improvement generations can stop the run. O(population_size*p) storage; no global certificate.
  /// @param function Borrowed ArbitraryFunction with matching finite output shapes.
  /// @param coordinates Owned-by-caller finite start; overwritten with the native returned point.
  /// @param lower_bound Matching finite inclusive lower box limits.
  /// @param upper_bound Matching finite inclusive upper limits, greater than lower limits.
  /// @return Objective at the native returned coordinates; not a stop-status code.
  /// @throws std::invalid_argument DifferentialEvolution::optimize: crossover_strategy must be 'binomial' or 'exponential'.
  /// @throws std::exception Exceptions from supplied callbacks propagate. Raw solvers do not uniformly reject nonfinite inputs.
  /// @par Example
  /// @code
  /// dm::optim::QuadraticObjective f({2,8},{1,-2});
  /// dm::optim::DEOptions controls;controls.max_generations=5;
  /// std::vector<double> x{0,0};
  /// const auto value=dm::optim::DifferentialEvolution(controls).optimize(f,x,std::vector<double>{-3,-3},std::vector<double>{3,3});
  /// std::print("{}\n",std::isfinite(value)); // true
  /// @endcode
  /// @see trace_optimize(), DEOptions
  double optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                     const std::vector<double>& lower_bound, const std::vector<double>& upper_bound) const;
  /// @brief Borrow this optimizer's stored configuration.
  /// @return Const option reference, valid during this object's lifetime; no mutation.
  [[nodiscard]] const DEOptions& options() const noexcept {return options_;}
 private:
  DEOptions options_; ///< Owned method configuration, copied at construction.
};
} // namespace datamunge::optim
