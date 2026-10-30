#pragma once
/// @file pso.hpp
/// @brief PSO controls and native optimization contract.
#include <datamunge/optim/function_types.hpp>
#include <cstddef>
#include <cstdint>
#include <string>
namespace datamunge::optim {
/// @brief Configuration for PSO; fields retain their documented native defaults.
/// Supply finite values in the supported ranges. Controls are copied at construction;
/// validation, stopping and loss scaling belong to optimize().
/// @par Example
/// @code
/// dm::optim::PSOOptions o;
/// std::print("{}\n",o.max_iterations); // 1000
/// @endcode
/// @see PSO
struct PSOOptions {
  std::size_t population_size{40}; ///< Population size; CMA-ES zero selects its dimension-based default. Default 40.
  std::size_t max_iterations{1000}; ///< Maximum native iterations/epochs as described by optimize(); zero performs no updates. Default 1000.
  double inertia_weight{0.7298}; ///< Initial or constant particle velocity carryover. Default 0.7298.
  double cognitive_coefficient{1.49618}; ///< Particle attraction to its own best position. Default 1.49618.
  double social_coefficient{1.49618}; ///< Particle attraction to its neighbourhood best. Default 1.49618.
  std::string topology{"global"}; ///< global or ring particle neighbourhood. Default "global".
  std::size_t ring_neighbors{2}; ///< Neighbours on each ring side when ring topology is selected. Default 2.
  std::string inertia_strategy{"constant"}; ///< constant or linear_decay. Default "constant".
  double final_inertia_weight{0.4}; ///< Final carryover for linear_decay inertia. Default 0.4.
  double tolerance{1e-10}; ///< Stopping threshold; its actual gradient/change/displacement/scale meaning is method-specific. Default 1e-10.
  std::uint64_t seed{42}; ///< Local mt19937_64 seed; same-build reproducibility, not guaranteed independent streams. Default 42.
};
/// @brief PSO on the ArbitraryFunction callback interface.
/// Box-constrained particle swarm with global or ring attraction and constant or linearly decayed inertia. Coordinates seed one clamped particle; returns the best visited feasible point. Stops after more than twenty small-best-improvement iterations or its budget. O(population_size*p) storage; no global-optimality certificate.
/// The caller supplies finite, dimensionally consistent callbacks. Inputs are borrowed;
/// coordinates mutate in place and can retain a partial state if a callback throws.
/// Calls use local optimizer state and propagate callback exceptions; sharing a mutable
/// objective across threads requires external synchronisation. Normal return alone
/// is not a uniform convergence certificate.
/// @par Example
/// @code
/// const dm::optim::PSO optimizer;
/// std::print("{}\n",optimizer.options().max_iterations); // 1000
/// @endcode
/// @par References
/// - J. Kennedy and R. Eberhart (1995), Particle Swarm Optimization, https://doi.org/10.1109/ICNN.1995.488968. This implementation adds the documented inertia, topology and clamping controls.
/// @see PSOOptions, trace_optimize()
class PSO {
 public:
  /// @brief Store a copy of the controls without running the objective.
  /// @param options Method-specific controls; defaults are listed by PSOOptions.
  /// @par Example
  /// @code
  /// const dm::optim::PSO optimizer;
  /// std::print("{}\n",optimizer.options().max_iterations); // 1000
  /// @endcode
  /// @see optimize()
  explicit PSO(PSOOptions options = {});
  /// @brief Run PSO from the supplied point using the configured finite budget.
  /// Box-constrained particle swarm with global or ring attraction and constant or linearly decayed inertia. Coordinates seed one clamped particle; returns the best visited feasible point. Stops after more than twenty small-best-improvement iterations or its budget. O(population_size*p) storage; no global-optimality certificate.
  /// @param function Borrowed ArbitraryFunction with matching finite output shapes.
  /// @param coordinates Owned-by-caller finite start; overwritten with the native returned point.
  /// @param lower_bound Matching finite inclusive lower box limits.
  /// @param upper_bound Matching finite inclusive upper limits, greater than lower limits.
  /// @return Objective at the native returned coordinates; not a stop-status code.
  /// @throws std::invalid_argument PSO::optimize: population_size must be at least 2; PSO::optimize: topology must be 'global' or 'ring'; PSO::optimize: inertia_strategy must be 'constant' or 'linear_decay'.
  /// @throws std::exception Exceptions from supplied callbacks propagate. Raw solvers do not uniformly reject nonfinite inputs.
  /// @par Example
  /// @code
  /// dm::optim::QuadraticObjective f({2,8},{1,-2});
  /// dm::optim::PSOOptions controls;controls.max_iterations=5;
  /// std::vector<double> x{0,0};
  /// const auto value=dm::optim::PSO(controls).optimize(f,x,std::vector<double>{-3,-3},std::vector<double>{3,3});
  /// std::print("{}\n",std::isfinite(value)); // true
  /// @endcode
  /// @see trace_optimize(), PSOOptions
  double optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                     const std::vector<double>& lower_bound, const std::vector<double>& upper_bound) const;
  /// @brief Borrow this optimizer's stored configuration.
  /// @return Const option reference, valid during this object's lifetime; no mutation.
  [[nodiscard]] const PSOOptions& options() const noexcept {return options_;}
 private:
  PSOOptions options_; ///< Owned method configuration, copied at construction.
};
} // namespace datamunge::optim
