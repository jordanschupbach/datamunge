#pragma once
/// @file genetic_algorithm.hpp
/// @brief GeneticAlgorithm controls and native optimization contract.
#include <datamunge/optim/function_types.hpp>
#include <cstddef>
#include <cstdint>
#include <string>
namespace datamunge::optim {
/// @brief Configuration for GeneticAlgorithm; fields retain their documented native defaults.
/// Supply finite values in the supported ranges. Controls are copied at construction;
/// validation, stopping and loss scaling belong to optimize().
/// @par Example
/// @code
/// dm::optim::GAOptions o;
/// std::print("{}\n",o.max_generations); // 500
/// @endcode
/// @see GeneticAlgorithm
struct GAOptions {
  std::size_t population_size{100}; ///< Population size; CMA-ES zero selects its dimension-based default. Default 100.
  std::size_t max_generations{500}; ///< Maximum population generations. Default 500.
  double crossover_rate{0.8}; ///< Crossover probability CR/per-gene inclusion as defined by the method. Default 0.8.
  double mutation_rate{0.1}; ///< Per-gene Gaussian mutation probability. Default 0.1.
  double mutation_std_dev{0.1}; ///< Mutation SD as a fraction of the gene bounds width. Default 0.1.
  std::string selection_strategy{"tournament"}; ///< tournament, roulette or rank. Default "tournament".
  std::size_t tournament_size{3}; ///< Tournament candidate count. Default 3.
  std::string crossover_strategy{"blend"}; ///< DE binomial/exponential or GA single_point/uniform/blend. Default "blend".
  double blend_alpha{0.5}; ///< BLX-alpha expansion for blend crossover. Default 0.5.
  bool elitism{true}; ///< Preserve elite population members. Default true.
  std::size_t elite_count{2}; ///< Number of members retained under elitism. Default 2.
  double tolerance{1e-10}; ///< Stopping threshold; its actual gradient/change/displacement/scale meaning is method-specific. Default 1e-10.
  std::uint64_t seed{42}; ///< Local mt19937_64 seed; same-build reproducibility, not guaranteed independent streams. Default 42.
};
/// @brief GeneticAlgorithm on the ArbitraryFunction callback interface.
/// Real-valued box-constrained selection, crossover, Gaussian mutation and optional elitism. Initial coordinates seed a clamped population member. Returns the best visited point and can stop after more than twenty small-best-improvement generations. O(population_size*p) storage; strategy/rate choices affect results, with no global certificate.
/// The caller supplies finite, dimensionally consistent callbacks. Inputs are borrowed;
/// coordinates mutate in place and can retain a partial state if a callback throws.
/// Calls use local optimizer state and propagate callback exceptions; sharing a mutable
/// objective across threads requires external synchronisation. Normal return alone
/// is not a uniform convergence certificate.
/// @par Example
/// @code
/// const dm::optim::GeneticAlgorithm optimizer;
/// std::print("{}\n",optimizer.options().max_generations); // 500
/// @endcode
/// @par References
/// - J. H. Holland (1992 reprint), Adaptation in Natural and Artificial Systems, MIT Press, https://mitpress.mit.edu/9780262581110/adaptation-in-natural-and-artificial-systems/. This implementation uses real-valued operators, as specified above, rather than binary chromosomes.
/// @see GAOptions, trace_optimize()
class GeneticAlgorithm {
 public:
  /// @brief Store a copy of the controls without running the objective.
  /// @param options Method-specific controls; defaults are listed by GAOptions.
  /// @par Example
  /// @code
  /// const dm::optim::GeneticAlgorithm optimizer;
  /// std::print("{}\n",optimizer.options().max_generations); // 500
  /// @endcode
  /// @see optimize()
  explicit GeneticAlgorithm(GAOptions options = {});
  /// @brief Run GeneticAlgorithm from the supplied point using the configured finite budget.
  /// Real-valued box-constrained selection, crossover, Gaussian mutation and optional elitism. Initial coordinates seed a clamped population member. Returns the best visited point and can stop after more than twenty small-best-improvement generations. O(population_size*p) storage; strategy/rate choices affect results, with no global certificate.
  /// @param function Borrowed ArbitraryFunction with matching finite output shapes.
  /// @param coordinates Owned-by-caller finite start; overwritten with the native returned point.
  /// @param lower_bound Matching finite inclusive lower box limits.
  /// @param upper_bound Matching finite inclusive upper limits, greater than lower limits.
  /// @return Objective at the native returned coordinates; not a stop-status code.
  /// @throws std::invalid_argument GeneticAlgorithm::optimize: population_size must be at least 2.
  /// @throws std::exception Exceptions from supplied callbacks propagate. Raw solvers do not uniformly reject nonfinite inputs.
  /// @par Example
  /// @code
  /// dm::optim::QuadraticObjective f({2,8},{1,-2});
  /// dm::optim::GAOptions controls;controls.max_generations=5;
  /// std::vector<double> x{0,0};
  /// const auto value=dm::optim::GeneticAlgorithm(controls).optimize(f,x,std::vector<double>{-3,-3},std::vector<double>{3,3});
  /// std::print("{}\n",std::isfinite(value)); // true
  /// @endcode
  /// @see trace_optimize(), GAOptions
  double optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                     const std::vector<double>& lower_bound, const std::vector<double>& upper_bound) const;
  /// @brief Borrow this optimizer's stored configuration.
  /// @return Const option reference, valid during this object's lifetime; no mutation.
  [[nodiscard]] const GAOptions& options() const noexcept {return options_;}
 private:
  GAOptions options_; ///< Owned method configuration, copied at construction.
};
} // namespace datamunge::optim
