#pragma once

/// @file
/// @brief Binary and permutation genetic search with steady-state and island modes.

#include <datamunge/optim/discrete_distribution_search.hpp>

namespace datamunge::optim {

/// @brief Feasible representation and crossover domain.
/// @par Example
/// @code
/// const auto encoding = dm::optim::GeneticEncoding::permutation;
/// @endcode
/// @see discrete_genetic_search()
enum class GeneticEncoding {
  binary,      ///< Independent bit mutation and uniform crossover.
  permutation  ///< Order crossover with swap or inversion mutation.
};

/// @brief Replacement, variation and island migration controls.
/// @par Example
/// @code
/// dm::optim::DiscreteGeneticOptions options;
/// options.islands = 3;
/// options.steady_state = true;
/// @endcode
/// @see discrete_genetic_search()
struct DiscreteGeneticOptions {
  std::size_t evaluations{1000}; ///< Positive objective-call budget, including initialization.
  std::uint64_t seed{42}; ///< Seed of the local engine.
  GeneticEncoding encoding{GeneticEncoding::binary}; ///< Representation and operators.
  std::size_t population{24}; ///< Members per island, at least two.
  std::size_t tournament{3}; ///< Positive number of candidates drawn per tournament.
  double crossover{0.9}; ///< Crossover probability in [0,1].
  double mutation{0.1}; ///< Binary probability per bit or probability per permutation in [0,1].
  double inversion{0.5}; ///< Conditional probability of inversion given permutation mutation, in [0,1].
  bool steady_state{false}; ///< Replace the worst in place after each accepted offspring.
  std::size_t islands{1}; ///< Positive number of independent populations.
  std::size_t migration_interval{8}; ///< Completed generations or steady-state sweeps per migration; zero disables.
};

/// @brief Optimize binary decisions or permutations using typed genetic operators.
///
/// Binary states contain zero or one per coordinate. Permutations contain
/// all indices from zero through dimensions minus one, each once. The
/// generational solver carries one elite and generates the rest by tournament
/// selection. Steady-state replacement compares each child with the worst
/// member. Islands initialize and evolve in round-robin order. A synchronous
/// ring migration copies each island's best member to the next island if
/// it improves the recipient's worst loss. Migrant scores are reused and
/// require a deterministic objective. Duplicate sampled offspring count.
/// Replay holds within a build. Exceptions from the callback propagate.
/// @param objective Borrowed callback returning finite losses on the selected domain.
/// @param dimensions Positive number of bits or permutation elements.
/// @param options Budget, seed, representation and evolution controls.
/// @return Earliest best evaluated state and exact objective-call count.
/// @throws std::invalid_argument Invalid dimension, objective or active options.
/// @throws std::domain_error The objective returns nonfinite loss.
/// @par Example
/// @code
/// const auto result = dm::optim::discrete_genetic_search(
///     [](const auto& point) { return static_cast<double>(point[0]); }, 1, {32, 7});
/// std::print("{}\n", result.value); // 0
/// @endcode
/// @par References
/// - Holland (1975), Adaptation in Natural and Artificial Systems.
/// - Davis (1985), Applying Adaptive Algorithms to Epistatic Domains.
/// - Whitley (1989), The GENITOR Algorithm and Selection Pressure.
/// @see discrete_distribution_search(), GeneticAlgorithm
DiscreteDistributionResult discrete_genetic_search(const CategoricalObjective& objective,
                                                    std::size_t dimensions,
                                                    const DiscreteGeneticOptions& options = {});

} // namespace datamunge::optim
