#pragma once

/// @file
/// @brief Budgeted iterated Lin–Kernighan with variable-depth sequential exchanges.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::optim {

/// @brief Evaluation cap and neighbor limits for iterated Lin–Kernighan.
/// @par Example
/// @code
/// const dm::optim::LinKernighanOptions options{500, 7, 5, 8};
/// @endcode
/// @see iterated_lin_kernighan()
struct LinKernighanOptions {
  std::size_t evaluations{1000}; ///< Positive budget of full tour scores, including the start and kicks.
  std::uint64_t seed{42}; ///< Seed of the local starting-tour and kick engine.
  std::size_t max_depth{5}; ///< Maximum sequential edge exchanges, at least two.
  std::size_t candidate_limit{8}; ///< Positive number of nearest feasible continuation edges per step.
};

/// @brief Scored tour and variable-depth work summary.
/// @par Example
/// @code
/// const dm::optim::LinKernighanResult run{{0,1,2,3},4,1,0,0};
/// @endcode
/// @see iterated_lin_kernighan()
struct LinKernighanResult {
  std::vector<std::size_t> tour; ///< Earliest best tour, with implicit closing edge.
  double length; ///< Observed cycle length.
  std::size_t evaluations; ///< Count of full tour scores.
  std::size_t kicks; ///< Scored randomized perturbations.
  std::size_t deepest_improvement; ///< Largest accepted improving sequential exchange, zero if none.
};

/// @brief Optimize symmetric tours by variable-depth exchanges and iterated kicks.
///
/// A start may be supplied or drawn uniformly. Each local search opens an
/// edge, then recursively adds an edge from the loose end and removes an
/// adjacent edge to preserve one open Hamiltonian path. It prunes nonpositive
/// gains and explores the nearest continuations up to the configured depth.
/// Eligible closures are fully scored. Once descent stalls, a double-bridge
/// perturbation rearranges four segments (a random 2-opt kick on small tours)
/// and descent restarts. Kicks start from the best recorded tour. Evaluation
/// counts include the start, kicks and scored closures; incremental edge
/// comparisons are not full tour scores. Seeded replay holds within a build.
/// @param distances Complete symmetric square matrix for at least four cities;
///        diagonal zero, off-diagonal positive finite, with finite tour sums.
/// @param initial Feasible permutation of all cities; empty samples a start.
/// @param options Full-tour budget, seed and search depth controls.
/// @return Best scored tour with actual score, kick and depth counts.
/// @throws std::invalid_argument Invalid distances, budget, controls or permutation.
/// @par Example
/// @code
/// const auto run = dm::optim::iterated_lin_kernighan(
///     {{0,1,2,1},{1,0,1,2},{2,1,0,1},{1,2,1,0}}, {0,2,1,3}, {40,7});
/// std::print("{}\n", run.length); // 4
/// @endcode
/// @par References
/// - Lin and Kernighan (1973). https://doi.org/10.1287/opre.21.2.498
/// @see sample_two_opt(), ant_colony_tour()
LinKernighanResult iterated_lin_kernighan(const std::vector<std::vector<double>>& distances,
                                          std::vector<std::size_t> initial = {},
                                          const LinKernighanOptions& options = {});

} // namespace datamunge::optim
