#pragma once

/// @file
/// @brief Pareto-ranked ant colony search for symmetric multiobjective tours.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::optim {

/// @brief Total tour budget and Pareto-ranked pheromone controls.
/// @par Example
/// @code
/// const dm::optim::ParetoAntOptions options{128, 7, 12, 24};
/// @endcode
/// @see pareto_ant_colony_tour()
struct ParetoAntOptions {
  std::size_t evaluations{1000}; ///< Positive full-vector tour score budget.
  std::uint64_t seed{42}; ///< Seed of the local engine.
  std::size_t ants{16}; ///< Positive number of tours per pheromone update.
  std::size_t capacity{32}; ///< Positive Pareto archive capacity.
  double alpha{1}; ///< Nonnegative finite pheromone exponent.
  double beta{2}; ///< Nonnegative finite inverse heuristic exponent.
  double evaporation{0.2}; ///< Evaporation fraction strictly between zero and one.
};

/// @brief One evaluated feasible tour and its vector of minimization objectives.
/// @par Example
/// @code
/// const dm::optim::ParetoTour point{{0,1,2}, {3,6}};
/// @endcode
/// @see ParetoAntResult
struct ParetoTour {
  std::vector<std::size_t> tour; ///< Permutation with implicit closing edge.
  std::vector<double> objectives; ///< Tour lengths in cost-matrix order.
};

/// @brief Nondominated tour archive and exact objective-call count.
/// @par Example
/// @code
/// const dm::optim::ParetoAntResult result{{}, 0};
/// @endcode
/// @see pareto_ant_colony_tour()
struct ParetoAntResult {
  std::vector<ParetoTour> front; ///< Unique nondominated objective vectors and corresponding tours.
  std::size_t evaluations; ///< Number of fully scored tours.
};

/// @brief Construct symmetric tours with Pareto-ranked colony reinforcement.
///
/// Each ant draws a random positive blend of the supplied objectives for
/// constructing a tour. The shared pheromone matrix evaporates each batch.
/// All ants reinforce edges inversely to their nondominated rank, and the
/// nondominated archive adds another reinforcement. The archive keeps
/// unique objective vectors and uses crowding to limit its size.
/// One evaluation computes every objective of one complete tour. Scalar
/// edge guidance and rank calculations do not consume evaluations.
/// Partial batches and duplicates count. Replay holds within a build.
/// @param costs At least two complete symmetric cost matrices of the same size, at least three cities.
/// @param options Budget, archive capacity, seed and pheromone controls.
/// @return Owned Pareto tour archive and exact evaluation count.
/// @throws std::invalid_argument Invalid matrices or controls.
/// @throws std::length_error Shared ranker rejects an oversized ant cohort.
/// @par Example
/// @code
/// const std::vector<std::vector<double>> triangle{{0,1,1},{1,0,1},{1,1,0}};
/// const auto result = dm::optim::pareto_ant_colony_tour({triangle,triangle}, {12});
/// std::print("{}\n", result.front[0].objectives[0]); // 3
/// @endcode
/// @par References
/// - Dorigo, Maniezzo and Colorni (1996). https://doi.org/10.1109/3477.484436
/// - García-Martínez, Cordón and Herrera (2007), A taxonomy and an empirical analysis of
///   multiple objective ant colony optimization algorithms for the bi-criteria TSP.
/// @see ant_colony_tour(), pareto_search()
ParetoAntResult pareto_ant_colony_tour(const std::vector<std::vector<std::vector<double>>>& costs,
                                      const ParetoAntOptions& options = {});

} // namespace datamunge::optim
