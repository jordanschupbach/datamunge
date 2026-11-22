#pragma once

/// @file
/// @brief Ant colony optimization for complete symmetric traveling-salesperson instances.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::optim {

/// @brief Pheromone update policy for ant-colony tour search.
/// @par Example
/// @code
/// const auto policy = dm::optim::AntColonyPolicy::ant_colony_system;
/// @endcode
/// @see ant_colony_tour()
enum class AntColonyPolicy {
  ant_system,          ///< Evaporate globally and reinforce all completed tours.
  ant_colony_system,   ///< Greedy or proportional selection, local decay, global-best reinforcement.
  max_min,             ///< Iteration-best update with bounded trails and stagnation restarts.
  rank_based,          ///< Reinforce the ranked top ants and the best overall tour.
  elitist              ///< Reinforce all ants and add an extra best-overall deposit.
};

/// @brief Budget, transition weights and pheromone settings for tour search.
/// @par Example
/// @code
/// dm::optim::AntColonyOptions options;
/// options.policy = dm::optim::AntColonyPolicy::max_min;
/// options.evaluations = 128;
/// @endcode
/// @see ant_colony_tour()
struct AntColonyOptions {
  std::size_t evaluations{1000}; ///< Positive full-tour budget, counting repeats.
  std::size_t ants{20}; ///< Positive number of tours between global updates.
  std::uint64_t seed{42}; ///< Seed of the local mt19937_64 engine.
  AntColonyPolicy policy{AntColonyPolicy::ant_system}; ///< Transition and update policy.
  double alpha{1}; ///< Pheromone exponent, finite and nonnegative; ACS uses one.
  double beta{2}; ///< Inverse-distance exponent, finite and nonnegative.
  double evaporation{0.2}; ///< Global evaporation fraction in (0, 1).
  double exploitation{0.9}; ///< ACS probability of greedy choice, in [0, 1].
  double local_decay{0.1}; ///< ACS local decay toward its starting trail, in [0, 1].
  std::size_t rank_count{6}; ///< Rank-based weights; best overall has this positive weight.
  double elite_weight{2}; ///< Elitist extra best-overall weight, finite and nonnegative.
  double min_trail_ratio{0.05}; ///< MMAS floor as fraction of its current ceiling, in (0, 1].
  std::size_t restart_after{25}; ///< MMAS stagnant iterations before reset; zero disables resets.
};

/// @brief Owned best tour and work summary.
/// @par Example
/// @code
/// const dm::optim::AntColonyResult result{{0, 1, 2}, 3, 1, 0};
/// @endcode
/// @see ant_colony_tour()
struct AntColonyResult {
  std::vector<std::size_t> tour; ///< Zero-based permutation; the closing edge is implicit.
  double length; ///< Length of the earliest shortest scored tour.
  std::size_t evaluations; ///< Number of tours scored against the budget.
  std::size_t restarts; ///< MMAS trail resets performed.
};

/// @brief Construct and score ant-colony tours on a complete symmetric graph.
///
/// Each ant starts at a random city and visits each remaining city once,
/// sampling with weights proportional to pheromone^alpha × distance^(-beta).
/// ACS also makes greedy choices, performs immediate local updates and globally
/// updates only best-overall edges. MMAS reinforces the iteration best and caps
/// trails using best-overall length, with a configurable floor ratio.
/// Ant System, rank-based and elitist policies differ in which tours deposit
/// pheromone. The algorithm spends the full positive budget, including partial
/// ant batches. A nearest-neighbor tour initializes ACS trails and enters
/// the incumbent competition as the first budgeted evaluation for ACS.
/// Replay is repeatable within a build. No callbacks or input mutation.
/// @param distances Square symmetric finite cost matrix for at least three cities.
///        Diagonal zero, off-diagonal strictly positive. n × max edge must fit a double.
/// @param options Budget, seed and update policy, defaulting to Ant System.
/// @return Owned best tour, its length and exact evaluation and restart counts.
/// @throws std::invalid_argument Invalid costs, shape or options.
/// @par Example
/// @code
/// const auto result = dm::optim::ant_colony_tour({{0,1,1},{1,0,1},{1,1,0}});
/// std::print("{}\n", result.length); // 3
/// @endcode
/// @par References
/// - Dorigo, Maniezzo and Colorni (1996). https://doi.org/10.1109/3477.484436
/// - Dorigo and Gambardella (1997). https://doi.org/10.1109/4235.585892
/// - Stützle and Hoos (2000). https://doi.org/10.1016/S0167-739X(00)00043-1
/// - Bullnheimer, Hartl and Strauss (1999). A New Rank Based Version of the Ant System.
/// @see AntColonyOptions, neighborhood_search()
AntColonyResult ant_colony_tour(const std::vector<std::vector<double>>& distances,
                               const AntColonyOptions& options = {});

} // namespace datamunge::optim
