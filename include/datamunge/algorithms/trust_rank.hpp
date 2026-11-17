#pragma once
/// @file trust_rank.hpp
/// @brief Checked trust rank operations and producer diagnostics.
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>
namespace datamunge::algorithms {
/// @brief Actual personalized random-surfer iteration.
/// @par Example
/// @code
/// auto r=dm::algorithms::trust_rank_work(2,{{0,1}},{0});
/// @endcode
/// @par References
/// - Actual canonical producer; report derives the recurrence.
/// @see Related legacy convenience overloads in this header.
struct TrustRankWork {
  std::vector<double> rank;     ///< Actual last iterate, including cap-limited result.
  std::vector<double> teleport; ///< Normalized seed multiplicities.
  std::vector<std::vector<double>>
                      iterates; ///< Initial iterate then actual updates, bounded by max_records.
  std::vector<double> deltas;   ///< Actual L1 successive-iterate differences.
  std::size_t         iterations{0};    ///< Actual completed updates.
  std::size_t         edge_visits{0};   ///< Actual directed edge contributions.
  bool                converged{false}; ///< Last successive L1 delta below tolerance.
};
/// @brief Run seeded PageRank with seed-distributed dangling mass.
/// Inputs borrowed; calls independent. L1 contraction alpha gives final fixed-point error <=alpha*last_delta/(1-alpha). O(iterations*(n+edges)) work; capped at100000000 vertex/edge visits.
/// @param n Positive vertex count, at most1000000.
/// @param edges Directed edges, duplicates preserve multiplicity, at most10000000.
/// @param seeds Nonempty in-range IDs; repeats add teleport mass.
/// @param damping Finite alpha in [0,1), default.85.
/// @param tolerance Finite positive successive-iterate L1 threshold, default1e-12.
/// @param max_iterations Update ceiling;0 returns initial seed distribution unresolved.
/// @param max_records Initial/update snapshots retained up to this count, default0; at most1000000 total values.
/// @return Actual iterates and explicit convergence state.
/// @throws std::invalid_argument Invalid dimensions, endpoints, seeds or finite parameters.
/// @throws std::length_error Storage/work budget exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::trust_rank_work(3,{{0,1},{1,2},{2,0}},{0});
/// @endcode
/// @par References
/// - Actual canonical producer; report derives the recurrence.
/// @see Related legacy convenience overloads in this header.
TrustRankWork trust_rank_work(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    const std::vector<std::size_t>&                         seeds,
    double                                                  damping        = .85,
    double                                                  tolerance      = 1e-12,
    std::size_t                                             max_iterations = 1000,
    std::size_t                                             max_records    = 0);

} // namespace datamunge::algorithms
