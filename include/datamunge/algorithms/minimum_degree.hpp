#pragma once
/// @file minimum_degree.hpp
/// @brief Actual minimum-current-degree elimination and fixed-order fill analysis.
#include <cstddef>
#include <utility>
#include <vector>
namespace datamunge::algorithms {
/// @brief Actual state immediately before and during one vertex elimination.
/// Neighbors are sorted live vertex IDs; fill contains each newly created
/// undirected edge once, with smaller endpoint first. No numeric factorization.
/// @par Example
/// @code
/// auto t=dm::algorithms::minimum_degree_trace({{1,2},{0},{0}});
/// std::print("{}\n",t.steps.front().vertex); // 1
/// @endcode
/// @see MinimumDegreeTrace
struct MinimumDegreeStep {
  int                              vertex{-1}; ///< Actually eliminated vertex.
  std::vector<int>                 neighbors;  ///< Actual sorted live neighbors before elimination.
  std::vector<std::pair<int, int>> fill;       ///< Actual newly inserted undirected edges.
};
/// @brief Owned input, actual elimination order and optional captured decisions.
/// Duplicate references and loops are ignored; asymmetric input is interpreted
/// as its symmetric union. Entry cap counts raw input references, input row
/// records, order IDs, step records, neighbor IDs and fill-edge records.
/// @par Example
/// @code
/// auto t=dm::algorithms::elimination_trace({{1,2},{0},{0}},{0,1,2});
/// std::print("{}\n",t.fill_edges); // 1
/// @endcode
/// @see minimum_degree_trace(), elimination_trace()
struct MinimumDegreeTrace {
  std::vector<std::vector<int>>  input; ///< Owned original lists, including duplicates/loops.
  std::vector<int>               order; ///< Actual completed permutation.
  std::vector<MinimumDegreeStep> steps; ///< Actual steps in execution order.
  std::size_t                    fill_edges{0};  ///< Total new undirected edges.
  std::size_t                    pair_trials{0}; ///< Actual neighbor-pair insertion attempts.
  std::size_t captured_entries{0}; ///< Actual captured units under the stated contract.
};
/// @brief Greedily eliminate minimum live degree, breaking ties by smallest ID.
/// Input IDs must be in [0,N); loops/duplicates ignored, asymmetric edges
/// symmetrized. The legacy order is preserved on symmetric valid inputs.
/// Set-based complete degree scans and fill trials give O(N^3 log(N+1))
/// worst-case work and O(N^2) driver storage. No minimum-fill guarantee.
/// @param adjacency Raw graph lists, including the empty graph.
/// @return Elimination permutation.
/// @throws std::invalid_argument If any ID is invalid.
/// @throws std::length_error If vertex count exceeds INT_MAX.
/// @par Example
/// @code
/// auto o=dm::algorithms::minimum_degree_ordering({{1,2},{0},{0}});
/// std::print("{} {} {}\n",o[0],o[1],o[2]); // 1 0 2
/// @endcode
/// @par References
/// - W. F. Tinney and J. W. Walker (1967), Direct solutions of sparse network equations by optimally ordered triangular factorization, Proceedings of the IEEE55,1801–1809, doi:10.1109/PROC.1967.6011.
/// @see minimum_degree_trace()
std::vector<int> minimum_degree_ordering(const std::vector<std::vector<int>>& adjacency);
/// @brief Capture actual greedy elimination, including every newly inserted fill edge.
/// Same producer as minimum_degree_ordering(), without replay. Actual entry
/// admission throws before exceeding the optional quota; no partial trace is
/// returned. Capture storage is proportional to admitted entries. Zero disables.
/// @param adjacency Raw graph lists with the ordinary normalization contract.
/// @param max_entries Capture cap, default1000000; zero disables.
/// @return Owned actual decisions and complete order.
/// @throws std::invalid_argument If an ID is invalid.
/// @throws std::length_error If graph geometry or capture quota is exceeded.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// auto t=dm::algorithms::minimum_degree_trace({{1,2},{0},{0}});
/// std::print("{} {}\n",t.order.front(),t.fill_edges); // 1 0
/// @endcode
/// @see minimum_degree_ordering(), elimination_trace()
MinimumDegreeTrace minimum_degree_trace(
    const std::vector<std::vector<int>>& adjacency,
    std::size_t                          max_entries = 1000000);
/// @brief Capture actual fill for a prescribed complete elimination permutation.
/// Uses the same fill producer as greedy ordering, replacing only vertex choice.
/// Same normalization, quota units, ownership and worst-case complexity.
/// @param adjacency Raw graph lists.
/// @param order Complete permutation, including empty for an empty graph.
/// @param max_entries Capture cap, default1000000; zero disables.
/// @return Owned actual fixed-order fill history.
/// @throws std::invalid_argument If any graph ID or permutation is invalid.
/// @throws std::length_error If geometry or quota exceeds its limit.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// auto t=dm::algorithms::elimination_trace({{1,2},{0},{0}},{0,1,2});
/// std::print("{}\n",t.fill_edges); // 1
/// @endcode
/// @see minimum_degree_trace()
MinimumDegreeTrace elimination_trace(
    const std::vector<std::vector<int>>& adjacency,
    const std::vector<int>&              order,
    std::size_t                          max_entries = 1000000);
/// @brief Construct the original arrow graph, with center0 adjacent to all others.
/// @param vertices Nonnegative vertex count, default16.
/// @return Owned symmetric simple adjacency lists, empty for zero.
/// @throws std::invalid_argument If vertices is negative.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::minimum_degree_arrow(4)[0].size()); // 3
/// @endcode
/// @see minimum_degree_trace()
std::vector<std::vector<int>> minimum_degree_arrow(int vertices = 16);
/// @brief Construct the natural increasing-ID elimination permutation.
/// @param vertices Nonnegative vertex count.
/// @return Owned IDs0 through vertices−1.
/// @throws std::invalid_argument If vertices is negative.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::elimination_natural_order(4).back()); // 3
/// @endcode
/// @see elimination_trace()
std::vector<int> elimination_natural_order(int vertices);
} // namespace datamunge::algorithms
