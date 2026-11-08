#pragma once
/// @file unweighted_search_result.hpp
/// @brief Shared owned result layout for unweighted graph searchers.
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {
/// @brief Owned path, producer-specific traversal sequence and success flag.
/// Field order/types retain the historical search_batch.hpp interface. Interpret
/// order using the producer's contract: BFS records dequeues, while the legacy
/// bidirectional_search wrapper reports the path in that field. Other searchers
/// may revisit vertices. No input graph storage is borrowed.
/// @par Example
/// @code
/// const auto result=dm::algorithms::breadth_first_search(3,{{0,1},{1,2}},0,2);
/// std::print("{} {}\n",result.path,result.found); // [0, 1, 2] true
/// @endcode
/// @see breadth_first_search(), breadth_first_search_trace()
struct UnweightedSearchResult {
    std::vector<std::size_t> path; ///< Producer's selected start-to-goal vertex path, or empty if not found.
    std::vector<std::size_t> order; ///< Producer-specific traversal sequence; BFS uses actual dequeue order including the goal.
    bool found{false}; ///< Whether the requested goal was reached.
};
} // namespace datamunge::algorithms
