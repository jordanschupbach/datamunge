#pragma once
/// @file basic_path_result.hpp
/// @brief Shared owned result layout for the search_more path-search producers.
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {
/// @brief Owned selected vertex path, historical visit counter and success flag.
/// Field order/types preserve the original search_more.hpp layout. Interpret
/// expanded by producer: brute_force_search counts entered simple-path prefixes,
/// including goal and bound-pruned visits; depth_first_search counts first vertex
/// entries; iterative_deepening_dfs accumulates path-prefix entries across depth
/// limits, including goals and cutoffs. Thus expanded is not generally a count
/// of unique vertices or scanned adjacency lists. No input storage is borrowed.
/// @par Example
/// @code
/// const auto result=dm::algorithms::brute_force_search(3,{{0,1},{1,2}},0,2);
/// std::print("{} {} {}\n",result.path,result.expanded,result.found); // [0, 1, 2] 3 true
/// @endcode
/// @see brute_force_search(), depth_first_search(), iterative_deepening_dfs()
struct BasicPathResult {
    std::vector<std::size_t> path; ///< Producer's selected start-to-goal vertex sequence, or empty on failure.
    std::size_t expanded{0}; ///< Producer-specific state-entry count, including terminal/cutoff visits.
    bool found{false}; ///< Whether a permitted path to the requested goal was found.
};
} // namespace datamunge::algorithms
