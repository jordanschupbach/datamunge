#pragma once

/// @file informed_search.hpp
/// @brief Directed single- and multi-goal A* search, with optional expansion snapshots.
#include <cstddef>
#include <tuple>
#include <vector>

namespace datamunge::algorithms {
/// @brief Weighted path-search output, distinct from the sorted-array SearchResult type.
/// Owns its path. Unreachable answers have found=false, cost zero and an empty path.
/// @par Example
/// @code
/// const auto result = dm::algorithms::a_star_search(2, {{0,1,3}}, 0, 1, {3,0});
/// std::print("{} {}\n", result.found, result.cost); // true 3
/// @endcode
/// @see a_star_search(), b_star_search()
struct InformedSearchResult {
  double cost{0}; ///< Total edge weight of a found path; zero when no path was found.
  std::vector<std::size_t> path; ///< Vertices from start through the reached goal, or empty.
  std::size_t expanded{0}; ///< Number of queue entries expanded by the search.
  bool found{false}; ///< Whether a goal was reached.
};
/// @brief Finds a directed path to one goal with reopening A*.
/// Same semantics and requirements as b_star_search(), with one goal. No shared state.
/// @param n Number of vertices, labelled 0 through n-1.
/// @param edges Directed (source,destination,weight) edges; weights are finite and nonnegative.
/// @param start Valid initial vertex.
/// @param goal Valid destination vertex.
/// @param heuristic Finite per-vertex estimates; admissibility and zero at the goal ensure optimality.
/// @return Owned path, cost and expansion count; found=false and an empty path if unreachable.
/// @throws std::invalid_argument For invalid vertices, weights, heuristic size or nonfinite estimates.
/// @throws std::overflow_error If an accumulated path cost or queue priority is not finite.
/// @par Example
/// @code
/// const auto r = dm::algorithms::a_star_search(2, {{0,1,3}}, 0, 1, {3,0});
/// std::print("{}\n", r.cost); // 3
/// @endcode
/// @par References
/// - P. E. Hart, N. J. Nilsson and B. Raphael (1968), A Formal Basis for the
///   Heuristic Determination of Minimum Cost Paths, doi:10.1109/TSSC.1968.300136.
/// @see b_star_search(), b_star_search_trace()
InformedSearchResult a_star_search(std::size_t n,const std::vector<std::tuple<std::size_t,std::size_t,double>>& edges,
 std::size_t start,std::size_t goal,const std::vector<double>& heuristic);
/// @brief Finds a cheapest directed path to a goal set using multi-goal A*.
/// The historical b_star name denotes multi-goal A*, not Berliner's proof-tree B*.
/// Finite nonnegative edges and a finite admissible bound with zero at every goal
/// give an optimal first removed goal (up to floating arithmetic). Admissibility
/// is the caller's responsibility and is not checked. An inconsistent bound is
/// supported: improved vertices reopen. Equal priorities prefer the smaller vertex;
/// equal-cost relaxations retain the first parent. Duplicate goals are allowed.
/// With a consistent bound, O((n+m) log(n+m)) time and O(n+m) storage for m edges;
/// inconsistent bounds can cause exponentially many re-expansions and queued labels.
/// No input mutation or shared state. The expansion count includes the removed goal
/// and repeated expansions, excludes stale queue entries, and is zero only if no
/// entry was expanded. Unreachable results have cost zero and an empty path.
/// @param n Positive vertex count, labelled 0 through n-1.
/// @param edges Directed (source,destination,weight) edges; finite nonnegative weights.
/// @param start Valid initial vertex; if already a goal, returns the singleton path at cost zero.
/// @param goals Nonempty list of valid goal vertices.
/// @param goal_set_lower_bound Finite vector of size n; estimates the cost to the nearest goal.
/// @return Owned path, cost, expansion count and found flag.
/// @throws std::invalid_argument For an empty goal set, invalid vertices, weights, size or nonfinite bound.
/// @throws std::overflow_error If an accumulated path cost or queue priority is not finite.
/// @par Example
/// @code
/// const auto r = dm::algorithms::b_star_search(4, {{0,1,1},{1,2,5},{0,3,2}},
///                                           0, {2,3}, {2,5,0,0});
/// std::print("{} {}\n", r.path.back(), r.cost); // 3 2
/// @endcode
/// @par References
/// - P. E. Hart, N. J. Nilsson and B. Raphael (1968), A Formal Basis for the
///   Heuristic Determination of Minimum Cost Paths, doi:10.1109/TSSC.1968.300136.
/// @see a_star_search(), b_star_search_trace()
InformedSearchResult b_star_search(std::size_t n,const std::vector<std::tuple<std::size_t,std::size_t,double>>& edges,
 std::size_t start,const std::vector<std::size_t>& goals,const std::vector<double>& goal_set_lower_bound);

/// @brief One non-stale queue removal, recorded before its outgoing edges are relaxed.
/// Values are snapshots owned by InformedSearchTrace; goal removals are included.
/// @par Example
/// @code
/// const auto trace = dm::algorithms::b_star_search_trace(2, {{0,1,3}}, 0, {1}, {3,0});
/// std::print("{}\n", trace.expansions.back().path_cost); // 3
/// @endcode
/// @see InformedSearchTrace, b_star_search_trace()
struct InformedSearchExpansion {
  std::size_t vertex{0}; ///< Removed vertex, possibly seen in an earlier expansion.
  double path_cost{0}; ///< Best known cost g at removal.
  double priority{0}; ///< Queue key g+h at removal.
  std::size_t queued{0}; ///< Remaining queue entries, including stale entries.
};

/// @brief Owned multi-goal search answer and chronological expansion snapshots.
/// Recording adds O(k) storage for k expansions; no mutable state is shared.
/// @par Example
/// @code
/// const auto trace = dm::algorithms::b_star_search_trace(2, {{0,1,3}}, 0, {1}, {3,0});
/// std::print("{} {}\n", trace.result.cost, trace.expansions.size()); // 3 2
/// @endcode
/// @see b_star_search_trace()
struct InformedSearchTrace {
  InformedSearchResult result; ///< Same answer and count as the unrecorded search.
  std::vector<InformedSearchExpansion> expansions; ///< Non-stale removals in execution order.
};

/// @brief Records multi-goal A* expansions using the same loop as b_star_search().
/// All search, tie, reopening and optimality semantics match b_star_search().
/// @param n Positive vertex count.
/// @param edges Directed finite nonnegative weighted edges.
/// @param start Valid initial vertex.
/// @param goals Nonempty list of valid goal vertices; duplicates are allowed.
/// @param goal_set_lower_bound Finite length-n estimates to the nearest goal.
/// @return Owned result and one snapshot per counted expansion, including the goal.
/// @throws std::invalid_argument For invalid vertices, weights, goal set or bound.
/// @throws std::overflow_error If path costs or queue priorities become nonfinite.
/// @par Example
/// @code
/// const auto trace = dm::algorithms::b_star_search_trace(2, {{0,1,3}}, 0, {1}, {3,0});
/// std::print("{} {}\n", trace.expansions.front().vertex, trace.result.cost); // 0 3
/// @endcode
/// @par References
/// - P. E. Hart, N. J. Nilsson and B. Raphael (1968), A Formal Basis for the
///   Heuristic Determination of Minimum Cost Paths, doi:10.1109/TSSC.1968.300136.
/// @see b_star_search(), InformedSearchExpansion
InformedSearchTrace b_star_search_trace(std::size_t n,
 const std::vector<std::tuple<std::size_t,std::size_t,double>>& edges,
 std::size_t start,const std::vector<std::size_t>& goals,
 const std::vector<double>& goal_set_lower_bound);
} // namespace datamunge::algorithms
