#pragma once
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>
namespace datamunge::algorithms {
struct TourResult{std::vector<std::size_t> tour;double cost{0};};
TourResult nearest_neighbor_tsp(const std::vector<std::vector<double>>& distance,std::size_t start=0);
TourResult christofides_tsp(const std::vector<std::vector<double>>& distance,std::size_t start=0);
struct VehicleRoutes{std::vector<std::vector<std::size_t>> routes;double total_distance{0};};
VehicleRoutes clarke_wright_savings(const std::vector<std::pair<double,double>>& points,
 const std::vector<double>& demand,double capacity,std::size_t depot=0);
/// @brief Attempt a deterministic Warnsdorff knight tour, returning only complete paths.
///
/// Chooses the unvisited legal neighbour with smallest onward degree, breaking
/// ties by row then column. Marks the start visited first and never backtracks.
/// A complete result visits every board cell exactly once; closure to the start
/// is not required. Failure returns an empty vector and does not prove that no
/// tour exists. O(rows*columns) time and storage, with at most eight candidate
/// and 64 onward-direction checks per decision. Calls share no mutable state.
/// @param rows Positive board height.
/// @param columns Positive board width; the product must fit addressable storage.
/// @param start_row Zero-based starting row, default zero.
/// @param start_column Zero-based starting column, default zero.
/// @return Owned (row,column) path on success; empty on a greedy dead end.
/// @throws std::invalid_argument If dimensions or the starting coordinate are invalid.
/// @throws std::length_error If the board cannot fit the supported storage range.
/// @par Example
/// @code
/// const auto path=dm::algorithms::warnsdorff_knight_tour(8,8);
/// const auto cells=path.size(); // 64
/// @endcode
/// @par References
/// - S. L. Marateck (2008), How good is the Warnsdorff's knight's tour heuristic?,
///   https://arxiv.org/abs/0803.4321 (this implementation uses coordinate ties,
///   rather than the fixed move-order ties studied in that paper).
/// @see warnsdorff_attempt(), warnsdorff_start_profile()
std::vector<std::pair<int,int>> warnsdorff_knight_tour(int rows,int columns,int start_row=0,int start_column=0);
/// @brief Weighted path-search output, distinct from the sorted-array SearchResult type.
struct InformedSearchResult {
  double cost{0}; ///< Total edge weight of a found path; zero when no path was found.
  std::vector<std::size_t> path; ///< Vertices from start through the reached goal, or empty.
  std::size_t expanded{0}; ///< Number of queue entries expanded by the search.
  bool found{false}; ///< Whether a goal was reached.
};
/// @brief Finds a path to goal using A* with a caller-supplied per-vertex heuristic.
/// @throws std::invalid_argument For invalid vertices, negative edge weights, or a wrong heuristic size.
InformedSearchResult a_star_search(std::size_t n,const std::vector<std::tuple<std::size_t,std::size_t,double>>& edges,
 std::size_t start,std::size_t goal,const std::vector<double>& heuristic);
/// @brief Finds a path to any listed goal using a per-vertex lower bound to the goal set.
/// @throws std::invalid_argument For an empty goal set, invalid vertices or weights, or wrong bound size.
InformedSearchResult b_star_search(std::size_t n,const std::vector<std::tuple<std::size_t,std::size_t,double>>& edges,
 std::size_t start,const std::vector<std::size_t>& goals,const std::vector<double>& goal_set_lower_bound);
}
