#pragma once
#include <datamunge/algorithms/informed_search.hpp>
#include <datamunge/algorithms/christofides.hpp>
#include <datamunge/algorithms/clarke_wright.hpp>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>
namespace datamunge::algorithms {
TourResult nearest_neighbor_tsp(const std::vector<std::vector<double>>& distance,std::size_t start=0);
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
}
