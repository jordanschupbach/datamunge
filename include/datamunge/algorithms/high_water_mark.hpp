#pragma once

/// @file high_water_mark.hpp
/// @brief The high-water mark of a heuristic, which brackets the work of greedy best-first search.
///
/// For a graph, a heuristic h and two vertices s and t, the high-water mark is
/// \f$H=\min_{P:s\to t}\max_{v\in P}h(v)\f$: every s-t path reaches \f$h\ge H\f$ somewhere, and some
/// path never exceeds \f$H\f$. Greedy best-first search from s to t (always expanding the open
/// vertex with the smallest h) expands every vertex reachable from s through vertices with
/// \f$h<H\f$ before it expands t, and never expands a vertex with \f$h>H\f$ (Wilt and Ruml 2014).

#include <cstddef>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

/// @brief Result of high_water_mark().
struct HighWaterMark {
    double      level = 0.0;   ///< \f$H\f$, the smallest achievable maximum of h along an s-t path.
    std::size_t below = 0;     ///< Vertices reachable from s through vertices with \f$h<H\f$.
    std::size_t at_most = 0;   ///< Vertices reachable from s through vertices with \f$h\le H\f$.
};

/// @brief Computes the high-water mark and the two regions that bracket greedy best-first search.
///
/// \f$H\f$ comes from a minimax variant of Dijkstra's algorithm (path "length" = largest h on the
/// path); the regions come from two floods from s. Greedy best-first search, as in
/// best_first_search(), expands between `below` and `at_most` vertices (plus t itself when t is
/// outside the first region).
///
/// @param n Number of vertices.
/// @param edges Edge list (u, v).
/// @param start The start vertex s.
/// @param goal The goal vertex t.
/// @param heuristic One value per vertex.
/// @param directed Whether edges are one-way.
/// @return H and the region sizes; H is infinite and both regions empty when t is unreachable.
/// @throws std::invalid_argument for out-of-range vertices or a heuristic of the wrong size.
///
/// @par Example
/// @code
/// const auto mark = dm::algorithms::high_water_mark(n, edges, s, t, h, false);
/// const auto run = dm::algorithms::best_first_search(n, edges, s, t, h, false);
/// // mark.below <= run.order.size() <= mark.at_most + 1
/// @endcode
/// @see best_first_search()
HighWaterMark high_water_mark(std::size_t n, const std::vector<std::pair<std::size_t, std::size_t>>& edges,
                              std::size_t start, std::size_t goal, const std::vector<double>& heuristic,
                              bool directed = true);

}  // namespace datamunge::algorithms
