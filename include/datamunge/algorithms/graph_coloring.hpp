#pragma once

#include <cstddef>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

/// @brief The result of a greedy vertex coloring.
struct ColoringResult {
    /// @brief color[v] is the color assigned to vertex v, an integer in 0..num_colors-1.
    std::vector<std::size_t> color;
    /// @brief Number of distinct colors used (the greedy always uses colors 0..num_colors-1).
    std::size_t num_colors{0};
};

/// @brief Greedy proper vertex coloring with the Welsh-Powell ordering (Welsh & Powell 1967).
///        A *proper coloring* assigns a color to each vertex so that no edge joins two vertices
///        of the same color; the smallest achievable number of colors is the *chromatic number*
///        chi(G), whose exact computation is NP-hard. This heuristic instead colors vertices one
///        at a time, always giving a vertex the smallest color not already used by one of its
///        already-colored neighbors. Processing vertices in order of *descending degree* (ties
///        broken by ascending index) -- the Welsh-Powell rule -- colors the hardest, highest-degree
///        vertices first and empirically keeps the count low. The greedy is *not* optimal, but it
///        always produces a proper coloring using at most Delta + 1 colors, where Delta is the
///        maximum degree.
///
///        The graph is treated as *undirected*: each pair {u, v} is an edge in both directions.
///        Self-loops (u == v) are *ignored* -- a vertex cannot conflict with itself, and admitting
///        them would make a proper coloring impossible. Duplicate/parallel edges are harmless.
///
/// @param n     number of vertices; valid vertex indices are 0..n-1.
/// @param edges undirected edges as index pairs; every endpoint must be < n.
/// @return a proper coloring of the n vertices.
/// @throws std::invalid_argument if any edge endpoint is >= n.
ColoringResult greedy_coloring(std::size_t n, const std::vector<std::pair<std::size_t, std::size_t>>& edges);

/// @brief Checks whether @p color is a proper coloring of the graph on @p n vertices with the
///        given @p edges: returns true iff no (non-self-loop) edge joins two equal-colored
///        vertices. Useful for certifying a coloring in tests and reports.
[[nodiscard]] bool is_proper_coloring(std::size_t n,
                                      const std::vector<std::pair<std::size_t, std::size_t>>& edges,
                                      const std::vector<std::size_t>& color);

} // namespace datamunge::algorithms
