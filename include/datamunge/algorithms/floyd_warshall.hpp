#pragma once

#include <cstddef>
#include <tuple>
#include <vector>

namespace datamunge::algorithms {

/// @brief Sentinel used in @c AllPairsShortestPaths::next to mark "no next hop" -- i.e. the pair
///        of vertices has no path between them (or is a diagonal entry).
inline constexpr std::size_t kNoVertex = static_cast<std::size_t>(-1);

struct AllPairsShortestPaths {
    /// @brief distance[i][j] is the weight of a shortest path from i to j: 0 on the diagonal,
    ///        +infinity when j is unreachable from i. When @c has_negative_cycle is true the
    ///        entries touched by a negative cycle are no longer meaningful.
    std::vector<std::vector<double>> distance;
    /// @brief next[i][j] is the vertex following i on a shortest i->j path (used by
    ///        @c reconstruct_path), or @c kNoVertex when no such path exists.
    std::vector<std::vector<std::size_t>> next;
    /// @brief True iff the graph contains a negative-weight cycle, detected by some
    ///        distance[i][i] < 0 after the algorithm runs.
    bool has_negative_cycle{false};
};

/// @brief The Floyd-Warshall algorithm (Floyd 1962; Warshall 1962) for *all-pairs shortest paths*
///        in a weighted *directed* graph. The graph is given as a vertex count @p n and an edge
///        list of (from, to, weight) tuples; weights may be negative, but the distances are only
///        meaningful when the graph has no negative-weight cycle. It is a dynamic program over the
///        set of *intermediate* vertices a path is allowed to use: letting \(d^{(k)}[i][j]\) be the
///        shortest \(i\!\to\!j\) distance using only intermediates drawn from \(\{0,\dots,k\}\),
///        \[ d^{(k)}[i][j] = \min\bigl(d^{(k-1)}[i][j],\; d^{(k-1)}[i][k] + d^{(k-1)}[k][j]\bigr), \]
///        which the triple loop evaluates in place in \(O(n^3)\) time and \(O(n^2)\) space. A
///        @c next matrix is maintained alongside so a concrete shortest path can be reconstructed.
///        After the run, a negative cycle exists iff some vertex can reach itself at negative cost,
///        i.e. some diagonal entry is negative.
///
/// @param n the number of vertices; valid vertex indices are \(0,\dots,n-1\).
/// @param edges the directed edges as (from, to, weight) tuples. Parallel edges are allowed; the
///        lightest one between a given ordered pair is kept.
/// @return the all-pairs distance and next-hop matrices, plus a negative-cycle flag.
/// @throws std::invalid_argument if any edge endpoint is >= @p n.
AllPairsShortestPaths floyd_warshall(std::size_t n,
                                     const std::vector<std::tuple<std::size_t, std::size_t, double>>& edges);

/// @brief Reconstructs a shortest path from @p i to @p j as the sequence of vertices visited,
///        following the @c next matrix of @p apsp. Returns {i} for i == j, and an empty vector when
///        no path from i to j exists.
/// @throws std::invalid_argument if @p i or @p j is >= the vertex count of @p apsp.
[[nodiscard]] std::vector<std::size_t> reconstruct_path(const AllPairsShortestPaths& apsp,
                                                        std::size_t i, std::size_t j);

} // namespace datamunge::algorithms
