#pragma once

#include <cstddef>
#include <tuple>
#include <vector>

namespace datamunge::algorithms {

/// @brief Sentinel stored in BellmanFordResult::predecessor for a vertex with no predecessor --
///        the source itself, or a vertex the source cannot reach.
inline constexpr std::size_t kNoPredecessor = static_cast<std::size_t>(-1);

struct BellmanFordResult {
    /// @brief distance[v] is the weight of a shortest path from the source to v, or
    ///        +infinity if v is unreachable. If has_negative_cycle is true these values are
    ///        not all meaningful (see @ref bellman_ford).
    std::vector<double> distance;
    /// @brief predecessor[v] is the vertex before v on a shortest path from the source
    ///        (kNoPredecessor for the source and for unreachable vertices); following it back
    ///        reconstructs the shortest-path tree.
    std::vector<std::size_t> predecessor;
    /// @brief True iff a negative-weight cycle is reachable from the source.
    bool has_negative_cycle{false};
};

/// @brief The Bellman-Ford single-source shortest-path algorithm (Bellman 1958; Ford 1956). Given
///        a weighted *directed* graph -- a vertex count @p n and an edge list of
///        (from, to, weight) triples whose weights *may be negative* -- it computes the shortest
///        path weight from @p source to every vertex. Unlike Dijkstra's algorithm it tolerates
///        negative edge weights, and it *detects* a negative-weight cycle reachable from the
///        source (a cycle around which one may loop to drive path weights to \(-\infty\), so no
///        shortest path exists). It works by *relaxation*: repeatedly, for every edge
///        \((u,v,w)\), replacing \(d[v]\) with \(d[u]+w\) whenever that is smaller. After
///        \(n-1\) full passes over the edge list every true shortest path -- which uses at most
///        \(n-1\) edges when no negative cycle exists -- has been found; one further pass that
///        can still relax an edge proves a reachable negative cycle exists.
///
/// @param n number of vertices; valid vertex indices are 0..n-1.
/// @param edges directed edges as (from, to, weight) triples; weights may be negative. Each
///        endpoint must be < @p n or std::invalid_argument is thrown.
/// @param source the source vertex; must be < @p n or std::invalid_argument is thrown.
/// @return the shortest-path distances and predecessor tree. If has_negative_cycle is true a
///         negative cycle is reachable from the source, in which case the distances of vertices
///         reachable through that cycle are not the true (nonexistent) shortest-path weights.
BellmanFordResult bellman_ford(std::size_t n,
                               const std::vector<std::tuple<std::size_t, std::size_t, double>>& edges,
                               std::size_t source);

} // namespace datamunge::algorithms
