#pragma once

/// @file bellman_ford.hpp
/// @brief Single-source shortest paths with negative edge weights: Bellman-Ford in passes and in
///        queue form (SPFA), with negative-cycle detection and extraction.

#include <cstddef>
#include <tuple>
#include <vector>

#include <datamunge/algorithms/path_common.hpp>

namespace datamunge::algorithms {

/// Result of bellman_ford() and bellman_ford_queue().
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
    /// @brief One reachable negative cycle, as vertices in edge order (the edge from the last
    ///        vertex back to the first closes it); empty when there is none.
    std::vector<std::size_t> negative_cycle;
    /// @brief Rounds run: full passes over the edge list for bellman_ford(), or queue "generations"
    ///        (vertices whose shortest-path tree depth is the same) for bellman_ford_queue().
    std::size_t passes{0};
    /// @brief Successful relaxations (distance decreases) in each round.
    std::vector<std::size_t> relaxations_per_pass;
    /// @brief Total number of edges examined.
    std::size_t edge_checks{0};
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

/// @brief Queue-based Bellman-Ford (Moore 1959; often called SPFA): only the out-edges of
///        vertices whose distance fell in the previous round are examined.
///
/// Gives the same distances as bellman_ford() and usually examines far fewer edges, though the
/// worst case is still \f$O(VE)\f$. A vertex whose shortest-path tree depth reaches \f$n\f$ proves a
/// reachable negative cycle, which is then extracted from the predecessor pointers.
/// @param n number of vertices.
/// @param edges directed (from, to, weight) triples; weights may be negative.
/// @param source the source vertex.
/// @return distances, predecessors, rounds and the negative cycle if any.
/// @throws std::invalid_argument if the source or an endpoint is out of range.
BellmanFordResult bellman_ford_queue(std::size_t n,
                                     const std::vector<std::tuple<std::size_t, std::size_t, double>>& edges,
                                     std::size_t source);

/// @brief Vertices of the path from the source to @p target, read from the predecessor tree;
///        empty if @p target is unreachable or the tree is broken by a negative cycle.
/// @param result output of bellman_ford() or bellman_ford_queue().
/// @param target the end vertex.
/// @return the path, starting at the source.
std::vector<std::size_t> shortest_path_to(const BellmanFordResult& result, std::size_t target);

} // namespace datamunge::algorithms
