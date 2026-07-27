#pragma once

#include <cstddef>
#include <tuple>
#include <vector>

namespace datamunge::algorithms {

/// @brief Sentinel for "no predecessor" in a ShortestPathResult: the source vertex, and every
///        vertex unreachable from it, carry this value in place of a real predecessor.
inline constexpr std::size_t kNoPredecessor = static_cast<std::size_t>(-1);

struct ShortestPathResult {
    /// @brief distance[v] is the length of a shortest path source -> v, or +infinity if v is
    ///        unreachable from the source.
    std::vector<double> distance;
    /// @brief predecessor[v] is the vertex preceding v on a shortest path from the source, or
    ///        kNoPredecessor for the source itself and for every unreachable vertex.
    std::vector<std::size_t> predecessor;
};

/// @brief Dijkstra's single-source shortest-path algorithm (Dijkstra 1959) on a weighted directed
///        graph with *non-negative* edge weights. Starting from @p source it settles vertices in
///        nondecreasing order of tentative distance, repeatedly extracting the closest unsettled
///        vertex from a binary min-heap and *relaxing* its outgoing edges. Because every weight is
///        non-negative, a vertex's tentative distance is provably final the moment it is the heap
///        minimum -- no later path through a farther vertex can improve it -- so each vertex is
///        settled exactly once. The heap uses *lazy deletion*: a vertex may be pushed several
///        times as its distance falls, and stale entries (those whose stored key exceeds the
///        settled distance) are simply skipped when popped. Runs in O((V+E) log V).
///
///        Negative edge weights break the settling invariant (a cheaper path may still arrive via
///        an as-yet-unsettled vertex), so they are rejected; use Bellman-Ford for those. An
///        *undirected* graph is handled by adding each edge in both directions.
///
/// @param n the number of vertices; valid vertex indices are 0..n-1.
/// @param edges the directed edges as (from, to, weight) tuples; every endpoint must be < n and
///        every weight must be >= 0.
/// @param source the vertex to compute shortest paths from; must be < n.
/// @return the shortest-path distances and predecessor tree rooted at @p source.
/// @throws std::invalid_argument if @p source or any endpoint is >= n, or any weight is negative.
ShortestPathResult dijkstra(std::size_t n,
                            const std::vector<std::tuple<std::size_t, std::size_t, double>>& edges,
                            std::size_t source);

/// @brief Reconstructs the shortest path from the source to @p target by walking the predecessor
///        tree in @p result back to the root and reversing it.
/// @return the vertex sequence source, ..., target, or an empty vector if @p target is unreachable.
[[nodiscard]] std::vector<std::size_t> reconstruct_path(const ShortestPathResult& result, std::size_t target);

} // namespace datamunge::algorithms
