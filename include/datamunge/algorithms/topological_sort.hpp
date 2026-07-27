#pragma once

#include <cstddef>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

struct TopologicalSortResult {
    /// @brief A linear order of the vertices in which every edge points forward. Empty when the
    ///        graph is not a DAG (see @c is_dag).
    std::vector<std::size_t> order;
    /// @brief True iff the graph is acyclic; when false a cycle was detected and @c order is left empty.
    bool is_dag{true};
};

/// @brief Kahn's algorithm (Kahn 1962) for topologically sorting a directed graph given as a
///        vertex count @p n and an edge list. A *topological order* lists the vertices so that for
///        every directed edge \(u \to v\), vertex \(u\) appears before \(v\); it exists iff the
///        graph is a *directed acyclic graph* (DAG). The method computes each vertex's in-degree,
///        seeds a queue with the sources (in-degree 0), then repeatedly emits a source and
///        decrements the in-degree of its successors, enqueuing any that drop to zero. If fewer
///        than @p n vertices are emitted, the remaining vertices lie on a cycle: no ordering
///        exists, so @c is_dag is set to false and @c order is cleared (left empty). Vertices are
///        modelled as the indices \(0,\dots,n-1\).
///
/// @param n the number of vertices; valid vertex indices are \(0,\dots,n-1\).
/// @param edges the directed edges as (from, to) index pairs.
/// @return a topological order (with @c is_dag true) or, on a cycle, an empty order (@c is_dag false).
/// @throws std::invalid_argument if any edge endpoint is >= @p n.
TopologicalSortResult topological_sort(std::size_t n,
                                       const std::vector<std::pair<std::size_t, std::size_t>>& edges);

/// @brief Checks whether @p order is a valid topological order of the graph: a permutation of
///        \(0,\dots,n-1\) in which every edge \(u \to v\) places @c u strictly before @c v.
[[nodiscard]] bool is_topological_order(std::size_t n,
                                        const std::vector<std::pair<std::size_t, std::size_t>>& edges,
                                        const std::vector<std::size_t>& order);

} // namespace datamunge::algorithms
