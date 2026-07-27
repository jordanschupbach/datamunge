#pragma once

#include <cstddef>
#include <tuple>
#include <vector>

#include <datamunge/algorithms/mst_result.hpp>

namespace datamunge::algorithms {

/// @brief Prim's minimum-spanning-tree algorithm (Jarnik 1930; Prim 1957; Dijkstra 1959) on a
///        weighted *undirected* graph. A *spanning tree* of a connected graph is an acyclic,
///        connected subgraph touching every vertex (hence n-1 edges); a *minimum* spanning tree
///        minimizes the total edge weight. Prim grows a single tree outward from @p start: it
///        maintains the set of in-tree vertices and, at each step, adds the cheapest edge crossing
///        the *cut* between the tree and the rest, attaching one new vertex. This greedy choice is
///        always safe by the *cut property* -- for any cut that no tree edge yet crosses, a minimum
///        weight edge across it belongs to some minimum spanning tree -- so the tree Prim builds is
///        optimal.
///
///        The frontier of candidate crossing edges is a binary min-heap (std::priority_queue)
///        keyed by weight, with *lazy deletion*: an edge to a vertex may sit in the heap after that
///        vertex has already been attached by a cheaper edge, and such stale entries are simply
///        skipped when popped. Runs in O(E log V).
///
///        If the graph is *disconnected*, only @p start's connected component is spanned: the
///        returned tree touches just that component and @ref MinimumSpanningTree::is_connected is
///        set to false. (A full minimum spanning *forest* would restart the growth from each
///        still-unvisited vertex; that is left to the caller.) The connected case is always handled
///        exactly, and connectivity is reported.
///
/// @param n the number of vertices; valid vertex indices are 0..n-1.
/// @param edges the undirected edges as (u, v, weight) tuples; each is used in both directions.
///        Every endpoint must be < n. Parallel edges and self-loops are permitted (a self-loop can
///        never cross a cut and is therefore never selected).
/// @param start the vertex from which the tree is grown; must be < n.
/// @return the minimum spanning tree of @p start's component, its total weight, and whether it
///         spans the entire graph.
/// @throws std::invalid_argument if @p start or any endpoint is >= n.
MinimumSpanningTree prim(std::size_t n,
                         const std::vector<std::tuple<std::size_t, std::size_t, double>>& edges,
                         std::size_t start = 0);

} // namespace datamunge::algorithms
