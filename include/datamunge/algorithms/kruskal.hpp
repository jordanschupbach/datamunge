#pragma once

#include <cstddef>
#include <tuple>
#include <vector>

#include <datamunge/algorithms/mst_result.hpp>

namespace datamunge::algorithms {

/// @brief Kruskal's algorithm (Kruskal 1956) for the minimum spanning tree of a weighted,
///        undirected graph. The graph is given as a vertex count @p n (vertices are 0..n-1) and an
///        edge list of (u, v, weight) tuples. A *spanning tree* is an acyclic connected subgraph
///        touching every vertex; its *weight* is the sum of its edge weights, and a *minimum*
///        spanning tree minimizes that sum. Kruskal is a greedy method: it sorts the edges
///        ascending by weight and scans them, accepting an edge exactly when its two endpoints lie
///        in *different* components so far -- i.e. when it does not close a cycle -- and rejecting
///        it otherwise. Correctness rests on the *cut property*: for any partition of the vertices
///        into two sides, the minimum-weight edge crossing it belongs to some minimum spanning
///        tree; the first accepted edge joining two components is always such a lightest crossing
///        edge, hence safe. The cycle test is answered in near-constant amortized time by a
///        *union-find* (disjoint-set) structure with path compression and union by rank, giving an
///        overall O(E log E) cost dominated by the sort. If fewer than n-1 edges are accepted the
///        graph is disconnected: the result is then a minimum spanning *forest* and
///        @ref MinimumSpanningTree::is_connected is false.
///
/// @param n number of vertices; valid vertex indices are 0..n-1.
/// @param edges undirected weighted edges as (u, v, weight); u and v must both be < n.
/// @return the minimum spanning tree (or, if the graph is disconnected, minimum spanning forest).
/// @throws std::invalid_argument if any edge endpoint is >= n.
MinimumSpanningTree kruskal(std::size_t n,
                            const std::vector<std::tuple<std::size_t, std::size_t, double>>& edges);

} // namespace datamunge::algorithms
