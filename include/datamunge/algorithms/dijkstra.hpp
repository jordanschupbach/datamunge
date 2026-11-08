#pragma once
/// @file dijkstra.hpp
/// @brief Nonnegative directed shortest paths, safe path recovery and actual lazy-heap traces.
#include <cstddef>
#include <tuple>
#include <vector>
#include <datamunge/algorithms/path_common.hpp>
namespace datamunge::algorithms {
/// @brief Owned shortest distances and one strict-improvement predecessor per vertex.
/// The original two-field layout is retained. Unreachable distances are positive infinity;
/// source/unreachable parents are kNoPredecessor. This public record can also be constructed
/// manually; reconstruct_path() validates the shape and visited chain before indexing it.
/// @par Example
/// @code
/// const auto r=dm::algorithms::dijkstra(3,{{0,1,4},{0,2,1},{2,1,1}},0);
/// std::print("{}\n",r.distance); // [0, 2, 1]
/// @endcode
/// @see dijkstra(), reconstruct_path()
struct ShortestPathResult {
    std::vector<double> distance; ///< Computed nonnegative route costs, or positive infinity if unreachable.
    std::vector<std::size_t> predecessor; ///< Vertex parents; source/unreachable use kNoPredecessor.
};
/// @brief Logical operation counts of the actual lazy binary-heap execution.
/// Counts exclude validation, adjacency construction and binary-heap comparator calls.
/// Heap pushes include the source; every pushed entry is eventually popped, including stale ones.
/// @par Example
/// @code
/// const auto t=dm::algorithms::dijkstra_trace(3,{{0,1,4},{0,2,1},{2,1,1}},0);
/// std::print("{} {} {}\n",t.work.heap_pushes,t.work.settled_vertices,t.work.stale_pops); // 4 3 1
/// @endcode
/// @see DijkstraTrace
struct DijkstraWork {
    std::size_t input_edges{0}; ///< Validated input arcs, including unreachable arcs, loops and duplicates.
    std::size_t edges_examined{0}; ///< Outgoing arcs visited from settled reachable tails.
    std::size_t successful_relaxations{0}; ///< Strict distance decreases, R <= edges_examined.
    std::size_t heap_pushes{0}; ///< R+1, including the initial source entry.
    std::size_t heap_pops{0}; ///< All removals, equal to heap_pushes on successful full traversal.
    std::size_t stale_pops{0}; ///< Removed keys strictly greater than their current vertex distance.
    std::size_t settled_vertices{0}; ///< Fresh removals; every source-reachable vertex exactly once.
    std::size_t max_heap_size{0}; ///< Peak live entries after any push; at least one.
};
/// @brief Owned state of one actual heap removal, fresh or stale.
/// A fresh removal settles the vertex before its outgoing arcs are inspected. Its
/// relaxation_begin/count index this pop's contiguous range in DijkstraTrace::relaxations.
/// A stale pop has zero relaxations. Heap_after is the size after the whole expansion.
/// @par Example
/// See dijkstra_trace().
/// @see DijkstraRelaxation
struct DijkstraPop {
    std::size_t vertex; ///< Removed vertex id.
    double key; ///< Removed stored tentative distance.
    double live_distance; ///< Current best distance at removal, before any outgoing relaxations.
    bool stale; ///< key > live_distance; such an entry does not expand the vertex.
    std::size_t heap_before; ///< Entry count immediately before removal.
    std::size_t heap_after; ///< Entry count after skipping or completing this expansion.
    std::size_t relaxation_begin; ///< First associated relaxation record index.
    std::size_t relaxation_count; ///< Number of associated outgoing-edge inspections.
};
/// @brief Owned operands and outcome of one actual outgoing-edge inspection.
/// Original edge identity is retained for multigraphs. Equal candidates do not update
/// either the predecessor or heap; no decrease-key or hidden replay occurs.
/// @par Example
/// See dijkstra_trace().
/// @see DijkstraPop
struct DijkstraRelaxation {
    std::size_t edge_index; ///< Index into the caller's original edge list.
    std::size_t from; ///< Settled tail vertex.
    std::size_t to; ///< Destination vertex.
    double weight; ///< Finite nonnegative edge weight.
    double from_distance; ///< Final distance of the settled tail.
    double distance_before; ///< Destination distance before this inspection, possibly infinity.
    double candidate; ///< Checked finite sum from_distance + weight.
    std::size_t predecessor_before; ///< Previous parent, or kNoPredecessor.
    bool improved; ///< Whether candidate is strictly smaller than distance_before.
    std::size_t heap_after; ///< Heap size after this inspection and any resulting push.
};
/// @brief Owned actual removals, relaxations, chosen edge identities and logical work.
/// No input views survive. Histories use O(V+E) storage without full distance/heap snapshots.
/// The trace cap counts pops plus edge inspections, not result arrays or adjacency storage.
/// @par Example
/// See dijkstra_trace().
/// @see DijkstraWork
struct DijkstraTrace {
    std::size_t source{0}; ///< Requested source id.
    ShortestPathResult result; ///< Same final distances and parents as ordinary dijkstra().
    DijkstraWork work; ///< Actual counters, not estimates from input sizes.
    std::vector<std::size_t> predecessor_edge; ///< Chosen original arc per vertex; kNoPredecessor at source/unreachable.
    std::vector<std::size_t> settled_order; ///< Fresh-pop vertices in final nondecreasing distance order.
    std::vector<DijkstraPop> pops; ///< All actual removals, including stale entries.
    std::vector<DijkstraRelaxation> relaxations; ///< Every actual inspected outgoing arc in execution order.
};
/// @brief Computes all source distances with Dijkstra's nonnegative directed shortest-path algorithm.
/// Uses a lazy binary min-heap of (distance,vertex); equal keys choose smaller vertex id.
/// Adjacency preserves input order and only strict improvements update parents. Self-loops,
/// zero weights and parallel arcs are accepted. Encode an undirected edge twice. All input
/// arcs, including unreachable ones, are validated before adjacency allocation. Rejects
/// nonfinite/negative weights and any nonfinite examined candidate sum, even when that
/// candidate would not improve a known shorter route. Distances use rounded double sums,
/// not exact-real arithmetic. No input mutation/shared state; no trace allocation.
/// Work O(V+E+(E+1)log(E+2)), storage O(V+E); for simple graphs the usual bound is
/// O((V+E)log V). The lazy heap may hold O(E) entries in a multigraph.
/// @param n Vertex count; source requires n>0.
/// @param edges Borrowed directed (from,to,weight) triples; endpoints <n, weights finite >=0.
/// @param source Vertex in [0,n).
/// @return Owned original-layout distance/predecessor vectors, each of length n.
/// @throws std::invalid_argument Invalid source/endpoint or nonfinite/negative weight.
/// @throws std::length_error Container or captured-counter capacity exceeded.
/// @throws std::overflow_error An examined reachable candidate sum is nonfinite.
/// @par Example
/// @code
/// const auto r=dm::algorithms::dijkstra(3,{{0,1,4},{0,2,1},{2,1,1}},0);
/// std::print("{}\n",dm::algorithms::reconstruct_path(r,1)); // [0, 2, 1]
/// @endcode
/// @par References
/// - E. W. Dijkstra (1959), A note on two problems in connexion with graphs.
///   Numerische Mathematik 1, 269-271. https://doi.org/10.1007/BF01386390
///   This variant exhausts all reachable vertices and uses a lazy binary heap.
/// @see dijkstra_trace(), bellman_ford()
ShortestPathResult dijkstra(std::size_t n,
    const std::vector<std::tuple<std::size_t,std::size_t,double>>& edges,std::size_t source);
/// @brief Records actual lazy-heap pops and outgoing relaxations without replay.
/// Uses the exact ordinary dijkstra() kernel, validation and tie rules. Caps are
/// checked online before the corresponding event is performed; failures return no partial
/// trace. A zero cap disables only the optional history budget, not capacity checks.
/// No full heap or distance snapshots; capture adds O(V+E) work/storage to the base bound.
/// @param n Vertex count, positive.
/// @param edges Borrowed finite nonnegative directed arcs, in adjacency order.
/// @param source Valid source vertex.
/// @param max_events Maximum pops plus edge inspections, default 1000000; zero disables.
/// @return Owned trace with exactly heap_pops pop records and edges_examined relaxations.
/// @throws std::invalid_argument Invalid source/endpoint/weight as in dijkstra().
/// @throws std::length_error Online event cap, counter or container capacity exceeded.
/// @throws std::overflow_error An examined candidate sum is nonfinite.
/// @par Example
/// @code
/// const auto t=dm::algorithms::dijkstra_trace(3,{{0,1,4},{0,2,1},{2,1,1}},0,7);
/// std::print("{} {} {}\n",t.settled_order,t.pops.back().stale,t.relaxations[2].candidate); // [0, 2, 1] true 2
/// @endcode
/// @par References
/// - Dijkstra (1959), doi:10.1007/BF01386390; lazy-heap event capture is an implementation extension.
/// @see DijkstraTrace, dijkstra()
[[nodiscard]] DijkstraTrace dijkstra_trace(std::size_t n,
    const std::vector<std::tuple<std::size_t,std::size_t,double>>& edges,std::size_t source,
    std::size_t max_events=1000000);
/// @brief Recovers a finite predecessor chain, safely rejecting malformed public records.
/// For an out-of-range or positive-infinity target returns empty before examining other
/// fields, preserving the original convenience behavior. A reachable target requires equal
/// vector lengths and a visited chain of finite nonnegative distances, valid parents and
/// no cycle; an early sentinel is a root. Only the visited chain is checked. Without edges
/// or a stored source this does not certify that a manually supplied root is the source or
/// that parent distances satisfy edge recurrences. No mutation/shared state; O(path length)
/// work/storage with a length bound n, avoiding unchecked indexing and infinite loops.
/// @param result Borrowed shortest-path record.
/// @param target Destination index; out of range returns empty.
/// @return Owned root-to-target ids, singleton at a root, or empty if unavailable.
/// @throws std::invalid_argument Reachable record has mismatched vectors, invalid/nonfinite
/// visited distances, out-of-range parents or a predecessor cycle.
/// @par Example
/// @code
/// const auto r=dm::algorithms::dijkstra(3,{{0,1,1}},0);
/// std::print("{} {}\n",dm::algorithms::reconstruct_path(r,0),dm::algorithms::reconstruct_path(r,2).size()); // [0] 0
/// @endcode
/// @see dijkstra(), ShortestPathResult
[[nodiscard]] std::vector<std::size_t> reconstruct_path(const ShortestPathResult& result,std::size_t target);
/// @brief Generates a two-vertex stress graph with strictly improving parallel arcs 0->1.
/// Returns weights count,count-1,...,1 in that order. At positive count m the actual
/// lazy heap peaks at m entries, has m+1 pushes/pops and m-1 stale removals, despite V=2.
/// Zero count returns no edges. Complete output guards precede allocation. No random or
/// shared state; O(count) work/storage. Intended for operation-count checks, not timing claims.
/// @param count Arc count, at most 2^53 so every integer weight is exactly binary64 representable.
/// @param max_edges Complete output cap, default 1000000; zero disables only this cap.
/// @return Owned descending-weight parallel arcs.
/// @throws std::length_error Count exceeds 2^53, optional cap or container capacity.
/// @par Example
/// @code
/// const auto edges=dm::algorithms::dijkstra_parallel_workload(3);
/// std::print("{}\n",dm::algorithms::dijkstra_trace(2,edges,0).work.stale_pops); // 2
/// @endcode
/// @see dijkstra_trace(), dag_forward_band_edges()
[[nodiscard]] std::vector<std::tuple<std::size_t,std::size_t,double>> dijkstra_parallel_workload(
    std::size_t count,std::size_t max_edges=1000000);
} // namespace datamunge::algorithms
