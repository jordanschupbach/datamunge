#pragma once
/// @file boruvka.hpp
/// @brief Deterministic Boruvka minimum spanning forests and real component/merge traces.
#include <datamunge/algorithms/mst_result.hpp>

namespace datamunge::algorithms {
/// @brief One component's cheapest outgoing edge at the start of a phase.
/// Choices are recorded before any unions, so multiple components may choose
/// the same edge. Acceptance is recorded separately in BoruvkaPhase.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::boruvka_trace(2,{{0,1,7}});
/// std::print("{}\n",trace.phases[0].choices.size()); // 2
/// @endcode
/// @see BoruvkaPhase
struct BoruvkaChoice {
  std::size_t component_label{0}; ///< Minimum vertex in the selecting component, independent of union-find roots.
  std::size_t edge_index{0}; ///< Index in the caller's original edge list; retains parallel-edge identity.
};
/// @brief One actual full edge scan followed by cycle-checked unions.
/// Labels refer to components before this phase; choices use ascending internal
/// root order, the same order used for acceptance. A disconnected forest can end
/// with a phase with no choices or merges. Empty and singleton graphs have no phases.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::boruvka_trace(2,{{0,1,7}});
/// std::print("{} {}\n",trace.phases[0].components_before,trace.phases[0].components_after); // 2 1
/// @endcode
/// @see BoruvkaChoice, boruvka_trace()
struct BoruvkaPhase {
  std::vector<std::size_t> component_labels; ///< One minimum-vertex component label per vertex before the scan.
  std::vector<BoruvkaChoice> choices; ///< One cheapest outgoing choice per nonisolated component, including duplicates.
  std::vector<std::size_t> accepted_edge_indices; ///< Original edge indices in actual successful-union order.
  std::size_t components_before{0}; ///< Number of components before this phase, including isolated vertices.
  std::size_t components_after{0}; ///< Number after this phase's successful unions.
  std::size_t edges_scanned{0}; ///< Every input edge is inspected once, including loops and internal edges.
};
/// @brief Owned forest, component snapshots and logical edge-scan counters from the canonical kernel.
/// Recording costs O(n*P) storage for P scans in addition to O(n) forest/work
/// storage. No input or shared state is modified; edge indices refer to input order.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::boruvka_trace(4,{{0,1,1},{1,2,2},{2,3,1},{0,3,9}});
/// std::print("{} {}\n",trace.phases.size(),trace.forest.total_weight); // 2 4
/// @endcode
/// @see boruvka_trace(), BoruvkaPhase
struct BoruvkaTrace {
  MinimumSpanningTree forest; ///< Same selected edges, order, total and connectivity as boruvka().
  std::size_t vertex_count{0}; ///< Original number of vertices.
  std::size_t component_count{0}; ///< Final component count, including isolated vertices; zero for an empty graph.
  std::size_t edge_scans{0}; ///< Sum of phase edge inspections, excluding the initial validation pass.
  std::vector<BoruvkaPhase> phases; ///< Actual scans and merges, including any final unproductive scan.
};
/// @brief Find a minimum spanning forest by each current component's cheapest outgoing edge.
/// Vertices are 0..n-1. Edges are undirected; finite negative/zero weights, loops
/// and parallel edges are accepted. Loops never join components. Equal weights
/// prefer the first input edge. Choices are fixed before merging and processed
/// in ascending union-find root order; union by rank retains the first root on ties.
/// Each accepted edge joins different components. Disconnected inputs return an
/// MST per component; empty/singleton graphs are connected by convention.
/// For P scans, work is O(n+m+P*(n+m)*alpha(n)), P=O(log(n+1)); O(n) auxiliary
/// space and O(n) owned output. The n term matters for sparse disconnected graphs.
/// Weights accumulate in long double and the final double total is checked.
/// @param n Number of vertices, including isolated vertices.
/// @param edges Borrowed (u,v,finite weight) tuples in deterministic tie order.
/// @return Owned minimum spanning forest, total weight and whole-graph connectivity.
/// @throws std::invalid_argument If any endpoint is outside [0,n) or weight is nonfinite.
/// @throws std::overflow_error If the accumulated total cannot be represented as finite double.
/// @par Example
/// @code
/// const auto tree=dm::algorithms::boruvka(4,{{0,1,1},{1,2,2},{2,3,1},{0,3,9}});
/// std::print("{} {}\n",tree.edges.size(),tree.total_weight); // 3 4
/// @endcode
/// @par References
/// - O. Boruvka (1926), O jistem problemu minimalnim, Prace Moravske prirodovedecke
///   spolecnosti III, 3, 37-58, https://dml.cz/handle/10338.dmlcz/500115.
/// @see boruvka_trace(), MinimumSpanningTree
MinimumSpanningTree boruvka(std::size_t n,const std::vector<WeightedEdge>& edges);
/// @brief Record component labels, cheapest choices and accepted merges from the same Boruvka kernel.
/// Forest edge order and tie behavior equal boruvka(). Canonical labels are
/// minimum original vertices, not implementation roots. Counter edge_scans
/// counts whole scans, not elapsed time. Recording takes O(n*P) owned storage
/// and O(n*P*alpha(n)) additional label work; caller data are unchanged.
/// @param n Number of vertices.
/// @param edges Borrowed undirected finite-weight edges in stable input order.
/// @return Owned forest, final component count and every actual phase.
/// @throws std::invalid_argument If an endpoint or weight is invalid.
/// @throws std::overflow_error If the final weight is outside finite double range.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::boruvka_trace(4,{{0,1,1},{1,2,2},{2,3,1},{0,3,9}});
/// std::print("{} {}\n",trace.phases[0].components_after,trace.edge_scans); // 2 8
/// @endcode
/// @par References
/// - O. Boruvka (1926), https://dml.cz/handle/10338.dmlcz/500115.
/// @see boruvka(), BoruvkaTrace
BoruvkaTrace boruvka_trace(std::size_t n,const std::vector<WeightedEdge>& edges);
/// @brief Build a dyadically weighted path, useful for reproducible spanning-forest studies.
/// Edge i connects i-1 to i with weight 1+v2(i), where v2 counts trailing zero
/// bits. Edges are in ascending i order. For n=2^k, Boruvka merges adjacent
/// equal-sized blocks in exactly k productive phases. All path edges are needed
/// in its unique spanning tree even though weights repeat. O(n) work/storage.
/// No shared state or randomness; n<=1 returns an empty edge list.
/// @param n Number of path vertices, including zero or one.
/// @return Owned n-1 edges for n>0, otherwise empty.
/// @throws std::length_error If n-1 exceeds vector capacity; allocation errors propagate.
/// @par Example
/// @code
/// const auto edges=dm::algorithms::dyadic_path_edges(8);
/// const auto trace=dm::algorithms::boruvka_trace(8,edges);
/// std::print("{} {}\n",edges.size(),trace.phases.size()); // 7 3
/// @endcode
/// @see boruvka_trace()
std::vector<WeightedEdge> dyadic_path_edges(std::size_t n);
} // namespace datamunge::algorithms
