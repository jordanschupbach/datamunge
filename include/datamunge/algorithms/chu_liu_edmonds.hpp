#pragma once
/// @file chu_liu_edmonds.hpp
/// @brief Checked minimum rooted directed arborescences with original-edge recovery and actual contractions.
#include <datamunge/algorithms/mst_result.hpp>
#include <limits>
namespace datamunge::algorithms {
/// @brief No incoming edge for the root or for a missing phase choice.
/// This is a storage sentinel, never an original input edge index.
/// @par Example
/// @code
/// const auto t=dm::algorithms::chu_liu_edmonds_tree(2,{{0,1,3}},0);
/// std::print("{}\n",t.incoming_edges[0]==dm::algorithms::kNoBranchingEdge); // true
/// @endcode
/// @see DirectedArborescence
inline constexpr std::size_t kNoBranchingEdge=std::numeric_limits<std::size_t>::max();
/// @brief Legacy owned minimum rooted-arborescence cost and existence verdict.
/// Field types/order/defaults are retained. Success spans all vertices from the
/// prescribed root with outgoing paths; finite negative weights are supported.
/// Failure returns {0,false}; default storage alone has not solved a graph.
/// Manually assigned values are not validated or optimality certificates.
/// @par Example
/// @code
/// const auto r=dm::algorithms::chu_liu_edmonds(3,{{0,1,5},{0,2,6},{1,2,1},{2,1,1}},0);
/// std::print("{} {}\n",r.exists,r.total_weight); // true 6
/// @endcode
/// @see chu_liu_edmonds(), DirectedArborescence
struct DirectedBranchingResult {
  double total_weight{0.0}; ///< Checked sum of recovered original selected edge weights on success; zero on failure.
  bool exists{false}; ///< Whether a spanning out-arborescence from the prescribed root exists.
};
/// @brief Owned original-vertex parent and input-edge identities of an optimum arborescence.
/// Success has one entry per original vertex: the root parents itself and has
/// kNoBranchingEdge; every other entry names its selected original incoming edge.
/// Failure leaves both arrays empty and result={0,false}, with no partial tree.
/// Parallel edges retain distinct input identities. Calls own all returned data.
/// @par Example
/// @code
/// const auto t=dm::algorithms::chu_liu_edmonds_tree(3,{{0,1,5},{0,2,6},{1,2,1},{2,1,1}},0);
/// std::print("{} {}\n",t.parent,t.incoming_edges[2]); // [0, 0, 1] 2
/// @endcode
/// @see chu_liu_edmonds_tree(), BranchingTrace
struct DirectedArborescence {
  DirectedBranchingResult result; ///< Same result as the cost-only public entry point.
  std::vector<std::size_t> parent; ///< Original parent per vertex, with root self-parented; empty on failure.
  std::vector<std::size_t> incoming_edges; ///< Original input edge id per vertex; root uses the sentinel.
};
/// @brief Whole-call phase and named edge-visit limits for the iterative construction.
/// Zero means unlimited. Every input edge validates before either cap applies.
/// Edge visits count incoming-choice scans and contraction reweighting, including
/// ignored self loops and root-entering arcs. Recovery/map copies are not edge
/// visits. Failure throws with no partial tree/trace. These are work limits, not
/// complete memory limits; a phase cap also limits retained expansion levels.
/// @par Example
/// @code
/// dm::algorithms::BranchingOptions options;options.max_phases=1;
/// const auto t=dm::algorithms::chu_liu_edmonds_tree(2,{{0,1,3}},0,options);
/// std::print("{}\n",t.result.total_weight); // 3
/// @endcode
/// @see chu_liu_edmonds_trace(), BranchingWork
struct BranchingOptions {
  std::size_t max_phases{0}; ///< Actual incoming-choice rounds, including a final failed round; zero is unlimited.
  std::size_t max_edge_visits{0}; ///< Incoming scan plus reweight visits across the whole call; zero is unlimited.
};
/// @brief Owned actual working arc with wide reduced weight and stable original identity.
/// All input arcs remain in input order through every contraction, even when
/// mapped to irrelevant self loops. weight is an actual long-double working
/// cost, not necessarily representable as double or equal to the original cost.
/// @par Example
/// @code
/// const auto t=dm::algorithms::chu_liu_edmonds_trace(3,{{0,1,5},{0,2,6},{1,2,1},{2,1,1}},0);
/// std::print("{} {}\n",t.phases[1].edges[0].original_index,t.phases[1].edges[0].weight); // 0 4
/// @endcode
/// @see BranchingPhase
struct BranchingArc {
  std::size_t from{0}; ///< Current contracted tail.
  std::size_t to{0}; ///< Current contracted head.
  long double weight{0}; ///< Actual finite working reduced cost.
  std::size_t original_index{0}; ///< Identity in the caller's original ordered edge list.
};
/// @brief Owned actual incoming choices, cycle partition and costs of one round.
/// Root choice uses the sentinel and weight zero; missing choices use the
/// sentinel and positive infinity. missing_vertices is nonempty only on failure,
/// before any baseline addition or cycle search. cycles lists vertices in
/// predecessor-following order, opposite to the directed edge traversal order.
/// On contraction, cycle components come first in discovery order, then singleton
/// components in vertex order. component is empty on acyclic/failed rounds.
/// original_to_vertex maps every original vertex into this round's graph.
/// @par Example
/// @code
/// const auto t=dm::algorithms::chu_liu_edmonds_trace(3,{{0,1,5},{0,2,6},{1,2,1},{2,1,1}},0);
/// std::print("{} {}\n",t.phases[0].cycles.size(),t.phases[0].added_cost); // 1 2
/// @endcode
/// @see BranchingTrace, BranchingArc
struct BranchingPhase {
  std::size_t vertex_count{0}; ///< Actual current vertex count.
  std::size_t root{0}; ///< Actual current root id.
  std::vector<BranchingArc> edges; ///< Actual arcs before this incoming-choice scan, in original input order.
  std::vector<std::size_t> original_to_vertex; ///< Current component per original vertex.
  std::vector<std::size_t> incoming_edges; ///< Actual chosen original edge id per current vertex; root/missing use sentinel.
  std::vector<long double> incoming_weights; ///< Actual chosen costs; root zero, missing positive infinity.
  std::vector<std::size_t> missing_vertices; ///< Non-root current vertices without an eligible incoming arc.
  std::vector<std::vector<std::size_t>> cycles; ///< Actual disjoint selected-edge cycles, in predecessor order.
  std::vector<std::size_t> component; ///< Actual next-round component ids when cycles were contracted; otherwise empty.
  long double added_cost{0}; ///< Actual sum added to the reduced baseline; zero when a missing choice aborts the round.
  long double cumulative_cost{0}; ///< Actual reduced baseline after this round, or prior baseline on failure.
};
/// @brief Actual named validation, scan, cycle and expansion counts.
/// Counters describe executed work, not seconds or a bit-complexity model.
/// Saved vertex entries count compact incoming-choice and original-map copies
/// for contraction recovery; full trace snapshots and final output are excluded.
/// @par Example
/// @code
/// const auto t=dm::algorithms::chu_liu_edmonds_trace(3,{{0,1,5},{0,2,6},{1,2,1},{2,1,1}},0);
/// std::print("{} {} {}\n",t.work.phases,t.work.incoming_edge_visits,t.work.reweight_edge_visits); // 2 8 4
/// @endcode
/// @see BranchingTrace, BranchingOptions
struct BranchingWork {
  std::size_t validated_edges{0}; ///< Original endpoints and finite weights validated.
  std::size_t phases{0}; ///< Actual incoming-choice rounds, including a final failure.
  std::size_t incoming_edge_visits{0}; ///< Every actual incoming scan inspection, including ignored arcs.
  std::size_t reweight_edge_visits{0}; ///< Every actual contraction transform inspection, including internal arcs.
  std::size_t predecessor_steps{0}; ///< Actual previously unprocessed non-root vertices followed during cycle detection.
  std::size_t cycles_contracted{0}; ///< Actual selected-edge cycles replaced by components.
  std::size_t expansion_replacements{0}; ///< Actual incoming-id assignments while expanding upper-level choices, including unchanged assignments.
  std::size_t saved_vertex_entries{0}; ///< Actual compact expansion-history vector entries copied.
};
/// @brief Owned actual contraction rounds, recovered original tree and operation counts.
/// One executing kernel serves legacy, tree and trace calls without replay.
/// reduced_total retains the actual sum of round baselines, including a prefix on
/// failure. The returned cost instead sums the recovered original edges in
/// original destination order; floating roundoff can distinguish those totals.
/// A missing incoming choice proves nonexistence, including after a closed cycle
/// is contracted. Failure returns no partial tree, but retains the actual rounds.
/// All storage is independent of caller inputs and other calls.
/// @par Example
/// @code
/// const auto t=dm::algorithms::chu_liu_edmonds_trace(3,{{0,1,5},{0,2,6},{1,2,1},{2,1,1}},0);
/// std::print("{} {}\n",t.tree.parent,t.reduced_total); // [0, 0, 1] 6
/// @endcode
/// @see chu_liu_edmonds_trace(), DirectedArborescence
struct BranchingTrace {
  DirectedArborescence tree; ///< Recovered original optimum or empty failure arrays.
  std::vector<BranchingPhase> phases; ///< Actual rounds, including a final missing-incoming round.
  long double reduced_total{0}; ///< Actual accumulated reduced baselines, not a partial returned tree cost.
  BranchingWork work; ///< Actual named counts.
};
/// @brief Find the minimum-cost spanning directed out-arborescence rooted at root.
/// Preserves the legacy call/result layout. Vertices are 0..vertex_count-1 and
/// every vertex must be reachable from root in the returned tree. Finite negative
/// weights and parallel edges are supported. Self loops and edges entering the
/// root validate but never become choices. Equal working costs prefer the first
/// original edge. Nonempty input graphs are required; a singleton costs zero.
/// Original edge recovery now occurs before summing the final cost. Working
/// differences and sums use checked long double, with checked final conversion
/// to double. This is rounded arithmetic, not exact-real optimality certification.
/// No vertex ids narrow to int, recursion, input mutation or shared state is used.
/// For P rounds, O(P*(V+E)) work and O(E+P*V) auxiliary storage, plus owned output;
/// P<=V. Compact expansion history is retained even by this cost-returning call.
/// @param vertex_count Positive original number of vertices.
/// @param directed_edges Borrowed finite (tail,head,weight) tuples in deterministic tie order.
/// @param root Valid prescribed original root.
/// @return Checked optimum weight and existence; {0,false} if no spanning arborescence exists.
/// @throws std::invalid_argument If root/endpoints are out of range or any weight is nonfinite.
/// @throws std::overflow_error If a working cost/count overflows or the final total is outside finite double range.
/// @throws std::length_error If container capacity is exceeded.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto r=dm::algorithms::chu_liu_edmonds(3,{{0,1,5},{0,2,6},{1,2,1},{2,1,1}},0);
/// std::print("{}\n",r.total_weight); // 6
/// @endcode
/// @par References
/// - Y.-J. Chu and T.-H. Liu (1965), On the shortest arborescence of a directed graph,
///   Scientia Sinica 14(10), 1396-1400, https://www.cs.cmu.edu/~15850/handouts/chu-liu_1965.pdf.
/// - J. Edmonds (1967), Optimum Branchings, J. Res. NBS 71B(4), 233-240,
///   https://doi.org/10.6028/jres.071B.032.
/// @see chu_liu_edmonds_tree(), chu_liu_edmonds_trace()
DirectedBranchingResult chu_liu_edmonds(std::size_t vertex_count,const std::vector<WeightedEdge>& directed_edges,std::size_t root);
/// @brief Recover the selected original input edges and parents with optional whole-call caps.
/// Same executing kernel, cost, validation and deterministic tie rules as
/// chu_liu_edmonds(). No partial tree escapes a failure or exhausted cap. Root
/// parents itself on success; its incoming id is kNoBranchingEdge. O(E+P*V)
/// auxiliary storage for iterative compact expansion history, plus owned output.
/// @param vertex_count Positive original vertex count.
/// @param directed_edges Borrowed directed finite-weight edges in input tie order.
/// @param root Valid prescribed original root.
/// @param options Phase/edge-visit limits, default unlimited.
/// @return Owned complete original tree and legacy result, or empty arrays on nonexistence.
/// @throws std::invalid_argument If root/endpoints are invalid or a weight is nonfinite.
/// @throws std::length_error If a positive work cap or container capacity is exceeded.
/// @throws std::overflow_error If a working cost/count or final double cost overflows.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto t=dm::algorithms::chu_liu_edmonds_tree(2,{{0,1,-3},{0,1,-3}},0);
/// std::print("{} {}\n",t.parent,t.incoming_edges[1]); // [0, 0] 0
/// @endcode
/// @see chu_liu_edmonds(), DirectedArborescence
DirectedArborescence chu_liu_edmonds_tree(std::size_t vertex_count,const std::vector<WeightedEdge>& directed_edges,std::size_t root,BranchingOptions options={});
/// @brief Retain actual selection, contraction and original-edge expansion evidence without replay.
/// Uses the same kernel as chu_liu_edmonds_tree(). Records include stable original
/// edge identities, wide reduced costs, current partitions, predecessor cycles,
/// missing vertices and baseline additions. Full trace storage is O(P*(V+E));
/// no partial trace escapes work-cap/allocation failure. Mathematical nonexistence
/// returns its failed round and an empty tree. Numerical overflow still throws.
/// @param vertex_count Positive original vertex count.
/// @param directed_edges Borrowed finite directed edges in deterministic input order.
/// @param root Valid prescribed original root.
/// @param options Whole-call phase/edge-visit limits, default unlimited.
/// @return Owned actual rounds, recovered original tree, reduced total and work.
/// @throws std::invalid_argument If root/endpoints are invalid or a weight is nonfinite.
/// @throws std::length_error If a work cap or container capacity is exceeded.
/// @throws std::overflow_error If a working cost/count or final double cost overflows.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto t=dm::algorithms::chu_liu_edmonds_trace(3,{{1,2,1},{2,1,1}},0);
/// std::print("{} {}\n",t.tree.result.exists,t.phases.size()); // false 2
/// @endcode
/// @par References
/// - J. Edmonds (1967), Optimum Branchings, https://doi.org/10.6028/jres.071B.032.
/// @see chu_liu_edmonds_tree(), BranchingTrace
BranchingTrace chu_liu_edmonds_trace(std::size_t vertex_count,const std::vector<WeightedEdge>& directed_edges,std::size_t root,BranchingOptions options={});
/// @brief Generate root-entry arcs and a directed cycle for controlled contraction studies.
/// Vertices are root 0 and cycle vertices 1..cycle_vertices. Root-entry arcs come
/// first in ascending target order, followed by 1->2,...,k->1. With entry_cost>
/// cycle_cost, each initial cheapest incoming choice is a cycle edge; ties can
/// instead favour earlier root edges. Costs are synthetic, not observed data.
/// @param cycle_vertices At least two cycle vertices; total vertex count is this plus one.
/// @param cycle_cost Finite cycle arc cost, default one; negatives are allowed.
/// @param entry_cost Finite root-to-cycle arc cost, default five.
/// @return Owned ordered 2*cycle_vertices finite weighted arcs.
/// @throws std::invalid_argument If the cycle has fewer than two vertices or costs are nonfinite.
/// @throws std::length_error If vertex/edge dimensions exceed capacity.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto edges=dm::algorithms::rooted_cycle_edges(3);
/// std::print("{} {}\n",edges.size(),dm::algorithms::chu_liu_edmonds(4,edges,0).total_weight); // 6 7
/// @endcode
/// @see chu_liu_edmonds_trace()
std::vector<WeightedEdge> rooted_cycle_edges(std::size_t cycle_vertices,double cycle_cost=1,double entry_cost=5);
}
