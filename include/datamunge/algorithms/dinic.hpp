#pragma once
/// @file dinic.hpp
/// @brief Checked iterative Dinic flow, actual blocking phases and numerical primal/dual checks.
#include <datamunge/algorithms/edmonds_karp.hpp>
#include <cstddef>
#include <tuple>
#include <vector>
namespace datamunge::algorithms {
/// @brief Directed (tail,head,capacity) input arc; each parallel/antiparallel arc has its own identity.
/// @par Example
/// See dinic_max_flow().
/// @see FlowNetwork
using CapacityEdge=std::tuple<std::size_t,std::size_t,double>;
/// @brief Owned directed capacitated fixture, with explicit terminals and isolated vertices.
/// Generators validate capacities and checked output counts before allocation.
/// @par Example
/// See dinic_disjoint_paths().
/// @see dinic_staircase_paths()
struct FlowNetwork {
    std::size_t vertex_count{0}; ///< Number of vertices, including isolated ones.
    std::vector<CapacityEdge> edges; ///< Directed finite nonnegative capacities in input order.
    std::size_t source{0}; ///< Source id.
    std::size_t sink{1}; ///< Distinct sink id.
};
/// @brief Logical operations in one actual Dinic execution, excluding validation/allocation.
/// BFS scans include zero-capacity, reverse and self arcs of reached vertices. DFS scans
/// include repeated use of a current arc after successful pushes. A pointer advance is
/// permanent within its phase. Path updates count oriented residual pairs, not both writes.
/// @par Example
/// @code
/// const auto t=dm::algorithms::dinic_max_flow_trace(2,{{0,1,7.5}},0,1);
/// std::print("{} {} {}\n",t.work.bfs_calls,t.work.blocking_phases,t.work.augmentations); // 2 1 1
/// @endcode
/// @see DinicTrace
struct DinicWork {
    std::size_t bfs_calls{0}; ///< Includes the final unsuccessful search.
    std::size_t blocking_phases{0}; ///< Searches that reach the sink and compute a blocking flow.
    std::size_t bfs_arc_inspections{0}; ///< Residual arcs inspected by all complete BFS traversals.
    std::size_t dfs_arc_inspections{0}; ///< Current-arc tests by the explicit depth-first stack.
    std::size_t current_arc_advances{0}; ///< Permanent current-pointer increments within phases.
    std::size_t augmentations{0}; ///< Positive source-to-sink path pushes.
    std::size_t path_arc_updates{0}; ///< Sum of pushed path lengths.
    std::size_t max_path_edges{0}; ///< Longest successful augmenting path.
};
/// @brief Owned before/after values for one oriented residual pair on an actual pushed path.
/// Arc ids 2*i and 2*i+1 denote the forward and cancellation arcs of input edge i.
/// A reverse use decreases the original edge flow. Self-loop pairing remains unambiguous.
/// @par Example
/// See dinic_max_flow_trace().
/// @see DinicAugmentation
struct DinicPathArc {
    std::size_t input_edge; ///< Original input edge identity.
    std::size_t from; ///< Residual tail.
    std::size_t to; ///< Residual head.
    bool reverse; ///< True for cancellation of original edge flow.
    double residual_before; ///< Capacity of the used oriented arc before the push.
    double residual_after; ///< Capacity of the used oriented arc after the push.
    double paired_before; ///< Capacity of its opposite arc before the push.
    double paired_after; ///< Capacity of its opposite arc after the push.
};
/// @brief Owned actual path push, with residual operands copied before mutation.
/// @par Example
/// See dinic_max_flow_trace().
/// @see DinicPhase
struct DinicAugmentation {
    std::size_t phase; ///< Index of the owning BFS phase record.
    double amount; ///< Positive path bottleneck.
    double flow_before; ///< Total flow before this push.
    double flow_after; ///< Total flow after this push.
    std::vector<DinicPathArc> path; ///< Oriented source-to-sink residual arc sequence.
};
/// @brief Complete BFS levels and actual blocking-flow work of one search, including termination.
/// Levels stay fixed during a blocking phase. Unreachable vertices have sentinel vertex_count.
/// The final record has unreachable sink, no augmentations and unchanged flow.
/// @par Example
/// See dinic_max_flow_trace().
/// @see DinicWork
struct DinicPhase {
    std::vector<std::size_t> levels; ///< BFS distances in the actual pre-phase residual graph.
    std::size_t sink_level; ///< Shortest residual path length, or vertex_count on failure.
    std::size_t augmentation_begin; ///< First associated augmentation index.
    std::size_t augmentation_count; ///< Associated path-push count.
    double flow_before; ///< Flow entering this phase.
    double flow_after; ///< Flow after the level graph is blocked, or unchanged on termination.
    std::size_t bfs_arc_inspections; ///< Residual arc visits in this BFS.
    std::size_t dfs_arc_inspections; ///< Residual arc tests in this phase's DFS calls.
    std::size_t current_arc_advances; ///< Current-pointer advances in this phase.
};
/// @brief Value-owned original input and actual phases/path pushes, captured without replay.
/// Ordinary graph/work storage is O(V+E). Histories add C=P*(V+1)+A+L entries,
/// where P includes final BFS, A counts augmentations and L sums path lengths.
/// Each level, phase, augmentation and path-arc record counts one entry; owned original
/// edges/results/work arrays are outside this optional history budget. No input views survive.
/// @par Example
/// See dinic_max_flow_trace().
/// @see DinicPhase
struct DinicTrace {
    FlowNetwork network; ///< Owned original graph and terminals.
    MaxFlowResult result; ///< Same final value, per-edge flow and cut as plain Dinic.
    DinicWork work; ///< Actual logical counters.
    std::vector<DinicPhase> phases; ///< Successful blocking phases plus final failed BFS.
    std::vector<DinicAugmentation> augmentations; ///< Actual positive residual path pushes.
};
/// @brief Computes directed maximum flow using BFS levels and iterative current-arc blocking flow.
/// Preserves the original four-argument API and MaxFlowResult layout. Adjacency follows
/// input order, including separate forward/reverse pairs for parallel/antiparallel arcs
/// and self-loops. Uses strictly positive residuals, replacing the former absolute 1e-12
/// cutoff; tiny positive capacities are not discarded. No recursive DFS or history storage.
/// Complete terminal/finite-capacity validation precedes adjacency allocation. Returned
/// per-edge flows come from each pair's cancellation residual, retaining small represented
/// flows. Every residual subtraction/addition and total flow must remain finite and make
/// positive representable progress; otherwise the fit fails explicitly, with no partial result.
/// This is double arithmetic, not an exact-real or arbitrary-precision solver. Calls borrow
/// inputs unchanged and share no mutable state. Exact-arithmetic bound O(V^2 E) after graph
/// initialization, O(V+E) storage; the terminal BFS adds O(V+E) for empty/disconnected cases.
/// @param vertex_count Vertex count admitting distinct terminals.
/// @param edges Borrowed directed finite nonnegative capacities; endpoints in range.
/// @param source Source id.
/// @param sink Distinct sink id.
/// @return Owned value, per-input-edge flows and final residual BFS source-side order.
/// @throws std::invalid_argument Invalid terminal, endpoint or nonfinite/negative capacity.
/// @throws std::length_error Residual-pool or logical-counter capacity exceeded.
/// @throws std::overflow_error Residual or total arithmetic becomes nonfinite.
/// @throws std::runtime_error A positive push loses representable residual/total progress.
/// @par Example
/// @code
/// const auto r=dm::algorithms::dinic_max_flow(2,{{0,1,7.5}},0,1);
/// std::print("{:.1f} {}\n",r.max_flow,r.min_cut_source_side); // 7.5 [0]
/// @endcode
/// @par References
/// - E. A. Dinic (1970), Algorithm for solution of a problem of maximum flow in a
///   network with power estimation. Soviet Math. Dokl. 11, 1277-1280.
///   https://www.cs.bgu.ac.il/~dinitz/D70.pdf
///   This variant uses current arcs and an explicit DFS stack, with checked double arithmetic.
/// @see dinic_max_flow_trace(), max_flow_certificate()
MaxFlowResult dinic_max_flow(std::size_t vertex_count,const std::vector<CapacityEdge>& edges,
    std::size_t source,std::size_t sink);
/// @brief Captures actual BFS levels and residual path updates in the same Dinic kernel.
/// The history budget is reserved before each BFS or path update, including the terminal
/// failed BFS. Exceeding it returns no partial trace. Storage/work for capture adds O(C)
/// to the base algorithm, with C defined in DinicTrace. Zero disables only the optional cap.
/// @param vertex_count Vertex count admitting terminals.
/// @param edges Borrowed directed finite nonnegative capacities.
/// @param source Source id.
/// @param sink Distinct sink id.
/// @param max_trace_entries Level/phase/augmentation/path-arc entry cap, default 1000000.
/// @return Owned exact execution trace, original input copy and ordinary result.
/// @throws std::invalid_argument Invalid graph as in dinic_max_flow().
/// @throws std::length_error History budget, residual-pool or counter capacity exceeded.
/// @throws std::overflow_error Nonfinite residual or total arithmetic.
/// @throws std::runtime_error Lost positive arithmetic progress.
/// @par Example
/// @code
/// const auto t=dm::algorithms::dinic_max_flow_trace(2,{{0,1,7.5}},0,1,8);
/// std::print("{} {:.1f} {}\n",t.phases[0].sink_level,t.augmentations[0].amount,t.phases.back().sink_level); // 1 7.5 2
/// @endcode
/// @par References
/// - Dinic (1970), https://www.cs.bgu.ac.il/~dinitz/D70.pdf; records the executing blocking-flow variant.
/// @see DinicTrace, dinic_max_flow()
[[nodiscard]] DinicTrace dinic_max_flow_trace(std::size_t vertex_count,const std::vector<CapacityEdge>& edges,
    std::size_t source,std::size_t sink,std::size_t max_trace_entries=1000000);
/// @brief Numerical primal/dual checks for a public maximum-flow result.
/// Feasibility and cut equality are tested separately. A nonzero tolerance gives numerical
/// agreement, not an exact proof; every error and the signed gap remain visible.
/// @par Example
/// See max_flow_certificate().
/// @see MaxFlowResult
struct FlowCertificate {
    double cut_capacity; ///< Sum of original capacities crossing supplied source side to complement.
    double max_capacity_violation; ///< Maximum of zero, -edge_flow and edge_flow-capacity.
    double max_conservation_error; ///< Maximum absolute net outflow at nonterminals.
    double source_net_flow; ///< Net outflow from source.
    double sink_net_flow; ///< Net inflow to sink.
    double value_balance_error; ///< Maximum source/sink disagreement with reported value.
    double duality_gap; ///< Signed cut_capacity minus reported max_flow.
    bool cut_separates; ///< Supplied cut includes source and excludes sink.
    bool feasible; ///< Capacity, conservation, value-balance and nonnegative-value checks pass.
    bool optimal_within_tolerance; ///< Feasible, separating and absolute gap <= tolerance.
    std::vector<std::size_t> crossing_edges; ///< Original crossing edge indices, including parallel arcs.
};
/// @brief Checks edge bounds, conservation, reported value and a supplied cut independently of the solver.
/// Uses long-double sums followed by checked finite-double conversion. Does not run Dinic,
/// infer a different cut or silently adjust flows. Work O(V+E), owned work/output O(V+E).
/// Default absolute tolerance is zero, in capacity units; a positive tolerance explicitly
/// relaxes every scalar check. Inputs are unchanged and calls share no state.
/// @param vertex_count Vertex count admitting terminals.
/// @param edges Original finite nonnegative capacities.
/// @param source Source id.
/// @param sink Distinct sink id.
/// @param result Borrowed result with exactly one finite flow per edge and finite reported value.
/// @param tolerance Finite nonnegative absolute tolerance, default zero.
/// @return Owned errors, cut capacity, crossing identities and separate feasible/optimal flags.
/// @throws std::invalid_argument Invalid graph/tolerance/result shape, nonfinite result values,
/// duplicated or out-of-range source-side cut ids. Finite infeasibility is returned as false.
/// @throws std::overflow_error A computed summary is outside finite double range.
/// @par Example
/// @code
/// const dm::algorithms::MaxFlowResult r{7,{7},{0}};
/// const auto c=dm::algorithms::max_flow_certificate(2,{{0,1,10}},0,1,r);
/// std::print("{} {} {:.0f}\n",c.feasible,c.optimal_within_tolerance,c.duality_gap); // true false 3
/// @endcode
/// @par References
/// - Ford and Fulkerson (1956), Maximal flow through a network. Canadian Journal of
///   Mathematics 8, 399-404. doi:10.4153/CJM-1956-045-5 (flow/cut dual bound).
/// @see FlowCertificate, dinic_max_flow()
[[nodiscard]] FlowCertificate max_flow_certificate(std::size_t vertex_count,const std::vector<CapacityEdge>& edges,
    std::size_t source,std::size_t sink,const MaxFlowResult& result,double tolerance=0);
/// @brief Generates equally long internally disjoint source-to-sink paths for one blocking phase.
/// Source=0, sink=1; paths are emitted in order, each with fresh internal vertices. With p
/// paths of length l>=1, E=p*l and V=2+p*(l-1). Zero paths yields two isolated terminals.
/// For length one the paths are distinct parallel arcs. Counts/capacity guards precede
/// allocation; no shared state or randomness. Work/storage O(V+E).
/// @param paths Number of paths.
/// @param length Edges per path, positive, default two.
/// @param capacity Common finite nonnegative capacity, default one.
/// @param max_edges Complete output cap, default 1000000; zero disables.
/// @return Owned network.
/// @throws std::invalid_argument Zero length or invalid capacity, including for empty output.
/// @throws std::length_error Count arithmetic, container or optional edge cap exceeded.
/// @par Example
/// @code
/// const auto g=dm::algorithms::dinic_disjoint_paths(3,2);
/// std::print("{} {}\n",g.vertex_count,dm::algorithms::dinic_max_flow(g.vertex_count,g.edges,g.source,g.sink).max_flow); // 5 3
/// @endcode
/// @see dinic_staircase_paths()
[[nodiscard]] FlowNetwork dinic_disjoint_paths(std::size_t paths,std::size_t length=2,
    double capacity=1,std::size_t max_edges=1000000);
/// @brief Generates disjoint unit-capacity paths of lengths one through paths.
/// Exactly one path augments in each successive phase. E=p*(p+1)/2 and V=2+p*(p-1)/2;
/// checked factors are divided before multiplication. Zero paths gives isolated terminals.
/// Counts/optional cap precede allocation; O(V+E) work/storage, no random/shared state.
/// @param paths Longest path length and number of paths.
/// @param max_edges Complete output cap, default 1000000; zero disables.
/// @return Owned network with source zero and sink one.
/// @throws std::length_error Count arithmetic, container or optional cap exceeded.
/// @par Example
/// @code
/// const auto g=dm::algorithms::dinic_staircase_paths(3);
/// std::print("{} {}\n",g.vertex_count,g.edges.size()); // 5 6
/// @endcode
/// @see dinic_disjoint_paths(), DinicPhase
[[nodiscard]] FlowNetwork dinic_staircase_paths(std::size_t paths,std::size_t max_edges=1000000);
} // namespace datamunge::algorithms
