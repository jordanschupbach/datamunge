#pragma once
/// @file johnson.hpp
/// @brief Checked directed all-pairs paths with actual potential and lazy-heap histories.
#include <cstddef>
#include <datamunge/algorithms/mst_result.hpp>
#include <vector>
namespace datamunge::algorithms {
/// @brief Legacy aggregate of owned all-pairs distances and global negative-cycle status.
/// Rows/columns are original vertices; unreachable entries are positive infinity.
/// A detected negative cycle anywhere invalidates the entire problem: all entries
/// remain positive infinity, including the diagonal. Empty graphs return an empty
/// matrix and false. Field order and types retain the legacy aggregate contract.
/// @par Example
/// @code
/// const auto r=dm::algorithms::johnson_all_pairs_shortest_paths(3,{{0,1,1},{1,2,-2},{0,2,4}});
/// std::print("{}\n",r.distance[0][2]); // -1
/// @endcode
/// @see johnson_all_pairs_shortest_paths()
struct JohnsonAllPairsResult {
  std::vector<std::vector<double>>
       distance; ///< Owned original-weight distances, or all infinity on a negative cycle.
  bool has_negative_cycle{
      false}; ///< Global detection, including components unreachable from vertex zero.
};
/// @brief Optional complete input/output and online event ceilings, measured in records.
/// Zero disables a ceiling, never arithmetic/container representability checks.
/// Validate endpoints and finite weights before admission; input count is n+m,
/// matrix count is n*n. No partial result is returned on failure. These limits
/// exclude vector capacities, allocator overhead, byte counts and peak RAM.
/// @par Example
/// @code
/// const auto a=dm::algorithms::johnson_analyze(2,{{0,1,3}},{3,4,1});
/// std::print("{}\n",a.result.distance[0][1]); // 3
/// @endcode
/// @see johnson_analyze(), johnson_trace()
struct JohnsonOptions {
  std::size_t max_input_records{1000000}; ///< Complete n+m ceiling; default million, zero off.
  std::size_t max_distance_cells{
      1000000}; ///< Complete n*n ceiling before matrix allocation; default million, zero off.
  std::size_t max_events{
      1000000}; ///< Actual event ceiling before each operation; ignored by untraced calls.
};
/// @brief Selected logical work from the producing execution, excluding validation/allocation.
/// Heap counts are operations, not internal comparator calls or elapsed time.
/// Passes include a final unchanged pass. Matrix cells count n*n initialization;
/// restored_cells counts reachable entries only. All counters are checked.
/// @par Example
/// @code
/// const auto a=dm::algorithms::johnson_analyze(2,{});
/// std::print("{} {}\n",a.work.matrix_cells,a.work.heap_pops); // 4 2
/// @endcode
/// @see JohnsonAnalysis
struct JohnsonWork {
  std::size_t matrix_cells{0};      ///< Initialized output cells, even on a negative cycle.
  std::size_t potential_passes{0};  ///< Completed edge passes, including unchanged terminal pass.
  std::size_t potential_checks{0};  ///< Actual potential relaxation attempts.
  std::size_t potential_updates{0}; ///< Strict potential decreases.
  std::size_t reweighted_edges{0};  ///< Actual original-edge reweight operations.
  std::size_t heap_pushes{0};       ///< Source seeds plus successful relaxation insertions.
  std::size_t heap_pops{0};         ///< All popped entries, stale included.
  std::size_t stale_pops{0};        ///< Popped distances unequal to the current label.
  std::size_t dijkstra_checks{0};   ///< Outgoing edges examined from nonstale pops.
  std::size_t dijkstra_updates{0};  ///< Strict reduced-distance decreases.
  std::size_t restored_cells{0};    ///< Finite labels restored to original weights.
};
/// @brief Owned output, terminal potentials, reduced edges and actual selected work.
/// Potentials/reduced weights use long double arithmetic on finite binary64
/// inputs. These are rounded machine values, not exact-real certificates. A
/// negative-cycle result retains terminal potentials but has no reduced edges.
/// Edge order matches input, including parallel edges/loops. No borrowed storage.
/// @par Example
/// @code
/// const auto a=dm::algorithms::johnson_analyze(3,{{0,1,1},{1,2,-2},{0,2,4}});
/// std::print("{} {}\n",a.potentials[2],a.reduced_weights[1]); // -2 0
/// @endcode
/// @see JohnsonTrace
struct JohnsonAnalysis {
  JohnsonAllPairsResult    result;          ///< Owned original-weight matrix and cycle status.
  std::vector<long double> potentials;      ///< Final actual h, initially zero at every vertex.
  std::vector<long double> reduced_weights; ///< Actual w+h[u]-h[v] in original edge order.
  JohnsonWork              work;            ///< Selected actual counts, with no replay.
};
/// @brief Phase of an actual Johnson operation; metadata meanings are phase-specific.
/// @par Example
/// @code
/// const auto t=dm::algorithms::johnson_trace(1,{});
/// std::print("{}\n",t.events[0].action==dm::algorithms::JohnsonAction::Seed); // true
/// @endcode
/// @see JohnsonEvent
enum class JohnsonAction {
  Potential, ///< Edge attempt: pass, edge, u/v and before/candidate/after potential.
  Reweight,  ///< Edge conversion: edge/u/v, before original weight, candidate=after reduced weight.
  Seed,      ///< Source initialization: source=u=v, before infinity, candidate=after zero.
  Pop, ///< Actual heap removal: source/u=v, before=after current label, candidate popped key, stale flag meaningful.
  Relax, ///< Actual outgoing edge attempt: source/edge/u/v and before/candidate/after reduced label.
  Restore ///< Reachable output conversion: source/u=source/v, before reduced label, candidate=after original distance.
};
/// @brief Owned scalar actual operation without repeated solver runs or full snapshots.
/// Unused size_t fields are zero. Potential uses source=0 placeholder; pass is
/// zero-based and meaningful only there. Edge is original input identity and is
/// meaningful only for Potential/Reweight/Relax. Stale is false outside Pop.
/// Seed/Pop use u=v; Restore uses u=source. Infinity denotes an absent old label
/// only; finite arithmetic results are checked. After records the committed value.
/// @par Example
/// @code
/// const auto t=dm::algorithms::johnson_trace(2,{{0,1,-3}});
/// std::print("{} {}\n",t.events[0].before,t.events[0].after); // 0 -3
/// @endcode
/// @see JohnsonAction
struct JohnsonEvent {
  JohnsonAction action{JohnsonAction::Potential}; ///< Actual operation kind.
  std::size_t   pass{0};      ///< Potential pass, zero placeholder for other phases.
  std::size_t   source{0};    ///< Dijkstra/restore source, zero placeholder in preparation.
  std::size_t   edge{0};      ///< Original edge index where meaningful.
  std::size_t   u{0};         ///< Origin vertex or source/pop vertex by phase.
  std::size_t   v{0};         ///< Target vertex or source/pop vertex by phase.
  long double   before{0};    ///< Label/weight immediately before the operation.
  long double   candidate{0}; ///< Actually computed candidate/key/converted weight.
  long double   after{0};     ///< Label/weight immediately after the operation.
  bool          stale{false}; ///< Actual stale-entry decision on Pop only.
};
/// @brief Owned same-execution analysis, original graph and chronological scalar history.
/// Default event cap is a million records; no truncation, replay or input views.
/// Recording adds O(K+m) owned storage for K actual events and copied m edges.
/// @par Example
/// @code
/// const auto t=dm::algorithms::johnson_trace(2,{{0,1,3}});
/// std::print("{} {}\n",t.vertex_count,t.edges.size()); // 2 1
/// @endcode
/// @see johnson_trace()
struct JohnsonTrace {
  JohnsonAnalysis           analysis; ///< Actual output, potentials, reduced weights and counts.
  std::size_t               vertex_count{0}; ///< Original vertex count, isolates included.
  std::vector<WeightedEdge> edges;           ///< Owned original directed edges in input order.
  std::vector<JohnsonEvent> events;          ///< Actual admitted operations in chronological order.
};
/// @brief Compute directed all-pairs paths by zero-source potentials and lazy binary heaps.
/// Accepts finite weights, loops, parallel edges and disconnected vertices. All h
/// start at zero, equivalent to an implicit zero-edge source reaching every vertex.
/// In-place input-order Bellman-Ford runs at most n passes; a decrease on pass n
/// detects a global negative cycle. Otherwise edges must have nonnegative reduced
/// weight. Each source uses a lazy priority queue ordered by (distance,vertex),
/// strict decreases, input adjacency order and stale-key rejection; restore with
/// d=d'-h[source]+h[target]. Matrix output is owned; input/shared state unchanged.
/// Work O(n*n+n*m+n*(m+1)*log(m+2)), including matrix and all source seeds;
/// O(n*n+n+m) storage plus O(m+1) heap. Lazy parallel edges explain log(m+2).
/// Arithmetic uses long double, rejecting nonfinite intermediates, negative reduced
/// weights and restored results outside finite double range; finite underflow on
/// conversion may round to zero. No tolerance/clamping/exact-real cycle guarantee
/// is promised. Wider arithmetic can change last bits versus the old double kernel.
/// @param n Number of vertices, including isolates, possibly zero.
/// @param edges Borrowed directed (u,v,weight) tuples with endpoints below n.
/// @return Owned matrix and cycle status; all infinity on detected negative cycle.
/// @throws std::invalid_argument On invalid endpoints or nonfinite input weights.
/// @throws std::length_error On representability/counter/container overflow; legacy optional caps are disabled.
/// @throws std::overflow_error On nonfinite arithmetic or out-of-range finite output.
/// @throws std::domain_error If rounded reweighting produces a negative edge.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::johnson_all_pairs_shortest_paths(2,{{0,1,-1},{1,0,-1}}).has_negative_cycle); // true
/// @endcode
/// @par References
/// - D. B. Johnson (1977), Efficient algorithms for shortest paths in sparse networks,
///   Journal of the ACM 24(1),1-13, https://doi.org/10.1145/321992.321993.
/// @see johnson_analyze(), johnson_trace()
JohnsonAllPairsResult johnson_all_pairs_shortest_paths(
    std::size_t                      n,
    const std::vector<WeightedEdge>& edges);
/// @brief Compute Johnson paths and selected work without event history.
/// Uses the same kernel/contracts/arithmetic as johnson_all_pairs_shortest_paths().
/// Complete finite validation precedes n+m/n*n admission and matrix allocation.
/// max_events is ignored. Owned potentials/reduced edges add O(n+m) output.
/// @param n Number of vertices, including isolates.
/// @param edges Borrowed directed finite edges in deterministic input order.
/// @param options Complete input/output ceilings; defaults a million of each, zero off.
/// @return Owned analysis from one execution, no input references or event replay.
/// @throws std::invalid_argument On endpoint/weight violations.
/// @throws std::length_error On enabled ceilings or size/counter/container overflow.
/// @throws std::overflow_error On nonfinite arithmetic or finite output overflow.
/// @throws std::domain_error On negative rounded reduced weight.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto a=dm::algorithms::johnson_analyze(2,{{0,1,4}});
/// std::print("{} {}\n",a.work.dijkstra_checks,a.result.distance[0][1]); // 1 4
/// @endcode
/// @par References
/// - D. B. Johnson (1977), https://doi.org/10.1145/321992.321993.
/// @see johnson_trace()
JohnsonAnalysis
johnson_analyze(std::size_t n, const std::vector<WeightedEdge>& edges, JohnsonOptions options = {});
/// @brief Capture actual potential/edge/heap/restore operations from the shared kernel.
/// Same numeric and complete-admission contracts as johnson_analyze(). Copies the
/// original edges only after admission. Each event is admitted before its operation;
/// a failed call throws and exposes no partial result. Trace adds O(K+m) storage.
/// @param n Original vertex count, including isolates.
/// @param edges Borrowed original directed finite edges in stable order.
/// @param options Complete n+m/n*n ceilings and online actual event ceiling.
/// @return Owned analysis, graph and complete actual chronological event sequence.
/// @throws std::invalid_argument On invalid endpoints or nonfinite weights.
/// @throws std::length_error On enabled ceilings or size/counter/container overflow.
/// @throws std::overflow_error On nonfinite arithmetic or output overflow.
/// @throws std::domain_error On negative rounded reduced weight.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::johnson_trace(2,{{0,1,-3}});
/// std::print("{} {}\n",t.analysis.potentials[1],t.analysis.result.distance[0][1]); // -3 -3
/// @endcode
/// @par References
/// - D. B. Johnson (1977), https://doi.org/10.1145/321992.321993.
/// @see JohnsonEvent, JohnsonOptions
JohnsonTrace
johnson_trace(std::size_t n, const std::vector<WeightedEdge>& edges, JohnsonOptions options = {});
} // namespace datamunge::algorithms
