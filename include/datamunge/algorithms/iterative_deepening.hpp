#pragma once
/// @file iterative_deepening.hpp
/// @brief Explicit-frame directed iterative deepening with actual pass and transition records.
#include <cstddef>
#include <datamunge/algorithms/basic_path_result.hpp>
#include <utility>
#include <vector>
namespace datamunge::algorithms {
/// @brief Actual chronological transition kind in a depth-limited search.
/// Goal stops before its adjacency and leaves the successful stack active; no
/// synthetic leave events are added. Cutoff means a non-goal entry at the limit,
/// even if that vertex has no outgoing arcs. Examine precedes the cycle decision.
/// @par Example
/// @code
/// const auto k=dm::algorithms::DeepeningEventKind::enter;
/// std::print("{}\n",k==dm::algorithms::DeepeningEventKind::enter); // true
/// @endcode
/// @see DeepeningEvent
enum class DeepeningEventKind {
  pass_start, ///< A new actual limit is about to enter the start.
  enter,      ///< Vertex was marked on the current path and pushed.
  examine,    ///< One original outgoing arc is being considered.
  cutoff,     ///< Non-goal entry reached the current depth limit.
  leave,      ///< A failed/cutoff activation was popped and unmarked.
  goal,       ///< Entered goal ends this pass successfully.
  pass_end    ///< Actual pass ended; successful path remains active.
};
/// @brief Optional complete-graph and retained-event count ceilings.
/// Domain validation precedes caps/allocation, including irrelevant components.
/// Zero disables each optional cap. Event cap excludes input copies, pass
/// summaries, ordinary adjacency/path membership/frames, counters, metadata,
/// capacity and byte/peak RAM. It is checked before each actual event append.
/// Failures return no partial trace; untraced calls ignore max_events.
/// @par Example
/// @code
/// const dm::algorithms::DeepeningOptions o;
/// std::print("{} {}\n",o.max_vertices,o.max_events); // 1000000 1000000
/// @endcode
/// @see iterative_deepening_dfs_trace()
struct DeepeningOptions {
  std::size_t max_vertices{
      1000000}; ///< Complete n cap, including isolated/unreachable vertices; zero disables.
  std::size_t max_edges{
      1000000}; ///< Complete original directed arc count, including duplicates/loops; zero disables.
  std::size_t max_events{1000000}; ///< Actual retained scalar-event count cap; zero disables.
};
/// @brief Owned scalar record emitted at an actual shared-kernel transition.
/// limit and depth count edges, start depth zero. neighbor is n except for
/// examine. edge_index is an original arc id for examine and the activation's
/// incoming id for enter/leave/cutoff/goal; root and pass boundaries use E.
/// Reconstruct active paths from enter/leave; successful stop has no fake pops.
/// @par Example
/// @code
/// const auto t=dm::algorithms::iterative_deepening_dfs_trace(2,{{0,1}},0,1,1);
/// std::print("{}\n",t.events.back().kind==dm::algorithms::DeepeningEventKind::pass_end); // true
/// @endcode
/// @see iterative_deepening_dfs_trace()
struct DeepeningEvent {
  DeepeningEventKind kind{DeepeningEventKind::pass_start}; ///< Actual transition kind.
  std::size_t        limit{0};                             ///< Current permitted edge depth.
  std::size_t        vertex{0};     ///< Current/new/popped vertex, or start at pass boundaries.
  std::size_t        depth{0};      ///< Current/new/popped vertex depth; zero for pass boundaries.
  std::size_t        neighbor{0};   ///< Examined successor or sentinel n.
  std::size_t        edge_index{0}; ///< Original arc id or sentinel E, as described above.
  bool cycle_blocked{false};        ///< Examine-only flag: successor is already on this path.
};
/// @brief Selected actual counters for one produced depth-limited pass.
/// Entries include goal and cutoff vertices and can revisit a graph vertex on
/// different simple prefixes; these are not unique-vertex or adjacency-scan counts.
/// All count arithmetic throws on overflow. No elapsed runtime is inferred.
/// @par Example
/// @code
/// const auto r=dm::algorithms::iterative_deepening_dfs_run(2,{{0,1}},0,1,1);
/// std::print("{} {}\n",r.passes[0].entered,r.passes[1].entered); // 1 2
/// @endcode
/// @see iterative_deepening_dfs_run()
struct DeepeningPass {
  std::size_t limit{0};   ///< Actual depth limit for this pass.
  std::size_t entered{0}; ///< Actual path-prefix entries, including cutoffs/goals.
  std::size_t edges_examined{
      0};                     ///< Actual original outgoing adjacency entries read below the limit.
  std::size_t cycle_skips{0}; ///< Examined arcs blocked by current-path membership.
  std::size_t cutoffs{0};     ///< Non-goal entries at exactly the depth limit.
  std::size_t maximum_stack{0}; ///< Largest actual number of active frames/vertices.
  bool        found{false};     ///< Goal actually entered in this pass.
};
/// @brief Final legacy-compatible result and actual pass summaries from one search.
/// effective_max_depth=min(requested,n-1): every reachable vertex has a simple
/// path within n-1 edges. This avoids redundant larger passes and SIZE_MAX loop
/// wrap. Ordinary paths/counts at requested depths <=n-1 are preserved. Requested
/// larger unsuccessful searches intentionally omit repeated impossible depths.
/// No mutation, retained references or shared state.
/// @par Example
/// @code
/// const auto r=dm::algorithms::iterative_deepening_dfs_run(2,{},0,1,std::size_t(-1));
/// std::print("{} {}\n",r.effective_max_depth,r.passes.size()); // 1 2
/// @endcode
/// @see iterative_deepening_dfs_run()
struct DeepeningRun {
  BasicPathResult
      result; ///< First input-order shallowest path; expanded is the sum of actual entries.
  std::vector<DeepeningPass>
              passes; ///< Actual limit-zero through success/effective-bound summaries, no replay.
  std::size_t effective_max_depth{
      0}; ///< Complete simple-path bound applied to the requested budget.
};
/// @brief Owned graph and chronological transitions from the producing search.
/// Every event is captured in the shared explicit-frame kernel, not by replaying
/// solved paths or repeated-prefix calls. On success, the last pass-end follows
/// goal without unwinding the retained successful active path.
/// @par Example
/// @code
/// const auto t=dm::algorithms::iterative_deepening_dfs_trace(2,{{0,1}},0,1,1);
/// std::print("{} {}\n",t.run.result.expanded,t.edges.size()); // 3 1
/// @endcode
/// @see iterative_deepening_dfs_trace()
struct DeepeningTrace {
  DeepeningRun run;      ///< Final result and summaries from this same execution.
  std::size_t  n{0};     ///< Complete vertex count.
  std::size_t  start{0}; ///< Valid requested source.
  std::size_t  goal{0};  ///< Valid requested target.
  std::vector<std::pair<std::size_t, std::size_t>>
      edges; ///< Owned complete original directed arcs in input order.
  std::vector<DeepeningEvent>
      events; ///< Actual ordered pass/enter/examine/cutoff/leave/goal transitions.
};
/// @brief Find a shallowest directed unweighted path by iterative deepening.
/// Runs limits zero through min(max_depth,n-1), stopping on the first goal.
/// Retains original per-source arc order, duplicates and self loops. Current-path
/// membership blocks only cycles, never globally marks a vertex across branches.
/// Thus shortest edge depth is guaranteed; weighted cost is not optimized.
/// Empty n/invalid terminals/endpoints are rejected, including unused arcs even
/// for start==goal. Equal terminals produce [start] in the limit-zero pass.
/// Explicit heap frames avoid recursive call-stack growth. Full graph storage
/// O(n+E), membership O(n), active frames O(min(depth,n)), and pass summaries
/// O(min(max_depth,n)) are separate from optional events/input copies. Work can
/// be exponential in permitted simple prefixes. On a complete b-ary tree the
/// visit sum is sum_(k=0)^d (d-k+1)*b^k; constant overhead claims require b>1.
/// Counters exclude validation, graph construction, comparisons, allocation and
/// copying. No caller mutation, retained references or shared state.
/// @param n Positive complete vertex count, vertices [0,n).
/// @param edges Borrowed complete directed arcs in input order.
/// @param start Valid source vertex.
/// @param goal Valid target vertex.
/// @param max_depth Maximum permitted edge depth; SIZE_MAX is safe and bounded by n-1.
/// Legacy entry point disables optional graph/event caps.
/// @return Owned path, cumulative actual entries and found flag.
/// @throws std::invalid_argument On any invalid terminal or original endpoint.
/// @throws std::length_error On enabled complete cap, count overflow or exhausted capacity.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto r=dm::algorithms::iterative_deepening_dfs(2,{{0,1}},0,1,1);
/// std::print("{}\n",r.expanded); // 3
/// @endcode
/// @par References
/// - R. E. Korf (1985), Depth-first iterative-deepening: An optimal admissible tree search,
///   Artificial Intelligence 27,97–109. doi:10.1016/0004-3702(85)90084-0.
///   This explicit finite-graph variant uses current-path cycle exclusion.
/// @see BasicPathResult, DeepeningOptions, iterative_deepening_dfs_run(), iterative_deepening_dfs_trace()
BasicPathResult iterative_deepening_dfs(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    std::size_t                                             start,
    std::size_t                                             goal,
    std::size_t                                             max_depth);
/// @brief Search once and retain selected actual per-pass work.
/// Runs limits zero through min(max_depth,n-1), stopping on the first goal.
/// Retains original per-source arc order, duplicates and self loops. Current-path
/// membership blocks only cycles, never globally marks a vertex across branches.
/// Thus shortest edge depth is guaranteed; weighted cost is not optimized.
/// Empty n/invalid terminals/endpoints are rejected, including unused arcs even
/// for start==goal. Equal terminals produce [start] in the limit-zero pass.
/// Explicit heap frames avoid recursive call-stack growth. Full graph storage
/// O(n+E), membership O(n), active frames O(min(depth,n)), and pass summaries
/// O(min(max_depth,n)) are separate from optional events/input copies. Work can
/// be exponential in permitted simple prefixes. On a complete b-ary tree the
/// visit sum is sum_(k=0)^d (d-k+1)*b^k; constant overhead claims require b>1.
/// Counters exclude validation, graph construction, comparisons, allocation and
/// copying. No caller mutation, retained references or shared state.
/// @param n Positive complete vertex count, vertices [0,n).
/// @param edges Borrowed complete directed arcs in input order.
/// @param start Valid source vertex.
/// @param goal Valid target vertex.
/// @param max_depth Maximum permitted edge depth; SIZE_MAX is safe and bounded by n-1.
/// @param options Complete graph/event policies, default million each; zero disables optional cap.
/// @return Owned actual result/pass summaries.
/// @throws std::invalid_argument On any invalid terminal or original endpoint.
/// @throws std::length_error On enabled complete cap, count overflow or exhausted capacity.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto r=dm::algorithms::iterative_deepening_dfs_run(2,{{0,1}},0,1,1);
/// std::print("{}\n",r.result.expanded); // 3
/// @endcode
/// @par References
/// - R. E. Korf (1985), Depth-first iterative-deepening: An optimal admissible tree search,
///   Artificial Intelligence 27,97–109. doi:10.1016/0004-3702(85)90084-0.
///   This explicit finite-graph variant uses current-path cycle exclusion.
/// @see BasicPathResult, DeepeningOptions, iterative_deepening_dfs_run(), iterative_deepening_dfs_trace()
DeepeningRun iterative_deepening_dfs_run(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    std::size_t                                             start,
    std::size_t                                             goal,
    std::size_t                                             max_depth,
    DeepeningOptions                                        options = {});
/// @brief Capture actual directed depth-limited passes and transitions.
/// Runs limits zero through min(max_depth,n-1), stopping on the first goal.
/// Retains original per-source arc order, duplicates and self loops. Current-path
/// membership blocks only cycles, never globally marks a vertex across branches.
/// Thus shortest edge depth is guaranteed; weighted cost is not optimized.
/// Empty n/invalid terminals/endpoints are rejected, including unused arcs even
/// for start==goal. Equal terminals produce [start] in the limit-zero pass.
/// Explicit heap frames avoid recursive call-stack growth. Full graph storage
/// O(n+E), membership O(n), active frames O(min(depth,n)), and pass summaries
/// O(min(max_depth,n)) are separate from optional events/input copies. Work can
/// be exponential in permitted simple prefixes. On a complete b-ary tree the
/// visit sum is sum_(k=0)^d (d-k+1)*b^k; constant overhead claims require b>1.
/// Counters exclude validation, graph construction, comparisons, allocation and
/// copying. No caller mutation, retained references or shared state.
/// @param n Positive complete vertex count, vertices [0,n).
/// @param edges Borrowed complete directed arcs in input order.
/// @param start Valid source vertex.
/// @param goal Valid target vertex.
/// @param max_depth Maximum permitted edge depth; SIZE_MAX is safe and bounded by n-1.
/// @param options Complete graph/event policies, default million each; zero disables optional cap.
/// @return Owned graph, actual events and final result.
/// @throws std::invalid_argument On any invalid terminal or original endpoint.
/// @throws std::length_error On enabled complete cap, count overflow or exhausted capacity.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto r=dm::algorithms::iterative_deepening_dfs_trace(2,{{0,1}},0,1,1);
/// std::print("{}\n",r.run.result.expanded); // 3
/// @endcode
/// @par References
/// - R. E. Korf (1985), Depth-first iterative-deepening: An optimal admissible tree search,
///   Artificial Intelligence 27,97–109. doi:10.1016/0004-3702(85)90084-0.
///   This explicit finite-graph variant uses current-path cycle exclusion.
/// @see BasicPathResult, DeepeningOptions, iterative_deepening_dfs_run(), iterative_deepening_dfs_trace()
DeepeningTrace iterative_deepening_dfs_trace(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    std::size_t                                             start,
    std::size_t                                             goal,
    std::size_t                                             max_depth,
    DeepeningOptions                                        options = {});
} // namespace datamunge::algorithms
