#pragma once
/// @file push_relabel.hpp
/// @brief Checked FIFO preflow discharge with actual pushes, heights and excess histories.
#include <datamunge/algorithms/dinic.hpp>
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {
/// @brief Selected-event and retained-entry policies for native FIFO push-relabel.
/// Zero independently disables either cap; validation/allocation are outside selected work.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::PushRelabelOptions{}.max_operations); // 10000000
/// @endcode
struct PushRelabelOptions {
  std::size_t max_operations{
      10000000}; ///< Actual arc inspections,pushes,relabels and active queue pops.
  std::size_t max_trace_entries{
      1000000}; ///< Owned network edge/header records, initial height/excess entries and actual update records.
};
/// @brief Actual mutation kind in native preflow discharge.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::push_relabel_trace(2,{{0,1,9}},0,1).events[0].kind==dm::algorithms::PreflowEventKind::Push); // true
/// @endcode
enum class PreflowEventKind {
  Push,   ///< Actual residual/excess push, including source preflow.
  Relabel ///< Actual height increase after exhausting the current adjacency scan.
};
/// @brief One actual residual push or height relabel with immediately used state.
/// For relabels, to=from, amount and residual fields are zero, and input is the size_t maximum sentinel. Height/excess fields refer to actual current vertices. Forward=false identifies an original-edge cancellation arc.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::push_relabel_trace(2,{{0,1,9}},0,1).events[0].amount); // 9
/// @endcode
struct PreflowEvent {
  PreflowEventKind kind;             ///< Actual completed mutation kind.
  bool             preflow{false};   ///< True only for initial source saturation.
  std::size_t      from{0};          ///< Actual changed/pushing vertex.
  std::size_t      to{0};            ///< Actual push target, or from for relabel.
  std::size_t      input{0};         ///< Original input edge identity, sentinel for relabel.
  bool             forward{false};   ///< Whether the actual residual arc is the original direction.
  std::size_t      height_before{0}; ///< Actual from height immediately before this mutation.
  std::size_t      height_after{0};  ///< Actual from height immediately afterward.
  std::size_t      target_height{0}; ///< Actual target height at this event.
  double           amount{0};        ///< Actual positive push amount; zero for relabel.
  double           residual_before{0};    ///< Actual oriented residual before push.
  double           residual_after{0};     ///< Actual oriented residual after push.
  double           reverse_before{0};     ///< Actual paired reverse residual before push.
  double           reverse_after{0};      ///< Actual paired reverse residual after push.
  double           from_excess_before{0}; ///< Actual source-vertex excess before mutation.
  double           from_excess_after{0};  ///< Actual source-vertex excess after mutation.
  double           to_excess_before{0};   ///< Actual target excess before mutation.
  double           to_excess_after{0};    ///< Actual target excess after mutation.
};
/// @brief Selected actual native FIFO preflow work.
/// No heap/gap/global-relabel heuristics are used; counts exclude validation and container internals.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::push_relabel_run(2,{{0,1,9}},0,1).work.pushes); // 1
/// @endcode
struct PushRelabelWork {
  std::size_t arc_inspections{0}; ///< Actual source/current/relabel/final-cut residual arc reads.
  std::size_t pushes{0};          ///< Actual positive pushes, including initial source saturation.
  std::size_t relabels{0};        ///< Actual completed height updates.
  std::size_t queue_pops{0};      ///< Actual active nonterminal discharges started.
  std::size_t operations{0};      ///< Sum of the four selected event counts.
};
/// @brief Owned flow/cut, final preflow state and selected actual work.
/// Final active nonterminal excess is zero on successful return; numerical certificate checks remain appropriate for double arithmetic.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::push_relabel_run(2,{{0,1,9}},0,1).result.max_flow); // 9
/// @endcode
struct PushRelabelRun {
  MaxFlowResult
      result; ///< Actual flow value, original-edge flows and ascending residual source-side cut.
  PushRelabelWork          work; ///< Actual complete selected work, including final cut traversal.
  std::vector<std::size_t> heights; ///< Actual final vertex heights.
  std::vector<double>      excess;  ///< Actual final net inflow at each vertex.
};
/// @brief Actual owned source network, initialization and producing preflow events.
/// Initial heights/excess are copied after setting source height=V and before source saturation; no second max-flow execution populates the history.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::push_relabel_trace(2,{{0,1,9}},0,1).initial_heights); // [2, 0]
/// @endcode
struct PushRelabelTrace {
  FlowNetwork              network;         ///< Owned exact input network and terminals.
  PushRelabelRun           run;             ///< Ordinary actual result/final arrays/work.
  std::vector<std::size_t> initial_heights; ///< Actual initialized labels before source saturation.
  std::vector<double>      initial_excess;  ///< Actual initial zero excesses.
  std::vector<PreflowEvent>
              events; ///< Actual source pushes, discharge pushes and relabels in producing order.
  std::size_t captured_entries{0}; ///< Actual admitted logical history entries.
};
/// @brief Compute flow by the single checked native FIFO preflow producer.
/// FIFO active vertices, current-arc scans and minimum-neighbor relabels implement the native preflow method. Complete graph validation precedes adjacency construction. Parallel/antiparallel edges own independent residual pairs; self-loops keep zero flow and are never source-saturated. Uses strictly positive residuals/excesses, replacing the historical absolute1e-12 cutoff, so tiny finite capacities are not silently discarded. Each positive residual/excess update must make finite representable progress or the call throws without returning a partial result. This is checked double arithmetic, not an exact-real solver. Original-edge flows are read from paired reverse residuals. Ordinary algorithmic bound O(V*V*E+V+E), O(V+E) working storage; default selected policy may stop a finite run. Borrowed inputs stay unchanged and state is local.
/// @param vertex_count Vertex count with two distinct valid terminals.
/// @param edges Borrowed directed finite nonnegative capacity records.
/// @param source Source id.
/// @param sink Distinct sink id.
/// @return Owned actual flow/cut.
/// @throws std::invalid_argument If terminals/endpoints are invalid or capacities are negative/nonfinite.
/// @throws std::length_error If selected work/history policies or representable residual/storage geometry are exceeded.
/// @throws std::overflow_error If an actual residual/excess update is nonfinite.
/// @throws std::runtime_error If a positive update cannot make representable progress or a checked residual/height invariant fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::push_relabel_max_flow(2,{{0,1,9}},0,1).max_flow); // 9
/// @endcode
/// @par References
/// - A. V. Goldberg and R. E. Tarjan (1988), A new approach to the maximum-flow problem, Journal of the ACM35,921–940, doi:10.1145/48014.61051.
/// @see max_flow_certificate(), edmonds_karp(), dinic_max_flow()
MaxFlowResult push_relabel_max_flow(
    std::size_t                      vertex_count,
    const std::vector<CapacityEdge>& edges,
    std::size_t                      source,
    std::size_t                      sink);
/// @brief Compute flow by the single checked native FIFO preflow producer.
/// FIFO active vertices, current-arc scans and minimum-neighbor relabels implement the native preflow method. Complete graph validation precedes adjacency construction. Parallel/antiparallel edges own independent residual pairs; self-loops keep zero flow and are never source-saturated. Uses strictly positive residuals/excesses, replacing the historical absolute1e-12 cutoff, so tiny finite capacities are not silently discarded. Each positive residual/excess update must make finite representable progress or the call throws without returning a partial result. This is checked double arithmetic, not an exact-real solver. Original-edge flows are read from paired reverse residuals. Ordinary algorithmic bound O(V*V*E+V+E), O(V+E) working storage; default selected policy may stop a finite run. Borrowed inputs stay unchanged and state is local.
/// @param vertex_count Vertex count with two distinct valid terminals.
/// @param edges Borrowed directed finite nonnegative capacity records.
/// @param source Source id.
/// @param sink Distinct sink id.
/// @param options Selected-operation and actual-history policies.
/// @return Owned actual flow/cut and final state/work.
/// @throws std::invalid_argument If terminals/endpoints are invalid or capacities are negative/nonfinite.
/// @throws std::length_error If selected work/history policies or representable residual/storage geometry are exceeded.
/// @throws std::overflow_error If an actual residual/excess update is nonfinite.
/// @throws std::runtime_error If a positive update cannot make representable progress or a checked residual/height invariant fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::push_relabel_run(2,{{0,1,9}},0,1).result.max_flow); // 9
/// @endcode
/// @par References
/// - A. V. Goldberg and R. E. Tarjan (1988), A new approach to the maximum-flow problem, Journal of the ACM35,921–940, doi:10.1145/48014.61051.
/// @see max_flow_certificate(), edmonds_karp(), dinic_max_flow()
PushRelabelRun push_relabel_run(
    std::size_t                      vertex_count,
    const std::vector<CapacityEdge>& edges,
    std::size_t                      source,
    std::size_t                      sink,
    PushRelabelOptions               options = {});
/// @brief Compute flow by the single checked native FIFO preflow producer.
/// FIFO active vertices, current-arc scans and minimum-neighbor relabels implement the native preflow method. Complete graph validation precedes adjacency construction. Parallel/antiparallel edges own independent residual pairs; self-loops keep zero flow and are never source-saturated. Uses strictly positive residuals/excesses, replacing the historical absolute1e-12 cutoff, so tiny finite capacities are not silently discarded. Each positive residual/excess update must make finite representable progress or the call throws without returning a partial result. This is checked double arithmetic, not an exact-real solver. Original-edge flows are read from paired reverse residuals. Ordinary algorithmic bound O(V*V*E+V+E), O(V+E) working storage; default selected policy may stop a finite run. Borrowed inputs stay unchanged and state is local.
/// @param vertex_count Vertex count with two distinct valid terminals.
/// @param edges Borrowed directed finite nonnegative capacity records.
/// @param source Source id.
/// @param sink Distinct sink id.
/// @param options Selected-operation and actual-history policies.
/// @return Owned actual flow/cut, final state/work and producing histories.
/// @throws std::invalid_argument If terminals/endpoints are invalid or capacities are negative/nonfinite.
/// @throws std::length_error If selected work/history policies or representable residual/storage geometry are exceeded.
/// @throws std::overflow_error If an actual residual/excess update is nonfinite.
/// @throws std::runtime_error If a positive update cannot make representable progress or a checked residual/height invariant fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::push_relabel_trace(2,{{0,1,9}},0,1).run.result.max_flow); // 9
/// @endcode
/// @par References
/// - A. V. Goldberg and R. E. Tarjan (1988), A new approach to the maximum-flow problem, Journal of the ACM35,921–940, doi:10.1145/48014.61051.
/// @see max_flow_certificate(), edmonds_karp(), dinic_max_flow()
PushRelabelTrace push_relabel_trace(
    std::size_t                      vertex_count,
    const std::vector<CapacityEdge>& edges,
    std::size_t                      source,
    std::size_t                      sink,
    PushRelabelOptions               options = {});
} // namespace datamunge::algorithms
