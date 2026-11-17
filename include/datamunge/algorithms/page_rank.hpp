#pragma once
/// @file page_rank.hpp
/// @brief Checked uniform-teleport PageRank and actual sparse edge-transfer histories.
#include <cstddef>
#include <utility>
#include <vector>
namespace datamunge::algorithms {
/// @brief Complete native power-iteration and optional owned-history controls.
/// @par Example
/// @code
/// dm::algorithms::PageRankOptions o;std::print("{}\n",o.damping); // 0.85
/// @endcode
/// @see page_rank_trace()
struct PageRankOptions {
  double      damping{.85};         ///< Finite link-following probability in[0,1).
  double      tolerance{1e-12};     ///< Finite positive strict L1 state-change threshold.
  std::size_t max_iterations{1000}; ///< Actual update budget,0 returns uniform initial scores.
  std::size_t max_trace_entries{
      1000000}; ///< Complete/online logical trace quota;0 disables, ignored without tracing.
};
/// @brief Owned actual returned scores, selected work and explicit stopping status.
/// @par Example
/// @code
/// auto r=dm::algorithms::page_rank_run(2,{});std::print("{} {}\n",r.iterations,r.converged); // 1 true
/// @endcode
/// @see page_rank_trace()
struct PageRankRun {
  std::vector<double> scores; ///< Actual final rank vector, initially uniform; empty for n=0.
  std::size_t         iterations{0};      ///< Actual complete power updates committed.
  std::size_t         edge_transfers{0};  ///< Actual link contribution additions, edges*iterations.
  std::size_t         dangling_checks{0}; ///< Actual source empty-outdegree checks, n*iterations.
  double last_change{0}; ///< Actual last committed L1 state change,0 placeholder if no update.
  double contraction_bound{
      0}; ///< d/(1-d)*last_change; exact-model posterior distance bound, not a floating certificate.
  bool converged{
      false}; ///< Strict last_change<tolerance, or true for empty graph; false for nonempty0budget.
};
/// @brief Actual committed destination addition for one directed edge occurrence.
/// @par Example
/// @code
/// dm::algorithms::PageRankTransfer s;std::print("{}\n",s.iteration); // 0
/// @endcode
/// @see page_rank_trace()
struct PageRankTransfer {
  std::size_t iteration{0};    ///< Actual one-based complete update ordinal.
  std::size_t source{0};       ///< Actual original source ID.
  std::size_t destination{0};  ///< Actual original destination ID.
  std::size_t out_degree{0};   ///< Actual source outdegree, including repeated/self edges.
  double      source_rank{0};  ///< Actual unchanged source rank from the before vector.
  double      contribution{0}; ///< Actual d*source_rank/out_degree in historical double order.
  double      destination_before{0}; ///< Actual destination next-buffer value before this addition.
  double      destination_after{0};  ///< Actual committed next-buffer value after this addition.
};
/// @brief Actual complete before/after vectors, uniform base and transfer range.
/// @par Example
/// @code
/// auto t=dm::algorithms::page_rank_trace(2,{});std::print("{}\n",t.iterations[0].dangling_mass); // 1
/// @endcode
/// @see page_rank_trace()
struct PageRankIteration {
  std::size_t         ordinal{0}; ///< Actual one-based update number.
  std::vector<double> before;     ///< Owned actual old source vector.
  std::vector<double> after;      ///< Owned actual committed destination vector.
  double dangling_mass{0}; ///< Actual ascending-source sum for vertices with no outgoing edge.
  double uniform_base{
      0}; ///< Actual(1-d)/n+d*dangling_mass/n assigned to every destination before transfers.
  double      l1_change{0};      ///< Actual ascending-ID sum abs(after-before).
  std::size_t transfer_begin{0}; ///< Inclusive offset into root transfers.
  std::size_t transfer_end{0};   ///< Exclusive offset into root transfers.
};
/// @brief Owned graph, initial vector and actual canonical power-update histories.
/// Quota counts vertex/numerical controls4, edge endpoints2E, initial/final
/// ranks2n, each iteration record1 with before/after2n, each fixed transfer1.
/// Fixed diagnostic fields/root vector objects are excluded, not a RAM claim.
/// @par Example
/// @code
/// auto t=dm::algorithms::page_rank_trace(2,{});std::print("{}\n",t.run.scores); // [0.5, 0.5]
/// @endcode
/// @see page_rank_trace()
struct PageRankTrace {
  std::size_t vertices{0}; ///< Original n.
  std::vector<std::pair<std::size_t, std::size_t>>
                                 edges; ///< Owned original directed occurrences, with multiplicity.
  PageRankOptions                options;    ///< Original numerical/history controls.
  std::vector<double>            initial;    ///< Owned actual uniform initial vector.
  PageRankRun                    run;        ///< Actual same-producer final result/work/status.
  std::vector<PageRankIteration> iterations; ///< Actual complete committed power updates.
  std::vector<PageRankTransfer>
      transfers; ///< Actual edge additions, grouped by source then original insertion order.
  std::size_t captured_entries{
      0}; ///< Actual logical units4+2E+2n + updates*(1+2n) + actual edge transfers.
};
/// @brief Compute actual uniform-teleport PageRank through sparse power iteration.
/// Complete endpoints and finite numerical controls are validated before graph
/// allocation/iteration; empty graph requires no edges. Repeated/self edges count
/// with multiplicity and every source chooses uniformly among edge occurrences.
/// Uniform teleport and uniform redistribution of dangling mass are both applied.
/// Old ranks remain fixed during each update; edges execute grouped by ascending
/// source, preserving insertion order within that source. Double operation and
/// convergence comparison order retain the historical valid-input trajectory.
/// Stops strictly when L1 change<tolerance, otherwise returns the budgeted iterate.
/// A nonempty0budget returns uniform scores without a convergence claim. No input
/// mutation, retained references, callbacks or shared mutable state. Scores are
/// computed doubles, not an exact sum-to-one or fixed-point accuracy certificate.
/// O(n+E) work per update and O(n+E) ordinary storage, excluding owned histories.
/// @param vertex_count Number of original vertices, zero accepted.
/// @param directed_edges Complete directed edge occurrences u->v with endpoints<n.
/// @param damping Finite d in[0,1), default.85.
/// @param tolerance Finite positive strict L1 change threshold, default1e-12.
/// @param max_iterations Actual complete update budget, default1000,0 allowed.
/// @return Owned actual std::vector<double>.
/// @throws std::invalid_argument If complete endpoints or finite numerical controls fail.
/// @throws std::length_error If complete graph/count geometry or selected history quota fails.
/// @throws std::overflow_error If produced arithmetic becomes nonfinite.
/// @throws std::bad_alloc If owned storage cannot be allocated.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::page_rank(2,{})); // [0.5, 0.5]
/// @endcode
/// @par References
/// - NetworkX upstream PageRank documentation, https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html. Native uniform-teleport/dangling model uses its documented strict L1 stop and explicit budget status.
/// @see PageRankRun, PageRankTrace
std::vector<double> page_rank(
    std::size_t                                             vertex_count,
    const std::vector<std::pair<std::size_t, std::size_t>>& directed_edges,
    double                                                  damping        = .85,
    double                                                  tolerance      = 1e-12,
    std::size_t                                             max_iterations = 1000);
/// @brief Compute actual uniform-teleport PageRank through sparse power iteration.
/// Complete endpoints and finite numerical controls are validated before graph
/// allocation/iteration; empty graph requires no edges. Repeated/self edges count
/// with multiplicity and every source chooses uniformly among edge occurrences.
/// Uniform teleport and uniform redistribution of dangling mass are both applied.
/// Old ranks remain fixed during each update; edges execute grouped by ascending
/// source, preserving insertion order within that source. Double operation and
/// convergence comparison order retain the historical valid-input trajectory.
/// Stops strictly when L1 change<tolerance, otherwise returns the budgeted iterate.
/// A nonempty0budget returns uniform scores without a convergence claim. No input
/// mutation, retained references, callbacks or shared mutable state. Scores are
/// computed doubles, not an exact sum-to-one or fixed-point accuracy certificate.
/// O(n+E) work per update and O(n+E) ordinary storage, excluding owned histories.
/// @param vertex_count Number of original vertices, zero accepted.
/// @param directed_edges Complete directed edge occurrences u->v with endpoints<n.
/// @param options Finite numerical controls and optional trace quota; default.85/1e-12/1000/1000000.
/// @return Owned actual PageRankRun.
/// @throws std::invalid_argument If complete endpoints or finite numerical controls fail.
/// @throws std::length_error If complete graph/count geometry or selected history quota fails.
/// @throws std::overflow_error If produced arithmetic becomes nonfinite.
/// @throws std::bad_alloc If owned storage cannot be allocated.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::page_rank_run(2,{}).converged); // true
/// @endcode
/// @par References
/// - NetworkX upstream PageRank documentation, https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html. Native uniform-teleport/dangling model uses its documented strict L1 stop and explicit budget status.
/// @see PageRankRun, PageRankTrace
PageRankRun page_rank_run(
    std::size_t                                             vertex_count,
    const std::vector<std::pair<std::size_t, std::size_t>>& directed_edges,
    PageRankOptions                                         options = {});
/// @brief Capture actual uniform-teleport PageRank edge additions and complete iterates.
/// Complete endpoints and finite numerical controls are validated before graph
/// allocation/iteration; empty graph requires no edges. Repeated/self edges count
/// with multiplicity and every source chooses uniformly among edge occurrences.
/// Uniform teleport and uniform redistribution of dangling mass are both applied.
/// Old ranks remain fixed during each update; edges execute grouped by ascending
/// source, preserving insertion order within that source. Double operation and
/// convergence comparison order retain the historical valid-input trajectory.
/// Stops strictly when L1 change<tolerance, otherwise returns the budgeted iterate.
/// A nonempty0budget returns uniform scores without a convergence claim. No input
/// mutation, retained references, callbacks or shared mutable state. Scores are
/// computed doubles, not an exact sum-to-one or fixed-point accuracy certificate.
/// O(n+E) work per update and O(n+E) ordinary storage, excluding owned histories.
/// History admission precedes graph/input copies and each iteration/transfer
/// record. Default1000000 logical units,0 disables; failure returns no partial
/// trace. Output storage grows with actual updates/edge transfers, not only n.
/// @param vertex_count Number of original vertices, zero accepted.
/// @param directed_edges Complete directed edge occurrences u->v with endpoints<n.
/// @param options Finite numerical controls and optional trace quota; default.85/1e-12/1000/1000000.
/// @return Owned actual PageRankTrace.
/// @throws std::invalid_argument If complete endpoints or finite numerical controls fail.
/// @throws std::length_error If complete graph/count geometry or selected history quota fails.
/// @throws std::overflow_error If produced arithmetic becomes nonfinite.
/// @throws std::bad_alloc If owned storage cannot be allocated.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::page_rank_trace(2,{}).captured_entries); // 13
/// @endcode
/// @par References
/// - NetworkX upstream PageRank documentation, https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html. Native uniform-teleport/dangling model uses its documented strict L1 stop and explicit budget status.
/// @see PageRankRun, PageRankTrace
PageRankTrace page_rank_trace(
    std::size_t                                             vertex_count,
    const std::vector<std::pair<std::size_t, std::size_t>>& directed_edges,
    PageRankOptions                                         options = {});
/// @brief Return the original three-page report graph0->2,1->2,2->2.
/// @return Fresh owned edge occurrences in original order.
/// @throws std::bad_alloc If vector allocation fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::page_rank_demo_edges().size()); // 3
/// @endcode
/// @see page_rank_trace()
std::vector<std::pair<std::size_t, std::size_t>> page_rank_demo_edges();
} // namespace datamunge::algorithms
