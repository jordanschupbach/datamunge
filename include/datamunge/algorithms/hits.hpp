#pragma once
/// @file hits.hpp
/// @brief Checked HITS power iteration with actual alternating-update histories.
#include <cstddef>
#include <utility>
#include <vector>
namespace datamunge::algorithms {
/// @brief Directed zero-based endorsement u->v, preserving loops, order and duplicate edges.
/// Repeated pairs contribute repeatedly, equivalent to integer adjacency weights.
/// The alias owns its two indices; validation belongs to the called algorithm.
/// @par Example
/// @code
/// const dm::algorithms::HitsEdge e{0,2};std::print("{} {}\n",e.first,e.second); // 0 2
/// @endcode
/// @see hits()
using HitsEdge = std::pair<std::size_t, std::size_t>;
/// @brief Owned historical hub/authority vectors and actual completed iteration count.
/// Ordinary nonzero updated vectors have approximately unit Euclidean norm;
/// edgeless updates are zero, and zero budget returns uniform initialization.
/// Nonempty initialization uses1/sqrt(n). No input references are retained.
/// @par Example
/// @code
/// const auto r=dm::algorithms::hits(2,{{0,1}});
/// std::print("{} {}\n",r.hubs[0],r.authorities[1]); // 1 1
/// @endcode
/// @see hits_run(), HitsTrace
struct HITSResult {
  std::vector<double> hubs;          ///< Owned actual hub scores in vertex order.
  std::vector<double> authorities;   ///< Owned actual authority scores in vertex order.
  std::size_t         iterations{0}; ///< Actual complete authority/hub update pairs applied.
};
/// @brief Distinguish a strict max-component change test from exhausted update budget.
/// Tolerance is not a spectral-residual or uniqueness certificate.
/// @par Example
/// @code
/// std::print("{}\n",int(dm::algorithms::HitsStop::Tolerance)); // 0
/// @endcode
/// @see HitsRun
enum class HitsStop {
  Tolerance,     ///< Last paired update had strict max-component change below tolerance.
  IterationLimit ///< Update budget exhausted; no claim change test passed.
};
/// @brief Complete input and online whole-iteration capture caps, each zero disabling policy.
/// Capture counts2n initial score values, then1+4n per complete iteration: one
/// record and raw/normalized authority/hub vectors. Scalar fields, ordinary
/// output, capacity and allocation overhead excluded; not bytes/peak memory.
/// @par Example
/// @code
/// const dm::algorithms::HitsOptions o;
/// std::print("{}\n",o.max_trace_entries); // 1000000
/// @endcode
/// @see hits_trace()
struct HitsOptions {
  std::size_t max_vertices{1000000}; ///< Complete vertex cap, default1m; zero disables.
  std::size_t max_edges{1000000};    ///< Complete edge-entry cap, default1m; zero disables.
  std::size_t max_trace_entries{
      1000000}; ///< Online record/score-value cap; ignored without capture.
};
/// @brief Selected actual update visits, excluding validation, initialization and capture copies.
/// Duplicate edges count repeatedly. Norm visits are squared-value sum terms;
/// divisions occur only for nonzero norms. Change visits compare both score
/// vectors at each vertex. Counts are not timing, FLOPs or total memory traffic.
/// @par Example
/// @code
/// const auto r=dm::algorithms::hits_run(2,{{0,1}},1e-12,1);
/// std::print("{} {}\n",r.work.authority_edges,r.work.hub_edges); // 1 1
/// @endcode
/// @see HitsRun
struct HitsWork {
  std::size_t authority_edges{0}; ///< Actual edge additions into raw authority scores.
  std::size_t hub_edges{0};       ///< Actual edge additions into raw hub scores.
  std::size_t norm_values{0};     ///< Actual values squared and summed in both normalizations.
  std::size_t divisions{0};       ///< Actual nonzero-norm score divisions.
  std::size_t change_vertices{0}; ///< Actual vertices visited by the joint change test.
};
/// @brief Owned actual HITS result, stop branch and selected producing work.
/// last_change is meaningful only when has_step; otherwise it is zero sentinel.
/// A small update change does not certify a unique principal direction, graph
/// relevance, convergence rate or absence of competing dominant components.
/// @par Example
/// @code
/// const auto r=dm::algorithms::hits_run(2,{{0,1}},1e-12,0);
/// std::print("{} {}\n",r.has_step,r.result.iterations); // false 0
/// @endcode
/// @see hits_run()
struct HitsRun {
  HITSResult result; ///< Owned actual summary, same three historical fields.
  HitsStop   stop{HitsStop::IterationLimit}; ///< Actual terminal branch.
  HitsWork   work;                           ///< Selected producing loop visits.
  double last_change{0};  ///< Actual final joint max absolute component change, or zero sentinel.
  bool   has_step{false}; ///< Whether any paired update was actually executed.
};
/// @brief Actual sequential authority-then-hub update captured during its producing pass.
/// Hub raw sums use this iteration's normalized authorities. Norms are actual
/// sequential sum-of-squares roots, not analytic substitutions or rescaling
/// certificates. All vectors own storage and copy independently.
/// @par Example
/// @code
/// const auto t=dm::algorithms::hits_trace(2,{{0,1}},1e-12,1);
/// std::print("{}\n",t.steps[0].raw_authorities.size()); // 2
/// @endcode
/// @see HitsTrace
struct HitsStep {
  std::size_t iteration{0}; ///< One-based actual paired update.
  std::vector<double>
      raw_authorities;             ///< Actual edge sums from preceding hubs, before normalization.
  std::vector<double> authorities; ///< Actual normalized authority vector, possibly zero.
  std::vector<double> raw_hubs;    ///< Actual edge sums using current normalized authorities.
  std::vector<double> hubs;        ///< Actual normalized hub vector, possibly zero.
  double              authority_norm{0}; ///< Actual raw authority Euclidean norm used.
  double              hub_norm{0};       ///< Actual raw hub Euclidean norm used.
  double              change{0};         ///< Actual joint max-component change used for stopping.
};
/// @brief Owned actual initialization, complete update records and producing result.
/// Initial2n values admitted before output allocation; each known1+4n update
/// record admitted before that iteration's allocations/edge work. No truncation,
/// replay, partial result or retained references. Capture adds O(Kn) score
/// values beyond O(n) ordinary working/output storage; fields/capacity excluded.
/// @par Example
/// @code
/// const auto t=dm::algorithms::hits_trace(2,{{0,1}},1e-12,1);
/// std::print("{}\n",t.captured_entries); // 13
/// @endcode
/// @see HitsOptions, hits_trace()
struct HitsTrace {
  HitsRun               run;                 ///< Actual producing result/stop/work.
  std::vector<double>   initial_hubs;        ///< Actual uniform initialization, empty for n0.
  std::vector<double>   initial_authorities; ///< Actual matching uniform initialization.
  std::vector<HitsStep> steps;               ///< Actual chronological complete paired updates.
  std::size_t captured_entries{0}; ///< Retained logical record/score-value units, not bytes.
};
/// @brief Apply preserved HITS authority-then-hub power updates with a strict change test.
/// Uses each edge in input order, counting loops/duplicates as endorsements.
/// Starts with both vectors1/sqrt(n), or empty for n0. Sequential sum-of-squares
/// normalization divides only nonzero norms, otherwise retaining zeros. Stops
/// when the largest absolute change in either vector is strictly below tolerance.
/// Zero budget returns initialization without a test. With default small tolerance,
/// a nonempty edgeless graph needs two paired zero updates; n0 takes one empty
/// paired update when budget permits. Unit norms are not promised in these cases.
/// No optional input/history caps; checked counts/container sizes still apply.
/// O(n+K(n+m)) time/O(n) working/output storage, excluding borrowed input edges;
/// no retained references or shared mutable state. Finite positive tolerance is
/// required; an iteration cap or tiny change is not a uniqueness/residual proof.
/// @param vertex_count Number n of vertices, zero allowed.
/// @param directed_edges Borrowed zero-based u->v pairs, loops/repeated entries permitted.
/// @param tolerance Finite strictly positive joint component tolerance, default1e-12.
/// @param max_iterations Nonnegative complete paired-update cap, default1000; zero allowed.
/// @return Owned historical scores and actual iterations, preserving ordinary legacy output.
/// @throws std::invalid_argument On any invalid endpoint or tolerance before caps/allocation.
/// @throws std::length_error On count/container limits.
/// @throws std::overflow_error On nonfinite update/norm arithmetic.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::hits(2,{{0,1}}).iterations); // 2
/// @endcode
/// @par References
/// - J. M. Kleinberg (1999), Authoritative Sources in a Hyperlinked Environment,
///   https://doi.org/10.1145/324133.324140 . Duplicate/self-edge counting is an
///   implementation input convention; no query-dependent base-set crawl is performed.
/// @see hits_run(), hits_trace()
HITSResult hits(
    std::size_t                  vertex_count,
    const std::vector<HitsEdge>& directed_edges,
    double                       tolerance      = 1e-12,
    std::size_t                  max_iterations = 1000);
/// @brief Execute the shared HITS kernel with explicit stop/work and complete input policies.
/// Same arithmetic, initialization, tie-free numeric order and domains as hits().
/// Complete graph validation precedes optional input caps and vector allocation.
/// Capture cap ignored without tracing. No extra producing score sweeps.
/// @param vertex_count Complete vertex count, zero allowed.
/// @param directed_edges Borrowed ordered directed endorsements.
/// @param tolerance Finite positive joint change tolerance, default1e-12.
/// @param max_iterations Complete paired-update budget, default1000; zero allowed.
/// @param options Complete vertex/edge caps, default1m each; zero disables either policy.
/// @return Owned actual scores, stop branch and selected work.
/// @throws std::invalid_argument On endpoints/tolerance before policy caps.
/// @throws std::length_error On graph/count/container caps.
/// @throws std::overflow_error On nonfinite producing arithmetic.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto r=dm::algorithms::hits_run(3,{});
/// std::print("{} {}\n",r.result.iterations,r.result.hubs[0]); // 2 0
/// @endcode
/// @see HitsRun, HitsOptions
HitsRun hits_run(
    std::size_t                  vertex_count,
    const std::vector<HitsEdge>& directed_edges,
    double                       tolerance      = 1e-12,
    std::size_t                  max_iterations = 1000,
    HitsOptions                  options        = {});
/// @brief Capture actual raw/normalized paired updates from the producing execution without replay.
/// Same contracts as hits_run(). Online whole iteration capture admits1+4n units
/// before allocation or work; input and initial2n admission precede vectors.
/// Failure returns no partial trace and never mutates input. Histories own all
/// score vectors and add memory beyond the ordinary O(n) working/output bound.
/// @param vertex_count Complete vertex count, zero allowed.
/// @param directed_edges Borrowed ordered directed endorsements.
/// @param tolerance Finite positive joint change tolerance, default1e-12.
/// @param max_iterations Complete paired-update budget, default1000; zero allowed.
/// @param options Vertex/edge/online capture caps, default1m each; zero disables each policy.
/// @return Owned complete actual update histories and producing run.
/// @throws std::invalid_argument On endpoints/tolerance before policy caps.
/// @throws std::length_error On complete input/initial or online iteration/count/container limits.
/// @throws std::overflow_error On nonfinite update/norm arithmetic.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::hits_trace(2,{{0,1}},1e-12,1,{2,1,13});
/// std::print("{}\n",t.steps.size()); // 1
/// @endcode
/// @see HitsTrace, HitsStep
HitsTrace hits_trace(
    std::size_t                  vertex_count,
    const std::vector<HitsEdge>& directed_edges,
    double                       tolerance      = 1e-12,
    std::size_t                  max_iterations = 1000,
    HitsOptions                  options        = {});
} // namespace datamunge::algorithms
