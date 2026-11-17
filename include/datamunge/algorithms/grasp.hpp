#pragma once
/// @file grasp.hpp
/// @brief Validated weighted MAX-CUT GRASP with actual construction, flip and restart records.
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>
namespace datamunge::algorithms {
/// @brief Owned weighted adjacency rows, each entry a neighbor ID and finite weight.
/// Public calls validate outer size, endpoints, no loops and exact reciprocal
/// weight multiplicities. Signed/zero weights and parallel entries are permitted.
/// Original row/entry ordering is preserved and affects sequential sums.
/// @par Example
/// @code
/// dm::algorithms::GraspAdjacency a{{{1,2}},{{0,2}}};
/// std::print("{}\n",a.size()); // 2
/// @endcode
/// @see grasp_max_cut()
using GraspAdjacency = std::vector<std::vector<std::pair<int, double>>>;
/// @brief Legacy owned feasible partition and producing final cut value.
/// side contains zeros/ones by vertex; value sums u<v crossing adjacency entries
/// once, including parallel entries. This is a feasible value, not an optimum
/// certificate. Empty graphs return zero with an empty partition.
/// @par Example
/// @code
/// const auto r=dm::algorithms::grasp_max_cut(2,{{{1,2}},{{0,2}}},1);
/// std::print("{}\n",r.value); // 2
/// @endcode
/// @see grasp_max_cut()
struct GraspMaxCutResult {
  double           value{0}; ///< Actual final objective of the best completed restart.
  std::vector<int> side;     ///< Owned best partition labels, one per vertex.
};
/// @brief Complete input, local-pass and actual optional capture limits.
/// Basic shape/endpoint/finite/no-loop/parameter validation precedes input caps;
/// reciprocal-multiplicity validation allocates only after those input caps.
/// A local pass includes a full cyclic scan, including the terminal no-flip scan.
/// @par Example
/// @code
/// const dm::algorithms::GraspOptions o;
/// std::print("{}\n",o.max_local_passes); // 1000000
/// @endcode
/// @see GraspTrace, grasp_max_cut_run()
struct GraspOptions {
  std::size_t max_vertices{1000000}; ///< Complete vertex cap, zero disables it.
  std::size_t max_adjacency_entries{
      1000000}; ///< Complete directed entry cap, duplicates included; zero disables.
  std::size_t max_local_passes{1000000}; ///< Per-restart full-scan cap; zero disables it.
  std::size_t max_trace_entries{
      1000000}; ///< Actual logical capture entries; zero disables; ignored without capture.
};
/// @brief Legacy local-state RNG retained for the GRASP implementation and fixtures.
namespace detail {
/// @brief Owned legacy xorshift(13,7,17) state, not a cryptographic or unbiased sampler.
/// Zero seeds map to the historical nonzero constant; methods mutate only s.
/// @par Example
/// @code
/// dm::algorithms::detail::GraspRng r(0);
/// std::print("{}\n",r.s!=0); // true
/// @endcode
/// @see grasp_max_cut()
struct GraspRng {
  std::uint64_t s; ///< Current owned unsigned wraparound xorshift state.

  /// @brief Initialize legacy state, replacing zero by a fixed fallback.
  /// @param seed Requested state; zero becomes 0x9E3779B97F4A7C15.
  /// @par Example
  /// @code
  /// dm::algorithms::detail::GraspRng r(1);
  /// std::print("{}\n",r.s); // 1
  /// @endcode
  explicit GraspRng(std::uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
  /// @brief Advance one original unsigned xorshift word.
  /// @return New state after left13, right7, left17 XOR shifts.
  /// @par Example
  /// @code
  /// dm::algorithms::detail::GraspRng r(1);
  /// std::print("{}\n",r.next()); // 1082269761
  /// @endcode
  std::uint64_t next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
  }
  /// @brief Convert one new word's high 53 bits to a double in [0,1).
  /// @return Actual generated grid value; not used by GRASP selection.
  /// @par Example
  /// @code
  /// dm::algorithms::detail::GraspRng r(1);const double u=r.uniform();
  /// std::print("{}\n",u>=0&&u<1); // true
  /// @endcode
  double uniform() { return (next() >> 11) * (1. / 9007199254740992.); }
  /// @brief Select by one original modulo reduction, with its possible bias.
  /// @param n Positive modulus, checked before advancing state.
  /// @return next()%n, converted to size_t.
  /// @throws std::invalid_argument If n is zero, leaving state unchanged.
  /// @par Example
  /// @code
  /// dm::algorithms::detail::GraspRng r(1);
  /// std::print("{}\n",r.below(1)); // 0
  /// @endcode
  std::size_t below(std::size_t n) {
    if (!n)
      throw std::invalid_argument("GRASP empty modulo range");
    return static_cast<std::size_t>(next() % n);
  }
};
} // namespace detail
/// @brief Selected actual construction, local-search and objective work, excluding
/// validation/copies. Construction candidates are scored once per unplaced vertex per step.
/// Neighbor visits count every inspected adjacency entry, even if unassigned or
/// not crossing. RNG words count exactly one modulo draw per placement.
/// @par Example
/// @code
/// const auto r=dm::algorithms::grasp_max_cut_run(1,{{}},1);
/// std::print("{} {}\n",r.work.candidate_scores,r.work.rng_words); // 1 1
/// @endcode
/// @see GraspRun
struct GraspWork {
  std::size_t candidate_scores{0};       ///< Actual better-side score computations.
  std::size_t construction_neighbors{0}; ///< Actual neighbor entries inspected while scoring.
  std::size_t rcl_tests{0};              ///< Actual gain/cutoff membership tests.
  std::size_t rng_words{0};              ///< Actual xorshift words consumed by modulo selection.
  std::size_t local_passes{0};           ///< Actual full cyclic scans, terminal scan included.
  std::size_t flip_tests{0};             ///< Actual vertex delta computations.
  std::size_t local_neighbors{0};        ///< Actual neighbor entries inspected for deltas.
  std::size_t accepted_flips{0};         ///< Actual immediate flips with computed delta>1e-12.
  std::size_t objective_calls{0};     ///< Actual final cut evaluations, one per completed restart.
  std::size_t objective_neighbors{0}; ///< Actual entries inspected by those evaluations.
  std::size_t restarts{0};            ///< Completed construction/local-search/objective restarts.
};
/// @brief Owned ordinary best result and actual selected work without retained histories.
/// Copying owns the partition; no input references or callback state survive.
/// @par Example
/// @code
/// const auto r=dm::algorithms::grasp_max_cut_run(1,{{}},2);
/// std::print("{}\n",r.work.restarts); // 2
/// @endcode
/// @see grasp_max_cut_run()
struct GraspRun {
  GraspMaxCutResult result; ///< Actual best completed restart.
  GraspWork         work;   ///< Actual selected producing work.
};
/// @brief Actual scores and RCL membership for one unplaced candidate.
/// Scores concern assigned neighbors only; side zero wins exact g0/g1 ties.
/// @par Example
/// @code
/// const dm::algorithms::GraspCandidate c{};
/// std::print("{}\n",c.vertex); // 0
/// @endcode
/// @see GraspConstruction
struct GraspCandidate {
  int    vertex{0};     ///< Actual candidate ID, in ascending vertex scan order.
  int    side{0};       ///< Actual better side, zero on a score tie.
  double gain0{0};      ///< Actual sequential gain for placement on side zero.
  double gain1{0};      ///< Actual sequential gain for placement on side one.
  double gain{0};       ///< Actual selected better-side score.
  bool   in_rcl{false}; ///< Actual gain>=threshold-1e-12 membership result.
};
/// @brief One actual construction step, owning all candidate scores and chosen RNG state.
/// pick is an index into candidates, not a vertex ID or RCL rank. Quota admission
/// precedes scores, candidate snapshots and the one modulo RNG draw.
/// @par Example
/// @code
/// const auto t=dm::algorithms::grasp_max_cut_trace(1,{{}},1);
/// std::print("{}\n",t.construction[0].candidates.size()); // 1
/// @endcode
/// @see GraspTrace
struct GraspConstruction {
  int           restart{0};    ///< Zero-based restart index.
  int           step{0};       ///< Zero-based placement count before this step.
  double        best_gain{0};  ///< Actual maximum candidate gain.
  double        worst_gain{0}; ///< Actual minimum candidate gain.
  double        threshold{0};  ///< Actual best-alpha*(best-worst), before the 1e-12 slack.
  std::uint64_t rng_before{0}; ///< Actual xorshift state before selection.
  std::uint64_t rng_after{0};  ///< Actual state after the single selection word.
  std::size_t   pick{0};       ///< Chosen candidate index from the actual RCL.
  std::vector<GraspCandidate> candidates; ///< Owned actual candidate records, no replay.
};
/// @brief One actual vertex-flip test in a cyclic immediate-improvement scan.
/// before_side is the label read before possible mutation; delta is the sequential
/// signed incident-weight sum. Later tests in the same pass see earlier flips.
/// @par Example
/// @code
/// const auto t=dm::algorithms::grasp_max_cut_trace(1,{{}},1);
/// std::print("{}\n",t.flips[0].accepted); // false
/// @endcode
/// @see GraspTrace
struct GraspFlip {
  int         restart{0};      ///< Zero-based producing restart.
  std::size_t pass{0};         ///< One-based full-scan index within that restart.
  int         vertex{0};       ///< Actual tested vertex in ascending scan order.
  int         before_side{0};  ///< Actual label before possible flip.
  double      delta{0};        ///< Actual rounded computed gain from flipping.
  bool        accepted{false}; ///< Actual delta>1e-12 decision.
};
/// @brief Actual completed restart with owned pre/post-local partitions and best-so-far value.
/// constructed_side is captured before local search, final_side after its terminal
/// scan. No constructed objective is evaluated by the producing kernel.
/// @par Example
/// @code
/// const auto t=dm::algorithms::grasp_max_cut_trace(1,{{}},1);
/// std::print("{}\n",t.restarts[0].final_value); // 0
/// @endcode
/// @see GraspTrace
struct GraspRestart {
  int              restart{0};       ///< Zero-based restart index.
  std::vector<int> constructed_side; ///< Owned actual construction result.
  std::vector<int> final_side;       ///< Owned actual local-search result.
  double           final_value{0};   ///< Actual final cut evaluation.
  double           best_value{0};    ///< Actual retained best after this restart.
  bool improved_best{false}; ///< Strict actual final-value improvement over the previous best.
};
/// @brief Owned actual construction, flip and completed-restart histories from one execution.
/// Logical capture units: 1 per construction step plus 1 per candidate record;
/// 1 per flip test; 1 per restart plus 2n retained partition labels. A restart's
/// whole 1+2n requirement is admitted before its construction; step and flip
/// requirements are admitted online before their producing score/delta work.
/// Excludes record fields, allocation overhead/capacities, validation and ordinary
/// result/work. This is not a byte/peak-RAM bound. No replay or partial result.
/// Histories own storage; ordinary/run calls allocate no optional histories.
/// @par Example
/// @code
/// const auto t=dm::algorithms::grasp_max_cut_trace(1,{{}},1);
/// std::print("{}\n",t.captured_entries); // 6
/// @endcode
/// @see GraspOptions, grasp_max_cut_trace()
struct GraspTrace {
  GraspRun                       run;          ///< Actual ordinary output and selected work.
  std::vector<GraspConstruction> construction; ///< Actual chronological scored construction steps.
  std::vector<GraspFlip>
      flips; ///< Actual chronological local tests, including rejected/terminal ones.
  std::vector<GraspRestart>
              restarts;            ///< Actual completed restart partitions and prefix best values.
  std::size_t captured_entries{0}; ///< Admitted logical record/partition-label units.
};
/// @brief Apply weighted MAX-CUT GRASP with the preserved xorshift/modulo construction and cyclic
/// flips. Requires finite symmetric loopless weighted adjacency, preserving entry order.
/// Parallel/signed/zero weights are supported; reciprocal multiplicities must
/// match exactly by numeric weight (signed zeros compare equal). Alpha is finite
/// in [0,1]; iterations is positive. Alpha zero still randomizes ties and gains
/// within 1e-12 of the best; alpha one includes all candidates only in the ideal
/// threshold model, subject to floating rounding. Modulo selection is biased in
/// general. One continuing RNG stream spans restarts; seed zero uses a fixed
/// nonzero fallback. Local scans test vertices 0..n-1 and immediately flip any
/// computed delta>1e-12 until a no-flip scan. This is tolerance-local, not a global
/// certificate. A finite partition space alone does not prove floating termination;
/// the legacy call uses a one-million-pass per-restart cap. No input/capture caps
/// or retained histories. Nonfinite score/range/delta/objective arithmetic throws.
/// Ordinary finite valid arithmetic and strict best-restart tie behavior remain;
/// initialization now uses infinities instead of arbitrary 1e300 sentinels.
/// O(T(n²+nA+LA)) selected loop work for T restarts, A adjacency entries and at
/// most L local passes per restart, plus O(A log A+n) reciprocal validation;
/// O(n+A) working storage. Independent calls share no mutable state.
/// @param n Nonnegative vertex count, equal to adj.size().
/// @param adj Borrowed finite symmetric loopless adjacency, unmodified.
/// @param iterations Positive restart budget, default 50.
/// @param alpha Finite RCL fraction in [0,1], default 0.3.
/// @param seed Xorshift state, default 1; zero maps to a fixed fallback.
/// @return Owned actual best completed feasible partition, not an optimum claim.
/// @throws std::invalid_argument On shape/endpoints/loops/weights/symmetry/parameters.
/// @throws std::length_error On local-pass cap, checked work or container exhaustion.
/// @throws std::overflow_error On nonfinite intermediate arithmetic.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto r=dm::algorithms::grasp_max_cut(2,{{{1,2}},{{0,2}}},1,.3,1);
/// std::print("{}\n",r.value); // 2
/// @endcode
/// @par References
/// - T. A. Feo, M. G. C. Resende (1995), Greedy Randomized Adaptive Search
///   Procedures, https://doi.org/10.1007/BF01096763 . This MAX-CUT variant uses
///   immediate cyclic flips, a 1e-12 decision slack and a continuing RNG stream.
/// @see grasp_max_cut_run(), grasp_max_cut_trace()
GraspMaxCutResult grasp_max_cut(
    int                   n,
    const GraspAdjacency& adj,
    int                   iterations = 50,
    double                alpha      = .3,
    std::uint64_t         seed       = 1);
/// @brief Execute the same producing kernel with actual work and explicit resource options.
/// Same numeric/ordering/RNG semantics as grasp_max_cut(). Basic validation
/// precedes complete input caps; reciprocal checking follows caps before search.
/// Trace cap is ignored. Zero disables any selected policy cap, including the
/// local-pass guard, so disabling it carries no floating termination guarantee.
/// @param n Nonnegative vertex count matching adjacency rows.
/// @param adj Borrowed finite symmetric loopless adjacency.
/// @param iterations Positive restart count, default 50.
/// @param alpha Finite RCL fraction in [0,1], default .3.
/// @param seed Continuing xorshift seed, default 1.
/// @param options Complete input/per-restart scan limits, defaults one million each.
/// @return Owned actual best output and selected work, without histories.
/// @throws std::invalid_argument On malformed graph or arithmetic parameters.
/// @throws std::length_error On limits, counters or containers.
/// @throws std::overflow_error On nonfinite arithmetic.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto r=dm::algorithms::grasp_max_cut_run(1,{{}},2);
/// std::print("{}\n",r.work.objective_calls); // 2
/// @endcode
/// @see GraspRun, GraspOptions
GraspRun grasp_max_cut_run(
    int                   n,
    const GraspAdjacency& adj,
    int                   iterations = 50,
    double                alpha      = .3,
    std::uint64_t         seed       = 1,
    GraspOptions          options    = {});
/// @brief Capture actual candidates/RCL/RNG, immediate flips and completed restart prefixes.
/// Same result and producing arithmetic work as run, with the capture units and
/// admission points documented by GraspTrace. Inputs are borrowed only during
/// the call. Failures return no partial best/history, even after completed earlier
/// restarts; already executed floating flags/errno are not rolled back.
/// @param n Nonnegative vertex count matching adjacency rows.
/// @param adj Borrowed valid weighted graph, unmodified.
/// @param iterations Positive restart count, default 50.
/// @param alpha Finite RCL fraction in [0,1], default .3.
/// @param seed Continuing xorshift seed, default 1.
/// @param options Complete input, local-pass and actual capture caps.
/// @return Owned producing result, actual states and prefix best values without reruns.
/// @throws std::invalid_argument On malformed graph or parameters.
/// @throws std::length_error On input/pass/capture/work/container limits.
/// @throws std::overflow_error On nonfinite arithmetic.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::grasp_max_cut_trace(1,{{}},2);
/// std::print("{}\n",t.restarts.size()); // 2
/// @endcode
/// @see GraspTrace, grasp_max_cut_run()
GraspTrace grasp_max_cut_trace(
    int                   n,
    const GraspAdjacency& adj,
    int                   iterations = 50,
    double                alpha      = .3,
    std::uint64_t         seed       = 1,
    GraspOptions          options    = {});
} // namespace datamunge::algorithms
