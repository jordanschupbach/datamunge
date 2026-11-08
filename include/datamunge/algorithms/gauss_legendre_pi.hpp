#pragma once
/// @file gauss_legendre_pi.hpp
/// @brief Actual fixed-precision AGM pi states, work, bounded histories and reference diagnostics.
#include <cstddef>
#include <vector>

namespace datamunge::algorithms {
/// @brief Actual arithmetic-geometric means, correction accumulator and doubling weight.
/// Default values are inert storage, not the initialized pi system. Produced
/// states start at a1,b1/sqrt2,t1/4,p1. At the maximal supported iteration the
/// final unused p overflows to infinity; no subsequent stage may use it.
/// @par Example
/// @code
/// const auto s = dm::algorithms::gauss_legendre_pi_run(0).initial;
/// std::cout << s.a << ' ' << s.t << ' ' << s.p; // 1 0.25 1
/// @endcode
/// @see GaussLegendreRun
struct GaussLegendreState {
    double a{}; ///< Actual arithmetic-mean state.
    double b{}; ///< Actual geometric-mean state, not forced equal to a.
    double t{}; ///< Actual rounded correction accumulator.
    double p{}; ///< Actual doubling weight; final unused overflow is explicitly allowed.
};

/// @brief Actual arithmetic counts, excluding diagnostics, allocation and snapshots.
/// Initial sqrt is included in square_roots. Each stage executes both means,
/// left-associated p*delta*delta subtraction and p*=2. Legacy scalar calls
/// evaluate only the final estimate; run/trace evaluate the initial estimate
/// and every stage to observe changes. One estimate is one quotient expression,
/// not one floating operation. No early stop is triggered by stagnation.
/// @par Example
/// @code
/// const auto w = dm::algorithms::gauss_legendre_pi_run(3).work;
/// std::cout << w.square_roots << ' ' << w.estimate_evaluations; // 4 4
/// @endcode
/// @see GaussLegendreRun
struct GaussLegendreWork {
    int square_roots{};              ///< One initial sqrt plus one per completed stage.
    int arithmetic_means{};          ///< Actual 0.5*(a+b) evaluations.
    int geometric_means{};           ///< Actual sqrt(a*b) stage evaluations.
    int correction_updates{};        ///< Actual t-=p*delta*delta assignments, including rounded no-ops.
    int power_updates{};             ///< Actual p*=2 assignments, including final unused overflow.
    int estimate_evaluations{};      ///< Actual (a+b)*(a+b)/(4*t) expressions.
};

/// @brief Owned fixed-count analysis result and actual initial/final states.
/// Executes exactly iterations stages, without replay or early stopping. Flags
/// record first equality of successive computed estimates and of (a,b,t), not
/// exact convergence or correct rounding; p is deliberately excluded from that
/// mean-state comparison. a and b can stall one ULP apart. Final unused p may
/// be infinite at the maximum iteration, while the estimate remains finite.
/// O(iterations) work/O(1) result storage; no shared mutable state.
/// @par Example
/// @code
/// const auto s = dm::algorithms::gauss_legendre_pi_run(0);
/// std::cout << s.iterations << ' ' << s.work.square_roots; // 0 1
/// @endcode
/// @see gauss_legendre_pi_run(), GaussLegendreTrace
struct GaussLegendreRun {
    double value{};                  ///< Actual final rounded estimate, not certified pi.
    int iterations{};                ///< Exact executed stage count.
    GaussLegendreState initial;      ///< Owned initialized state, counted once in trace admission.
    GaussLegendreState final;        ///< Owned final state; independent of every snapshot.
    GaussLegendreWork work;          ///< Actual executed arithmetic counts.
    int first_unchanged_estimate{-1}; ///< First one-based stage equal to prior estimate, or -1.
    int first_unchanged_mean_state{-1}; ///< First unchanged a/b/t stage, or -1; excludes p.
};

/// @brief Owned actual stage, including the used correction and before/after states.
/// correction is the actual left-associated (p*delta)*delta used in t-=correction;
/// it need not equal the subsequently rounded difference t_before-t_after.
/// estimate_changed compares rounded estimates; mean_state_changed compares
/// actual a,b,t, excluding p. Neither flag certifies mathematical accuracy.
/// @par Example
/// @code
/// const auto t = dm::algorithms::gauss_legendre_pi_trace(1);
/// std::cout << t.steps[0].iteration << ' ' << t.steps[0].estimate_changed; // 1 1
/// @endcode
/// @see GaussLegendreTrace
struct GaussLegendreStep {
    int iteration{};                 ///< One-based executed stage number.
    GaussLegendreState before;       ///< Actual state immediately before this stage.
    GaussLegendreState after;        ///< Actual state after all assignments, including p doubling.
    double correction{};            ///< Actual used left-associated correction expression.
    double estimate{};              ///< Actual rounded estimate from after-state means/t.
    bool estimate_changed{};        ///< Inequality of successive computed estimates.
    bool mean_state_changed{};      ///< Inequality of at least one of a,b,t.
};

/// @brief Owned actual fixed-count history and final run, admitted as complete records.
/// captured_records counts one initial-state record plus each fixed-schema stage
/// record, not scalar entries or bytes. Ordinary final output, root/work metadata,
/// working state and allocator bytes are excluded; this is not a peak RAM limit.
/// All state copies are independent. O(N) history/copy cost, no numerical replay.
/// @par Example
/// @code
/// const auto t = dm::algorithms::gauss_legendre_pi_trace(3);
/// std::cout << t.steps.size() << ' ' << t.captured_records; // 3 4
/// @endcode
/// @see gauss_legendre_pi_trace_records()
struct GaussLegendreTrace {
    GaussLegendreRun run;            ///< Actual analysis output from the same execution.
    std::vector<GaussLegendreStep> steps; ///< Owned actual-time records; empty for zero stages.
    std::size_t captured_records{};  ///< One initial record plus the actual stage count.
};

/// @brief Largest fixed stage count whose doubling weight is never used after overflow.
/// This model requires radix-two double at build time. The limit is its maximum
/// exponent (1024 for binary64). The last unused p*=2 can overflow, but every
/// used weight and mean/correction/estimate remains finite in this model.
/// @return Positive format-dependent stage limit.
/// @par Example
/// @code
/// std::cout << (dm::algorithms::gauss_legendre_pi_limit() > 0); // 1
/// @endcode
/// @see gauss_legendre_pi_run(), gauss_legendre_pi()
int gauss_legendre_pi_limit();

/// @brief Whole-history record admission for a checked fixed stage count.
/// Counts one initial-state record plus N fixed-schema stage records; zero stages
/// still require one record. No early-stop discount, scalar-entry or byte claim.
/// Checks the complete count domain and vector capacity without allocating.
/// @param iterations Stage count in [0,gauss_legendre_pi_limit()].
/// @return N+1 complete records.
/// @throws std::invalid_argument Negative count or a weight-overflowing suffix requested.
/// @throws std::length_error Stage vector capacity exceeded.
/// @par Example
/// @code
/// std::cout << dm::algorithms::gauss_legendre_pi_trace_records(3); // 4
/// @endcode
/// @see gauss_legendre_pi_trace()
std::size_t gauss_legendre_pi_trace_records(int iterations);

/// @brief Execute exactly N ordered double AGM stages and inspect actual changes.
/// Initialized at (1,1/sqrt2,1/4,1); preserves original expression order, including
/// left-associated p*delta*delta, rather than widening or using pow. No early
/// stop or a==b forcing. Initial and per-stage estimates are computed to measure
/// stagnation, unlike the legacy scalar API's single final estimate. Negative or
/// excessive counts are rejected before work. Zero gives the initialized estimate.
/// The final unused p can overflow at the format limit; a future use is forbidden.
/// An unchanged estimate is not correctly rounded pi or an error certificate.
/// @param iterations Stage count in [0,gauss_legendre_pi_limit()].
/// @return Owned states, estimate, actual counters and change indices.
/// @throws std::invalid_argument Count lies outside the supported interval.
/// @throws std::overflow_error A used weight, correction, mean, t or estimate is nonfinite.
/// @par Example
/// @code
/// const auto s = dm::algorithms::gauss_legendre_pi_run(2);
/// std::cout << s.iterations << ' ' << s.work.correction_updates; // 2 2
/// @endcode
/// @par References
/// - R. P. Brent (1976), Fast Multiple-Precision Evaluation of Elementary Functions,
///   J. ACM23(2),242–251, Section5, doi:10.1145/321941.321944,
///   https://maths-people.anu.edu.au/~brent/pd/rpb034.pdf . The paper uses precision
///   stopping/variable precision; this model uses a fixed count and double.
/// @see gauss_legendre_pi(), gauss_legendre_pi_trace()
GaussLegendreRun gauss_legendre_pi_run(int iterations);

/// @brief Capture actual AGM stages after complete count validation and whole admission.
/// Admission precedes initialization and history allocation. Zero max_records
/// disables only the optional cap; count-domain/container checks remain. Units
/// are one complete initial record plus N fixed-schema stage records, not scalar
/// entries/bytes or peak RAM. No replay, truncation, partial output or mutation.
/// The final unused p overflow at the maximum count is retained transparently;
/// it is never used by another stage. Executes all requested admitted stages.
/// @param iterations Stage count in [0,gauss_legendre_pi_limit()].
/// @param max_records Complete-record quota, default1000000; zero disables optional quota.
/// @return Owned initial state, actual records and final analysis output.
/// @throws std::invalid_argument Count is outside the supported interval.
/// @throws std::length_error Whole quota or vector capacity exceeded.
/// @throws std::overflow_error A used arithmetic state or estimate is nonfinite.
/// @throws std::bad_alloc History allocation fails.
/// @par Example
/// @code
/// const auto t = dm::algorithms::gauss_legendre_pi_trace(0,1);
/// std::cout << t.steps.size() << ' ' << t.captured_records; // 0 1
/// @endcode
/// @par References
/// - Brent (1976), Section5: https://maths-people.anu.edu.au/~brent/pd/rpb034.pdf .
/// @see gauss_legendre_pi_run(), gauss_legendre_pi_trace_records()
GaussLegendreTrace gauss_legendre_pi_trace(int iterations, std::size_t max_records = 1000000);

/// @brief Computed comparison with rounded double and long-double pi references.
/// These are reference differences, not certified errors against mathematical pi.
/// rounded_ulp_offset uses spacing at the double reference, not a bit-distance
/// metric for distant candidates. Wider precision is platform-dependent. Log
/// fields have zero placeholders when the corresponding difference is zero,
/// with explicit has-log flags. -log10(error) is an accuracy measure, not a
/// certified count of correct decimal digits. No shared state or allocation.
/// @par Example
/// @code
/// const auto s = dm::algorithms::pi_reference_accuracy(std::numbers::pi);
/// std::cout << s.rounded_absolute_error << ' ' << s.has_rounded_log_error; // 0 0
/// @endcode
/// @see pi_reference_accuracy()
struct PiReferenceAccuracy {
    double rounded_reference{};      ///< std::numbers::pi, itself rounded to double.
    long double wider_reference{};   ///< std::numbers::pi_v<long double>, also rounded.
    double rounded_absolute_error{}; ///< Actual abs(value-rounded double pi).
    long double wider_absolute_error{}; ///< Actual widened reference difference.
    long double rounded_ulp_offset{}; ///< Signed difference in units of spacing at rounded double pi.
    double rounded_log10_error{};    ///< log10 rounded difference when has_rounded_log_error.
    long double wider_log10_error{}; ///< log10 wider difference when has_wider_log_error.
    bool has_rounded_log_error{};    ///< Rounded difference is positive; zero is not plotted as -infinity.
    bool has_wider_log_error{};      ///< Wider difference is positive.
    bool wider_has_more_bits{};      ///< long double has more significand bits than double.
};

/// @brief Compare any finite candidate with both rounded pi references, without clipping errors.
/// Retains the original double reference subtraction and separately computes a
/// checked widened difference and local-reference ULP offset. No solver is called.
/// Zero differences get explicit unavailable-log flags, not a fabricated floor.
/// A wider reference can itself round and may add no bits on some platforms.
/// @param value Finite candidate approximation; negative values are allowed.
/// @return Reference differences, log-availability flags and precision metadata.
/// @throws std::invalid_argument Candidate is nonfinite.
/// @throws std::overflow_error Reference comparison or local ULP normalization is nonfinite.
/// @par Example
/// @code
/// const auto s = dm::algorithms::pi_reference_accuracy(std::numbers::pi);
/// std::cout << s.rounded_ulp_offset; // 0
/// @endcode
/// @par References
/// - C++ mathematical constants, https://eel.is/c++draft/numbers (pi is rounded
///   in the selected floating type, not an exact real value).
/// @see gauss_legendre_pi_run(), PiReferenceAccuracy
PiReferenceAccuracy pi_reference_accuracy(double value);
} // namespace datamunge::algorithms
