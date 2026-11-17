#pragma once
/// @file ruzzo_tompa.hpp
/// @brief Actual Ruzzo-Tompa candidate integration with explicitly counted direct predecessor scans.
#include <cstddef>
#include <vector>
/// @brief Canonical sequence scoring and owned integration records.
namespace datamunge::algorithms {
/// @brief One positive maximal-scoring half-open interval in caller coordinates.
/// @par Example
/// @code
/// const dm::algorithms::ScoringSegment value{};
/// std::print("{}\n",value.score); // 0
/// @endcode
struct ScoringSegment {
  std::size_t begin{}; ///< First input index, inclusive.
  std::size_t end{};   ///< First input index after the segment.
  double      score{}; ///< Positive finite represented prefix-score difference.
};
/// @brief Complete sample, actual predecessor scan and retained-list policies; zero disables a cap.
/// @par Example
/// @code
/// const dm::algorithms::RuzzoOptions value{};
/// std::print("{}\n",value.max_comparisons); // 10000000
/// @endcode
struct RuzzoOptions {
  std::size_t max_values{1000000}; ///< Complete score count.
  std::size_t max_comparisons{
      10000000}; ///< Actual predecessor left-prefix and right-prefix comparisons.
  std::size_t max_trace_entries{
      1000000}; ///< Actual current-list scalars plus one header per input event.
};
/// @brief One actual input event after its positive seed has been integrated.
/// @par Example
/// @code
/// const dm::algorithms::RuzzoStep value{};
/// std::print("{}\n",value.comparisons); // 0
/// @endcode
struct RuzzoStep {
  std::size_t input_index{};   ///< Actual processed input position.
  long double prefix_before{}; ///< Actual widened prefix before the value.
  long double prefix_after{};  ///< Actual widened prefix after the value.
  bool        seeded{};        ///< Whether this actual positive value created a candidate.
  std::size_t comparisons{};   ///< Actual predecessor/endpoint comparisons during this event.
  std::size_t absorbed{}; ///< Actual prior list entries removed by all merges during this event.
  std::vector<ScoringSegment>
      candidates; ///< Actual post-event candidate list, not final maximal segments yet.
};
/// @brief Owned actual final positive intervals and measured list-integration work.
/// @par Example
/// @code
/// const dm::algorithms::RuzzoRun value{};
/// std::print("{}\n",value.segments.size()); // 0
/// @endcode
struct RuzzoRun {
  std::vector<ScoringSegment> segments; ///< Actual final left-to-right positive segments.
  std::size_t values{};         ///< Actual input values visited, including nonpositive values.
  std::size_t positive_seeds{}; ///< Actual positive singleton candidates created.
  std::size_t
      comparisons{}; ///< Actual left-prefix tests plus successful-predecessor right-prefix tests.
  std::size_t            absorbed{}; ///< Actual existing candidate entries removed during merges.
  std::size_t            peak_candidates{}; ///< Largest actual post-event list length.
  std::vector<RuzzoStep> steps;             ///< Actual optional input-event candidate histories.
};
/// @brief Integrate positive scoring intervals through the native direct predecessor scan.
/// Preserves the original candidate-list algorithm: maintain widened cumulative scores; each strictly positive value seeds a candidate. Search existing candidates from right to left for the first smaller left prefix; merge it and all following entries only if its right prefix is also strictly smaller. Emit the final surviving list. Zero/negative values update the actual prefix but do not seed. Maximality uses strict improvement over every proper subinterval, so zero-score extensions are not added. This implementation deliberately measures a direct predecessor scan: worst-case O(N^2), unlike the linear-time linked algorithm in the original paper. Candidate removals themselves total at most N. Optional complete-list histories can be quadratic too. Finite inputs required; widened prefix arithmetic is not an exact-real certificate, and nonzero increments that vanish at its precision reject. Empty/all-nonpositive inputs yield no segments. Borrowed input unchanged, output owned; no callbacks/shared state.
/// @param scores Complete finite position scores.
/// @param options Input, actual-comparison and retained-list policies.
/// @return Actual final intervals, measured scan work and optional same-pass list histories.
/// @throws std::invalid_argument Nonfinite score.
/// @throws std::length_error Input/comparison/history/container capacity exceeded.
/// @throws std::overflow_error Nonfinite prefix or unrepresentable positive output score.
/// @throws std::underflow_error Nonzero score increment vanishes in widened prefix arithmetic.
/// @par Example
/// @code
/// auto r=dm::algorithms::ruzzo_tompa_run({3,-2,5});
/// std::print("{} {}\n",r.segments.size(),r.segments[0].score); // 1 6
/// @endcode
/// @par References
/// - Ruzzo and Tompa (1999), A Linear Time Algorithm for Finding All Maximal Scoring Subsequences, ISMB pp.234–241, https://homes.cs.washington.edu/~ruzzo/papers/maxseq.html. This implementation uses a direct predecessor scan rather than the paper's linear-time links.
/// @see ruzzo_tompa()
RuzzoRun ruzzo_tompa_run(const std::vector<double>& scores, const RuzzoOptions& options = {});
/// @brief Integrate positive scoring intervals with actual candidate histories.
/// Preserves the original candidate-list algorithm: maintain widened cumulative scores; each strictly positive value seeds a candidate. Search existing candidates from right to left for the first smaller left prefix; merge it and all following entries only if its right prefix is also strictly smaller. Emit the final surviving list. Zero/negative values update the actual prefix but do not seed. Maximality uses strict improvement over every proper subinterval, so zero-score extensions are not added. This implementation deliberately measures a direct predecessor scan: worst-case O(N^2), unlike the linear-time linked algorithm in the original paper. Candidate removals themselves total at most N. Optional complete-list histories can be quadratic too. Finite inputs required; widened prefix arithmetic is not an exact-real certificate, and nonzero increments that vanish at its precision reject. Empty/all-nonpositive inputs yield no segments. Borrowed input unchanged, output owned; no callbacks/shared state.
/// @param scores Complete finite position scores.
/// @param options Input, actual-comparison and retained-list policies.
/// @return Actual final intervals, measured scan work and optional same-pass list histories.
/// @throws std::invalid_argument Nonfinite score.
/// @throws std::length_error Input/comparison/history/container capacity exceeded.
/// @throws std::overflow_error Nonfinite prefix or unrepresentable positive output score.
/// @throws std::underflow_error Nonzero score increment vanishes in widened prefix arithmetic.
/// @par Example
/// @code
/// auto r=dm::algorithms::ruzzo_tompa_trace({3,-2,5});
/// std::print("{} {}\n",r.segments.size(),r.segments[0].score); // 1 6
/// @endcode
/// @par References
/// - Ruzzo and Tompa (1999), A Linear Time Algorithm for Finding All Maximal Scoring Subsequences, ISMB pp.234–241, https://homes.cs.washington.edu/~ruzzo/papers/maxseq.html. This implementation uses a direct predecessor scan rather than the paper's linear-time links.
/// @see ruzzo_tompa()
RuzzoRun ruzzo_tompa_trace(const std::vector<double>& scores, const RuzzoOptions& options = {});
/// @brief Return native maximal-scoring intervals through the preserved signature.
/// Uses the same direct predecessor-scan producer and default bounds; valid ordinary interval ordering is preserved. Worst-case quadratic work is explicit, replacing the old incorrect linear-time claim.
/// @param scores Complete finite scored sequence.
/// @return Owned left-to-right positive intervals.
/// @throws std::invalid_argument Nonfinite input.
/// @throws std::length_error Default policy exceeded.
/// @throws std::overflow_error Unrepresentable prefix/output score.
/// @throws std::underflow_error Unresolved nonzero prefix increment.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::ruzzo_tompa({-1,2,-1}).size()); // 1
/// @endcode
/// @see ruzzo_tompa_run()
std::vector<ScoringSegment> ruzzo_tompa(const std::vector<double>& scores);
/// @brief Generate a canonical controlled report input.
/// Construct the exact alternating1,-1 predecessor-scan worst-case fixture. Returns2*n-1 scores ending positive, or empty for zero; this is controlled synthetic work, not random data.
/// @param positive_count Positive singleton count, default8; at most500000.
/// @return Owned exact synthetic values.
/// @throws std::length_error Requested alternating fixture exceeds the fixed allocation policy.
/// @par Example
/// @code
/// auto x=dm::algorithms::ruzzo_alternating_scores(3);
/// std::print("{}\n",x.size()); // 5
/// @endcode
/// @see ruzzo_tompa_run()
std::vector<double> ruzzo_alternating_scores(std::size_t positive_count = 8);
} // namespace datamunge::algorithms
