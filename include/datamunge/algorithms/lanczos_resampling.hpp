#pragma once
/// @file lanczos_resampling.hpp
/// @brief Checked windowed-sinc reconstruction with explicit border/normalization contracts and actual taps.
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {
/// @brief Actual selected finite-window tap work, excluding validation, trig internals, copies and time.
/// Every complete query evaluates2*a candidate weights and reads their clamped
/// samples, including zero weights. Negative weights can cause ringing/overshoot.
/// @par Example
/// @code
/// const auto a=dm::algorithms::lanczos_resample_analyze({0,1,2},.5,3);
/// std::print("{}\n",a.work.taps); // 6
/// @endcode
/// @see LanczosResampleAnalysis
struct LanczosResampleWork {
  std::size_t taps{
      0}; ///< Actual candidate weight/sample/product evaluations, including zero weights.
  std::size_t clamped_taps{0};     ///< Actual virtual tap indices mapped to a boundary sample.
  std::size_t negative_weights{0}; ///< Actual weights strictly below zero.
};
/// @brief Owned actual query result, raw sums and selected producing work.
/// position is the original query, effective_position is clamped to[0,n-1].
/// raw_sum/weight_sum are actual original-order sums. value=raw_sum for legacy
/// raw mode or raw_sum/weight_sum for explicitly normalized mode. No replay.
/// @par Example
/// @code
/// const auto a=dm::algorithms::lanczos_resample_analyze({1,1,1},.5,3,true);
/// std::print("{}\n",a.value); // 1
/// @endcode
/// @see lanczos_resample_analyze()
struct LanczosResampleAnalysis {
  double position{0};           ///< Original finite caller query in sample-index units.
  double effective_position{0}; ///< Actual query after documented border-position clamp.
  int    lobes{3};              ///< Actual positive window half-width a.
  bool   normalized{false}; ///< Whether the actual final result divides by the actual weight sum.
  double raw_sum{0};        ///< Actual sequential unnormalized weighted-value sum.
  double weight_sum{0};     ///< Actual sequential raw kernel-weight sum.
  double value{0};          ///< Actual raw or explicitly normalized final value.
  LanczosResampleWork work; ///< Actual selected tap counts from this query.
};
/// @brief Owned original identity, weight and actual accumulation boundary for one producing tap.
/// virtual_index is the window lattice position; sample_index is its actual
/// boundary-clamped stored identity. Product/sum fields are unnormalized even
/// when the final analysis value divides by total weight.
/// @par Example
/// @code
/// const auto t=dm::algorithms::lanczos_resample_trace({1},0,1);
/// std::print("{} {}\n",t.taps[0].virtual_index,t.taps[0].sample_index); // 0 0
/// @endcode
/// @see LanczosResampleTrace
struct LanczosResampleTap {
  std::ptrdiff_t virtual_index{0};    ///< Actual integer lattice tap before border lookup clamping.
  std::size_t    sample_index{0};     ///< Actual stored sample identity read.
  double         sample{0};           ///< Actual original finite sample value read.
  double         weight{0};           ///< Actual finite raw Lanczos kernel weight.
  double         weighted_value{0};   ///< Actual rounded sample times weight product.
  double         sum_before{0};       ///< Actual raw accumulated sum before this tap.
  double         sum_after{0};        ///< Actual raw accumulated sum after this tap.
  double         weight_sum_after{0}; ///< Actual raw kernel-weight sum after this tap.
};
/// @brief Owned complete original samples, every actual window tap and actual query analysis.
/// captured_entries=n+2*a counts original sample values and tap records, not
/// individual fields, capacities, output/runtime copies or bytes/peak RAM.
/// Complete input, integer geometry and count/quota are checked before copying
/// samples or evaluating weights. No second query or synthetic history.
/// @par Example
/// @code
/// const auto t=dm::algorithms::lanczos_resample_trace({1},0,1,false,3);
/// std::print("{} {}\n",t.taps.size(),t.captured_entries); // 2 3
/// @endcode
/// @see lanczos_resample_trace()
struct LanczosResampleTrace {
  std::vector<double>     samples;  ///< Owned complete original finite nonempty samples.
  LanczosResampleAnalysis analysis; ///< Actual final sums/value/parameters and selected work.
  std::vector<LanczosResampleTap>
              taps;                ///< Actual chronological virtual-index-ordered tap evaluations.
  std::size_t captured_entries{0}; ///< Complete original-sample plus tap-record count.
};
/// @brief Evaluate sinc(distance)*sinc(distance/a) on the open window abs(distance)<a.
/// sinc(0)=1, and nonzero exactly integral sinc arguments explicitly return0,
/// preserving numeric integer-sample interpolation without sin(pi*k) remnants.
/// The historical tiny abs(x)<1e-12 sinc approximation returns1. Outside support
/// returns0. Finite distance and positive a required; no input mutation/state.
/// @param distance Finite signed offset in sample-index units.
/// @param lobes Positive integer half-width a, default3.
/// @return Actual finite double kernel weight, possibly negative.
/// @throws std::invalid_argument On nonfinite distance or nonpositive lobes.
/// @par Example
/// @code
/// std::print("{} {}\n",dm::algorithms::lanczos_kernel_weight(0),dm::algorithms::lanczos_kernel_weight(1)); // 1 0
/// @endcode
/// @par References
/// - Duchon(1979), Lanczos Filtering in One and Two Dimensions, J.Appl.Meteor.18(8),1016–1022, https://journals.ametsoc.org/view/journals/apme/18/8/1520-0450_1979_018_1016_lfioat_2_0_co_2.xml.
/// @see lanczos_resample_at()
double lanczos_kernel_weight(double distance, int lobes = 3);
/// @brief Reconstruct one finite query from finite nonempty uniformly indexed samples in raw mode.
/// Clamp the query position to[0,n-1], then examine virtual integer taps
/// floor(p)-a+1 through floor(p)+a, clamping each stored lookup to[0,n-1].
/// Sum sample*kernel in ascending virtual order. This intentionally retains the
/// unnormalized legacy interior formula: finite kernel weights need not sum1,
/// so constants can have phase-dependent gain. Exact integer queries preserve
/// numeric sample values using explicit integral sinc zeros. Corrects the old
/// outside-position comment/implementation mismatch and unsafe empty/nonfinite
/// casts; huge finite outside positions safely return the border value.
/// No low-pass cutoff adjustment for downsampling, universal superiority to
/// linear interpolation, monotonicity, DC normalization or alias-free claim.
/// Require2*a representable and n-1+a<=min(PTRDIFF_MAX,2^53-1) before integer
/// indexing. Finite raw products/sums required or throw. O(n+a) work including
/// complete sample validation, O(1) query state, no input copies/history/shared state.
/// @param samples Borrowed complete finite nonempty sample values at indices0..n-1.
/// @param p Finite query in sample-index units, clamped to sample-position range.
/// @param a Positive integer half-width, default3.
/// @return Actual finite unnormalized weighted sum.
/// @throws std::invalid_argument On empty/nonfinite samples, nonfinite query or nonpositive a.
/// @throws std::length_error On complete index/count geometry limits.
/// @throws std::overflow_error On nonfinite produced product/sum.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::lanczos_resample_at({0,1,4},1)); // 1
/// @endcode
/// @par References
/// - Duchon(1979), Lanczos Filtering in One and Two Dimensions, https://journals.ametsoc.org/view/journals/apme/18/8/1520-0450_1979_018_1016_lfioat_2_0_co_2.xml.
///   The native reconstruction window/border and optional normalization policies are explicit here.
/// @see lanczos_resample_analyze(), lanczos_resample_trace()
double lanczos_resample_at(const std::vector<double>& samples, double p, int a = 3);
/// @brief Reconstruct every complete finite query through the shared raw window producer.
/// Validate complete nonempty finite samples, all positions and integer geometry
/// before output/evaluation. Empty positions returns empty after validation.
/// Same raw border/integer/nonfinite policies as lanczos_resample_at(); no
/// per-query input copying/validation repetition. O(n+queries*a) work/output.
/// @param samples Borrowed complete finite nonempty original samples.
/// @param positions Borrowed complete finite query positions, possibly empty.
/// @param a Positive integer half-width, default3.
/// @return Owned raw reconstructed values in original query order.
/// @throws std::invalid_argument On any complete input/query/lobe violation.
/// @throws std::length_error On complete index/count/container geometry limits.
/// @throws std::overflow_error On nonfinite produced arithmetic.
/// @throws std::bad_alloc On output allocation failure.
/// @par Example
/// @code
/// const auto x=dm::algorithms::lanczos_resample({0,1,4},{0,1,2});
/// std::print("{} {}\n",x.size(),x[2]); // 3 4
/// @endcode
/// @par References
/// - Duchon(1979), Lanczos Filtering in One and Two Dimensions, https://journals.ametsoc.org/view/journals/apme/18/8/1520-0450_1979_018_1016_lfioat_2_0_co_2.xml.
/// @see lanczos_resample_at(), lanczos_resample_analyze()
std::vector<double> lanczos_resample(
    const std::vector<double>& samples,
    const std::vector<double>& positions,
    int                        a = 3);
/// @brief Return actual producing raw sums/work and optionally normalize by the actual weight sum.
/// Same validated window/index/arithmetic producer as lanczos_resample_at().
/// Normalization is an explicit additional mode, never silently changes raw API.
/// Does not remove negative-lobe ringing or implement an antialiasing prefilter.
/// O(n+a) including validation, no optional capture/input copies or replay.
/// @param samples Borrowed complete finite nonempty original samples.
/// @param p Actual finite query, position-clamped before window construction.
/// @param a Positive integer half-width, default3.
/// @param normalize Divide final raw sum by actual weight sum if true, default false.
/// @return Owned actual sums, value, parameters and selected work.
/// @throws std::invalid_argument On complete sample/query/lobe violations.
/// @throws std::length_error On index/count geometry limits.
/// @throws std::overflow_error On nonfinite produced arithmetic or normalized result.
/// @throws std::domain_error On zero actual weight sum in normalized mode.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::lanczos_resample_analyze({1,1,1},.5,3,true).value); // 1
/// @endcode
/// @par References
/// - Duchon(1979), Lanczos Filtering in One and Two Dimensions, https://journals.ametsoc.org/view/journals/apme/18/8/1520-0450_1979_018_1016_lfioat_2_0_co_2.xml.
/// @see LanczosResampleAnalysis, lanczos_resample_trace()
LanczosResampleAnalysis lanczos_resample_analyze(
    const std::vector<double>& samples,
    double                     p,
    int                        a         = 3,
    bool                       normalize = false);
/// @brief Capture complete original samples and every actual raw tap accumulation once.
/// Same raw/explicit-normalized producer as lanczos_resample_analyze(). Admit
/// complete n+2*a and optional quota before sample/history copies/evaluation;
/// adds owned O(n+a) capture, no partial result escapes failure or prefix replay.
/// @param samples Borrowed complete finite nonempty original samples.
/// @param p Actual finite query, clamped to stored position range.
/// @param a Positive integer half-width, default3.
/// @param normalize Divide only the final value by actual total weight if true, default false.
/// @param max_entries Complete original-sample+tap-record ceiling, default1000000, zero off.
/// @return Owned actual samples, chronological tap evidence and final analysis.
/// @throws std::invalid_argument On complete input/query/lobe violations before copies.
/// @throws std::length_error On enabled quota or complete geometry/count/container limits.
/// @throws std::overflow_error On nonfinite produced arithmetic.
/// @throws std::domain_error On zero weight sum in normalized mode.
/// @throws std::bad_alloc On original/history allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::lanczos_resample_trace({0,1,4},.5,3);
/// std::print("{}\n",t.analysis.work.taps); // 6
/// @endcode
/// @par References
/// - Duchon(1979), Lanczos Filtering in One and Two Dimensions, https://journals.ametsoc.org/view/journals/apme/18/8/1520-0450_1979_018_1016_lfioat_2_0_co_2.xml.
/// @see LanczosResampleTrace, lanczos_resample_analyze()
LanczosResampleTrace lanczos_resample_trace(
    const std::vector<double>& samples,
    double                     p,
    int                        a           = 3,
    bool                       normalize   = false,
    std::size_t                max_entries = 1000000);
} // namespace datamunge::algorithms
