#pragma once

/// @file wilcoxon_test.hpp
/// @brief Classical wilcoxon test contracts, assumptions and numerical results.

#include <datamunge/stats/hypothesis_test_result.hpp>
#include <vector>

namespace datamunge::stats {

/// @brief Normal-approximation Wilcoxon signed-rank test with tie/continuity corrections.
///
/// Tests symmetry of x-mu about zero under independent observations, not an arbitrary-population
/// mean or median null. For paired data pass aligned differences. Exact-zero differences are
/// removed; absolute differences receive average ranks. Always uses a normal approximation, even
/// for small tie-free samples; no exact option or CI. O(n log n) work/O(n) storage; input
/// unchanged, no shared state.
/// @param x Finite values; at least one nonzero difference from mu must remain.
/// @param mu Finite symmetry centre, default zero.
/// @param alternative TwoSided (default), Less or Greater for the signed location shift.
/// @return statistic=positive rank sum V, estimate1=mu (not an estimated pseudomedian),
/// approximate p_value; no CI.
/// @throws std::invalid_argument Nonfinite input/mu, invalid alternative, no nonzero differences
/// or zero rank variance.
/// @par Example
/// @code
/// const auto w=dm::stats::wilcoxon_signed_rank_test({1,2,3,4});
/// std::print("{:.0f}\n",w.statistic); // 10
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/wilcox.test.html
/// @see HypothesisTestResult
HypothesisTestResult wilcoxon_signed_rank_test(
    const std::vector<double>& x,
    double                     mu          = 0.0,
    Alternative                alternative = Alternative::TwoSided);

/// @brief Tie/continuity-corrected Mann-Whitney U test using a normal approximation.
///
/// Pools ranks, giving average ranks to ties, and reports U for the first sample. Its identical-
/// distribution null requires independent exchangeable samples. A common-shape shift model
/// supports a location interpretation; equal medians alone is insufficient. U/(n_x*n_y) estimates
/// P(X>Y)+P(X=Y)/2, not a mean difference. Approximation is used at every size. O(N log N)
/// work/O(N) storage, inputs unchanged, no shared state.
/// @param x Nonempty finite first sample.
/// @param y Nonempty finite independent second sample.
/// @param alternative TwoSided (default), Less or Greater for first-versus-second stochastic
/// location.
/// @return statistic=U=rank_sum(x)-n_x(n_x+1)/2 and approximate p_value; no CI or location
/// estimate.
/// @throws std::invalid_argument Empty/nonfinite samples, invalid alternative or zero rank
/// variance (all pooled observations tied).
/// @par Example
/// @code
/// const auto w=dm::stats::wilcoxon_rank_sum_test({1,2,3},{4,5,6});
/// std::print("{:.0f}\n",w.statistic); // 0
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/wilcox.test.html
/// @see HypothesisTestResult
HypothesisTestResult wilcoxon_rank_sum_test(
    const std::vector<double>& x,
    const std::vector<double>& y,
    Alternative                alternative = Alternative::TwoSided);

} // namespace datamunge::stats
