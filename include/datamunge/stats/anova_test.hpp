#pragma once

/// @file anova_test.hpp
/// @brief Classical anova test contracts, assumptions and numerical results.

#include <cstddef>
#include <datamunge/stats/hypothesis_test_result.hpp>
#include <vector>

namespace datamunge::stats {

/// @brief Classical equal-variance one-way ANOVA, like R oneway.test(var.equal=TRUE).
///
/// Tests equality of group means with between/within mean-square ratio. Independent normal
/// observations and a common within-group population variance give the exact F reference. This is
/// not Welch ANOVA. O(N) work, O(1) auxiliary storage; input unchanged, no omission or shared
/// mutable state.
/// @param values Finite observations concatenated group by group.
/// @param group_sizes At least two positive sizes summing exactly to values.size(); N must exceed
/// the group count.
/// @return statistic=F, parameter1=k-1, parameter2=N-k and upper-tail p_value; no CI or group
/// estimates.
/// @throws std::invalid_argument Invalid grouping/nonfinite input, no residual df, zero within-
/// group variation or arithmetic overflow.
/// @par Example
/// @code
/// const auto a=dm::stats::one_way_anova({1,2,3,4,5,6,7,8,9},{3,3,3});
/// std::print("{:.0f}\n",a.statistic); // 27
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/oneway.test.html
/// @see HypothesisTestResult
HypothesisTestResult one_way_anova(
    const std::vector<double>&      values,
    const std::vector<std::size_t>& group_sizes);

/// @brief Tie-corrected Kruskal-Wallis rank test, like R kruskal.test().
///
/// Ranks the pooled observations with average ties and uses the chi-square(k-1) approximation.
/// The usual exchangeability null is identical independent group distributions; a common-shape
/// shift model permits a location interpretation. It is not an omnibus test for every possible
/// distributional difference. Small groups can invalidate the approximation. O(N log N) work/O(N)
/// space, no mutation, omission or shared state.
/// @param values Finite observations concatenated by group.
/// @param group_sizes At least two positive group sizes summing to values.size().
/// @return statistic=H, parameter1=k-1 and upper-tail approximate p_value; no CI.
/// @throws std::invalid_argument Invalid groups, nonfinite input or all observations tied.
/// @par Example
/// @code
/// const auto h=dm::stats::kruskal_wallis_test({1,2,3,4,5,6,7,8,9},{3,3,3});
/// std::print("{:.1f}\n",h.statistic); // 7.2
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/kruskal.test.html
/// @see HypothesisTestResult
HypothesisTestResult kruskal_wallis_test(
    const std::vector<double>&      values,
    const std::vector<std::size_t>& group_sizes);

} // namespace datamunge::stats
