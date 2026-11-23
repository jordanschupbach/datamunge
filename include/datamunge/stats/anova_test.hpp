#pragma once

/// @file anova_test.hpp
/// @brief Classical anova test contracts, assumptions and numerical results.

#include <cstddef>
#include <datamunge/stats/hypothesis_test_result.hpp>
#include <datamunge/stats/p_adjust.hpp>
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

/// @brief Welch's one-way test of equal means in independent normal groups with unequal variances.
///
/// Weights means by inverse estimated sampling variances and uses an approximate F reference.
/// Inputs are unchanged. Work is linear in observations with O(k) storage for k groups.
/// @param values Finite observations concatenated by group.
/// @param group_sizes At least two groups of size two or larger, each with positive variance.
/// @return F statistic, numerator and denominator degrees of freedom, and upper-tail p-value.
/// @throws std::invalid_argument Invalid sizes, nonfinite observations, zero variance or overflow.
/// @par Example
/// @code
/// const auto fit = dm::stats::welch_anova({1, 2, 4, 3, 6, 9, 12, 15, 3, 4, 5, 5}, {3, 5, 4});
/// std::print("{:.4f}\n", fit.statistic); // 4.1238
/// @endcode
/// @par References
/// - Welch (1951), On the Comparison of Several Mean Values, doi:10.2307/2332579.
/// @see one_way_anova(), games_howell()
HypothesisTestResult welch_anova(const std::vector<double>& values,
                               const std::vector<std::size_t>& group_sizes);

/// @brief One Games–Howell contrast between two independent groups.
/// @par Example
/// @code
/// const auto row = dm::stats::games_howell({1, 2, 3, 4}, {2, 2}).front();
/// std::print("{:.0f}\n", row.difference); // 2
/// @endcode
/// @see games_howell()
struct GamesHowellComparison {
  std::size_t group1; ///< Zero-based index of the later group in the input.
  std::size_t group2; ///< Zero-based index of the earlier group in the input.
  double difference; ///< Mean of group1 minus mean of group2, in observation units.
  double degrees_of_freedom; ///< Pair-specific Welch–Satterthwaite degrees of freedom.
  double lower; ///< Lower approximate simultaneous confidence bound.
  double upper; ///< Upper approximate simultaneous confidence bound.
  double p_adjusted; ///< Studentized-range upper tail for the pair within the whole family.
};

/// @brief Compares all independent normal group means using Games–Howell intervals and p-values.
///
/// Allows unequal group sizes and variances. Uses a pair-specific variance and degrees of freedom
/// with a studentized range over all groups. Inputs are unchanged.
/// @param values Finite observations concatenated by group.
/// @param group_sizes At least two groups of size two or larger, each with positive variance.
/// @param confidence_level Desired simultaneous coverage in (0,1), default 0.95.
/// @return Owned comparisons in order (1,0), (2,0), (2,1), and so on.
/// @throws std::invalid_argument Invalid inputs, numerical overflow or an unresolved critical value.
/// @par Example
/// @code
/// const auto rows = dm::stats::games_howell({1, 2, 4, 3, 6, 9, 12, 15, 3, 4, 5, 5}, {3, 5, 4});
/// std::print("{:.5f}\n", rows.front().p_adjusted); // 0.07077
/// @endcode
/// @par References
/// - Games and Howell (1976), Pairwise Multiple Comparison Procedures with Unequal N's and/or
///   Variances, doi:10.3102/10769986001002113.
/// @see welch_anova(), studentized_range_cdf()
std::vector<GamesHowellComparison> games_howell(const std::vector<double>& values,
                                              const std::vector<std::size_t>& group_sizes,
                                              double confidence_level = 0.95);

/// @brief Tests group scale with Levene's mean or Brown–Forsythe's median centering.
///
/// Runs one-way ANOVA on absolute within-group deviations. Independent observations
/// and comparable group distribution shapes support interpreting the result as a scale test.
/// @param values Finite observations concatenated by group.
/// @param group_sizes At least two groups of size two or larger partitioning values.
/// @param median_center Use group medians when true (default), or group means when false.
/// @return F statistic, two degrees of freedom and an approximate upper-tail p-value.
/// @throws std::invalid_argument Invalid inputs, zero residual deviation variance or overflow.
/// @par Example
/// @code
/// const auto fit = dm::stats::levene_test({1, 2, 4, 3, 6, 9, 12, 15, 3, 4, 5, 5}, {3, 5, 4});
/// std::print("{:.3f}\n", fit.statistic); // 3.550
/// @endcode
/// @par References
/// - Brown and Forsythe (1974), Robust Tests for the Equality of Variances,
///   doi:10.1080/01621459.1974.10482955.
/// - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levene.html
/// @see bartlett_test(), fligner_killeen_test()
HypothesisTestResult levene_test(const std::vector<double>& values,
                                const std::vector<std::size_t>& group_sizes, bool median_center = true);

/// @brief Bartlett's test of equal variances for independent normal groups.
///
/// Compares the pooled log variance with the weighted group log variances and
/// applies Bartlett's finite-sample correction. Inputs are unchanged.
/// @param values Finite observations concatenated by group.
/// @param group_sizes At least two groups of size two or larger, each with positive variance.
/// @return K-squared statistic, k-1 degrees of freedom and approximate chi-square upper tail.
/// @throws std::invalid_argument Invalid inputs, zero group variance or overflow.
/// @par Example
/// @code
/// const auto fit = dm::stats::bartlett_test({1, 2, 4, 3, 6, 9, 12, 15, 3, 4, 5, 5}, {3, 5, 4});
/// std::print("{:.3f}\n", fit.statistic); // 6.496
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/bartlett.test.html
/// @see levene_test(), fligner_killeen_test()
HypothesisTestResult bartlett_test(const std::vector<double>& values,
                                  const std::vector<std::size_t>& group_sizes);

/// @brief Fligner–Killeen scale test using normal scores of median-centered absolute deviations.
///
/// Ranks the pooled deviations with midranks for ties. The chi-square reference
/// assumes independent groups and is asymptotic.
/// @param values Finite observations concatenated by group.
/// @param group_sizes At least two groups of size two or larger partitioning values.
/// @return Chi-square statistic, k-1 degrees of freedom and the approximate upper-tail p-value.
/// @throws std::invalid_argument Invalid inputs, zero variation in scores or overflow.
/// @par Example
/// @code
/// const auto fit = dm::stats::fligner_killeen_test({1, 2, 4, 3, 6, 9, 12, 15, 3, 4, 5, 5}, {3, 5, 4});
/// std::print("{:.3f}\n", fit.statistic); // 4.128
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/fligner.test.html
/// @see levene_test(), bartlett_test()
HypothesisTestResult fligner_killeen_test(const std::vector<double>& values,
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

/// @brief One pair in Dunn's multiple comparisons of pooled mean ranks.
/// @par Example
/// @code
/// const auto row = dm::stats::dunn_test({1, 2, 4, 5}, {2, 2}).front();
/// std::print("{:.0f}\n", row.rank_difference); // 2
/// @endcode
/// @see dunn_test()
struct DunnComparison {
  std::size_t group1;      ///< Zero-based index of the later group in the input.
  std::size_t group2;      ///< Zero-based index of the earlier group in the input.
  double rank_difference; ///< Pooled mean rank of group1 minus that of group2.
  double statistic;       ///< Signed normal z statistic.
  double p_value;         ///< Two-sided asymptotic p-value without adjustment.
  double p_adjusted;      ///< p-value adjusted over the returned comparisons.
};

/// @brief Performs Dunn pairwise comparisons on ranks pooled across all groups.
///
/// Tied observations get average ranks, and their reduced pooled rank variance
/// enters the normal approximation. The standard interpretation assumes
/// independent groups. A location interpretation uses a common-shape shift model.
/// @param values Finite observations, concatenated by group.
/// @param group_sizes At least two positive counts that partition values.
/// @param adjustment Multiplicity method for the whole pairwise family, Holm by default.
/// @return Owned comparisons in order (1,0), (2,0), (2,1), and so on.
/// @throws std::invalid_argument Invalid groups or zero pooled rank variation.
/// @par Example
/// @code
/// const auto rows = dm::stats::dunn_test({1, 2, 4, 5, 7, 8}, {2, 2, 2});
/// std::print("{}\n", rows.size()); // 3
/// @endcode
/// @par References
/// - Dunn (1964), Multiple Comparisons Using Rank Sums, doi:10.2307/1266041.
/// - https://cran.r-project.org/web/packages/dunn.test/index.html
/// @see kruskal_wallis_test(), p_adjust()
std::vector<DunnComparison> dunn_test(const std::vector<double>& values,
                                      const std::vector<std::size_t>& group_sizes,
                                      PAdjustMethod adjustment = PAdjustMethod::Holm);

} // namespace datamunge::stats
