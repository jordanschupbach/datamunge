#pragma once

/// @file nonparametric.hpp
/// @brief Rank, sign, median and distribution-free inference for independent and paired data.

#include <datamunge/plot/plot.hpp>
#include <datamunge/stats/hypothesis_test_result.hpp>
#include <datamunge/stats/ks_test.hpp>

#include <string>
#include <vector>

namespace datamunge::stats {

/// @brief Tests association using Kendall concordance and reports tau-b.
///
/// Requires at least two pairs and nonconstant margins. For up to 50 untied pairs,
/// uses the exact permutation distribution of the inversion count. Otherwise uses
/// the normal reference with tie-corrected variance and no continuity correction.
/// @param x Finite first coordinates, with one observation per pair.
/// @param y Equal-length finite second coordinates, paired by index.
/// @param alternative TwoSided (default), Less, or Greater for the signed effect.
/// @return Concordant minus discordant count, tau-b in estimate1, sample size in parameter1, and p-value.
/// @throws std::invalid_argument If inputs violate the stated constraints.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto result = dm::stats::kendall_correlation_test({1, 2, 3}, {1, 2, 3});
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/cor.test.html
/// @see kendall_tau(), spearman_correlation_test()
HypothesisTestResult kendall_correlation_test(const std::vector<double>& x,
                                              const std::vector<double>& y,
                                              Alternative alternative = Alternative::TwoSided);

/// @brief Tests association in the four quadrants defined by the sample medians.
///
/// Drops pairs on either median line. Requires nonempty row and column margins
/// after dropping. Uses a Fisher exact test conditional on the retained sign margins.
/// The coefficient is concordant-quadrant minus discordant-quadrant fraction.
/// @param x Finite first coordinates, with one observation per pair.
/// @param y Equal-length finite second coordinates, paired by index.
/// @param alternative TwoSided (default), Less, or Greater for the signed effect.
/// @return Upper-right count, retained pairs in parameter1, quadrant coefficient in estimate1, and p-value.
/// @throws std::invalid_argument If inputs violate the stated constraints.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto q = dm::stats::quadrant_test({1, 2, 3, 4}, {1, 2, 4, 3});
/// @endcode
/// @par References
/// - https://doi.org/10.1214/aoms/1177729754
/// @see kendall_correlation_test(), fisher_exact_test_2x2()
HypothesisTestResult quadrant_test(const std::vector<double>& x,
                                   const std::vector<double>& y,
                                   Alternative alternative = Alternative::TwoSided);

/// @brief Exact sign test for a prespecified median.
///
/// Drops observations equal to the null median and tests the positive fraction
/// against one-half. Requires at least one nonzero sign.
/// @param values Nonempty finite independent observations.
/// @param median Finite null median, default zero.
/// @param alternative TwoSided (default), Less, or Greater for the signed effect.
/// @return Positive count, retained count in parameter1, positive fraction in estimate1, and exact p-value.
/// @throws std::invalid_argument If inputs violate the stated constraints.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto s = dm::stats::sign_test({1, 2, 3, 4});
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/binom.test.html
/// @see paired_sign_test()
HypothesisTestResult sign_test(const std::vector<double>& values,
                               double median = 0.0,
                               Alternative alternative = Alternative::TwoSided);

/// @brief Exact sign test of paired differences x minus y.
///
/// Requires at least one unequal pair. Tests median paired difference zero.
/// @param x Finite first coordinates, with one observation per pair.
/// @param y Equal-length finite second coordinates, paired by index.
/// @param alternative TwoSided (default), Less, or Greater for the signed effect.
/// @return The result of sign_test() on paired differences, using the default null median.
/// @throws std::invalid_argument If inputs violate the stated constraints.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto s = dm::stats::paired_sign_test({3, 4, 6}, {1, 4, 2});
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/binom.test.html
/// @see sign_test(), wilcoxon_paired_test()
HypothesisTestResult paired_sign_test(const std::vector<double>& x,
                                      const std::vector<double>& y,
                                      Alternative alternative = Alternative::TwoSided);

/// @brief Paired Wilcoxon signed-rank test of x minus y.
///
/// Uses the exact sign-flip reference for up to 100 nonzero differences. Larger
/// samples use the existing signed-rank normal approximation with continuity and
/// tie corrections. The location interpretation assumes symmetric differences.
/// @param x Finite first coordinates, with one observation per pair.
/// @param y Equal-length finite second coordinates, paired by index.
/// @param alternative TwoSided (default), Less, or Greater for the signed effect.
/// @return Positive rank sum and p-value, with the null location zero.
/// @throws std::invalid_argument If inputs violate the stated constraints.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto w = dm::stats::wilcoxon_paired_test({3, 4, 6}, {1, 4, 2});
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/wilcox.test.html
/// @see paired_sign_test()
HypothesisTestResult wilcoxon_paired_test(const std::vector<double>& x,
                                               const std::vector<double>& y,
                                               Alternative alternative = Alternative::TwoSided);

/// @brief Exact one-sample Wilcoxon signed-rank test conditional on absolute ranks.
///
/// Omits zero differences and uses average ranks for tied magnitudes. Enumerates
/// their sign assignments by dynamic programming. The two-sided tail includes
/// scores at least as far from half the total rank as the observed score.
/// Assumes independent differences symmetric about the null location.
/// @param values Finite observations, with one through 100 nonzero centered differences.
/// @param mu Finite null location, default zero.
/// @param alternative TwoSided (default), Less, or Greater for the signed effect.
/// @return Positive rank sum, effective sample size in parameter1, null location in estimate1, and exact p-value.
/// @throws std::invalid_argument For invalid data, an invalid alternative or unsupported sample size.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto w = dm::stats::wilcoxon_signed_rank_exact_test({1, 1, 2});
/// std::print("{:.2f}\n", w.p_value); // 0.25
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/wilcox.test.html
/// @see wilcoxon_paired_test(), wilcoxon_signed_rank_test()
HypothesisTestResult wilcoxon_signed_rank_exact_test(const std::vector<double>& values,
                                                      double mu = 0.0,
                                                      Alternative alternative = Alternative::TwoSided);

/// @brief Mood median test for two or more independent groups.
///
/// Forms an above/below-pooled-median table and drops observations at the median.
/// Requires every group to retain observations and both pooled categories to occur.
/// Uses Pearson chi-square without continuity correction. Sparse tables weaken
/// the approximation. With ties, the test concerns the retained sign proportions.
/// @param groups At least two nonempty groups with finite observations.
/// @return Chi-square statistic, df in parameter1, pooled median in estimate1, and p-value.
/// @throws std::invalid_argument If inputs violate the stated constraints.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto m = dm::stats::median_test({{1, 2, 3}, {6, 7, 8}});
/// @endcode
/// @par References
/// - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.median_test.html
/// @see sign_test(), van_der_waerden_test()
HypothesisTestResult median_test(const std::vector<std::vector<double>>& groups);

/// @brief Normal-score test for two or more independent groups.
///
/// Transforms pooled average ranks r to Normal quantiles at r/(N+1).
/// The statistic divides between-group score variation by the sample score variance.
/// Uses a chi-square approximation and rejects constant pooled data.
/// @param groups At least two nonempty groups with finite observations.
/// @return Normal-score statistic, df in parameter1, and approximate p-value.
/// @throws std::invalid_argument If inputs violate the stated constraints.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto a = dm::stats::van_der_waerden_test({{1, 2, 3}, {6, 7, 8}});
/// @endcode
/// @par References
/// - https://itl.nist.gov/div898/software/dataplot/refman1/auxillar/vanderwa.htm
/// @see kruskal_wallis_test(), median_test()
HypothesisTestResult van_der_waerden_test(const std::vector<std::vector<double>>& groups);

/// @brief Kruskal–Wallis rank test accepting one vector per independent group.
///
/// Pools groups and calls the existing tie-corrected rank test. Identical pooled
/// values leave no rank variation and are rejected. The reference is chi-square.
/// @param groups At least two nonempty groups with finite observations.
/// @return Tie-corrected H, df in parameter1, and approximate p-value.
/// @throws std::invalid_argument If inputs violate the stated constraints.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto h = dm::stats::kruskal_wallis_test({{1, 2, 3}, {6, 7, 8}});
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/kruskal.test.html
/// @see van_der_waerden_test(), friedman_test()
HypothesisTestResult kruskal_wallis_test(const std::vector<std::vector<double>>& groups);

/// @brief Friedman rank test for randomized complete blocks or repeated measures.
///
/// Averages ties within each block and divides by the within-block tie correction.
/// Blocks are independent. Uses chi-square calibration, which is approximate for
/// small designs. Throws if all blocks have zero within-block rank variation.
/// @param blocks At least two finite rows, each containing the same two or more treatments in the same order.
/// @return Tie-corrected Friedman statistic, df in parameter1, block count in parameter2, and p-value.
/// @throws std::invalid_argument If inputs violate the stated constraints.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto f = dm::stats::friedman_test({{1, 2, 3}, {3, 2, 4}});
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/friedman.test.html
/// @see kruskal_wallis_test()
HypothesisTestResult friedman_test(const std::vector<std::vector<double>>& blocks);

/// @brief Plots two empirical cumulative distributions as right-continuous steps.
///
/// Sorts owned copies of both inputs and groups equal values into a single jump.
/// Useful for inspecting where distributions differ before reporting a KS statistic.
/// @param x First nonempty finite sample.
/// @param y Second nonempty finite sample.
/// @param x_label First series label, default Sample 1.
/// @param y_label Second series label, default Sample 2.
/// @return Owned editable native plot.
/// @throws std::invalid_argument If inputs violate the stated constraints.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// auto plot = dm::stats::empirical_cdf_plot({1, 2}, {2, 3});
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/ks.test.html
/// @see ks_test_two_sample()
graphics::RPlot empirical_cdf_plot(const std::vector<double>& x,
                                   const std::vector<double>& y,
                                   const std::string& x_label = "Sample 1",
                                   const std::string& y_label = "Sample 2");

} // namespace datamunge::stats
