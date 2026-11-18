#pragma once

/// @file classical_tests.hpp
/// @brief Classical rank, blocked and categorical hypothesis tests.

#include <datamunge/stats/hypothesis_test_result.hpp>

#include <array>
#include <cstddef>
#include <vector>

namespace datamunge::stats {

/// @brief Mann–Whitney U test for two independent samples.
///
/// Calls the Wilcoxon rank-sum test and adds the probability-of-superiority estimate.
/// Uses a tie- and continuity-corrected normal reference.
/// @param x First nonempty finite sample.
/// @param y Second nonempty finite independent sample.
/// @param alternative TwoSided (default), Less or Greater for the first sample effect.
/// @return First-sample U, p-value and U divided by sample-size product in estimate1.
/// @throws std::invalid_argument For invalid inputs or degenerate reference variance where noted.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto u = dm::stats::mann_whitney_test({3, 4, 5}, {1, 2, 3});
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/wilcox.test.html
/// @see wilcoxon_rank_sum_test()
HypothesisTestResult mann_whitney_test(const std::vector<double>& x,
                                       const std::vector<double>& y,
                                       Alternative alternative = Alternative::TwoSided);

/// @brief Siegel–Tukey two-sample scale test.
///
/// Scores increase inward, assigning the lowest observation first, followed by two
/// from the high end, two from the low end, and so on.
/// An odd-sized pool retains its middle observation and assigns the final score.
/// Averages scores across tied observations. Uses a conditional-score normal
/// approximation without continuity correction. A scale interpretation assumes
/// similarly shaped distributions with a common center. Zero score variance is rejected.
/// @param x First nonempty finite sample.
/// @param y Second nonempty finite independent sample.
/// @param alternative TwoSided (default), Less for smaller or Greater for larger first-sample scale.
/// @return First-sample score sum and approximate p-value.
/// @throws std::invalid_argument For invalid inputs or degenerate reference variance where noted.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto result = dm::stats::siegel_tukey_test({0, 1, 9, 10}, {3, 4, 6, 7});
/// @endcode
/// @par References
/// - https://itl.nist.gov/div898/software/dataplot/refman1/auxillar/siegel.htm
/// @see ansari_bradley_test()
HypothesisTestResult siegel_tukey_test(const std::vector<double>& x,
                                       const std::vector<double>& y,
                                       Alternative alternative = Alternative::TwoSided);

/// @brief Ansari–Bradley two-sample scale test.
///
/// Scores increase symmetrically inward as 1, 2, 3, … on each side.
/// Averages scores across tied observations. Uses a conditional-score normal
/// approximation without continuity correction. A scale interpretation assumes
/// similarly shaped distributions with a common center. Zero score variance is rejected.
/// @param x First nonempty finite sample.
/// @param y Second nonempty finite independent sample.
/// @param alternative TwoSided (default), Less for smaller or Greater for larger first-sample scale.
/// @return First-sample score sum and approximate p-value.
/// @throws std::invalid_argument For invalid inputs or degenerate reference variance where noted.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto result = dm::stats::ansari_bradley_test({0, 1, 9, 10}, {3, 4, 6, 7});
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/ansari.test.html
/// @see siegel_tukey_test()
HypothesisTestResult ansari_bradley_test(const std::vector<double>& x,
                                         const std::vector<double>& y,
                                         Alternative alternative = Alternative::TwoSided);

/// @brief Kendall’s concordance coefficient with a chi-square test.
///
/// Rows are judges or blocks and columns are the same objects in every row.
/// The tie correction and chi-square statistic are those of the Friedman test.
/// @param blocks At least two finite rows with the same two or more columns.
/// @return Friedman chi-square result with Kendall W in estimate1.
/// @throws std::invalid_argument For invalid inputs or degenerate reference variance where noted.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto w = dm::stats::kendall_w_test({{1, 2, 3}, {2, 4, 5}});
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/friedman.test.html
/// @see friedman_test(), kendall_tau()
HypothesisTestResult kendall_w_test(const std::vector<std::vector<double>>& blocks);

/// @brief Page’s rank test for a prespecified increasing trend across treatments.
///
/// Columns are in the prespecified order. Ranks within each row use average ties.
/// Uses a normal approximation with variance conditional on the observed ranks.
/// Rows are independent. Zero rank variance is rejected.
/// @param blocks At least two finite rows with the same two or more columns.
/// @return Page L, block count in parameter1, treatment count in parameter2, and one-sided p-value.
/// @throws std::invalid_argument For invalid inputs or degenerate reference variance where noted.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto trend = dm::stats::page_trend_test({{1, 2, 3}, {2, 4, 5}});
/// @endcode
/// @par References
/// - https://doi.org/10.1080/01621459.1963.10500843
/// @see friedman_test(), kendall_w_test()
HypothesisTestResult page_trend_test(const std::vector<std::vector<double>>& blocks);

/// @brief Cochran’s Q test of equal success probabilities for matched binary outcomes.
///
/// Rows are subjects or blocks, and columns are repeated conditions. Uses the
/// tie-corrected Friedman statistic for binary inputs, with chi-square calibration.
/// At least one row must contain both 0 and 1.
/// @param blocks At least two independent rows with the same two or more binary columns.
/// @return Cochran Q, df in parameter1, row count in parameter2, and approximate p-value.
/// @throws std::invalid_argument For invalid inputs or degenerate reference variance where noted.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto q = dm::stats::cochran_q_test({{0, 1, 1}, {0, 0, 1}, {1, 1, 1}});
/// @endcode
/// @par References
/// - https://doi.org/10.1093/biomet/37.3-4.256
/// @see mcnemar_test(), friedman_test()
HypothesisTestResult cochran_q_test(const std::vector<std::vector<int>>& blocks);

/// @brief Exact McNemar test of paired binary outcomes using discordant counts.
///
/// Conditions on the total discordant count and uses the binomial reference with
/// probability one-half. With no discordant pairs, p is one and the estimate is zero.
/// @param first_only Number of successes only in the first condition.
/// @param second_only Number of successes only in the second condition. Sum must fit size_t.
/// @param alternative TwoSided (default), Less or Greater for the first condition’s success probability.
/// @return First-only count, discordant total in parameter1, first-only fraction among discordant pairs in estimate1, and p-value.
/// @throws std::invalid_argument For invalid inputs or degenerate reference variance where noted.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto m = dm::stats::mcnemar_test(12, 3);
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/mcnemar.test.html
/// @see cochran_q_test()
HypothesisTestResult mcnemar_test(std::size_t first_only,
                                  std::size_t second_only,
                                  Alternative alternative = Alternative::TwoSided);

/// @brief Exact McNemar test directly on paired binary observations.
/// @param first Nonempty vector of zeros and ones in the first condition.
/// @param second Binary observations from the second condition in matching order.
/// @param alternative TwoSided (default), Less or Greater for the first condition.
/// @return Result of the discordant-count overload.
/// @throws std::invalid_argument If the data are empty, sizes differ, values are not binary or the alternative is invalid.
/// @par Example
/// @code
/// const std::vector<int> first{1, 1, 0}, second{0, 0, 1};
/// const auto result = dm::stats::mcnemar_test(first, second);
/// @endcode
/// @see mcnemar_test(std::size_t, std::size_t, Alternative), cochran_q_test()
HypothesisTestResult mcnemar_test(const std::vector<int>& first, const std::vector<int>& second,
                                 Alternative alternative = Alternative::TwoSided);

/// @brief Cochran–Mantel–Haenszel conditional association test for stratified 2×2 tables.
///
/// Uses the asymptotic chi-square reference, with optional continuity correction.
/// Zero-variance strata contribute no score information. At least one informative
/// stratum is required. The common-odds-ratio interpretation assumes homogeneity.
/// @param strata Nonempty sequence of row-major 2×2 counts, in a, b, c, d order.
/// @param correct Apply the 0.5 continuity correction with a floor at zero, default true.
/// @return Chi-square with one df, p-value, and Mantel–Haenszel odds ratio in estimate1 (infinite for a zero denominator).
/// @throws std::invalid_argument For invalid inputs or degenerate reference variance where noted.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto mh = dm::stats::mantel_haenszel_test({{8, 2, 4, 6}, {5, 3, 2, 6}});
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/mantelhaen.test.html
/// @see fisher_exact_test_2x2()
HypothesisTestResult mantel_haenszel_test(const std::vector<std::array<std::size_t, 4>>& strata,
                                          bool correct = true);

} // namespace datamunge::stats
