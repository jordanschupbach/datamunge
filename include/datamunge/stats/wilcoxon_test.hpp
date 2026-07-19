#pragma once

#include <datamunge/stats/hypothesis_test_result.hpp>

#include <vector>

namespace datamunge::stats {

/// @brief Wilcoxon signed-rank test: a non-parametric analog of the one-sample/paired
///        t-test, testing whether the distribution of @p x - @p mu is symmetric about zero.
///        For a paired test, pass the elementwise differences as @p x. Differences exactly
///        equal to @p mu are dropped (matching R's wilcox.test() default). Uses the normal
///        approximation with a continuity correction and a tie correction to the variance
///        (matching R's behavior once ties are present, which is the common case for real
///        data); no confidence interval is reported (has_conf_int stays false), matching the
///        default `conf.int = FALSE` behavior in R.
HypothesisTestResult wilcoxon_signed_rank_test(const std::vector<double>& x, double mu = 0.0,
                                                Alternative alternative = Alternative::TwoSided);

/// @brief Wilcoxon rank-sum test (equivalently, the Mann-Whitney U test): a non-parametric
///        analog of the two-sample t-test, testing whether @p x and @p y are drawn from
///        distributions with the same location. Uses the normal approximation with a
///        continuity correction and a tie correction to the variance.
HypothesisTestResult wilcoxon_rank_sum_test(const std::vector<double>& x, const std::vector<double>& y,
                                             Alternative alternative = Alternative::TwoSided);

} // namespace datamunge::stats
