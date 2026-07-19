#pragma once

#include <datamunge/stats/hypothesis_test_result.hpp>

#include <cstddef>
#include <vector>

namespace datamunge::stats {

/// @brief One-way ANOVA F-test across k >= 2 independent groups, testing whether all groups
///        share the same mean. @p values is every group's observations concatenated in
///        order, and @p group_sizes gives each group's length (must sum to values.size()).
///        statistic = F, parameter1 = between-groups df, parameter2 = within-groups df.
HypothesisTestResult one_way_anova(const std::vector<double>& values, const std::vector<std::size_t>& group_sizes);

/// @brief Kruskal-Wallis rank-sum test: a non-parametric analog of one-way ANOVA across
///        k >= 2 independent groups, testing whether they share the same distribution.
///        Same (values, group_sizes) layout as one_way_anova(). Uses the chi-squared
///        approximation to the H statistic with a tie correction; statistic = H,
///        parameter1 = df = k - 1.
HypothesisTestResult kruskal_wallis_test(const std::vector<double>& values, const std::vector<std::size_t>& group_sizes);

} // namespace datamunge::stats
