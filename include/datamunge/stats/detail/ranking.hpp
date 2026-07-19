#pragma once

#include <vector>

namespace datamunge::stats::detail {

/// @brief Returns the 1-indexed rank of each element of @p x, with tied values assigned
///        their average rank (the standard "fractional ranking" used by Wilcoxon,
///        Kruskal-Wallis, and Spearman's rank correlation).
std::vector<double> rank_with_ties(const std::vector<double>& x);

/// @brief Sum, over every group of tied values in @p x, of (t^3 - t) where t is the group's
///        size. Used to correct the variance of rank-based test statistics for ties
///        (Wilcoxon rank-sum/signed-rank, Kruskal-Wallis). Zero when there are no ties.
double tie_correction_sum(const std::vector<double>& x);

} // namespace datamunge::stats::detail
