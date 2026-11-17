#pragma once

/// @file pairwise_test.hpp
/// @brief Pairwise group comparisons with multiple-testing adjustment.

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/stats/hypothesis_test_result.hpp>
#include <datamunge/stats/p_adjust.hpp>
#include <string>
#include <vector>

namespace datamunge::stats {

/// @brief All pairwise two-sample t-tests between groups with adjusted p-values, like R
///        pairwise.t.test().
///
/// Groups are the distinct labels in @p groups, sorted lexicographically (R's default factor
/// level order). Rows follow R's order and sign: for levels a < b < c they are b-a, c-a, c-b,
/// so group1 is the later level and every column describes group1 - group2. With @p pool_sd true (R's default) every comparison uses the
/// pooled within-group standard deviation of *all* groups and N-k degrees of freedom, as in a
/// one-way ANOVA; with it false each pair is an independent Welch test, as
/// t_test_two_sample(). The raw p-values are then adjusted together with p_adjust().
///
/// A one-sided @p alternative is read in the group1 - group2 direction, as R's is. Inputs are
/// unchanged; O(N log k + k^2) work.
/// @param values Finite observations, one per element of @p groups.
/// @param groups Group label of each observation; at least two distinct labels.
/// @param method p-value adjustment across all k(k-1)/2 comparisons, default Holm (R's default).
/// @param pool_sd True (default) pools the standard deviation across all groups; false runs
/// Welch tests per pair.
/// @param alternative TwoSided (default), Less or Greater, for mean(group1) - mean(group2).
/// @return A DataFrame with one row per pair and columns group1, group2 (strings), estimate
/// (mean difference group1 - group2), statistic (t), df, p_value (unadjusted) and p_adjusted.
/// @throws std::invalid_argument Length mismatch, nonfinite values, fewer than two groups, no
/// residual df or zero pooled variance (pool_sd), a group with fewer than two observations or
/// zero standard error (Welch), or an invalid alternative.
/// @par Example
/// @code
/// const auto pw = dm::stats::pairwise_t_test({1, 2, 3, 4, 5, 6, 7, 8, 9},
///                                            {"a", "a", "a", "b", "b", "b", "c", "c", "c"});
/// std::print("{} {} {:.0f}\n", pw.nrows(), pw.string_at("group1", 0), pw.double_at("df", 0));
/// // 3 b 6
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/pairwise.t.test.html
/// - S. Holm (1979). "A Simple Sequentially Rejective Multiple Test Procedure". Scandinavian
///   Journal of Statistics 6(2), 65-70. https://www.jstor.org/stable/4615733
/// @see TukeyHSD, t_test_two_sample(), p_adjust(), one_way_anova()
dstruct::DataFrame pairwise_t_test(
    const std::vector<double>&      values,
    const std::vector<std::string>& groups,
    PAdjustMethod                   method      = PAdjustMethod::Holm,
    bool                            pool_sd     = true,
    Alternative                     alternative = Alternative::TwoSided);

/// @brief pairwise_t_test() on two columns of a DataFrame, like R
///        `with(df, pairwise.t.test(value, group))`.
///
/// Rows where either column is null are dropped first, as R drops NA. Otherwise identical to
/// the vector overload.
/// @param data Frame holding both columns; unchanged.
/// @param value_column Numeric column of observations.
/// @param group_column String column of group labels.
/// @param method p-value adjustment, default Holm.
/// @param pool_sd True (default) pools the standard deviation across groups; false uses Welch.
/// @param alternative TwoSided (default), Less or Greater, for mean(group1) - mean(group2).
/// @return As the vector overload.
/// @throws std::invalid_argument A missing column, a non-numeric @p value_column, a non-string
/// @p group_column, or any error of the vector overload.
/// @par Example
/// @code
/// const auto pw = dm::stats::pairwise_t_test(dm::datasets::iris(), "Sepal.Length", "Species");
/// std::print("{}-{}\n", pw.string_at("group1", 2), pw.string_at("group2", 2));
/// // virginica-versicolor
/// @endcode
/// @see pairwise_t_test(const std::vector<double>&, const std::vector<std::string>&, PAdjustMethod, bool, Alternative)
dstruct::DataFrame pairwise_t_test(
    const dstruct::DataFrame& data,
    const std::string&        value_column,
    const std::string&        group_column,
    PAdjustMethod             method      = PAdjustMethod::Holm,
    bool                      pool_sd     = true,
    Alternative               alternative = Alternative::TwoSided);

} // namespace datamunge::stats
