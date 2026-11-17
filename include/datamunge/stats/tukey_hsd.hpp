#pragma once

/// @file tukey_hsd.hpp
/// @brief Tukey's honest significant differences and the studentized range distribution.

#include <cstddef>
#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/plot/plot.hpp>
#include <datamunge/stats/formula.hpp>
#include <map>
#include <string>
#include <vector>

namespace datamunge::stats {

/// @brief CDF of the studentized range distribution, like R's `ptukey(q, nmeans, df)`.
///
/// P(Q <= q) for the range of @p nmeans independent standard normals divided by an independent
/// sqrt(chi-square(df) / df). Uses the Gauss-Legendre quadrature of Copenhaver and Holland
/// (1988) as in R's ptukey.c (nranges = 1), accurate to roughly 1e-8; df > 25000 uses the
/// infinite-df limit. Pure function, no shared state.
/// @param q Quantile; q <= 0 gives 0 and +infinity gives 1.
/// @param nmeans Number of means k, at least 2.
/// @param df Error degrees of freedom, at least 2.
/// @param lower_tail True (default) returns P(Q <= q); false returns P(Q > q).
/// @return Probability in [0, 1].
/// @throws std::invalid_argument nmeans < 2, df < 2 or NaN @p q.
/// @par Example
/// @code
/// std::print("{:.4f}\n", dm::stats::studentized_range_cdf(3.5, 3, 20));   // 0.9441
/// @endcode
/// @par References
/// - M. W. Copenhaver, B. Holland (1988). "Computation of the distribution of the maximum
///   studentized range statistic with application to multiple significance testing of simple
///   effects". Journal of Statistical Computation and Simulation 30(1), 1-15.
///   doi:10.1080/00949658808811082
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/Tukey.html
/// @see studentized_range_quantile(), TukeyHSD
double studentized_range_cdf(double q, std::size_t nmeans, double df, bool lower_tail = true);

/// @brief Quantile of the studentized range distribution, like R's `qtukey(p, nmeans, df)`.
///
/// Inverts studentized_range_cdf() by bracketing and bisection to about 1e-9 in q.
/// @param p Lower-tail probability strictly in (0, 1).
/// @param nmeans Number of means k, at least 2.
/// @param df Error degrees of freedom, at least 2.
/// @return q with P(Q <= q) = p.
/// @throws std::invalid_argument @p p outside (0, 1), nmeans < 2 or df < 2.
/// @par Example
/// @code
/// std::print("{:.3f}\n", dm::stats::studentized_range_quantile(0.95, 3, 20));   // 3.578
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/Tukey.html
/// @see studentized_range_cdf()
double studentized_range_quantile(double p, std::size_t nmeans, double df);

/// @brief Tukey's honest significant differences for a one-way layout, like R's
///        `TukeyHSD(aov(value ~ group))`.
///
/// Compares every pair of group means with family-wise confidence intervals and adjusted
/// p-values from the studentized range distribution, using the pooled within-group mean square
/// error (MSE) on N-k degrees of freedom; unequal group sizes use the Tukey-Kramer standard
/// error sqrt(MSE/2 * (1/n_i + 1/n_j)). Assumes independent normal observations with a common
/// variance. Groups are the distinct labels sorted lexicographically (R's default factor
/// order); comparisons follow R's order and sign: for levels a < b < c the rows are b-a, c-a,
/// c-b, each estimating mean(group1) - mean(group2). O(N log k + k^2) work.
///
/// @par Example
/// @code
/// const dm::stats::TukeyHSD hsd(dm::datasets::iris(), "Sepal.Length ~ Species");
/// std::print("{} {:.3f}\n", hsd.comparisons()[0].label(), hsd.comparisons()[0].diff);
/// // versicolor-setosa 0.930
/// @endcode
/// @par References
/// - J. W. Tukey (1949). "Comparing Individual Means in the Analysis of Variance". Biometrics
///   5(2), 99-114. doi:10.2307/3001913
/// - C. Y. Kramer (1956). "Extension of Multiple Range Tests to Group Means with Unequal
///   Numbers of Replications". Biometrics 12(3), 307-310. doi:10.2307/3001469
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/TukeyHSD.html
/// @see pairwise_t_test(), one_way_anova(), studentized_range_cdf()
class TukeyHSD {
 public:
  /// @brief One pairwise comparison: group1 - group2 with its family-wise interval.
  struct Comparison {
    std::string group1;     ///< The later level; its mean is the minuend.
    std::string group2;     ///< The earlier level; its mean is the subtrahend.
    double      diff;       ///< mean(group1) - mean(group2).
    double      lower;      ///< Lower family-wise confidence bound for diff.
    double      upper;      ///< Upper family-wise confidence bound for diff.
    double      p_adjusted; ///< Tukey-adjusted p-value for diff == 0.

    /// @brief R's row name, "group1-group2".
    /// @return The label, e.g. "versicolor-setosa".
    [[nodiscard]] std::string label() const { return group1 + "-" + group2; }
  };

  /// @brief Runs Tukey's HSD on observations and their group labels.
  /// @param values Finite observations, one per element of @p groups; unchanged.
  /// @param groups Group label of each observation; at least two distinct labels.
  /// @param conf_level Family-wise confidence level strictly in (0, 1), default 0.95.
  /// @throws std::invalid_argument Length mismatch, nonfinite values, fewer than two groups,
  /// fewer than two residual df, zero within-group variance or an invalid @p conf_level.
  /// @par Example
  /// @code
  /// const dm::stats::TukeyHSD hsd({1, 2, 3, 4, 5, 6, 7, 8, 9},
  ///                               {"a", "a", "a", "b", "b", "b", "c", "c", "c"});
  /// std::print("{} {:.0f}\n", hsd.comparisons().size(), hsd.df());   // 3 6
  /// @endcode
  TukeyHSD(const std::vector<double>&      values,
           const std::vector<std::string>& groups,
           double                          conf_level = 0.95);

  /// @brief Runs Tukey's HSD on a numeric and a string column of a DataFrame.
  ///
  /// Groups come from DataFrame::split(), so rows where either column is null are dropped, as
  /// R's aov() drops NA.
  /// @param data Frame holding both columns; unchanged.
  /// @param value_column Numeric response column.
  /// @param group_column String grouping column.
  /// @param conf_level Family-wise confidence level, default 0.95.
  /// @throws std::out_of_range A missing column.
  /// @throws std::invalid_argument A mistyped column, or any error of the vector constructor.
  /// @par Example
  /// See the class example.
  TukeyHSD(const dstruct::DataFrame& data,
           const std::string&        value_column,
           const std::string&        group_column,
           double                    conf_level = 0.95);

  /// @brief Runs Tukey's HSD from a one-way formula, like R's
  ///        `TukeyHSD(aov(value ~ group, data))`.
  ///
  /// The formula must be `response ~ group` with a numeric response column and a single string
  /// grouping column; anything else on the right-hand side is rejected. Strings convert
  /// implicitly to Formula, so `TukeyHSD(iris, "Sepal.Length ~ Species")` works. Otherwise
  /// identical to the column-name constructor.
  /// @param data Frame holding both columns; unchanged.
  /// @param formula `value ~ group`.
  /// @param conf_level Family-wise confidence level, default 0.95.
  /// @throws std::out_of_range A missing column.
  /// @throws std::invalid_argument A right-hand side that is not a single column name, a
  /// mistyped column (the message suggests the swapped formula when the sides look reversed),
  /// or any error of the vector constructor.
  /// @par Example
  /// @code
  /// const dm::stats::TukeyHSD hsd(dm::datasets::iris(), "Sepal.Length ~ Species");
  /// std::print("{}\n", hsd.comparisons()[0].label());   // versicolor-setosa
  /// @endcode
  TukeyHSD(const dstruct::DataFrame& data, const Formula& formula, double conf_level = 0.95);

  /// @brief The comparisons, in R's order (see the class description).
  /// @return k(k-1)/2 comparisons.
  [[nodiscard]] const std::vector<Comparison>& comparisons() const noexcept { return comparisons_; }

  /// @brief The comparisons as a DataFrame, like R's `TukeyHSD(...)$group`.
  /// @return Columns comparison ("b-a"), group1, group2, diff, lwr, upr and p_adj.
  /// @par Example
  /// @code
  /// const dm::stats::TukeyHSD hsd(dm::datasets::iris(), "Sepal.Length", "Species");
  /// std::print("{}\n", hsd.frame().nrows());   // 3
  /// @endcode
  [[nodiscard]] dstruct::DataFrame frame() const;

  /// @brief Plots each difference with its family-wise interval, like R's
  ///        `plot(TukeyHSD(...))`.
  ///
  /// One horizontal interval per comparison (first comparison on top) with the estimate as a
  /// point and a reference line at zero; the left margin is widened to fit the labels. Intervals that exclude zero, i.e. pairs whose
  /// adjusted p-value is below 1 - conf_level, are drawn in @p significant; the rest in
  /// @p not_significant. A legend names both.
  /// @param significant Color of significantly different pairs.
  /// @param not_significant Color of the remaining pairs.
  /// @return A new plot; title, labels and size can be changed by chaining.
  /// @par Example
  /// @code
  /// const dm::stats::TukeyHSD hsd(dm::datasets::iris(), "Sepal.Length", "Species");
  /// hsd.plot().save("tukey.png");
  /// @endcode
  [[nodiscard]] graphics::RPlot plot(graphics::RGBA significant     = {220, 38, 38},
                                     graphics::RGBA not_significant = {100, 116, 139}) const;

  /// @brief The family-wise confidence level.
  /// @return The level passed to the constructor.
  [[nodiscard]] double conf_level() const noexcept { return conf_level_; }

  /// @brief Residual degrees of freedom N - k.
  /// @return The error df used by every comparison.
  [[nodiscard]] double df() const noexcept { return df_; }

  /// @brief Pooled within-group mean square error.
  /// @return Residual sum of squares / (N - k).
  [[nodiscard]] double mse() const noexcept { return mse_; }

 private:
  /// @brief Computes the comparisons; shared by both constructors.
  /// @param by Group label -> that group's finite values, as DataFrame::split() returns.
  /// @throws std::invalid_argument As the vector constructor.
  void fit(const std::map<std::string, std::vector<double>>& by);

  std::vector<Comparison> comparisons_; ///< Pairwise results in R's order.
  double                  conf_level_;  ///< Family-wise confidence level.
  double                  df_{};        ///< Residual degrees of freedom.
  double                  mse_{};       ///< Pooled within-group mean square.
};

} // namespace datamunge::stats
