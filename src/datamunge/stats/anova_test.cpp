/// @file anova_test.cpp
/// @brief Classical anova test statistics, domain checks and stable probabilities.
#include <datamunge/random/distributions.hpp>
#include <datamunge/stats/anova_test.hpp>
#include <datamunge/stats/detail/ranking.hpp>
#include <datamunge/stats/detail/test_common.hpp>
#include <datamunge/stats/tukey_hsd.hpp>
#include <limits>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <string>

namespace datamunge::stats {

/// @brief Shared finite-sample and concatenated-group validation.
namespace {

/// @brief Requires nonempty groups whose counts exactly partition finite values.
/// @param values Concatenated observations.
/// @param group_sizes Positive group counts.
/// @param fn Caller label for errors.
void validate_groups(
    const std::vector<double>&      values,
    const std::vector<std::size_t>& group_sizes,
    const char*                     fn) {
  if (group_sizes.size() < 2)
    throw std::invalid_argument(std::string(fn) + ": need at least 2 groups");
  detail::check_sample(values, 2);
  std::size_t total = 0;
  for (auto n : group_sizes) {
    if (n == 0 || n > values.size() - total)
      throw std::invalid_argument(std::string(fn) + ": invalid group size");
    total += n;
  }
  if (total != values.size())
    throw std::invalid_argument(std::string(fn) + ": group sizes must sum to observations");
}

/// @brief Group moments used in unequal-variance comparisons.
struct GroupMoments {
  long double mean;          ///< Sample mean.
  long double mean_variance; ///< Unbiased sample variance divided by group size.
  long double df;            ///< Group size minus one.
};

/// @brief Computes centered, widened moments for nonconstant independent groups.
/// @param values Concatenated observations.
/// @param group_sizes Counts partitioning values.
/// @param fn Caller label for errors.
/// @return One summary per group in input order.
/// @throws std::invalid_argument Malformed groups, nonfinite values or nonpositive mean variance.
std::vector<GroupMoments> group_moments(const std::vector<double>& values,
                                        const std::vector<std::size_t>& group_sizes,
                                        const char* fn) {
  validate_groups(values, group_sizes, fn);
  std::vector<GroupMoments> groups;
  groups.reserve(group_sizes.size());
  std::size_t offset = 0;
  for (const auto size : group_sizes) {
    if (size < 2)
      throw std::invalid_argument(std::string(fn) + ": each group needs at least two observations");
    const long double origin = values[offset];
    long double mean = 0, m2 = 0;
    for (std::size_t i = 0; i < size; ++i) {
      const long double centered = values[offset + i] - origin;
      const long double delta = centered - mean;
      mean += delta / (i + 1);
      m2 += delta * (centered - mean);
    }
    const auto df = static_cast<long double>(size - 1);
    const auto variance = m2 / df / size;
    if (!(variance > 0) || !std::isfinite(variance))
      throw std::invalid_argument(std::string(fn) + ": each group needs positive finite variance");
    groups.push_back({origin + mean, variance, df});
    offset += size;
  }
  return groups;
}

/// @brief Narrows a computed result while checking representability.
/// @param value Widened result.
/// @return Finite double result.
/// @throws std::invalid_argument Nonfinite or out-of-range result.
double finite_result(long double value) {
  if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<double>::max())
    throw std::invalid_argument("unequal-variance comparison: result overflow");
  return static_cast<double>(value);
}

/// @brief Computes absolute deviations from group means or medians.
/// @param values Concatenated finite observations.
/// @param group_sizes Group counts, each at least two.
/// @param median_center Whether to use group medians.
/// @return Deviations in original group and row order.
/// @throws std::invalid_argument Invalid groups, nonfinite input or out-of-range deviation.
std::vector<double> absolute_group_deviations(const std::vector<double>& values,
                                              const std::vector<std::size_t>& group_sizes,
                                              bool median_center) {
  validate_groups(values, group_sizes, "group scale test");
  std::vector<double> deviations;
  deviations.reserve(values.size());
  auto first = values.begin();
  for (const auto size : group_sizes) {
    if (size < 2)
      throw std::invalid_argument("group scale test: each group needs two observations");
    const auto last = first + size;
    long double center = 0;
    if (median_center) {
      std::vector<double> sorted(first, last);
      std::sort(sorted.begin(), sorted.end());
      center = std::midpoint(static_cast<long double>(sorted[(size - 1) / 2]),
                             static_cast<long double>(sorted[size / 2]));
    } else {
      const long double origin = *first;
      for (auto it = first; it != last; ++it)
        center += (*it - origin) / size;
      center += origin;
    }
    for (auto it = first; it != last; ++it)
      deviations.push_back(finite_result(std::abs(*it - center)));
    first = last;
  }
  return deviations;
}

} // namespace

HypothesisTestResult welch_anova(const std::vector<double>& values,
                               const std::vector<std::size_t>& group_sizes) {
  const auto groups = group_moments(values, group_sizes, "welch_anova");
  const auto k = static_cast<long double>(groups.size());
  const auto origin = groups.front().mean;
  long double weight_sum = 0, weighted_sum = 0;
  for (const auto& group : groups) {
    weight_sum += 1 / group.mean_variance;
    weighted_sum += (group.mean - origin) / group.mean_variance;
  }
  const auto center = weighted_sum / weight_sum;
  long double between = 0, correction = 0;
  for (const auto& group : groups) {
    const auto residual = (group.mean - origin) - center;
    const auto complement = 1 - 1 / (group.mean_variance * weight_sum);
    between += residual * residual / group.mean_variance;
    correction += complement * complement / group.df;
  }
  HypothesisTestResult result;
  result.parameter1 = static_cast<double>(k - 1);
  result.parameter2 = finite_result((k * k - 1) / (3 * correction));
  result.statistic = finite_result(between / (k - 1) / (1 + 2 * (k - 2) * correction / (k * k - 1)));
  result.p_value = detail::f_test_survival(result.statistic, result.parameter1, result.parameter2);
  result.method = "Welch one-way analysis of means";
  return result;
}

std::vector<GamesHowellComparison> games_howell(const std::vector<double>& values,
                                              const std::vector<std::size_t>& group_sizes,
                                              double confidence_level) {
  detail::check_confidence(confidence_level);
  const auto groups = group_moments(values, group_sizes, "games_howell");
  std::vector<GamesHowellComparison> rows;
  for (std::size_t j = 1; j < groups.size(); ++j) {
    for (std::size_t i = 0; i < j; ++i) {
      const auto variance = groups[j].mean_variance + groups[i].mean_variance;
      const auto a = groups[j].mean_variance / variance;
      const auto b = groups[i].mean_variance / variance;
      const auto df = static_cast<double>(1 / (a * a / groups[j].df + b * b / groups[i].df));
      const auto difference = groups[j].mean - groups[i].mean;
      const auto scale = std::sqrt(variance / 2);
      const auto critical = studentized_range_quantile(confidence_level, groups.size(), df);
      const auto width = critical * scale;
      const auto q = static_cast<double>(std::abs(difference) / scale);
      rows.push_back({j, i, finite_result(difference), df, finite_result(difference - width),
                      finite_result(difference + width),
                      studentized_range_cdf(q, groups.size(), df, false)});
    }
  }
  return rows;
}

HypothesisTestResult levene_test(const std::vector<double>& values,
                                const std::vector<std::size_t>& group_sizes, bool median_center) {
  auto result = one_way_anova(absolute_group_deviations(values, group_sizes, median_center), group_sizes);
  result.method = median_center ? "Brown-Forsythe test of equal variances" : "Levene test of equal variances";
  return result;
}

HypothesisTestResult bartlett_test(const std::vector<double>& values,
                                  const std::vector<std::size_t>& group_sizes) {
  const auto groups = group_moments(values, group_sizes, "bartlett_test");
  long double pooled_sum = 0, reciprocal_df = 0;
  for (const auto& group : groups) {
    pooled_sum += group.df * group.mean_variance * (group.df + 1);
    reciprocal_df += 1 / group.df;
  }
  const long double residual_df = values.size() - groups.size();
  const auto log_pooled = std::log(pooled_sum / residual_df);
  long double numerator = 0;
  for (const auto& group : groups)
    numerator += group.df * (log_pooled - std::log(group.mean_variance * (group.df + 1)));
  HypothesisTestResult result;
  result.parameter1 = static_cast<double>(groups.size() - 1);
  result.statistic = std::max(0.0, finite_result(numerator / (1 + (reciprocal_df - 1 / residual_df) /
                                                                  (3 * result.parameter1))));
  result.p_value = detail::chi_squared_test_survival(result.statistic, result.parameter1);
  result.method = "Bartlett test of homogeneity of variances";
  return result;
}

HypothesisTestResult fligner_killeen_test(const std::vector<double>& values,
                                         const std::vector<std::size_t>& group_sizes) {
  auto scores = detail::rank_with_ties(absolute_group_deviations(values, group_sizes, true));
  const auto n = static_cast<double>(scores.size());
  for (auto& score : scores)
    score = random::normal_quantile(0.5 + score / (2 * (n + 1)));
  const auto mean = std::accumulate(scores.begin(), scores.end(), 0.0) / n;
  double within_total = 0, between = 0;
  auto first = scores.begin();
  for (const auto size : group_sizes) {
    const auto last = first + size;
    double sum = 0;
    for (auto it = first; it != last; ++it) {
      const auto centered = *it - mean;
      sum += centered;
      within_total += centered * centered;
    }
    between += sum * sum / static_cast<double>(size);
    first = last;
  }
  if (!(within_total > 0) || std::all_of(scores.begin(), scores.end(),
      [&scores](double score) { return score == scores.front(); }))
    throw std::invalid_argument("fligner_killeen_test: all absolute deviations have the same rank");
  HypothesisTestResult result;
  result.parameter1 = static_cast<double>(group_sizes.size() - 1);
  result.statistic = between * (n - 1) / within_total;
  result.p_value = detail::chi_squared_test_survival(result.statistic, result.parameter1);
  result.method = "Fligner-Killeen test of homogeneity of variances";
  return result;
}

HypothesisTestResult one_way_anova(
    const std::vector<double>&      values,
    const std::vector<std::size_t>& group_sizes) {
  validate_groups(values, group_sizes, "one_way_anova");

  const auto n_total    = static_cast<double>(values.size());
  double     grand_mean = 0.0;
  for (const double v : values)
    grand_mean += v;
  grand_mean /= n_total;

  double      ss_between = 0.0, ss_within = 0.0;
  std::size_t offset = 0;
  for (const std::size_t g : group_sizes) {
    double group_mean = 0.0;
    for (std::size_t i = 0; i < g; ++i)
      group_mean += values[offset + i];
    group_mean /= static_cast<double>(g);
    for (std::size_t i = 0; i < g; ++i) {
      const double d = values[offset + i] - group_mean;
      ss_within += d * d;
    }
    const double dg = group_mean - grand_mean;
    ss_between += static_cast<double>(g) * dg * dg;
    offset += g;
  }

  const double df1 = static_cast<double>(group_sizes.size() - 1);
  const double df2 = n_total - static_cast<double>(group_sizes.size());
  if (df2 <= 0 || !(ss_within > 0) || !std::isfinite(ss_between) || !std::isfinite(ss_within))
    throw std::invalid_argument(
        "one_way_anova: positive finite within-group variation and residual df required");
  const double f_stat = (ss_between / df1) / (ss_within / df2);

  HypothesisTestResult result;
  result.statistic  = f_stat;
  result.parameter1 = df1;
  result.parameter2 = df2;
  result.p_value    = detail::f_test_survival(f_stat, df1, df2);
  result.method     = "One-way analysis of means";
  return result;
}

HypothesisTestResult kruskal_wallis_test(
    const std::vector<double>&      values,
    const std::vector<std::size_t>& group_sizes) {
  validate_groups(values, group_sizes, "kruskal_wallis_test");

  const auto ranks   = detail::rank_with_ties(values);
  const auto n_total = static_cast<double>(values.size());

  double      h      = 0.0;
  std::size_t offset = 0;
  for (const std::size_t g : group_sizes) {
    double rank_sum = 0.0;
    for (std::size_t i = 0; i < g; ++i)
      rank_sum += ranks[offset + i];
    h += rank_sum * rank_sum / static_cast<double>(g);
    offset += g;
  }
  h = 12.0 / (n_total * (n_total + 1.0)) * h - 3.0 * (n_total + 1.0);

  const double tie_sum    = detail::tie_correction_sum(values);
  const double correction = 1.0 - tie_sum / (n_total * n_total * n_total - n_total);
  if (!(correction > 0))
    throw std::invalid_argument("kruskal_wallis_test: all values are tied");
  h = std::max(0.0, h / correction);

  const double df = static_cast<double>(group_sizes.size() - 1);

  HypothesisTestResult result;
  result.statistic  = h;
  result.parameter1 = df;
  result.p_value    = detail::chi_squared_test_survival(h, df);
  result.method     = "Kruskal-Wallis rank sum test";
  return result;
}

std::vector<DunnComparison> dunn_test(const std::vector<double>& values,
                                      const std::vector<std::size_t>& group_sizes,
                                      PAdjustMethod adjustment) {
  validate_groups(values, group_sizes, "dunn_test");
  const auto ranks = detail::rank_with_ties(values);
  const long double center = (static_cast<long double>(ranks.size()) + 1) / 2;
  long double sum_squares = 0;
  std::vector<long double> group_means;
  group_means.reserve(group_sizes.size());
  std::size_t offset = 0;
  for (const auto size : group_sizes) {
    long double group_sum = 0;
    for (std::size_t i = 0; i < size; ++i) {
      const auto centered_rank = ranks[offset + i] - center;
      group_sum += centered_rank;
      sum_squares += centered_rank * centered_rank;
    }
    group_means.push_back(group_sum / size);
    offset += size;
  }
  if (sum_squares == 0)
    throw std::invalid_argument("dunn_test: all observations are tied");
  const auto variance = sum_squares / (ranks.size() - 1);
  std::vector<DunnComparison> rows;
  std::vector<double> raw;
  for (std::size_t j = 1; j < group_sizes.size(); ++j) {
    for (std::size_t i = 0; i < j; ++i) {
      const auto difference = group_means[j] - group_means[i];
      const auto error = std::sqrt(variance * (1.0L / group_sizes[i] + 1.0L / group_sizes[j]));
      const auto z = static_cast<double>(difference / error);
      const auto p = detail::normal_test_p(z, Alternative::TwoSided);
      rows.push_back({j, i, static_cast<double>(difference), z, p, p});
      raw.push_back(p);
    }
  }
  const auto adjusted = p_adjust(raw, adjustment);
  for (std::size_t i = 0; i < rows.size(); ++i)
    rows[i].p_adjusted = adjusted[i];
  return rows;
}

} // namespace datamunge::stats
