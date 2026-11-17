/// @file pairwise_test.cpp
/// @brief Pairwise t-tests over every pair of groups, with p-value adjustment.
#include <algorithm>
#include <cmath>
#include <datamunge/stats/detail/test_common.hpp>
#include <datamunge/stats/pairwise_test.hpp>
#include <datamunge/stats/t_test.hpp>
#include <map>
#include <stdexcept>

namespace datamunge::stats {

namespace {

/// @brief The comparisons for groups already split by label (see DataFrame::split()).
/// @param by Group label -> that group's values, in level order.
/// @param method p-value adjustment.
/// @param pool_sd Pool the standard deviation across all groups.
/// @param alternative Tail for group1 - group2.
/// @return The result frame documented on pairwise_t_test().
/// @throws std::invalid_argument As pairwise_t_test().
dstruct::DataFrame pairwise_t_test_groups(std::map<std::string, std::vector<double>> by, PAdjustMethod method,
                                          bool pool_sd, Alternative alternative) {
  detail::check_alternative(alternative);
  if (by.size() < 2) throw std::invalid_argument("pairwise_t_test: need at least two groups");
  std::size_t total = 0;
  for (const auto& [name, sample] : by) {
    detail::check_sample(sample);
    total += sample.size();
  }

  std::vector<std::string>         names;
  std::vector<std::vector<double>> samples;
  for (auto& [name, sample] : by) {
    names.push_back(name);
    samples.push_back(std::move(sample));
  }
  const std::size_t k = samples.size();

  // Group means, plus the pooled within-group variance over all k groups when pooling.
  std::vector<long double> means(k);
  long double              within_ss = 0;
  for (std::size_t g = 0; g < k; ++g) {
    for (double v : samples[g]) means[g] += static_cast<long double>(v) / samples[g].size();
    for (double v : samples[g]) within_ss += (v - means[g]) * (v - means[g]);
  }
  const double pooled_df  = static_cast<double>(total) - static_cast<double>(k);
  double       pooled_var = 0;
  if (pool_sd) {
    if (!(pooled_df > 0)) throw std::invalid_argument("pairwise_t_test: no residual degrees of freedom");
    pooled_var = static_cast<double>(within_ss / pooled_df);
    if (!(pooled_var > 0) || !std::isfinite(pooled_var))
      throw std::invalid_argument("pairwise_t_test: pooled variance must be positive and finite");
  }

  std::vector<std::string> group1, group2;
  std::vector<double>      estimate, statistic, df, p_value;
  // R's order and sign: for levels a < b < c the rows are b-a, c-a, c-b.
  for (std::size_t i = 0; i < k; ++i)
    for (std::size_t j = i + 1; j < k; ++j) {
      group1.push_back(names[j]);
      group2.push_back(names[i]);
      if (pool_sd) {
        const double diff = static_cast<double>(means[j] - means[i]);
        const double se   = std::sqrt(pooled_var * (1.0 / samples[i].size() + 1.0 / samples[j].size()));
        const double t    = diff / se;
        estimate.push_back(diff);
        statistic.push_back(t);
        df.push_back(pooled_df);
        p_value.push_back(detail::student_test_p(t, pooled_df, alternative));
      } else {
        const auto r = t_test_two_sample(samples[j], samples[i], false, alternative);
        estimate.push_back(r.estimate1 - r.estimate2);
        statistic.push_back(r.statistic);
        df.push_back(r.parameter1);
        p_value.push_back(r.p_value);
      }
    }

  dstruct::DataFrame out;
  out.add_column("group1", group1);
  out.add_column("group2", group2);
  out.add_column("estimate", estimate);
  out.add_column("statistic", statistic);
  out.add_column("df", df);
  const auto adjusted = p_adjust(p_value, method);
  out.add_column("p_value", p_value);
  out.add_column("p_adjusted", adjusted);
  return out;
}

} // namespace

dstruct::DataFrame pairwise_t_test(
    const std::vector<double>&      values,
    const std::vector<std::string>& groups,
    PAdjustMethod                   method,
    bool                            pool_sd,
    Alternative                     alternative) {
  if (values.size() != groups.size())
    throw std::invalid_argument("pairwise_t_test: values and groups must have the same length");
  // std::map orders the levels lexicographically, like R's factor().
  std::map<std::string, std::vector<double>> by;
  for (std::size_t i = 0; i < values.size(); ++i) by[groups[i]].push_back(values[i]);
  return pairwise_t_test_groups(std::move(by), method, pool_sd, alternative);
}

dstruct::DataFrame pairwise_t_test(
    const dstruct::DataFrame& data,
    const std::string&        value_column,
    const std::string&        group_column,
    PAdjustMethod             method,
    bool                      pool_sd,
    Alternative               alternative) {
  return pairwise_t_test_groups(data.split(value_column, group_column), method, pool_sd, alternative);
}

} // namespace datamunge::stats
