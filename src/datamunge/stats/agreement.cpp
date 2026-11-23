/// @file agreement.cpp
/// @brief Chance correction, paired difference limits and concordance inference.

#include <datamunge/stats/agreement.hpp>
#include <datamunge/stats/detail/test_common.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Shared agreement domain checks and result conversion.
namespace {

/// @brief Applies chance correction using disagreement directly for stable small probabilities.
/// @param observed Observed disagreement.
/// @param expected Expected disagreement under the marginal model.
/// @return Coefficient and complementary agreement probabilities.
KappaAgreement chance_correct(long double observed, long double expected) {
  if (!(expected > 0))
    throw std::invalid_argument("kappa: expected disagreement must be positive");
  return {static_cast<double>(1 - observed / expected), static_cast<double>(1 - observed),
          static_cast<double>(1 - expected)};
}

/// @brief Validates two finite measurement vectors with matching lengths.
/// @param first First measurements.
/// @param second Second measurements.
/// @param minimum Minimum pair count.
void check_pairs(const std::vector<double>& first, const std::vector<double>& second, std::size_t minimum) {
  if (first.size() != second.size())
    throw std::invalid_argument("agreement: measurement vectors must have equal lengths");
  detail::check_sample(first, minimum);
  detail::check_sample(second, minimum);
}

/// @brief Converts a finite agreement estimate in measurement units.
/// @param value Statistic or interval endpoint.
/// @return The rounded double result.
double checked_result(long double value) {
  if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<double>::max())
    throw std::overflow_error("agreement: result exceeds finite double range");
  return static_cast<double>(value);
}

} // namespace

KappaAgreement cohen_kappa(const std::vector<std::vector<std::size_t>>& table, KappaWeights weights) {
  if (table.size() < 2 || (weights != KappaWeights::Unweighted &&
      weights != KappaWeights::Linear && weights != KappaWeights::Quadratic))
    throw std::invalid_argument("Cohen kappa: invalid categories or weight mode");
  const auto k = table.size();
  std::vector<long double> rows(k), columns(k);
  long double total = 0, observed = 0;
  /// @brief Returns the scaled ordinal distance for a pair of categories.
  const auto penalty = [&](std::size_t i, std::size_t j) {
    if (weights == KappaWeights::Unweighted)
      return static_cast<long double>(i != j);
    const auto distance = static_cast<long double>(std::max(i, j) - std::min(i, j)) / (k - 1);
    return weights == KappaWeights::Linear ? distance : distance * distance;
  };
  for (std::size_t i = 0; i < k; ++i) {
    if (table[i].size() != k)
      throw std::invalid_argument("Cohen kappa: count table must be square");
    for (std::size_t j = 0; j < k; ++j) {
      const auto count = table[i][j];
      total += count;
      rows[i] += count;
      columns[j] += count;
      observed += penalty(i, j) * count;
    }
  }
  if (total == 0)
    throw std::invalid_argument("Cohen kappa: count table is empty");
  long double expected = 0;
  for (std::size_t i = 0; i < k; ++i)
    for (std::size_t j = 0; j < k; ++j)
      expected += penalty(i, j) * (rows[i] / total) * (columns[j] / total);
  return chance_correct(observed / total, expected);
}

KappaAgreement fleiss_kappa(const std::vector<std::vector<std::size_t>>& ratings) {
  if (ratings.empty() || ratings.front().size() < 2)
    throw std::invalid_argument("Fleiss kappa: need items and at least two categories");
  std::vector<long double> columns(ratings.front().size());
  std::size_t panel = 0;
  long double observed = 0;
  for (const auto& row : ratings) {
    if (row.size() != columns.size())
      throw std::invalid_argument("Fleiss kappa: ragged rating table");
    std::size_t size = 0;
    long double pairs = 0;
    for (std::size_t j = 0; j < row.size(); ++j) {
      if (row[j] > std::numeric_limits<std::size_t>::max() - size)
        throw std::invalid_argument("Fleiss kappa: panel size exceeds index range");
      pairs += static_cast<long double>(size) * row[j];
      size += row[j];
      columns[j] += row[j];
    }
    if (panel == 0)
      panel = size;
    if (size < 2 || size != panel)
      throw std::invalid_argument("Fleiss kappa: items require equal panels of at least two raters");
    observed += 2 * pairs / (static_cast<long double>(panel) * (panel - 1));
  }
  const auto total = static_cast<long double>(panel) * ratings.size();
  long double expected = 0, prior = 0;
  for (const auto count : columns) {
    const auto share = count / total;
    expected += 2 * share * prior;
    prior += share;
  }
  return chance_correct(observed / ratings.size(), expected);
}

BlandAltmanAgreement bland_altman(const std::vector<double>& first, const std::vector<double>& second,
                                  double population_coverage, double confidence_level) {
  check_pairs(first, second, 2);
  detail::check_confidence(population_coverage);
  detail::check_confidence(confidence_level);
  const long double n = first.size();
  const long double origin = static_cast<long double>(first.front()) - second.front();
  long double offset = 0;
  for (std::size_t i = 0; i < first.size(); ++i)
    offset += ((static_cast<long double>(first[i]) - second[i]) - origin) / n;
  long double ss = 0;
  for (std::size_t i = 0; i < first.size(); ++i) {
    const auto d = ((static_cast<long double>(first[i]) - second[i]) - origin) - offset;
    ss += d * d;
  }
  const auto mean = origin + offset;
  const auto sd = std::sqrt(ss / (n - 1));
  const long double z = -random::normal_quantile((1 - population_coverage) / 2);
  const long double t = -random::student_t_quantile((1 - confidence_level) / 2, static_cast<double>(n - 1));
  const auto half_width = z * sd;
  const auto bias_error = t * sd / std::sqrt(n);
  const auto limit_error = t * sd * std::sqrt(1 / n + z * z / (2 * (n - 1)));
  const auto lower = mean - half_width, upper = mean + half_width;
  return {checked_result(mean), checked_result(sd), {checked_result(lower), checked_result(upper)},
          {checked_result(mean - bias_error), checked_result(mean + bias_error)},
          {checked_result(lower - limit_error), checked_result(lower + limit_error)},
          {checked_result(upper - limit_error), checked_result(upper + limit_error)}};
}

LinConcordance lin_concordance(const std::vector<double>& first, const std::vector<double>& second,
                              double confidence_level) {
  check_pairs(first, second, 3);
  detail::check_confidence(confidence_level);
  const long double n = first.size();
  const long double ox = first.front(), oy = second.front();
  long double scale = std::abs(ox - oy);
  for (std::size_t i = 0; i < first.size(); ++i)
    scale = std::max({scale, std::abs(first[i] - ox), std::abs(second[i] - oy)});
  if (scale == 0)
    throw std::invalid_argument("Lin concordance: zero concordance denominator");

  long double mx = 0, my = 0;
  for (std::size_t i = 0; i < first.size(); ++i) {
    mx += ((first[i] - ox) / scale) / n;
    my += ((second[i] - oy) / scale) / n;
  }
  const auto gap = (ox - oy) / scale + (mx - my);
  long double variance = 0, covariance = 0, mismatch = 0, opposition = 0;
  for (std::size_t i = 0; i < first.size(); ++i) {
    const auto x = (first[i] - ox) / scale - mx;
    const auto y = (second[i] - oy) / scale - my;
    variance += (x * x + y * y) / n;
    covariance += x * y / n;
    mismatch += (x - y) * (x - y) / n;
    opposition += (x + y) * (x + y) / n;
  }
  const auto denominator = variance + gap * gap;
  const auto coefficient = std::clamp(2 * covariance / denominator, -1.0L, 1.0L);
  const auto estimate = static_cast<double>(coefficient);
  const auto from_one = (mismatch + gap * gap) / denominator;
  const auto from_minus_one = (opposition + gap * gap) / denominator;
  if (from_one == 0 || from_minus_one == 0)
    return {estimate, 0, {estimate, estimate}};

  long double squared = 0;
  for (std::size_t i = 0; i < first.size(); ++i) {
    const auto x = (first[i] - ox) / scale - mx;
    const auto y = (second[i] - oy) / scale - my;
    const auto v = x * x + y * y + 2 * gap * (x - y);
    // Equivalent derivatives expressed using distances to +/-1 near a boundary.
    const auto influence = coefficient >= 0
        ? ((mismatch - (x - y) * (x - y)) - 2 * gap * (x - y) + from_one * (v - variance)) / denominator
        : (((x + y) * (x + y) - opposition) + 2 * gap * (x - y) - from_minus_one * (v - variance)) / denominator;
    squared += influence * influence;
  }
  const auto se = std::sqrt(squared / (n * (n - 1)));
  const auto z = .5L * (std::log(from_minus_one) - std::log(from_one));
  const auto margin = -random::normal_quantile((1 - confidence_level) / 2) * se /
                      (from_one * from_minus_one);
  return {estimate, static_cast<double>(se),
          {static_cast<double>(std::tanh(z - margin)), static_cast<double>(std::tanh(z + margin))}};
}

} // namespace datamunge::stats
