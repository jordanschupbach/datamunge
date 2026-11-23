/// @file edf_tests.cpp
/// @brief Finite-sample simulation for empirical distribution function tests.

#include <datamunge/stats/edf_tests.hpp>
#include <datamunge/stats/detail/test_common.hpp>
#include "pooled_permutations.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>

namespace datamunge::stats {

/// @brief Private empirical distribution statistics and calibration.
namespace {

/// @brief Transforms sorted observations to their specified null probabilities.
/// @param sample Nonempty finite observations.
/// @param cdf Continuous cumulative distribution function to evaluate.
/// @return Nondecreasing probabilities in [0,1].
/// @throws std::invalid_argument Missing callable or invalid probabilities.
std::vector<double> null_probabilities(const std::vector<double>& sample,
                                       const std::function<double(double)>& cdf) {
  detail::check_sample(sample);
  if (!cdf)
    throw std::invalid_argument("EDF test: a CDF is required");
  auto values = sample;
  std::ranges::sort(values);
  double previous = 0;
  for (double& x : values) {
    const auto p = cdf(x);
    if (!(p >= previous && p <= 1))
      throw std::invalid_argument("EDF test: CDF must be nondecreasing with probabilities in [0,1]");
    previous = x = p;
  }
  return values;
}

/// @brief Computes Anderson–Darling from the logarithms of both CDF tails.
/// @tparam LogTerm Callable supplying the sum of the lower and reflected upper log probabilities.
/// @param n Number of observations.
/// @param log_term Function of a zero-based order statistic index.
/// @return Anderson–Darling statistic, possibly positive infinity.
template <class LogTerm>
long double ad_statistic(std::size_t n, const LogTerm& log_term) {
  long double result = 0;
  for (std::size_t i = 0; i < n; ++i)
    result -= ((2.L * i + 1) / n) * log_term(i);
  return std::max(0.L, result - n);
}

/// @brief Calculates the Anderson–Darling statistic from sorted uniform probabilities.
/// @param probabilities Nondecreasing values in [0,1].
/// @return Statistic, infinite for probability zero or one.
long double ad_uniform(const std::vector<double>& probabilities) {
  return ad_statistic(probabilities.size(), [&](std::size_t i) {
    return std::log(static_cast<long double>(probabilities[i])) +
           std::log1p(-static_cast<long double>(probabilities[probabilities.size() - 1 - i]));
  });
}

/// @brief Calculates the Cramér–von Mises statistic from sorted uniform probabilities.
/// @param probabilities Nondecreasing values in [0,1].
/// @return Squared CDF discrepancy including the finite-sample centering term.
long double cvm_uniform(const std::vector<double>& probabilities) {
  const long double n = probabilities.size();
  auto score = 1 / (12 * n);
  for (std::size_t i = 0; i < probabilities.size(); ++i) {
    const auto delta = probabilities[i] - (i + .5L) / n;
    score += delta * delta;
  }
  return score;
}

/// @brief Evaluates the log standard normal CDF, including extreme left tails.
/// @param z Standardized coordinate.
/// @return Natural logarithm of Phi(z).
long double normal_log_cdf(long double z) {
  const auto tail = std::erfc(std::abs(z) / std::numbers::sqrt2_v<long double>) / 2;
  if (z >= 0)
    return std::log1p(-tail);
  if (tail >= std::numeric_limits<long double>::min())
    return std::log(tail);
  // At erfc underflow, the leading Mills-ratio series is already accurate.
  const auto inv_square = (1 / z) / z;
  return -z * z / 2 - std::log(-z) - std::log(2 * std::numbers::pi_v<long double>) / 2 +
         std::log1p(inv_square * (-1 + inv_square * (3 - 15 * inv_square)));
}

/// @brief Anderson–Darling statistic and the parameters estimated from a normal sample.
struct FittedNormalScore {
  long double statistic; ///< Discrepancy from the fitted normal CDF.
  long double mean;      ///< Fitted location in original units.
  long double sd;        ///< Sample standard deviation in original units.
};

/// @brief Standardizes a sample using stable location and sample-variance arithmetic.
/// @param sorted Nonempty finite observations in sorted order.
/// @return Fitted normality statistic, mean and sample SD.
/// @throws std::invalid_argument A constant sample.
FittedNormalScore ad_normal(const std::vector<double>& sorted) {
  const long double origin = sorted.front();
  const auto scale = std::max(std::abs(origin), std::abs(static_cast<long double>(sorted.back())));
  if (origin == sorted.back())
    throw std::invalid_argument("Anderson–Darling normality test: sample must vary");
  /// @brief Removes an offset and rescales to prevent overflow when fitting moments.
  const auto centered = [&](long double x) {
    const auto difference = x - origin;
    return std::isfinite(difference) ? difference / scale : x / scale - origin / scale;
  };
  long double mean = 0, ss = 0;
  std::size_t n = 0;
  for (const auto x : sorted) {
    const auto y = centered(x);
    const auto delta = y - mean;
    mean += delta / ++n;
    ss += delta * (y - mean);
  }
  const auto sd = std::sqrt(ss / (n - 1));
  const auto statistic = ad_statistic(n, [&](std::size_t i) {
    return normal_log_cdf((centered(sorted[i]) - mean) / sd) +
           normal_log_cdf((mean - centered(sorted[n - 1 - i])) / sd);
  });
  return {statistic, std::fma(mean, scale, origin), sd * scale};
}

/// @brief Constructs an EDF result from either an exact or a simulated reference.
/// @param observed Observed test statistic.
/// @param draws Positive number of reference samples or assignments.
/// @param extreme Number of reference statistics at least as large as observed.
/// @param exact Whether the reference enumerated all assignments.
/// @param method Display method.
/// @return Observed statistic, assignment counts and calibrated p-value.
HypothesisTestResult edf_result(long double observed, std::size_t draws, std::size_t extreme,
                                bool exact, const char* method) {
  const long double correction = exact ? 0 : 1;
  HypothesisTestResult result;
  result.statistic = static_cast<double>(observed);
  result.p_value = static_cast<double>((correction + extreme) / (correction + draws));
  result.parameter1 = static_cast<double>(draws);
  result.parameter2 = static_cast<double>(extreme);
  result.alternative = Alternative::Greater;
  result.method = method;
  return result;
}

/// @brief Calibrates an EDF statistic from independent simulated reference statistics.
/// @tparam Draw Callable yielding a reference statistic.
/// @param observed Observed test statistic.
/// @param draws Positive number of reference samples.
/// @param sample_score Reference statistic generator.
/// @param method Display method.
/// @return Observed statistic, assignment counts and plus-one p-value.
template <class Draw>
HypothesisTestResult simulate_reference(long double observed, std::size_t draws,
                                        const Draw& sample_score, const char* method) {
  std::size_t extreme = 0;
  for (std::size_t i = 0; i < draws; ++i)
    extreme += sample_score() >= observed;
  return edf_result(observed, draws, extreme, false, method);
}

/// @brief Tests a specified continuous distribution using sorted uniform reference samples.
/// @param sample Observations under test.
/// @param cdf Specified null CDF.
/// @param draws Positive number of independent simulations.
/// @param rng Caller-owned uniform generator.
/// @param score Statistic computed from sorted null probabilities.
/// @param method Display method.
/// @return Observed statistic and simulated reference.
HypothesisTestResult uniform_reference(const std::vector<double>& sample, const std::function<double(double)>& cdf,
                                        std::size_t draws, random::RandomGenerator& rng,
                                        long double (*score)(const std::vector<double>&), const char* method) {
  if (draws == 0)
    throw std::invalid_argument("EDF test: draw count must be positive");
  auto probabilities = null_probabilities(sample, cdf);
  return simulate_reference(score(probabilities), draws, [&] {
    for (auto& value : probabilities) {
      do {
        value = rng.uniform01();
      } while (value == 0);
    }
    std::ranges::sort(probabilities);
    return score(probabilities);
  }, method);
}

/// @brief Validates observations and counts pooled rows for a two-sample EDF test.
/// @param first First sample.
/// @param second Second sample.
/// @return Positive combined size.
/// @throws std::length_error Pooled sample exceeds capacity.
std::size_t pooled_size(const std::vector<double>& first, const std::vector<double>& second) {
  detail::check_sample(first);
  detail::check_sample(second);
  if (second.size() > first.max_size() - first.size())
    throw std::length_error("Cramér–von Mises test: pooled sample is too large");
  return first.size() + second.size();
}

/// @brief Pooled observations mapped to their sorted tie blocks.
struct PooledTieBlocks {
  std::vector<std::size_t> blocks; ///< Sorted tie-block index for each original pooled observation.
  std::vector<std::size_t> totals; ///< Number of observations in each sorted tie block.
};

/// @brief Sorts observation indices and assigns equal values to one block.
/// @param values Validated finite pooled observations in input order.
/// @return Block index of each observation and block sizes in sorted order.
PooledTieBlocks rank_pool(const std::vector<double>& values) {
  PooledTieBlocks result;
  result.blocks.resize(values.size());
  std::vector<std::size_t> order(values.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::ranges::sort(order, {}, [&](std::size_t row) { return values[row]; });
  for (std::size_t i = 0; i < order.size(); ++i) {
    if (i == 0 || values[order[i]] != values[order[i - 1]])
      result.totals.push_back(0);
    result.blocks[order[i]] = result.totals.size() - 1;
    ++result.totals.back();
  }
  return result;
}

/// @brief Reuses sorted tie blocks and a count workspace when calculating Cramér–von Mises statistics.
struct CvmPool {
  PooledTieBlocks ties;            ///< Tie-block indices and their multiplicities.
  std::vector<std::size_t> hits;   ///< Reused counts assigned to the selected group.

  /// @brief Sorts the pooled observations and records the multiplicity of each value.
  /// @param first Validated first sample.
  /// @param second Validated second sample.
  CvmPool(const std::vector<double>& first, const std::vector<double>& second) {
    auto values = first;
    values.insert(values.end(), second.begin(), second.end());
    ties = rank_pool(values);
    hits.resize(ties.totals.size());
  }

  /// @brief Calculates the statistic for one assignment of rows to the selected group.
  /// @param group Distinct pooled observation indices in the selected group.
  /// @return Scaled integrated squared CDF difference.
  long double score(std::span<const std::size_t> group) {
    std::ranges::fill(hits, 0);
    for (const auto id : group)
      ++hits[ties.blocks[id]];
    const long double n = group.size(), total = ties.blocks.size(), m = total - n;
    std::size_t first = 0, second = 0;
    long double sum = 0;
    for (std::size_t i = 0; i < hits.size(); ++i) {
      first += hits[i];
      second += ties.totals[i] - hits[i];
      const auto delta = first * m - second * n;
      sum += ties.totals[i] * delta * delta;
    }
    return sum / (total * total * n * m);
  }
};

/// @brief Validates independent groups and returns prefix offsets defining their sizes.
/// @param groups Finite samples to compare.
/// @return Group starts followed by the total number of pooled observations.
/// @throws std::invalid_argument Fewer than two groups, an empty group or a nonfinite value.
/// @throws std::length_error The combined observations exceed container capacity.
std::vector<std::size_t> group_offsets(const std::vector<std::vector<double>>& groups) {
  if (groups.size() < 2)
    throw std::invalid_argument("k-sample Anderson–Darling: need at least two groups");
  std::vector<std::size_t> offsets{0};
  for (const auto& group : groups) {
    detail::check_sample(group);
    if (group.size() > group.max_size() - offsets.back())
      throw std::length_error("k-sample Anderson–Darling: pooled sample is too large");
    offsets.push_back(offsets.back() + group.size());
  }
  return offsets;
}

/// @brief Holds the pooled midranks and weight factors for repeated k-sample scoring.
struct AdMidrankPool {
  /// @brief Fixed contributions from one distinct pooled value.
  struct Block {
    long double double_midrank; ///< Twice the cumulative pooled count at the midpoint of this jump.
    long double weight;         ///< Block size divided by the fourfold Scholz–Stephens denominator.
  };

  std::vector<std::size_t> block_of_row; ///< Tie-block index of every pooled observation.
  std::vector<Block> blocks;            ///< Distinct-value contributions in sorted order.
  std::vector<std::size_t> hits;         ///< Counts of group observations in each tie block, reused across groups.
  std::vector<long double> group_scores; ///< Group terms summed in magnitude order.

  /// @brief Caches the pooled tie blocks and weights.
  /// @param groups Validated independent samples.
  /// @param total Number of pooled rows.
  AdMidrankPool(const std::vector<std::vector<double>>& groups, std::size_t total) {
    std::vector<double> values;
    values.reserve(total);
    for (const auto& group : groups)
      values.insert(values.end(), group.begin(), group.end());
    auto ties = rank_pool(values);
    block_of_row = std::move(ties.blocks);
    if (ties.totals.size() == 1)
      return;
    std::size_t before = 0;
    for (const auto tied : ties.totals) {
      const long double after = total - before - tied;
      // Four times B(N-B)-N*l/4, expanded into nonnegative terms.
      const auto denominator = 4.L * before * after + tied * (before + after);
      blocks.push_back({2.L * before + tied, tied / denominator});
      before += tied;
    }
    hits.resize(blocks.size());
    group_scores.resize(groups.size());
  }

  /// @brief Evaluates equation 7 using half jumps for each group's empirical CDF.
  /// @param order Row indices grouped into successive sample slots.
  /// @param offsets Boundaries between group slots, including the pooled size.
  /// @return Unstandardized midrank Anderson–Darling statistic.
  long double score(const std::vector<std::size_t>& order, const std::vector<std::size_t>& offsets) {
    if (blocks.empty())
      return 0;
    const long double total = order.size();
    for (std::size_t group = 0; group < group_scores.size(); ++group) {
      std::ranges::fill(hits, 0);
      for (auto i = offsets[group]; i < offsets[group + 1]; ++i)
        ++hits[block_of_row[order[i]]];
      const long double n = offsets[group + 1] - offsets[group];
      std::size_t before = 0;
      long double sum = 0;
      for (std::size_t i = 0; i < blocks.size(); ++i) {
        const auto difference = total * (2.L * before + hits[i]) - n * blocks[i].double_midrank;
        sum += blocks[i].weight * difference * difference;
        before += hits[i];
      }
      group_scores[group] = sum / n;
    }
    std::ranges::sort(group_scores);
    const auto sum = std::accumulate(group_scores.begin(), group_scores.end(), 0.L);
    return (total - 1) / total * (sum / total);
  }
};

/// @brief Includes an assignment in the upper tail, allowing near-equality from floating-point sums.
/// @param candidate Permuted midrank statistic.
/// @param observed Midrank statistic for the observed group labels.
/// @return True when the assignment is at least as extreme as observed.
bool ad_k_upper(long double candidate, long double observed) {
  return observed - candidate <= 64 * std::numeric_limits<long double>::epsilon() * std::max(candidate, observed);
}

} // namespace

HypothesisTestResult anderson_darling_test(const std::vector<double>& sample,
                                           const std::function<double(double)>& cdf,
                                           std::size_t draws, random::RandomGenerator& rng) {
  return uniform_reference(sample, cdf, draws, rng, ad_uniform, "Anderson–Darling test (specified continuous CDF)");
}

HypothesisTestResult anderson_darling_normal_test(const std::vector<double>& sample,
                                                  std::size_t draws, random::RandomGenerator& rng) {
  if (draws == 0)
    throw std::invalid_argument("EDF test: draw count must be positive");
  detail::check_sample(sample, 3);
  auto values = sample;
  std::ranges::sort(values);
  const auto observed = ad_normal(values);
  auto result = simulate_reference(observed.statistic, draws, [&] {
    for (auto& value : values)
      value = rng.normal();
    std::ranges::sort(values);
    return ad_normal(values).statistic;
  }, "Anderson–Darling test (fitted normal)");
  result.estimate1 = static_cast<double>(observed.mean);
  result.estimate2 = observed.sd > std::numeric_limits<double>::max() ? std::numeric_limits<double>::infinity()
                                                                     : static_cast<double>(observed.sd);
  return result;
}

HypothesisTestResult anderson_darling_k_sample_test(const std::vector<std::vector<double>>& groups,
                                                    std::size_t draws, random::RandomGenerator& rng) {
  if (draws == 0)
    throw std::invalid_argument("EDF test: draw count must be positive");
  const auto offsets = group_offsets(groups);
  auto pool = AdMidrankPool(groups, offsets.back());
  std::vector<std::size_t> order(offsets.back());
  std::iota(order.begin(), order.end(), std::size_t{0});
  const auto observed = pool.score(order, offsets);
  std::size_t tail = 0;
  for (std::size_t i = 0; i < draws; ++i) {
    detail::shuffle_assignment(order, offsets[offsets.size() - 2], rng);
    tail += ad_k_upper(pool.score(order, offsets), observed);
  }
  return edf_result(observed, draws, tail, false, "Midrank Anderson–Darling k-sample test (simulated pooled labels)");
}

HypothesisTestResult anderson_darling_k_sample_test_exact(const std::vector<std::vector<double>>& groups,
                                                          std::size_t max_assignments) {
  if (max_assignments == 0)
    throw std::invalid_argument("EDF test: exact assignment cap must be positive");
  const auto offsets = group_offsets(groups);
  std::size_t count = 1;
  for (std::size_t i = 0; i < groups.size(); ++i) {
    const auto selected = std::min(offsets[i], groups[i].size());
    count *= detail::fixed_assignment_count(offsets[i + 1], selected, max_assignments / count);
  }
  auto pool = AdMidrankPool(groups, offsets.back());
  std::vector<std::size_t> order(offsets.back());
  std::iota(order.begin(), order.end(), std::size_t{0});
  const auto observed = pool.score(order, offsets);

  std::vector<std::size_t> labels;
  labels.reserve(order.size());
  for (std::size_t i = 0; i < groups.size(); ++i)
    labels.insert(labels.end(), groups[i].size(), i);
  auto cursor = offsets;
  std::size_t tail = 0;
  do {
    std::ranges::copy(offsets, cursor.begin());
    for (std::size_t i = 0; i < labels.size(); ++i)
      order[cursor[labels[i]]++] = i;
    tail += ad_k_upper(pool.score(order, offsets), observed);
  } while (std::ranges::next_permutation(labels).found);
  return edf_result(observed, count, tail, true, "Midrank Anderson–Darling k-sample test (exact pooled labels)");
}

HypothesisTestResult cramer_von_mises_test(const std::vector<double>& sample,
                                           const std::function<double(double)>& cdf,
                                           std::size_t draws, random::RandomGenerator& rng) {
  return uniform_reference(sample, cdf, draws, rng, cvm_uniform, "Cramér–von Mises test (specified continuous CDF)");
}

HypothesisTestResult cramer_von_mises_test_two_sample(const std::vector<double>& first,
                                                      const std::vector<double>& second,
                                                      std::size_t draws, random::RandomGenerator& rng) {
  if (draws == 0)
    throw std::invalid_argument("EDF test: draw count must be positive");
  const auto total = pooled_size(first, second), selected = std::min(first.size(), second.size());
  const bool first_is_smaller = first.size() <= second.size();
  CvmPool pool(first_is_smaller ? first : second, first_is_smaller ? second : first);
  std::vector<std::size_t> order(total);
  std::iota(order.begin(), order.end(), std::size_t{0});
  const auto group = std::span(order).first(selected);
  return simulate_reference(pool.score(group), draws, [&] {
    detail::shuffle_assignment(order, selected, rng);
    return pool.score(group);
  }, "Cramér–von Mises test (simulated pooled labels)");
}

HypothesisTestResult cramer_von_mises_test_two_sample_exact(const std::vector<double>& first,
                                                            const std::vector<double>& second,
                                                            std::size_t max_assignments) {
  if (max_assignments == 0)
    throw std::invalid_argument("EDF test: exact assignment cap must be positive");
  const auto total = pooled_size(first, second), selected = std::min(first.size(), second.size());
  const auto count = detail::fixed_assignment_count(total, selected, max_assignments);
  const bool first_is_smaller = first.size() <= second.size();
  CvmPool pool(first_is_smaller ? first : second, first_is_smaller ? second : first);
  std::vector<std::size_t> group(selected);
  std::iota(group.begin(), group.end(), std::size_t{0});
  const auto observed = pool.score(group);
  std::size_t tail = 0;
  do {
    tail += pool.score(group) >= observed;
  } while (detail::next_fixed_assignment(group, total));
  return edf_result(observed, count, tail, true, "Cramér–von Mises test (exact pooled labels)");
}

} // namespace datamunge::stats
