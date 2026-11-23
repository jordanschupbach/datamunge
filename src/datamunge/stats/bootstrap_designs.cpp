/// @file bootstrap_designs.cpp
/// @brief Row-resampling geometry and fitted-null bootstrap calibration.

#include <datamunge/stats/bootstrap_designs.hpp>
#include <datamunge/stats/detail/test_common.hpp>
#include <datamunge/random/simplex.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace datamunge::stats {

/// @brief Shared geometry and comparison utilities.
namespace {

/// @brief Groups row indices in first-occurrence label order.
/// @param labels One label per source row, with at least one row.
/// @return Rows for each group, in original order within each group.
std::vector<std::vector<std::size_t>> grouped_rows(const std::vector<std::size_t>& labels) {
  if (labels.empty())
    throw std::invalid_argument("bootstrap: need at least one row label");
  std::unordered_map<std::size_t, std::size_t> position;
  std::vector<std::vector<std::size_t>> groups;
  for (std::size_t row = 0; row < labels.size(); ++row) {
    const auto [entry, inserted] = position.try_emplace(labels[row], groups.size());
    if (inserted)
      groups.emplace_back();
    groups[entry->second].push_back(row);
  }
  return groups;
}

/// @brief Draws one uniform index from a nonempty range of choices.
/// @param rng Source of random words.
/// @param size Number of choices.
/// @return Index between zero and size minus one.
std::size_t choose_index(random::RandomGenerator& rng, std::size_t size) {
  return static_cast<std::size_t>(rng.uniform_u64(0, size - 1));
}

/// @brief Advances one step around a circular series.
/// @param index Current valid series index.
/// @param size Positive series length.
/// @return Next index, wrapping to zero at the end.
std::size_t next_index(std::size_t index, std::size_t size) {
  return index == size - 1 ? 0 : index + 1;
}

/// @brief Compares bootstrap and observed statistics using the requested tail.
/// @param replicate Finite refitted statistic.
/// @param observed Finite observed statistic.
/// @param alternative Prespecified valid tail.
/// @return True if the replicate is at least as extreme as observed.
bool is_extreme(double replicate, double observed, Alternative alternative) {
  if (alternative == Alternative::Greater)
    return replicate >= observed;
  if (alternative == Alternative::Less)
    return replicate <= observed;
  return std::abs(replicate) >= std::abs(observed);
}

} // namespace

std::vector<std::size_t> stratified_bootstrap_indices(const std::vector<std::size_t>& strata,
                                                        random::RandomGenerator& rng) {
  const auto groups = grouped_rows(strata);
  std::vector<std::size_t> indices(strata.size());
  for (const auto& group : groups)
    for (const auto row : group)
      indices[row] = group[choose_index(rng, group.size())];
  return indices;
}

ClusterBootstrapSample cluster_bootstrap_indices(const std::vector<std::size_t>& clusters,
                                                  random::RandomGenerator& rng) {
  const auto groups = grouped_rows(clusters);
  ClusterBootstrapSample result;
  result.indices.reserve(clusters.size());
  result.cluster_offsets.push_back(0);
  for (std::size_t i = 0; i < groups.size(); ++i) {
    const auto& group = groups[choose_index(rng, groups.size())];
    result.indices.insert(result.indices.end(), group.begin(), group.end());
    result.cluster_offsets.push_back(result.indices.size());
  }
  return result;
}

std::vector<std::size_t> block_bootstrap_indices(std::size_t series_size, std::size_t block_size,
                                                   random::RandomGenerator& rng, bool circular) {
  if (block_size == 0 || block_size > series_size)
    throw std::invalid_argument("block bootstrap: require 1 <= block size <= series size");
  const auto starts = circular ? series_size : series_size - block_size + 1;
  std::vector<std::size_t> indices(series_size);
  for (std::size_t copied = 0; copied < series_size;) {
    auto row = choose_index(rng, starts);
    const auto length = std::min(block_size, series_size - copied);
    for (std::size_t i = 0; i < length; ++i) {
      indices[copied++] = row;
      row = next_index(row, series_size);
    }
  }
  return indices;
}

std::vector<std::size_t> stationary_bootstrap_indices(std::size_t series_size, double mean_block_size,
                                                        random::RandomGenerator& rng) {
  if (series_size == 0 || !(mean_block_size >= 1 && std::isfinite(mean_block_size)))
    throw std::invalid_argument("stationary bootstrap: invalid series size or mean block length");
  const auto restart_probability = 1 / mean_block_size;
  std::vector<std::size_t> indices(series_size);
  indices.front() = choose_index(rng, series_size);
  for (std::size_t i = 1; i < indices.size(); ++i)
    indices[i] = rng.bernoulli(restart_probability) ? choose_index(rng, series_size) :
                                                    next_index(indices[i - 1], series_size);
  return indices;
}

std::vector<std::vector<std::size_t>> balanced_bootstrap_indices(std::size_t sample_size,
                                                                  std::size_t replicates,
                                                                  random::RandomGenerator& rng) {
  if (sample_size == 0 || replicates == 0)
    throw std::invalid_argument("balanced bootstrap: positive dimensions required");
  if (replicates > std::numeric_limits<std::size_t>::max() / sample_size)
    throw std::length_error("balanced bootstrap: index pool size overflow");

  std::vector<std::vector<std::size_t>> result(replicates, std::vector<std::size_t>(sample_size));
  for (auto& row : result)
    for (std::size_t i = 0; i < row.size(); ++i)
      row[i] = i;
  /// @brief Accesses an entry in the logical flattened pool.
  const auto at = [&](std::size_t index) -> std::size_t& {
    return result[index / sample_size][index % sample_size];
  };
  for (auto remaining = sample_size * replicates; remaining > 1; --remaining)
    std::swap(at(remaining - 1), at(choose_index(rng, remaining)));
  return result;
}

HypothesisTestResult parametric_bootstrap_test(
    double observed, std::size_t draws,
    const std::function<std::vector<double>(random::RandomGenerator&)>& simulate,
    const SampleStatistic& refit_statistic, random::RandomGenerator& rng, Alternative alternative) {
  detail::check_alternative(alternative);
  if (!std::isfinite(observed) || draws == 0 || !simulate || !refit_statistic)
    throw std::invalid_argument("parametric bootstrap: invalid statistic, draw count or callback");
  std::size_t extreme = 0;
  for (std::size_t iteration = 0; iteration < draws; ++iteration) {
    const auto sample = simulate(rng);
    detail::check_sample(sample);
    const auto replicate = refit_statistic(sample);
    if (!std::isfinite(replicate))
      throw std::domain_error("parametric bootstrap: nonfinite refitted statistic");
    extreme += is_extreme(replicate, observed, alternative);
  }

  HypothesisTestResult result;
  result.statistic = observed;
  result.parameter1 = static_cast<double>(draws);
  result.parameter2 = static_cast<double>(extreme);
  result.p_value = static_cast<double>((1.0L + extreme) / (1.0L + draws));
  result.alternative = alternative;
  result.method = "Parametric bootstrap test";
  return result;
}

HypothesisTestResult wild_bootstrap_test(
    double observed, const std::vector<double>& fitted_null, const std::vector<double>& residuals,
    std::size_t draws, const SampleStatistic& refit_statistic, random::RandomGenerator& rng,
    WildBootstrapWeight weight, Alternative alternative) {
  detail::check_sample(fitted_null);
  detail::check_sample(residuals);
  if (fitted_null.size() != residuals.size())
    throw std::invalid_argument("wild bootstrap: unequal lengths for fitted values and residuals");
  if (weight != WildBootstrapWeight::Rademacher && weight != WildBootstrapWeight::Mammen)
    throw std::invalid_argument("wild bootstrap: invalid multiplier distribution");

  const bool rademacher = weight == WildBootstrapWeight::Rademacher;
  const double root5 = std::sqrt(5.0);
  const double low = rademacher ? -1 : (1 - root5) / 2;
  const double high = rademacher ? 1 : (1 + root5) / 2;
  const double probability_low = rademacher ? .5 : high / root5;
  /// @brief Simulates one response while holding the already fitted null and design fixed.
  const auto simulate = [&](random::RandomGenerator& stream) {
    std::vector<double> response(fitted_null.size());
    for (std::size_t i = 0; i < response.size(); ++i)
      response[i] = std::fma(stream.bernoulli(probability_low) ? low : high, residuals[i], fitted_null[i]);
    return response;
  };
  auto result = parametric_bootstrap_test(observed, draws, simulate, refit_statistic, rng, alternative);
  result.method = rademacher ? "Wild bootstrap test (Rademacher weights)" : "Wild bootstrap test (Mammen weights)";
  return result;
}

std::vector<double> bayesian_bootstrap(std::size_t observations, std::size_t draws,
                                       const SampleStatistic& weighted_functional, random::RandomGenerator& rng) {
  if (observations == 0 || draws == 0 || !weighted_functional)
    throw std::invalid_argument("Bayesian bootstrap: need observations, posterior draws, and a functional");
  std::vector<double> posterior;
  posterior.reserve(draws);
  const std::vector<double> concentration(observations, 1);
  for (std::size_t i = 0; i < draws; ++i) {
    const auto weights = random::dirichlet_sample(rng, concentration);
    const auto estimate = weighted_functional(weights);
    if (!std::isfinite(estimate))
      throw std::domain_error("Bayesian bootstrap: functional must return a finite value");
    posterior.push_back(estimate);
  }
  return posterior;
}

} // namespace datamunge::stats
