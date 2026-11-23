/// @file simplex_sampling.cpp
/// @brief Seeded gamma normalization and conditional binomial allocation.

#include <datamunge/random/simplex.hpp>
#include "gamma_sampling.hpp"
#include "simplex_detail.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>

namespace datamunge::random {

/// @brief Local sampling algorithms using caller-owned random state.
namespace {

/// @brief Draws unnormalized Dirichlet weights with the largest weight scaled to one.
/// @param rng Engine supplying independent gamma draws.
/// @param concentration Validated positive finite shapes.
/// @return Max-normalized weights, with small weights allowed to underflow to zero.
std::vector<long double> simplex_weights(RandomGenerator& rng, const std::vector<double>& concentration) {
  const long double smallest = *std::min_element(concentration.begin(), concentration.end());
  const auto log_scale = smallest < 128 / std::numeric_limits<long double>::max() ? smallest : 1.L;
  std::vector<long double> weights(concentration.size());
  for (std::size_t i = 0; i < concentration.size(); ++i)
    weights[i] = detail::gamma_draw_log(rng, concentration[i], log_scale);
  const auto maximum = *std::max_element(weights.begin(), weights.end());
  for (auto& weight : weights)
    weight = std::exp((weight - maximum) / log_scale);
  return weights;
}

/// @brief Adapts the raw engine interface to the standard binomial sampler.
struct WordEngine {
  /// @brief Raw word type required by standard distributions.
  using result_type = std::uint64_t;

  RandomGenerator& engine; ///< Borrowed caller-owned engine.

  /// @brief Returns the inclusive minimum raw word.
  static constexpr result_type min() { return 0; }

  /// @brief Returns the inclusive maximum raw word.
  static constexpr result_type max() { return std::numeric_limits<result_type>::max(); }

  /// @brief Advances the underlying engine once and returns its next raw word.
  result_type operator()() { return engine.next_u64(); }
};

/// @brief Draws a binomial count by adding independent chunks for very large trial sizes.
/// @param rng Engine adapter supplying random words.
/// @param trials Number of Bernoulli trials.
/// @param probability Probability of success in [0, 1/2].
/// @return Sum of the binomial chunk counts.
std::size_t binomial_draw(WordEngine& rng, std::size_t trials, double probability) {
  if (probability == 0)
    return 0;
  // Keep integer trial counts exactly representable in the standard
  // sampler's double intermediates.
  constexpr auto chunk_size = std::uint64_t{1} << 52;
  std::size_t result = 0;
  while (trials > 0) {
    const auto chunk = std::min<std::uint64_t>(trials, chunk_size);
    result += std::binomial_distribution<std::int64_t>(chunk, probability)(rng);
    trials -= chunk;
  }
  return result;
}

} // namespace

std::vector<double> dirichlet_sample(RandomGenerator& rng, const std::vector<double>& concentration) {
  simplex_detail::check_concentration(concentration);
  if (concentration.size() == 1)
    return {1};
  const auto weights = simplex_weights(rng, concentration);
  const auto total = std::accumulate(weights.begin(), weights.end(), 0.L);
  std::vector<double> result(weights.size());
  for (std::size_t i = 0; i < weights.size(); ++i)
    result[i] = static_cast<double>(weights[i] / total);
  return result;
}

std::vector<std::size_t> dirichlet_multinomial_sample(RandomGenerator& rng, std::size_t trials,
                                                       const std::vector<double>& concentration) {
  simplex_detail::check_concentration(concentration);
  std::vector<std::size_t> result(concentration.size());
  if (trials == 0 || result.size() == 1) {
    result.back() = trials;
    return result;
  }
  const auto weights = simplex_weights(rng, concentration);
  std::vector<long double> suffix(weights.size());
  std::partial_sum(weights.rbegin(), weights.rend(), suffix.rbegin());
  WordEngine engine{rng};
  for (std::size_t i = 0; i + 1 < weights.size() && trials > 0; ++i) {
    const auto rest = suffix[i + 1];
    const auto probability = static_cast<double>(std::min(weights[i], rest) / suffix[i]);
    const auto count = binomial_draw(engine, trials, probability);
    result[i] = weights[i] <= rest ? count : trials - count;
    trials -= result[i];
  }
  result.back() = trials;
  return result;
}

} // namespace datamunge::random
