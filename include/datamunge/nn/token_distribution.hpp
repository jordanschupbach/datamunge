#pragma once
/// @file token_distribution.hpp
/// @brief Deterministic temperature/top-k/nucleus filtering and one caller-supplied uniform draw.
#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <vector>
namespace datamunge::nn::models {
/// @brief Owning temperature/top-k/nucleus probability distribution with stable index ties.
///
/// Vectors own their elements. Retained IDs are in descending-logit order with index-order ties;
/// probabilities remain vocabulary-indexed and assign0 to discarded tokens. No RNG or generation
/// history is owned.
///
/// @par Example
/// @code
/// auto distribution=dm::nn::models::FilterTokens({0,0,0,0},1,2,1);
/// @endcode
/// @see FilterTokens(), DrawToken()
struct TokenDistribution {
  std::vector<double> probability; ///< Vocabulary-indexed normalized retained probabilities.
  std::vector<int>    retained;    ///< Kept token IDs, descending logits then ascending IDs.
  double retained_mass; ///< Pre-renormalization mass retained from the original tempered softmax.
};

/// softmax(logits / temperature) restricted to the top_k largest logits (0 =
/// all) and then to the smallest prefix (by logit, ties in index order) whose
/// mass reaches top_p of the available mass; renormalised over what is kept.
/// @brief Compute source stable temperature/top-k/nucleus filtering without drawing a token.
///
/// Does not mutate inputs or own RNG. Ties retain lower token index. Nucleus includes the token
/// crossing its threshold and always retains at least one token. Source cumulative/re-normalization
/// order is preserved.
///
/// @param logits Nonempty finite-or-negative-infinity vocabulary scores; at least one finite entry.
/// @param temperature Finite positive divisor, default1.
/// @param top_k Nonnegative number of highest-ranked tokens;0 keeps all, values above vocabulary
/// keep all.
/// @param top_p Finite cumulative mass fraction in(0,1], default1; applied after top-k relative to
/// its available mass.
/// @return Owned vocabulary probabilities, ordered retained IDs and their original tempered mass.
/// @throws std::invalid_argument Invalid logits, temperature, count or mass threshold.
/// @par Example
/// @code
/// auto
/// d=dm::nn::models::FilterTokens({std::log(.4),std::log(.3),std::log(.2),std::log(.1)},1,0,.6);
/// @endcode
/// @par References
/// - Holtzman et al. (2020), The Curious Case of Neural Text Degeneration,
/// https://arxiv.org/abs/1904.09751 .
/// @see DrawToken()
inline TokenDistribution FilterTokens(
    const std::vector<double>& logits,
    double                     temperature = 1,
    int                        top_k       = 0,
    double                     top_p       = 1) {
  if (logits.empty() || !std::isfinite(temperature) || temperature <= 0 || top_k < 0 ||
      !std::isfinite(top_p) || top_p <= 0 || top_p > 1)
    throw std::invalid_argument("valid logits/temperature/top-k/top-p required");
  double maximum = -INFINITY;
  for (double x : logits) {
    if (std::isnan(x) || x == INFINITY)
      throw std::invalid_argument("finite or negative-infinity logits required");
    maximum = std::max(maximum, x);
  }
  if (!std::isfinite(maximum))
    throw std::invalid_argument("at least one finite logit required");
  std::vector<double> probabilities(logits.size());
  double              mass = 0;
  for (std::size_t i = 0; i < logits.size(); ++i) {
    probabilities[i] = std::exp((logits[i] - maximum) / temperature);
    mass += probabilities[i];
  }
  for (auto& x : probabilities)
    x /= mass;
  std::vector<int> order(logits.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return logits[a] > logits[b]; });
  if (top_k > 0 && top_k < static_cast<int>(order.size()))
    order.resize(static_cast<std::size_t>(top_k));
  double available = 0;
  for (int i : order)
    available += probabilities[static_cast<std::size_t>(i)];
  std::vector<int> retained;
  double           cumulative = 0;
  for (int i : order) {
    if (probabilities[static_cast<std::size_t>(i)] == 0)
      continue;
    retained.push_back(i);
    cumulative += probabilities[static_cast<std::size_t>(i)];
    if (top_p < 1 && cumulative >= top_p * available)
      break;
  }
  std::vector<double> result(logits.size(), 0);
  for (int i : retained)
    result[static_cast<std::size_t>(i)] = probabilities[static_cast<std::size_t>(i)] / cumulative;
  return {result, retained, cumulative};
}

/// Inverse-CDF draw: the first index whose cumulative probability exceeds `uniform`.
/// @brief Select one categorical token using an explicit caller-supplied uniform.
///
/// Uses the original cumulative-probability order and final positive-entry fallback for rounding.
/// No engine state is owned or consumed.
///
/// @param p Nonempty finite nonnegative probabilities normalized within source tolerance.
/// @param uniform Finite number in[0,1); caller owns random generation.
/// @return Vocabulary index of the selected positive-mass entry.
/// @throws std::invalid_argument Invalid probabilities, normalization or uniform.
/// @par Example
/// @code
/// auto token=dm::nn::models::DrawToken({.4,.3,.2,.1},.8);
/// @endcode
/// @see FilterTokens()
inline int DrawToken(const std::vector<double>& p, double uniform) {
  if (!std::isfinite(uniform) || uniform < 0 || uniform >= 1 || p.empty())
    throw std::invalid_argument("uniform in [0,1) and nonempty probability vector required");
  double mass = 0;
  int    last = -1;
  for (std::size_t i = 0; i < p.size(); ++i) {
    if (!std::isfinite(p[i]) || p[i] < 0)
      throw std::invalid_argument("nonnegative finite probabilities required");
    mass += p[i];
    if (p[i] > 0)
      last = static_cast<int>(i);
  }
  if (std::abs(mass - 1) > 1e-9 || last < 0)
    throw std::invalid_argument("normalized probability vector required");
  double cumulative = 0;
  for (std::size_t i = 0; i < p.size(); ++i) {
    cumulative += p[i];
    if (uniform < cumulative)
      return static_cast<int>(i);
  }
  return last;
}

/// @brief Draw a token by visiting the retained IDs in ranked order.
/// Use this function to replay a sampler that accumulates the most likely
/// tokens first. DrawToken() visits vocabulary indices in order.
/// @param distribution The filtered token distribution returned by FilterTokens().
/// @param uniform Finite number in [0, 1).
/// @return The selected vocabulary index.
/// @throws std::invalid_argument Invalid probabilities, retained IDs or uniform.
/// @code
/// auto d = dm::nn::models::FilterTokens({0.0, 1.0, 2.0}, 1.0, 2);
/// auto token = dm::nn::models::DrawRankedToken(d, 0.25);
/// @endcode
inline int DrawRankedToken(const TokenDistribution& distribution, double uniform) {
  std::vector<bool> seen(distribution.probability.size(), false);
  std::vector<double> ranked;
  ranked.reserve(distribution.retained.size());
  for (const int id : distribution.retained) {
    if (id < 0 || static_cast<std::size_t>(id) >= seen.size() || seen[id]) {
      throw std::invalid_argument("invalid ranked token IDs");
    }
    seen[id] = true;
    ranked.push_back(distribution.probability[id]);
  }
  for (std::size_t i = 0; i < seen.size(); ++i) {
    if (!seen[i] && distribution.probability[i] != 0.0)
      throw std::invalid_argument("ranked distribution omits a positive-mass token");
  }
  return distribution.retained[DrawToken(ranked, uniform)];
}

/// Largest logit (lowest index on ties).
/// @brief Return the first highest-scoring finite token with source tie order.
///
/// @param logits Nonempty finite-or-negative-infinity scores with at least one finite entry.
/// @return Lowest vocabulary index among the maxima.
/// @throws std::invalid_argument Invalid or all-negative-infinity scores.
/// @par Example
/// @code
/// auto token=dm::nn::models::GreedyToken({0,1,1});
/// @endcode
/// @see FilterTokens()
inline int GreedyToken(const std::vector<double>& logits) {
  return FilterTokens(logits).retained.front();
}

/// @brief Caller-owned history-to-logits callable for report-owned decoding/search loops.
/// @par Example
/// @code
/// dm::nn::models::TokenPredictor predict=[](const std::vector<int>& history){return std::vector<double>{0,0};};
/// @endcode
/// @see FilterTokens()
using TokenPredictor = std::function<std::vector<double>(const std::vector<int>&)>;

} // namespace datamunge::nn::models
