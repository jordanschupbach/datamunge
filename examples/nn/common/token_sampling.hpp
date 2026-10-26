// Deterministic ranking, temperature/top-k/nucleus sampling and log-score beam
// search: port of tflow's common/ml/token_sampling.h (plain C++ over double
// logits, identical rules and guards).
#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace lm {

struct TokenDistribution {
  std::vector<double> probability;
  std::vector<int>    retained;
  double              retained_mass;
};

/// softmax(logits / temperature) restricted to the top_k largest logits (0 =
/// all) and then to the smallest prefix (by logit, ties in index order) whose
/// mass reaches top_p of the available mass; renormalised over what is kept.
inline TokenDistribution FilterTokens(const std::vector<double>& logits, double temperature = 1, int top_k = 0,
                                      double top_p = 1) {
  if (logits.empty() || !std::isfinite(temperature) || temperature <= 0 || top_k < 0 || !std::isfinite(top_p) ||
      top_p <= 0 || top_p > 1)
    throw std::invalid_argument("valid logits/temperature/top-k/top-p required");
  double maximum = -INFINITY;
  for (double x : logits) {
    if (std::isnan(x) || x == INFINITY) throw std::invalid_argument("finite or negative-infinity logits required");
    maximum = std::max(maximum, x);
  }
  if (!std::isfinite(maximum)) throw std::invalid_argument("at least one finite logit required");
  std::vector<double> probabilities(logits.size());
  double              mass = 0;
  for (std::size_t i = 0; i < logits.size(); ++i) {
    probabilities[i] = std::exp((logits[i] - maximum) / temperature);
    mass += probabilities[i];
  }
  for (auto& x : probabilities) x /= mass;
  std::vector<int> order(logits.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return logits[a] > logits[b]; });
  if (top_k > 0 && top_k < static_cast<int>(order.size())) order.resize(static_cast<std::size_t>(top_k));
  double available = 0;
  for (int i : order) available += probabilities[static_cast<std::size_t>(i)];
  std::vector<int> retained;
  double           cumulative = 0;
  for (int i : order) {
    if (probabilities[static_cast<std::size_t>(i)] == 0) continue;
    retained.push_back(i);
    cumulative += probabilities[static_cast<std::size_t>(i)];
    if (top_p < 1 && cumulative >= top_p * available) break;
  }
  std::vector<double> result(logits.size(), 0);
  for (int i : retained) result[static_cast<std::size_t>(i)] = probabilities[static_cast<std::size_t>(i)] / cumulative;
  return {result, retained, cumulative};
}

/// Inverse-CDF draw: the first index whose cumulative probability exceeds `uniform`.
inline int DrawToken(const std::vector<double>& p, double uniform) {
  if (!std::isfinite(uniform) || uniform < 0 || uniform >= 1 || p.empty())
    throw std::invalid_argument("uniform in [0,1) and nonempty probability vector required");
  double mass = 0;
  int    last = -1;
  for (std::size_t i = 0; i < p.size(); ++i) {
    if (!std::isfinite(p[i]) || p[i] < 0) throw std::invalid_argument("nonnegative finite probabilities required");
    mass += p[i];
    if (p[i] > 0) last = static_cast<int>(i);
  }
  if (std::abs(mass - 1) > 1e-9 || last < 0) throw std::invalid_argument("normalized probability vector required");
  double cumulative = 0;
  for (std::size_t i = 0; i < p.size(); ++i) {
    cumulative += p[i];
    if (uniform < cumulative) return static_cast<int>(i);
  }
  return last;
}

/// Largest logit (lowest index on ties).
inline int GreedyToken(const std::vector<double>& logits) { return FilterTokens(logits).retained.front(); }

using TokenPredictor = std::function<std::vector<double>(const std::vector<int>&)>;

struct TokenBeam {
  std::vector<int> tokens;
  double           score    = 0;
  bool             finished = false;
};

/// Width-limited beam search on summed log-probabilities; beams that emit
/// `eos` are kept but not extended; ties are broken by token sequence.
inline std::vector<TokenBeam> BeamTokens(const TokenPredictor& predict, const std::vector<int>& prompt, int steps,
                                         int width, int eos = -1) {
  if (prompt.empty() || steps < 1 || width < 1) throw std::invalid_argument("beam requires prompt,positive steps/width");
  std::vector<TokenBeam> beam{{prompt, 0, false}};
  for (int t = 0; t < steps; ++t) {
    std::vector<TokenBeam> expanded;
    bool                   all_done = true;
    for (auto& b : beam) {
      if (b.finished) {
        expanded.push_back(b);
        continue;
      }
      all_done    = false;
      auto logits = predict(b.tokens);
      FilterTokens(logits);  // validates the logits as tflow does
      double maximum = *std::max_element(logits.begin(), logits.end()), mass = 0;
      for (double z : logits) mass += std::exp(z - maximum);
      const double logmass = std::log(mass);
      for (std::size_t k = 0; k < logits.size(); ++k)
        if (std::isfinite(logits[k])) {
          auto next = b;
          next.tokens.push_back(static_cast<int>(k));
          next.score += (logits[k] - maximum) - logmass;
          next.finished = static_cast<int>(k) == eos;
          expanded.push_back(std::move(next));
        }
    }
    if (all_done) break;
    std::sort(expanded.begin(), expanded.end(), [](const TokenBeam& a, const TokenBeam& b) {
      return a.score != b.score ? a.score > b.score : a.tokens < b.tokens;
    });
    if (static_cast<int>(expanded.size()) > width) expanded.resize(static_cast<std::size_t>(width));
    beam = std::move(expanded);
  }
  return beam;
}

}  // namespace lm
