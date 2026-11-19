#pragma once

/// @file beam_search.hpp
/// @brief Bounded autoregressive decoding with independent hypothesis states.

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace datamunge::nn {

/// @brief A ranked prefix and the decoder state before consuming its last token.
/// @tparam State Copyable decoder state type.
template<class State>
struct BeamHypothesis {
  std::vector<int> tokens;        ///< Emitted tokens, excluding BOS and including EOS if reached.
  double log_probability = 0.0;  ///< Sum of the normalized log probabilities.
  State state;                   ///< State used to expand this prefix if it has not ended.
};

/// @brief Decode up to a cap with stable cumulative log probabilities.
/// Finished paths keep their scores and compete with unfinished paths at
/// each depth. Equal scores are ordered lexicographically by token IDs.
/// @tparam State Copyable decoder state type.
/// @tparam Decode Caller-provided decoder callable.
/// @param initial Initial state, before consuming BOS.
/// @param bos Token passed to the first decoder call.
/// @param eos Terminal token ID in the decoder vocabulary.
/// @param width Positive number of hypotheses to retain per step.
/// @param max_steps Positive maximum number of output tokens.
/// @param decode Callable taking the previous token and parent hypothesis, returning
/// a pair containing the next state and a vector of finite logits.
/// @return Up to width hypotheses in decreasing score order.
/// @throws std::invalid_argument Invalid bounds, EOS ID or decoder logits.
/// @code
/// auto beams = dm::nn::BeamSearch(0, 2, 1, 3, 5, [](int, const auto& parent) {
///   return std::pair{parent.state + 1, std::vector<double>{0.0, 1.0}};
/// });
/// @endcode
template<class State, class Decode>
std::vector<BeamHypothesis<State>> BeamSearch(
    State initial, int bos, int eos, int width, int max_steps, Decode decode) {
  if (width <= 0 || max_steps <= 0 || eos < 0) {
    throw std::invalid_argument("beam width, step cap or EOS");
  }
  using Hypothesis = BeamHypothesis<State>;
  const auto better = [](const Hypothesis& a, const Hypothesis& b) {
    return a.log_probability == b.log_probability ? a.tokens < b.tokens
                                                   : a.log_probability > b.log_probability;
  };
  std::vector<Hypothesis> beam{{{}, 0.0, std::move(initial)}};
  for (int depth = 0; depth < max_steps; ++depth) {
    std::vector<Hypothesis> candidates;
    bool expanded = false;
    for (const auto& parent : beam) {
      const int last = parent.tokens.empty() ? bos : parent.tokens.back();
      if (!parent.tokens.empty() && last == eos) {
        candidates.push_back(parent);
        continue;
      }

      expanded = true;
      auto [state, logits] = decode(last, parent);
      if (std::size_t(eos) >= logits.size() || logits.size() > std::size_t(std::numeric_limits<int>::max()) ||
          !std::all_of(logits.begin(), logits.end(), [](double v) { return std::isfinite(v); })) {
        throw std::invalid_argument("invalid beam decoder logits");
      }
      const double largest = *std::max_element(logits.begin(), logits.end());
      double sum = 0.0;
      for (const double value : logits) {
        sum += std::exp(value - largest);
      }
      const double log_sum = std::log(sum);
      for (std::size_t token = 0; token < logits.size(); ++token) {
        auto prefix = parent.tokens;
        prefix.push_back(static_cast<int>(token));
        const double contribution = (double(logits[token]) - largest) - log_sum;
        candidates.push_back({std::move(prefix), parent.log_probability + contribution, state});
      }
    }
    if (!expanded) break;
    const auto keep = std::min(candidates.size(), std::size_t(width));
    auto end = candidates.begin() + keep;
    std::partial_sort(candidates.begin(), end, candidates.end(), better);
    candidates.erase(end, candidates.end());
    beam = std::move(candidates);
  }
  return beam;
}

}  // namespace datamunge::nn
