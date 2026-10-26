// Autoregressive beam search with additive log probabilities and terminal EOS
// retention (port of tflow's common/ml/beam_search.h). Each hypothesis owns
// the decoder state before consuming its most recent predicted token; the
// decoder callback turns a parent hypothesis into next-token logits and the
// next state. Ties are broken by the token sequence, so results are
// deterministic.
#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

namespace beam {

struct Hypothesis {
  std::vector<int>   tokens;
  double             score = 0;
  bool               ended = false;
  std::vector<float> h, c;
};

struct Step {
  std::vector<float> logits, h, c;
};

inline std::vector<double> LogSoftmax(const std::vector<float>& logits) {
  if (logits.empty()) throw std::invalid_argument("empty beam vocabulary");
  double maximum = *std::max_element(logits.begin(), logits.end()), sum = 0;
  for (auto v : logits) sum += std::exp(v - maximum);
  const double        logz = maximum + std::log(sum);
  std::vector<double> result;
  for (auto v : logits) result.push_back(v - logz);
  return result;
}

inline bool Better(const Hypothesis& a, const Hypothesis& b) {
  return a.score != b.score ? a.score > b.score : a.tokens < b.tokens;
}

/// Keeps the `width` best of all one-token extensions (ended hypotheses carry
/// over unchanged) for at most `cap` tokens or until every beam has ended.
inline std::vector<Hypothesis> BeamSearch(Hypothesis initial, int width, int cap, int eos,
                                          const std::function<Step(const Hypothesis&)>& decoder) {
  if (width < 1 || cap < 1) throw std::invalid_argument("positive beam width/cap required");
  std::vector<Hypothesis> beam{initial};
  for (int depth = 0; depth < cap; ++depth) {
    std::vector<Hypothesis> candidates;
    bool                    all_ended = true;
    for (const auto& parent : beam) {
      if (parent.ended) {
        candidates.push_back(parent);
        continue;
      }
      all_ended       = false;
      const auto next = decoder(parent);
      const auto lp   = LogSoftmax(next.logits);
      if (eos < 0 || eos >= static_cast<int>(lp.size())) throw std::invalid_argument("EOS outside vocabulary");
      for (std::size_t token = 0; token < lp.size(); ++token) {
        Hypothesis child = parent;
        child.tokens.push_back(static_cast<int>(token));
        child.score += lp[token];
        child.ended = static_cast<int>(token) == eos;
        child.h     = next.h;
        child.c     = next.c;
        candidates.push_back(std::move(child));
      }
    }
    if (all_ended) break;
    std::sort(candidates.begin(), candidates.end(), Better);
    if (static_cast<int>(candidates.size()) > width) candidates.resize(width);
    beam = std::move(candidates);
  }
  return beam;
}

/// The 3-token fixture where greedy is not optimal: hypothesis states are a
/// hash of the parent's state and last token, so each branch owns its state.
inline Step FixtureStep(const Hypothesis& parent) {
  const int          last = parent.tokens.empty() ? 3 : parent.tokens.back();
  const float        hash = parent.h[0] * 4 + last + 1;
  std::vector<float> probabilities = hash == 4    ? std::vector<float>{.6f, .35f, .05f}
                                     : hash == 17 ? std::vector<float>{.3f, .3f, .4f}
                                     : hash == 18 ? std::vector<float>{.025f, .025f, .95f}
                                                  : std::vector<float>{.05f, .05f, .9f};
  std::vector<float> logits;
  for (float p : probabilities) logits.push_back(std::log(p));
  return {logits, {hash}, {parent.c[0] + hash}};
}

}  // namespace beam
