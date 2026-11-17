/// @file viterbi.cpp
/// @brief Actual log-space Viterbi tables, backpointers and explicit impossibility.
#include <cmath>
#include <datamunge/algorithms/viterbi.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Execute the validated joint-path recurrence once and retain its actual tables.
ViterbiWork viterbi_work(const HiddenMarkovModel& hmm, const std::vector<std::size_t>& obs) {
  hmm_validate(hmm, obs);
  std::size_t n = hmm.num_states(), t = obs.size();
  ViterbiWork w;
  if (!t) {
    w.result.possible = true;
    return w;
  }
  if (!n || t > 1000000 / n || n > 20000000 / n || (t - 1) > 20000000 / (n * n))
    throw std::length_error("Viterbi budget");
  double neg = -std::numeric_limits<double>::infinity();
  // Map zero transition/emission weights to impossible log mass.
  auto log = [](double p) {
    return p > 0 ? std::log(p) : -std::numeric_limits<double>::infinity();
  };
  w.delta.assign(t, std::vector<double>(n, neg));
  w.predecessor.assign(t, std::vector<std::size_t>(n));
  for (std::size_t i = 0; i < n; ++i)
    w.delta[0][i] = log(hmm.initial[i]) + log(hmm.emission[i][obs[0]]);
  for (std::size_t step = 1; step < t; ++step)
    for (std::size_t j = 0; j < n; ++j) {
      double      best = neg;
      std::size_t arg  = 0;
      for (std::size_t i = 0; i < n; ++i) {
        ++w.transition_terms;
        double candidate = w.delta[step - 1][i] + log(hmm.transition[i][j]);
        if (candidate > best) {
          best = candidate;
          arg  = i;
        }
      }
      w.delta[step][j]       = best + log(hmm.emission[j][obs[step]]);
      w.predecessor[step][j] = arg;
    }
  double      best = neg;
  std::size_t last = 0;
  for (std::size_t i = 0; i < n; ++i)
    if (w.delta[t - 1][i] > best) {
      best = w.delta[t - 1][i];
      last = i;
    }
  w.result.log_probability = best;
  if (best == neg)
    return w;
  w.result.possible = true;
  w.result.states.assign(t, 0);
  w.result.states[t - 1] = last;
  for (std::size_t step = t - 1; step > 0; --step)
    w.result.states[step - 1] = w.predecessor[step][w.result.states[step]];
  return w;
}
/// @brief Return the same actual producer result through the historical decoding interface.
ViterbiResult viterbi(const HiddenMarkovModel& hmm, const std::vector<std::size_t>& obs) {
  return viterbi_work(hmm, obs).result;
}
} // namespace datamunge::algorithms
