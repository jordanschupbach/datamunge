// datamunge::nn port: copied from tflow_examples src/common/rl/utils.h (TF-free code, unchanged).
// Small helpers shared by the RL examples: episode statistics, random and
// greedy rollouts, and printing.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

#include "common/rl/env.hpp"

namespace rl {

struct EpisodeStats {
  std::vector<float> returns;
  std::vector<int> lengths;
  float mean_return() const {
    return returns.empty() ? 0.f
                           : std::accumulate(returns.begin(), returns.end(), 0.f) / returns.size();
  }
  float mean_length() const {
    return lengths.empty() ? 0.f
                           : std::accumulate(lengths.begin(), lengths.end(), 0.f) / lengths.size();
  }
  float min_return() const { return *std::min_element(returns.begin(), returns.end()); }
  float max_return() const { return *std::max_element(returns.begin(), returns.end()); }
};

// Run `episodes` episodes of `env` choosing actions with `policy(obs)`.
inline EpisodeStats RunEpisodes(Env& env, int episodes, uint64_t seed,
                                const std::function<Action(const Obs&)>& policy) {
  EpisodeStats stats;
  for (int e = 0; e < episodes; ++e) {
    Obs obs = env.reset(seed + e);
    float ret = 0;
    int len = 0;
    while (true) {
      StepResult r = env.step(policy(obs));
      ret += r.reward;
      ++len;
      obs = r.obs;
      if (r.done()) break;
    }
    stats.returns.push_back(ret);
    stats.lengths.push_back(len);
  }
  return stats;
}

// Uniformly random actions.
inline EpisodeStats RunRandom(Env& env, int episodes, uint64_t seed) {
  std::mt19937_64 rng(seed);
  const ActionSpace space = env.action_space();
  return RunEpisodes(env, episodes, seed, [&](const Obs&) { return space.Sample(rng); });
}

inline void PrintStats(const std::string& label, const EpisodeStats& s) {
  std::printf("  %-32s return mean %8.2f  min %8.2f  max %8.2f   length mean %6.1f\n",
              label.c_str(), s.mean_return(), s.min_return(), s.max_return(), s.mean_length());
}

inline int ArgMax(const std::vector<float>& v) {
  return static_cast<int>(std::max_element(v.begin(), v.end()) - v.begin());
}

// Argmax with ties broken uniformly at random.
inline int ArgMaxRandomTie(const float* q, int n, std::mt19937_64& rng) {
  float best = q[0];
  for (int i = 1; i < n; ++i) best = std::max(best, q[i]);
  std::vector<int> ties;
  for (int i = 0; i < n; ++i)
    if (q[i] == best) ties.push_back(i);
  return ties[rng() % ties.size()];
}

inline std::string FormatObs(const Obs& o, int max_elems = 8) {
  std::string s = "[";
  for (size_t i = 0; i < o.size() && static_cast<int>(i) < max_elems; ++i) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%s%.3f", i ? " " : "", o[i]);
    s += buf;
  }
  if (static_cast<int>(o.size()) > max_elems) s += " ...";
  return s + "]";
}

}  // namespace rl

