// datamunge::nn port of tflow_examples src/common/rl/tabular.h: the Q-table is backed by a
// std::vector instead of a TF tensor (tensor() dropped); the rest is unchanged.
// Helpers for tabular methods: a Q-table,
// epsilon-greedy action selection, and printing policies and values on grids.
#pragma once

#include <cstdio>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "common/rl/grid_world.hpp"
#include "common/rl/utils.hpp"

namespace rl {

class QTable {
 public:
  QTable(int num_states, int num_actions, float init = 0.f)
      : S(num_states), A(num_actions), q_(static_cast<size_t>(num_states) * num_actions, init) {}
  float& operator()(int s, int a) { return q_[static_cast<size_t>(s) * A + a]; }
  float operator()(int s, int a) const { return q_[static_cast<size_t>(s) * A + a]; }
  const float* Row(int s) const { return q_.data() + static_cast<size_t>(s) * A; }
  float Max(int s) const {
    const float* r = Row(s);
    float m = r[0];
    for (int a = 1; a < A; ++a) m = std::max(m, r[a]);
    return m;
  }
  int Greedy(int s, std::mt19937_64& rng) const { return ArgMaxRandomTie(Row(s), A, rng); }
  int Greedy(int s) const {
    std::vector<float> r(Row(s), Row(s) + A);
    return ArgMax(r);
  }
  int EpsilonGreedy(int s, float eps, std::mt19937_64& rng) const {
    if (std::uniform_real_distribution<float>(0, 1)(rng) < eps) return static_cast<int>(rng() % A);
    return Greedy(s, rng);
  }
  int S, A;

 private:
  std::vector<float> q_;
};

// Greedy policy arrows on the grid.
inline std::string PolicyGrid(const GridBase& env, const QTable& q) {
  return env.Render([&](int r, int c) -> std::string {
    if (env.IsWall(r, c)) return "#";
    if (env.IsGoal(r, c)) return "G";
    return GridBase::kArrows[q.Greedy(env.Index(r, c))];
  });
}

// Greedy policy arrows from an explicit policy vector.
inline std::string PolicyGrid(const GridBase& env, const std::vector<int>& policy) {
  return env.Render([&](int r, int c) -> std::string {
    if (env.IsWall(r, c)) return "#";
    if (env.IsGoal(r, c)) return "G";
    return GridBase::kArrows[policy[env.Index(r, c)]];
  });
}

// State values printed as a table.
inline std::string ValueGrid(const GridBase& env, const std::vector<float>& v) {
  std::string s;
  for (int r = 0; r < env.height(); ++r) {
    for (int c = 0; c < env.width(); ++c) {
      char buf[16];
      if (env.IsWall(r, c)) std::snprintf(buf, sizeof buf, "%7s", "#");
      else std::snprintf(buf, sizeof buf, "%7.2f", v[env.Index(r, c)]);
      s += buf;
    }
    s += "\n";
  }
  return s;
}

// Run the greedy policy of a Q-table (ties broken by lowest index).

inline float Mean(const std::vector<float>& v, size_t from, size_t to) {
  if (to > v.size()) to = v.size();
  if (from >= to) return 0.f;
  return std::accumulate(v.begin() + from, v.begin() + to, 0.f) / (to - from);
}

}  // namespace rl

