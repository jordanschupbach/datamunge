// CMD: just run-nn sarsa
// SARSA and expected SARSA.
//
// Port of tflow's sarsa (TF-free apart from the Q-table, so the numbers match
// tflow exactly). On-policy TD control. SARSA updates Q(s, a) toward
// r + gamma Q(s', a') using the action a' the epsilon-greedy policy actually
// takes next; expected SARSA uses the expectation over the policy instead,
// which removes the variance from the sampled a'. The classic cliff-walking
// comparison shows the on-policy effect: because exploration sometimes steps
// off the cliff, SARSA learns the safer path along the top, while Q-learning
// (shown for reference) learns the optimal but risky path along the edge.
// The vsim videos of each greedy policy are omitted: the same seeded episode
// is played and reported instead. The port adds checks (exit 1 on failure).

#include <cstdio>
#include <iostream>
#include <vector>

#include "common/checks.hpp"
#include "common/rl/grid_world.hpp"
#include "common/rl/hud.hpp"
#include "common/rl/playback.hpp"
#include "common/rl/tabular.hpp"

// Experiment workflow from examples/nn/common/rl/utils.hpp.
namespace rl {
using namespace ::rl;
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
}  // namespace rl

// Experiment workflow from examples/nn/common/rl/tabular.hpp.
namespace rl {
using namespace ::rl;
inline EpisodeStats GreedyEval(TabularEnv& env, const QTable& q, int episodes, uint64_t seed) {
  return RunEpisodes(env, episodes, seed,
                     [&](const Obs& o) { return Action(q.Greedy(static_cast<int>(o[0]))); });
}
}  // namespace rl



namespace {

enum Method { kSarsa, kExpectedSarsa, kQLearning };
const char* kNames[] = {"SARSA", "Expected SARSA", "Q-learning"};

struct Result {
  std::vector<float> returns;  // online return per episode
  rl::QTable         Q;
};

Result Train(rl::GridBase& env, Method method, int episodes, float alpha, float eps, float gamma, uint64_t seed) {
  const int       S = env.num_states(), A = env.action_space().n;
  Result          res{{}, rl::QTable(S, A)};
  rl::QTable&     Q = res.Q;
  std::mt19937_64 rng(seed);
  for (int ep = 0; ep < episodes; ++ep) {
    int   s = static_cast<int>(env.reset(seed * 10007 + ep)[0]);
    int   a = Q.EpsilonGreedy(s, eps, rng);
    float ret = 0;
    while (true) {
      rl::StepResult r = env.step(a);
      ret += r.reward;
      const int s2 = static_cast<int>(r.obs[0]);
      const int a2 = Q.EpsilonGreedy(s2, eps, rng);  // the action we will take next
      float     next_value = 0;
      if (!r.terminated) {
        switch (method) {
          case kSarsa:
            next_value = Q(s2, a2);
            break;
          case kExpectedSarsa: {
            // E_pi[Q(s', .)] under epsilon-greedy: eps/A on every action plus
            // (1 - eps) on the greedy one.
            const float best = Q.Max(s2);
            float       expect = 0;
            for (int b = 0; b < A; ++b) expect += eps / A * Q(s2, b);
            next_value = expect + (1 - eps) * best;
            break;
          }
          case kQLearning:
            next_value = Q.Max(s2);
            break;
        }
      }
      Q(s, a) += alpha * (r.reward + gamma * next_value - Q(s, a));
      s = s2;
      a = a2;
      if (r.done()) break;
    }
    res.returns.push_back(ret);
  }
  return res;
}

}  // namespace

int main() {
  try {
    rl::CliffWalking env;
    const int        episodes = 500, runs = 20;
    const float      alpha = 0.5f, eps = 0.1f, gamma = 1.f;
    std::printf("== CliffWalking: alpha %.1f, epsilon %.1f, gamma %.0f, %d episodes, averaged over %d runs\n", alpha,
                eps, gamma, episodes, runs);
    std::printf("   mean online return (reward collected while exploring) per block of 50 episodes\n");
    std::printf("  %-16s", "episodes");
    for (int b = 0; b < episodes; b += 50) std::printf(" %7d", b + 50);
    std::printf("\n");

    double late_online[3], greedy_return[3];  // mean online return of episodes 101-500; greedy return
    for (Method m : {kSarsa, kExpectedSarsa, kQLearning}) {
      std::vector<double>     avg(episodes, 0.0);
      std::vector<rl::QTable> tables;
      for (int run = 0; run < runs; ++run) {
        Result r = Train(env, m, episodes, alpha, eps, gamma, run + 1);
        for (int e = 0; e < episodes; ++e) avg[e] += r.returns[e] / runs;
        if (run == runs - 1) tables.push_back(r.Q);
      }
      std::printf("  %-16s", kNames[m]);
      for (int b = 0; b < episodes; b += 50) {
        double block = 0;
        for (int e = b; e < b + 50; ++e) block += avg[e];
        std::printf(" %7.1f", block / 50);
      }
      std::printf("\n");
      late_online[m] = 0;
      for (int e = 100; e < episodes; ++e) late_online[m] += avg[e] / (episodes - 100);
      std::cout << "    greedy policy after training (last run):\n";
      std::string grid = rl::PolicyGrid(env, tables.back());
      for (size_t pos = 0; pos < grid.size();) {
        size_t end = grid.find('\n', pos);
        std::cout << "      " << grid.substr(pos, end - pos) << "\n";
        pos = end + 1;
      }
      const rl::EpisodeStats stats = rl::GreedyEval(env, tables.back(), 100, 0);
      rl::PrintStats("  greedy policy, 100 episodes", stats);
      greedy_return[m] = stats.mean_return();
      // tflow films this greedy episode (seed 0) here.
      const rl::QTable& Q = tables.back();
      rl::PlayEpisode(
          env, [&](const rl::Obs& o) { return rl::GreedyQ(Q.Row(static_cast<int>(o[0])), Q.A); },
          std::string("sarsa_greedy_") + kNames[m], 0);
    }
    std::cout << "\n  SARSA and expected SARSA earn more *while learning* (safer path); Q-learning's greedy\n"
                 "  policy is optimal (-13) but its epsilon-greedy behaviour keeps falling off the cliff.\n";

    // Checks: the on-policy effect.
    check::Require(late_online[kSarsa] > late_online[kQLearning] + 10,
                   "SARSA should earn more than Q-learning while learning");
    check::Require(late_online[kExpectedSarsa] > late_online[kQLearning] + 10,
                   "expected SARSA should earn more than Q-learning while learning");
    check::Require(greedy_return[kQLearning] == -13, "Q-learning's greedy policy should take the optimal edge (-13)");
    for (Method m : {kSarsa, kExpectedSarsa})
      check::Require(greedy_return[m] < -13 && greedy_return[m] >= -21,
                     std::string(kNames[m]) + "'s greedy policy should take a safer, longer path");
    std::cout << "\nchecks passed (vsim videos omitted)\n";
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
