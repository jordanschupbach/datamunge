// CMD: just run-nn monte_carlo_control
// Monte Carlo control.
//
// Port of tflow's monte_carlo_control (TF-free apart from the Q-table, so the
// numbers match tflow exactly). Learn Q(s, a) from complete episodes: every
// state-action pair visited gets the discounted return that followed it. No
// model, no bootstrapping. Exploration comes from an epsilon-soft policy
// whose epsilon decays, in place of the "exploring starts" assumption. Q lives
// in the Q-table of common/rl/tabular.hpp (a std::vector here; a
// tensorflow::Tensor in tflow). The vsim videos of the greedy policy are
// omitted: the best of the same 3 seeded episodes is played and reported
// instead. The port adds checks on the learned policies (exit 1 on failure).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

#include "common/checks.hpp"
#include "common/rl/grid_world.hpp"
#include "common/rl/hud.hpp"
#include "common/rl/play_best.hpp"
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

struct Options {
  int   episodes = 3000;
  float gamma = 0.95f;
  bool  first_visit = true;
  float eps_final = 0.05f;
};

struct Outcome {
  float first_block, last_block;  // mean online return of the first / last 500 episodes
  float greedy_mean;              // greedy policy, 500 evaluation episodes
};

Outcome Train(rl::GridBase& env, const Options& opt, uint64_t seed) {
  std::printf("\n==== %s: %s-visit MC control, %d episodes, gamma %.2f\n", env.name().c_str(),
              opt.first_visit ? "first" : "every", opt.episodes, opt.gamma);
  const int          S = env.num_states(), A = env.action_space().n;
  rl::QTable         Q(S, A);
  std::vector<int>   visits(S * A, 0);  // for sample-average step sizes
  std::mt19937_64    rng(seed);
  std::vector<float> returns;

  struct Step {
    int   s, a;
    float r;
  };
  std::vector<Step> episode;
  Outcome           out{};

  for (int ep = 0; ep < opt.episodes; ++ep) {
    // epsilon decays from 1 toward eps_final so the policy becomes greedy in the limit.
    const float eps = std::max(opt.eps_final, 1.f / (1.f + ep / 100.f));
    episode.clear();
    rl::Obs obs = env.reset(seed * 7919 + ep);
    float   ret = 0;
    while (true) {
      const int      s = static_cast<int>(obs[0]);
      const int      a = Q.EpsilonGreedy(s, eps, rng);
      rl::StepResult r = env.step(a);
      episode.push_back({s, a, r.reward});
      ret += r.reward;
      obs = r.obs;
      if (r.done()) break;
    }
    returns.push_back(ret);

    // Mark first visits by scanning forward.
    std::vector<bool> is_first(episode.size(), true);
    if (opt.first_visit) {
      std::vector<bool> marked(S * A, false);
      for (size_t t = 0; t < episode.size(); ++t) {
        const int key = episode[t].s * A + episode[t].a;
        is_first[t] = !marked[key];
        marked[key] = true;
      }
    }
    // Backward pass: G accumulates the discounted return from each step on.
    float G = 0;
    for (int t = static_cast<int>(episode.size()) - 1; t >= 0; --t) {
      G = episode[t].r + opt.gamma * G;
      if (!is_first[t]) continue;
      const int key = episode[t].s * A + episode[t].a;
      visits[key]++;
      Q(episode[t].s, episode[t].a) += (G - Q(episode[t].s, episode[t].a)) / visits[key];
    }

    if ((ep + 1) % 500 == 0) {
      const float block = rl::Mean(returns, ep - 499, ep + 1);
      std::printf("  episodes %4d-%4d  eps %.2f  mean online return %8.2f\n", ep - 498, ep + 1, eps, block);
      if (ep + 1 == 500) out.first_block = block;
      out.last_block = block;
    }
  }

  std::cout << "  greedy policy:\n" << rl::PolicyGrid(env, Q);
  std::vector<float> V(S);
  for (int s = 0; s < S; ++s) V[s] = Q.Max(s);
  std::cout << "  max_a Q(s, a):\n" << rl::ValueGrid(env, V);
  const rl::EpisodeStats stats = rl::GreedyEval(env, Q, 500, 99);
  rl::PrintStats("greedy policy, 500 episodes", stats);
  out.greedy_mean = stats.mean_return();
  rl::PlayBestOfSeeds(
      env, [&](const rl::Obs& o) { return rl::GreedyQ(Q.Row(static_cast<int>(o[0])), Q.A); },
      "monte_carlo_control_greedy", 4242, 3);
  return out;
}

}  // namespace

int main() {
  try {
    Options      opt;
    rl::GridWorld grid(/*slip_prob=*/0.2f);
    const Outcome first = Train(grid, opt, 1);
    opt.first_visit = false;
    const Outcome every = Train(grid, opt, 1);

    // The cliff: MC's long, noisy returns make this slower than TD methods
    // (compare with sarsa and q_learning).
    rl::CliffWalking cliff(/*max_steps=*/200);
    opt.first_visit = true;
    opt.gamma = 1.f;
    opt.episodes = 5000;
    const Outcome cl = Train(cliff, opt, 2);

    // Checks. The shortest GridWorld path is 9 steps (return -9); with 20% slip
    // a good greedy policy averages about -12.
    for (const Outcome* o : {&first, &every}) {
      check::Require(o->greedy_mean > -14.f, "GridWorld greedy policy should average better than -14");
      check::Require(o->last_block > o->first_block + 5.f, "GridWorld online return should improve with training");
    }
    // On the cliff MC settles for a safe path (the optimum along the edge is -13).
    check::Require(cl.greedy_mean >= -25.f && cl.greedy_mean <= -13.f,
                   "CliffWalking greedy policy should reach the goal in 13-25 steps");
    check::Require(cl.last_block > cl.first_block + 100.f, "CliffWalking online return should improve with training");
    std::cout << "\nchecks passed (vsim videos omitted)\n";
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
