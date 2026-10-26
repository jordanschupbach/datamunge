// CMD: just run-nn q_learning
// Q-learning.
//
// Port of tflow's q_learning (TF-free apart from the Q-table, so the numbers
// match tflow exactly). Off-policy TD control: Q(s, a) moves toward
// r + gamma max_a' Q(s', a') regardless of which action the behaviour policy
// takes next. This program trains on the cliff and on a slippery grid world,
// tracks the gap between the behaviour policy's return and the greedy
// policy's return, and shows the maximisation bias that Double Q-learning
// removes. The vsim videos of the greedy policies are omitted: the best of the
// same 3 seeded episodes is played and reported instead. The port adds checks
// (exit 1 on failure).

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

struct Curve {
  std::vector<float> online, greedy;  // per evaluation point
};

Curve QLearn(rl::GridBase& env, int episodes, float alpha, float eps, float gamma, int eval_every, uint64_t seed,
             rl::QTable* out_Q, bool double_q = false) {
  const int          S = env.num_states(), A = env.action_space().n;
  rl::QTable         Q1(S, A), Q2(S, A);
  std::mt19937_64    rng(seed);
  Curve              curve;
  std::vector<float> online;
  for (int ep = 0; ep < episodes; ++ep) {
    int   s = static_cast<int>(env.reset(seed * 7 + ep)[0]);
    float ret = 0;
    while (true) {
      // Behaviour: epsilon-greedy on Q1 + Q2 (just Q1 in the single case).
      int a;
      if (std::uniform_real_distribution<float>(0, 1)(rng) < eps) {
        a = static_cast<int>(rng() % A);
      } else {
        std::vector<float> sum(A);
        for (int b = 0; b < A; ++b) sum[b] = Q1(s, b) + (double_q ? Q2(s, b) : 0.f);
        a = rl::ArgMaxRandomTie(sum.data(), A, rng);
      }
      rl::StepResult r = env.step(a);
      ret += r.reward;
      const int s2 = static_cast<int>(r.obs[0]);
      if (!double_q) {
        const float target = r.reward + (r.terminated ? 0.f : gamma * Q1.Max(s2));
        Q1(s, a) += alpha * (target - Q1(s, a));
      } else {
        // Double Q-learning: one table picks the action, the other scores it.
        rl::QTable& upd = (rng() % 2) ? Q1 : Q2;
        rl::QTable& other = (&upd == &Q1) ? Q2 : Q1;
        const int   best = upd.Greedy(s2, rng);
        const float target = r.reward + (r.terminated ? 0.f : gamma * other(s2, best));
        upd(s, a) += alpha * (target - upd(s, a));
      }
      s = s2;
      if (r.done()) break;
    }
    online.push_back(ret);
    if ((ep + 1) % eval_every == 0) {
      curve.online.push_back(rl::Mean(online, ep + 1 - eval_every, ep + 1));
      curve.greedy.push_back(rl::GreedyEval(env, Q1, 50, 12345).mean_return());
    }
  }
  if (out_Q) *out_Q = Q1;
  return curve;
}

Curve Report(rl::GridBase& env, int episodes, float alpha, float eps, float gamma) {
  std::printf("\n==== %s: alpha %.2f, epsilon %.2f, gamma %.2f, %d episodes\n", env.name().c_str(), alpha, eps, gamma,
              episodes);
  const int  eval_every = episodes / 10;
  rl::QTable Q(1, 1);
  Curve      c = QLearn(env, episodes, alpha, eps, gamma, eval_every, 1, &Q);
  std::printf("  %-10s", "episode");
  for (size_t i = 0; i < c.online.size(); ++i) std::printf(" %7d", static_cast<int>((i + 1) * eval_every));
  std::printf("\n  %-10s", "online");
  for (float v : c.online) std::printf(" %7.1f", v);
  std::printf("\n  %-10s", "greedy");
  for (float v : c.greedy) std::printf(" %7.1f", v);
  std::printf("\n  greedy policy:\n%s", rl::PolicyGrid(env, Q).c_str());
  // tflow films the greedy policy here (the overlay showing Q(s, .) at every cell).
  rl::PlayBestOfSeeds(
      env, [&](const rl::Obs& o) { return rl::GreedyQ(Q.Row(static_cast<int>(o[0])), Q.A); }, "q_learning_greedy",
      4242, 3);
  return c;
}

// Sutton & Barto example 6.7: from A, "left" leads to B where every action
// pays N(-0.1, 1); "right" ends with 0.
struct TwoState : public rl::TabularEnv {
  TwoState() : rl::TabularEnv(10) {}
  std::string                 name() const override { return "TwoState"; }
  int                         num_states() const override { return 2; }
  int                         state() const override { return s_; }
  rl::ActionSpace             action_space() const override { return rl::ActionSpace::Discrete(10); }
  std::vector<rl::Transition> transitions(int, int) const override { return {}; }
  rl::Obs                     Reset() override {
    BeginEpisode();
    s_ = 0;
    return StateObs();
  }
  rl::StepResult step(const rl::Action& a) override {
    rl::StepResult r;
    r.terminated = true;
    if (s_ == 0) {
      if (a.discrete == 0) {  // left -> B
        s_ = 1;
        r.terminated = false;
      }
    } else {
      r.reward = Normal(-0.1f, 1.f);
    }
    r.obs = StateObs();
    return r;
  }
  int s_ = 0;
};

}  // namespace

int main() {
  try {
    rl::CliffWalking cliff;
    const Curve      cc = Report(cliff, 500, 0.5f, 0.1f, 1.f);
    rl::GridWorld    grid(/*slip_prob=*/0.2f);
    const Curve      gc = Report(grid, 2000, 0.1f, 0.1f, 0.95f);

    // Maximisation bias: Q-learning's max over B's noisy estimates makes
    // "left" look good for a long time.
    std::cout << "\n==== Maximisation bias (Sutton & Barto 6.7): fraction of 'left' actions from A\n";
    TwoState two;
    double   left_frac[2][3];
    for (bool dbl : {false, true}) {
      int       lefts[3] = {0, 0, 0};
      const int runs = 1000, episodes = 300;
      for (int run = 0; run < runs; ++run) {
        rl::QTable      Q1(2, 10), Q2(2, 10);
        std::mt19937_64 rng(run);
        for (int ep = 0; ep < episodes; ++ep) {
          int s = static_cast<int>(two.reset(run * 1000 + ep)[0]);
          while (true) {
            int a;
            if (std::uniform_real_distribution<float>(0, 1)(rng) < 0.1f) {
              a = static_cast<int>(rng() % (s == 0 ? 2 : 10));
            } else {
              std::vector<float> sum(s == 0 ? 2 : 10);
              for (size_t b = 0; b < sum.size(); ++b) sum[b] = Q1(s, b) + (dbl ? Q2(s, b) : 0.f);
              a = rl::ArgMaxRandomTie(sum.data(), sum.size(), rng);
            }
            if (s == 0 && a == 0) lefts[ep < 100 ? 0 : (ep < 200 ? 1 : 2)]++;
            rl::StepResult r = two.step(a);
            const int      s2 = static_cast<int>(r.obs[0]);
            auto           maxq = [&](rl::QTable& q, int st) {
              float m = q(st, 0);
              for (int b = 1; b < (st == 0 ? 2 : 10); ++b) m = std::max(m, q(st, b));
              return m;
            };
            if (!dbl) {
              Q1(s, a) += 0.1f * (r.reward + (r.terminated ? 0.f : maxq(Q1, s2)) - Q1(s, a));
            } else {
              rl::QTable& upd = (rng() % 2) ? Q1 : Q2;
              rl::QTable& oth = (&upd == &Q1) ? Q2 : Q1;
              int         best = 0;
              for (int b = 1; b < (s2 == 0 ? 2 : 10); ++b)
                if (upd(s2, b) > upd(s2, best)) best = b;
              upd(s, a) += 0.1f * (r.reward + (r.terminated ? 0.f : oth(s2, best)) - upd(s, a));
            }
            s = s2;
            if (r.done()) break;
          }
        }
      }
      for (int k = 0; k < 3; ++k) left_frac[dbl][k] = static_cast<double>(lefts[k]) / (runs * 100);
      std::printf("  %-18s episodes 1-100: %.0f%%   101-200: %.0f%%   201-300: %.0f%%   (optimal: 5%%)\n",
                  dbl ? "Double Q-learning" : "Q-learning", 100.0 * left_frac[dbl][0], 100.0 * left_frac[dbl][1],
                  100.0 * left_frac[dbl][2]);
    }

    // Checks.
    check::Require(cc.greedy.back() == -13.f, "Q-learning's greedy cliff policy should be the optimal edge path (-13)");
    check::Require(cc.online.back() < -20.f, "epsilon-greedy behaviour on the cliff should still fall off sometimes");
    check::Require(gc.greedy.back() > -14.f, "GridWorld greedy policy should average better than -14");
    check::Require(gc.online.back() > gc.online.front() + 10.f, "GridWorld online return should improve");
    check::Require(left_frac[0][0] > 0.5, "Q-learning should be biased toward 'left' early on");
    check::Require(left_frac[1][0] < left_frac[0][0] - 0.3, "Double Q-learning should remove most of the bias");
    check::Require(left_frac[1][2] < 0.1, "Double Q-learning should approach the optimal 5% left");
    std::cout << "\nchecks passed (vsim videos omitted)\n";
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
