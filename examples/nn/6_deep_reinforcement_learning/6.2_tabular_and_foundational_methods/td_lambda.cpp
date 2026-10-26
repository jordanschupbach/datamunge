// CMD: just run-nn td_lambda
// n-step methods and TD(lambda) with eligibility traces.
//
// Port of tflow's td_lambda (TF-free apart from the Q-table, so the numbers
// match tflow exactly). One-step TD bootstraps immediately; Monte Carlo waits
// for the whole return. n-step SARSA sits in between by backing up n rewards
// before bootstrapping. SARSA(lambda) blends all n in one pass with an
// eligibility trace e(s, a): every visited pair keeps a decaying share of
// each TD error. The experiment measures how fast each variant learns a
// slippery grid. The vsim video of the final greedy policy is omitted: the
// best of the same 3 seeded episodes is played and reported instead. The
// port adds checks (exit 1 on failure).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <map>
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

const float kAlpha = 0.2f, kEps = 0.1f, kGamma = 0.95f;

// Mean online return over the first `episodes` episodes.
float NStepSarsa(rl::GridBase& env, int n, int episodes, uint64_t seed) {
  const int       S = env.num_states(), A = env.action_space().n;
  rl::QTable      Q(S, A);
  std::mt19937_64 rng(seed);
  double          total = 0;
  for (int ep = 0; ep < episodes; ++ep) {
    std::vector<int>   states{static_cast<int>(env.reset(seed * 31 + ep)[0])}, actions;
    std::vector<float> rewards{0.f};  // rewards[t+1] is the reward after step t
    actions.push_back(Q.EpsilonGreedy(states[0], kEps, rng));
    int T = 1 << 30;
    for (int t = 0;; ++t) {
      if (t < T) {
        rl::StepResult r = env.step(actions[t]);
        rewards.push_back(r.reward);
        total += r.reward;
        states.push_back(static_cast<int>(r.obs[0]));
        if (r.terminated) T = t + 1;
        else if (r.truncated) T = t + 1;  // treat the cut as the end of the episode
        else actions.push_back(Q.EpsilonGreedy(states[t + 1], kEps, rng));
        if (r.truncated) actions.push_back(0);  // placeholder; unused when T = t + 1
      }
      const int tau = t - n + 1;  // the time whose estimate is updated
      if (tau >= 0) {
        float G = 0;
        for (int i = tau + 1; i <= std::min(tau + n, T); ++i) G += std::pow(kGamma, i - tau - 1) * rewards[i];
        if (tau + n < T) G += std::pow(kGamma, n) * Q(states[tau + n], actions[tau + n]);
        Q(states[tau], actions[tau]) += kAlpha * (G - Q(states[tau], actions[tau]));
      }
      if (tau == T - 1) break;
    }
  }
  return static_cast<float>(total / episodes);
}

float SarsaLambda(rl::GridBase& env, float lambda, bool replacing, int episodes, uint64_t seed,
                  rl::QTable* out_Q = nullptr, std::vector<float>* trace_snapshot = nullptr) {
  const int          S = env.num_states(), A = env.action_space().n;
  rl::QTable         Q(S, A);
  std::vector<float> e(S * A);
  std::mt19937_64    rng(seed);
  double             total = 0;
  for (int ep = 0; ep < episodes; ++ep) {
    std::fill(e.begin(), e.end(), 0.f);
    int s = static_cast<int>(env.reset(seed * 31 + ep)[0]);
    int a = Q.EpsilonGreedy(s, kEps, rng);
    for (int t = 0;; ++t) {
      rl::StepResult r = env.step(a);
      total += r.reward;
      const int   s2 = static_cast<int>(r.obs[0]);
      const int   a2 = Q.EpsilonGreedy(s2, kEps, rng);
      const float delta = r.reward + (r.terminated ? 0.f : kGamma * Q(s2, a2)) - Q(s, a);
      if (replacing) e[s * A + a] = 1.f;
      else e[s * A + a] += 1.f;
      // Every pair with a trace shares the TD error, then traces decay.
      for (int i = 0; i < S * A; ++i) {
        if (e[i] == 0.f) continue;
        Q(i / A, i % A) += kAlpha * delta * e[i];
        e[i] *= kGamma * lambda;
      }
      if (trace_snapshot && ep == 0 && t == 12) *trace_snapshot = e;
      s = s2;
      a = a2;
      if (r.done()) break;
    }
  }
  if (out_Q) *out_Q = Q;
  return static_cast<float>(total / episodes);
}

}  // namespace

int main() {
  try {
    rl::GridWorld env(/*slip_prob=*/0.1f);
    const int     episodes = 150, runs = 20;
    std::printf("== %s (slip 0.1): mean return over the first %d episodes, averaged over %d runs\n"
                "   alpha %.2f, epsilon %.2f, gamma %.2f\n",
                env.name().c_str(), episodes, runs, kAlpha, kEps, kGamma);

    std::printf("\n-- n-step SARSA\n");
    std::map<int, double> nstep;
    for (int n : {1, 2, 4, 8, 16}) {
      double sum = 0;
      for (int run = 0; run < runs; ++run) sum += NStepSarsa(env, n, episodes, run + 1);
      nstep[n] = sum / runs;
      std::printf("  n = %2d   mean return %7.2f\n", n, sum / runs);
    }

    std::printf("\n-- SARSA(lambda)\n");
    std::printf("  %-8s %18s %18s\n", "lambda", "accumulating", "replacing");
    std::map<float, double> acc_ret, rep_ret;
    for (float lambda : {0.f, 0.5f, 0.8f, 0.9f, 0.95f}) {
      double acc = 0, rep = 0;
      for (int run = 0; run < runs; ++run) {
        acc += SarsaLambda(env, lambda, false, episodes, run + 1);
        rep += SarsaLambda(env, lambda, true, episodes, run + 1);
      }
      acc_ret[lambda] = acc / runs, rep_ret[lambda] = rep / runs;
      std::printf("  %-8.2f %18.2f %18.2f\n", lambda, acc / runs, rep / runs);
    }

    std::printf("\n-- Eligibility trace after 13 steps of the first episode (lambda 0.9, replacing)\n");
    std::vector<float> trace;
    rl::QTable         Q(1, 1);
    SarsaLambda(env, 0.9f, true, 1, 3, &Q, &trace);
    std::vector<float> per_state(env.num_states(), 0.f);
    for (int s = 0; s < env.num_states(); ++s)
      for (int a = 0; a < 4; ++a) per_state[s] = std::max(per_state[s], trace[s * 4 + a]);
    std::cout << "  max_a e(s, a); recent states carry the most credit:\n" << rl::ValueGrid(env, per_state);

    SarsaLambda(env, 0.9f, true, 1000, 1, &Q);
    std::cout << "\n-- Greedy policy after 1000 episodes of SARSA(0.9):\n" << rl::PolicyGrid(env, Q);
    const rl::EpisodeStats stats = rl::GreedyEval(env, Q, 200, 0);
    rl::PrintStats("greedy policy, 200 episodes", stats);
    rl::PlayBestOfSeeds(
        env, [&](const rl::Obs& o) { return rl::GreedyQ(Q.Row(static_cast<int>(o[0])), Q.A); },
        "td_lambda_sarsa_lambda_greedy", 4242, 3);

    // Checks. lambda = 0 is one-step SARSA (identical updates for both trace kinds).
    check::Near(acc_ret[0.f], rep_ret[0.f], 1e-9, "SARSA(0) should not depend on the trace kind");
    check::Require(nstep[4] > nstep[1] + 3, "4-step SARSA should learn faster than 1-step");
    check::Require(nstep[4] > nstep[16], "very long backups should lose to intermediate n");
    check::Require(acc_ret[0.9f] > acc_ret[0.f] + 5, "SARSA(0.9) should learn much faster than SARSA(0)");
    float max_trace = *std::max_element(per_state.begin(), per_state.end());
    check::Near(max_trace, kGamma * 0.9f, 1e-6, "the last visited pair's replacing trace after one decay");
    check::Require(stats.mean_return() > -12.f, "greedy policy after SARSA(0.9) should average better than -12");
    std::cout << "\nchecks passed (vsim video omitted)\n";
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
