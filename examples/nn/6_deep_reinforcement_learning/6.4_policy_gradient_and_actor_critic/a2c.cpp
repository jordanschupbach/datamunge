// CMD: just run-nn a2c
// Advantage actor-critic (A2C).
//
// Port of tflow's a2c. Synchronous, batched actor-critic: n_envs
// environments are stepped together for n_steps, the n-step bootstrapped
// returns
//   R_t = r_t + ... + gamma^{k-1} r_{t+k-1} + gamma^k V(s_{t+k})
// give advantages R_t - V(s_t), and one gradient step is taken on the
// whole batch with the policy, value and entropy terms combined. The batch
// of decorrelated environments replaces A3C's asynchronous threads.
//
// The TF graph (actor-critic, loss, Adam with global-norm clipping) becomes
// eager nn ops over a ParameterStore. The vectorised CartPole and the action
// sampling use the same std::mt19937_64 streams as TF; the weights use nn's
// generator, so results match TF statistically, not digit for digit. The
// vsim video of the best evaluation episode is replaced by PlayBestOfSeeds,
// which replays the same episode and prints its length and return.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <datamunge/nn/nn.hpp>
#include <memory>
#include <vector>

#include "common/checks.hpp"
#include "common/rl/classic_control.hpp"
#include "common/rl/pg_common.hpp"
#include "common/rl/play_best.hpp"
#include "common/rl/vec_env.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

int main(int argc, char** argv) {
  rl::PgConfig cfg;
  rl::ParsePgConfig(argc, argv, &cfg);
  rl::VecEnv venv([] { return std::make_unique<rl::CartPole>(); }, cfg.n_envs, cfg.seed);
  rl::CartPole eval_env;
  const int obs_dim = venv.obs_size(), A = venv.action_space().n;
  std::printf("== A2C on CartPole: %d envs x %d steps per update, %d steps total, lr %.0e\n", cfg.n_envs, cfg.n_steps,
              cfg.steps, cfg.lr);

  // n-step return with a bootstrap, and a terminal that cuts it.
  const std::vector<float> R = rl::DiscountedReturns({1, 1, 1}, {0, 1, 0}, 0.5f, 4.f);
  check::Near(R[2], 1 + 0.5 * 4, 1e-6, "R_2 = r_2 + gamma V(s_3)");
  check::Near(R[0], 1 + 0.5 * 1, 1e-6, "R_0 stops at the terminal step 1");

  nn::ParameterStore params(cfg.seed);
  auto net = [&](const Tensor& obs) { return rl::BuildActorCritic(params, obs, A, cfg.hidden); };
  {
    nn::NoGradGuard no_grad;
    net(nn::zeros({1, obs_dim}));
  }
  rl::Adam adam(params, cfg.lr, cfg.clip_norm);

  rl::Monitor monitor(rl::MonitorConfig(cfg), "entropy");
  auto policy = [&](const std::vector<rl::Obs>& o, std::vector<float>* probs, std::vector<float>* values) {
    nn::NoGradGuard no_grad;
    const rl::ActorCritic ac = net(rl::ObsMatrix(o));
    *probs = ac.probs.to_vector<float>();
    *values = ac.value.to_vector<float>();
  };
  auto explain = [&](const rl::Obs& o) {  // pi(.|s) and V(s): the most probable action, for evaluation and the replay
    nn::NoGradGuard no_grad;
    const rl::ActorCritic ac = net(rl::Row(o));
    return rl::GreedyPi(ac.probs.to_vector<float>(), ac.value.item<float>());
  };
  auto greedy = [&](const rl::Obs& o) { return explain(o).action.discrete; };
  rl::RolloutCollector collector(venv, A, cfg.seed + 1);

  float ent = 0, best_eval = 0;
  long next_log = cfg.log_every;
  while (collector.total_steps() < cfg.steps) {
    rl::Rollout ro = collector.Collect(cfg.n_steps, policy, &monitor);
    // n-step returns per environment, bootstrapped from V(last obs).
    std::vector<float> advantages, returns;
    for (int i = 0; i < ro.n_envs; ++i) {
      const std::vector<float> Ri = rl::DiscountedReturns(ro.rew[i], ro.done[i], cfg.gamma, ro.last_val[i]);
      for (int t = 0; t < ro.n_steps; ++t) {
        returns.push_back(Ri[t]);
        advantages.push_back(Ri[t] - ro.val[i][t]);
      }
    }
    const rl::ActorCritic ac = net(rl::ObsMatrix(ro.FlatObs()));
    const Tensor logp_a = rl::LogProbOf(ac.logp, rl::IntVector(ro.FlatAct()), A);
    const Tensor policy_loss = -nn::mean(logp_a * rl::FloatVector(advantages), {0});
    const Tensor value_loss = nn::mean(nn::square(rl::FloatVector(returns) - ac.value), {0});
    const Tensor entropy = nn::mean(ac.entropy, {0});
    const Tensor loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy;
    adam.Minimize(loss);
    monitor.AddLoss(loss.item<float>());
    ent = entropy.item<float>();
    check::Require(std::isfinite(loss.item<float>()) && ent >= 0 && ent <= std::log(2.0) + 1e-5,
                   "finite loss, entropy in [0, log 2]");
    if (collector.total_steps() >= next_log) {
      const float ev =
          next_log % cfg.eval_every == 0 ? rl::Evaluate(eval_env, greedy, cfg.eval_episodes, 1000 + next_log) : NAN;
      if (!std::isnan(ev)) best_eval = std::max(best_eval, ev);
      monitor.Log(static_cast<int>(collector.total_steps()), ent, ev);
      next_log += cfg.log_every;
    }
  }
  const float final_eval = rl::Evaluate(eval_env, greedy, 20, 777);
  std::printf("  final greedy evaluation over 20 episodes: %.1f\n", final_eval);
  rl::PlayBestOfSeeds(eval_env, explain, "a2c_cartpole", 4242);
  // A random policy averages about 22. Over seeds 0..4 the best greedy
  // evaluation is 500 for every seed in TF and here; final evaluations average
  // 413 (TF) and 345 (this port).
  check::TrainingQuality(std::max(best_eval, final_eval) >= 150, "A2C learned to balance (best greedy evaluation >= 150)");
  return 0;
}
