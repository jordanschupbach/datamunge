// CMD: just run-nn gae
// Generalized advantage estimation (GAE).
//
// Port of tflow's gae. The advantage estimate can bootstrap after 1 step
// (low variance, biased by the critic), after the whole episode (unbiased,
// noisy), or anything in between. GAE(lambda) (Schulman et al. 2016) averages
// all n-step estimates with weights lambda^(n-1), exactly like TD(lambda):
//   A_t = sum_l (gamma lambda)^l delta_{t+l},   delta_t = r_t + gamma V(s_{t+1}) - V(s_t).
// lambda = 0 is the one-step TD advantage, lambda = 1 the Monte Carlo
// advantage. A2C with 20-step rollouts is trained with several lambdas.
//
// The TF graph becomes eager nn ops over a ParameterStore. The vectorised
// CartPole and the action sampling use the same std::mt19937_64 streams as
// TF; the weights use nn's generator, so results match TF statistically, not
// digit for digit. The vsim videos are replaced by PlayBestOfSeeds (same
// episodes, printed). tflow has no checks; the ones below test the GAE
// identities and that the agent learns.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <datamunge/nn/nn.hpp>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common/checks.hpp"
#include "common/rl/classic_control.hpp"
#include "common/rl/pg_common.hpp"
#include "common/rl/play_best.hpp"
#include "common/rl/vec_env.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

struct Result {
  float final_eval, best_eval;
};

Result Train(float lam, const rl::PgConfig& cfg) {
  std::printf("\n== A2C + GAE(lambda = %.2f) on CartPole: %d envs x %d steps, %d steps total\n", lam, cfg.n_envs,
              cfg.n_steps, cfg.steps);
  rl::VecEnv venv([] { return std::make_unique<rl::CartPole>(); }, cfg.n_envs, cfg.seed);
  rl::CartPole eval_env;
  const int obs_dim = venv.obs_size(), A = venv.action_space().n;

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
  float ent = 0, best = 0;
  long next_log = cfg.log_every;
  while (collector.total_steps() < cfg.steps) {
    rl::Rollout ro = collector.Collect(cfg.n_steps, policy, &monitor);
    std::vector<float> advantages, returns, a_i, r_i;
    for (int i = 0; i < ro.n_envs; ++i) {
      rl::Gae(ro.rew[i], ro.val[i], ro.done[i], ro.last_val[i], cfg.gamma, lam, &a_i, &r_i);
      advantages.insert(advantages.end(), a_i.begin(), a_i.end());
      returns.insert(returns.end(), r_i.begin(), r_i.end());
    }
    const rl::ActorCritic ac = net(rl::ObsMatrix(ro.FlatObs()));
    const Tensor logp_a = rl::LogProbOf(ac.logp, rl::IntVector(ro.FlatAct()), A);
    const Tensor entropy = nn::mean(ac.entropy, {0});
    const Tensor loss = -nn::mean(logp_a * rl::FloatVector(advantages), {0}) +
                        cfg.value_coef * nn::mean(nn::square(rl::FloatVector(returns) - ac.value), {0}) -
                        cfg.entropy_coef * entropy;
    adam.Minimize(loss);
    monitor.AddLoss(loss.item<float>());
    ent = entropy.item<float>();
    check::Require(std::isfinite(loss.item<float>()) && ent >= 0 && ent <= std::log(2.0) + 1e-5,
                   "finite loss, entropy in [0, log 2]");
    if (collector.total_steps() >= next_log) {
      const float ev =
          next_log % cfg.eval_every == 0 ? rl::Evaluate(eval_env, greedy, cfg.eval_episodes, 1000 + next_log) : NAN;
      if (!std::isnan(ev)) best = std::max(best, ev);
      monitor.Log(static_cast<int>(collector.total_steps()), ent, ev);
      next_log += cfg.log_every;
    }
  }
  const float ev = rl::Evaluate(eval_env, greedy, 20, 777);
  std::printf("  final greedy evaluation over 20 episodes: %.1f\n", ev);
  std::string tag = std::to_string(lam).substr(0, 4);
  std::replace(tag.begin(), tag.end(), '.', '_');
  rl::PlayBestOfSeeds(eval_env, explain, "gae_lambda_" + tag + "_cartpole", 4242);
  return {ev, std::max(best, ev)};
}

}  // namespace

int main(int argc, char** argv) {
  rl::PgConfig cfg;
  cfg.n_steps = 20;
  cfg.steps = 80000;
  cfg.log_every = 20000;
  cfg.eval_every = 20000;
  rl::ParsePgConfig(argc, argv, &cfg);

  // GAE identities on a hand-made trajectory (gamma 0.9, V = 0.5 everywhere, bootstrap V = 2).
  {
    const std::vector<float> r = {1, 0, 2}, v = {0.5f, 0.5f, 0.5f}, done = {0, 0, 0};
    std::vector<float> adv, ret, mc = rl::DiscountedReturns(r, done, 0.9f, 2.f);
    rl::Gae(r, v, done, 2.f, 0.9f, 0.f, &adv, &ret);
    check::Near(adv[0], 1 + 0.9 * 0.5 - 0.5, 1e-6, "GAE(0) is the one-step TD error");
    rl::Gae(r, v, done, 2.f, 0.9f, 1.f, &adv, &ret);
    for (int t = 0; t < 3; ++t) check::Near(ret[t], mc[t], 1e-5, "GAE(1) returns are the bootstrapped Monte Carlo returns");
    const std::vector<float> stop = {0, 1, 0};
    rl::Gae(r, v, stop, 2.f, 0.9f, 0.95f, &adv, &ret);
    check::Near(adv[1], 0 - 0.5, 1e-6, "a terminal step cuts the bootstrap and the trace");
  }

  std::vector<std::pair<float, Result>> results;
  for (float lam : {0.f, 0.5f, 0.95f, 1.f}) results.push_back({lam, Train(lam, cfg)});
  std::printf("\n  final greedy return by lambda:");
  for (auto& [lam, r] : results) std::printf("  %.2f: %.1f", lam, r.final_eval);
  std::printf("\n");
  // A random policy averages about 22. TF seed 0: final 64 / 223 / 374 / 500 for
  // lambda 0 / 0.5 / 0.95 / 1; lambda = 0 learns slowest (critic bias).
  for (auto& [lam, r] : results)
    if (lam >= 0.9f)
      check::TrainingQuality(r.best_eval >= 150, "A2C + GAE(" + std::to_string(lam).substr(0, 4) +
                                             ") learned to balance (best greedy evaluation >= 150)");
  return 0;
}
