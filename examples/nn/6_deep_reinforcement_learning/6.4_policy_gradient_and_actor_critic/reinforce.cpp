// CMD: just run-nn reinforce
// REINFORCE (Monte Carlo policy gradient).
//
// Port of tflow's reinforce. The simplest policy-gradient method (Williams
// 1992): play a whole episode with the stochastic policy, then push up the
// log-probability of every action taken in proportion to the return that
// followed it,
//   grad J = E[ sum_t grad log pi(a_t | s_t) G_t ].
// No critic, no replay, no bootstrapping. Monte Carlo returns make the
// estimate unbiased but noisy; the gradient norm is logged to show that.
//
// The TF graph (policy, loss, Adam with global-norm clipping) becomes eager
// nn ops over a ParameterStore. CartPole and the action sampling use the
// same std::mt19937_64 streams as TF; the weight initialisation uses nn's
// generator, so the run matches TF statistically, not digit for digit. The
// vsim video of the best evaluation episode is omitted.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <datamunge/nn/nn.hpp>
#include <numeric>
#include <random>
#include <vector>

#include "common/checks.hpp"
#include "common/rl/classic_control.hpp"
#include "common/rl/pg_common.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

int main(int argc, char** argv) {
  rl::PgConfig cfg;
  cfg.steps = 150000;
  rl::ParsePgConfig(argc, argv, &cfg);
  rl::CartPole env, eval_env;
  const int obs_dim = env.obs_size(), A = env.action_space().n;
  std::printf("== REINFORCE on CartPole, %d steps, lr %.0e, gamma %.2f\n", cfg.steps, cfg.lr, cfg.gamma);

  // Discounted returns of a short hand-checked episode.
  const std::vector<float> G3 = rl::DiscountedReturns({1, 1, 1}, {0, 0, 1}, 0.5f);
  check::Near(G3[0], 1.75, 1e-6, "G_0 = 1 + 0.5 + 0.25");
  check::Near(G3[2], 1.0, 1e-6, "G_2 = r_2 at the terminal step");

  nn::ParameterStore params(cfg.seed);
  auto policy = [&](const Tensor& obs) { return rl::BuildActorCritic(params, obs, A, cfg.hidden, /*with_value=*/false); };
  {
    nn::NoGradGuard no_grad;
    policy(nn::zeros({1, obs_dim}));
  }
  rl::Adam adam(params, cfg.lr, cfg.clip_norm);

  rl::Monitor monitor(rl::MonitorConfig(cfg), "entropy");
  std::mt19937_64 rng(cfg.seed);
  auto probs_of = [&](const rl::Obs& o) {
    nn::NoGradGuard no_grad;
    return policy(rl::Row(o)).probs.to_vector<float>();
  };
  auto greedy = [&](const rl::Obs& o) { return rl::ArgMax(probs_of(o)); };

  std::vector<rl::Obs> ep_obs;
  std::vector<int> ep_act;
  std::vector<float> ep_rew, ep_done, grad_norms;
  int step = 0, next_log = cfg.log_every;
  float ent = 0, best_eval = 0;
  while (step < cfg.steps) {
    // One full episode with the current stochastic policy.
    ep_obs.clear(), ep_act.clear(), ep_rew.clear(), ep_done.clear();
    rl::Obs o = env.reset();
    while (true) {
      const int a = rl::SampleAction(probs_of(o).data(), A, rng);
      rl::StepResult sr = env.step(a);
      ep_obs.push_back(o), ep_act.push_back(a), ep_rew.push_back(sr.reward), ep_done.push_back(sr.terminated ? 1.f : 0.f);
      o = sr.obs;
      ++step;
      if (sr.done()) break;
    }
    monitor.EndEpisode(std::accumulate(ep_rew.begin(), ep_rew.end(), 0.f));
    // Monte Carlo returns, then one gradient step on the whole episode.
    const std::vector<float> G = rl::DiscountedReturns(ep_rew, ep_done, cfg.gamma);
    const rl::ActorCritic ac = policy(rl::ObsMatrix(ep_obs));
    const Tensor logp_a = rl::LogProbOf(ac.logp, rl::IntVector(ep_act), A);
    // Minimising -mean(log pi(a_t|s_t) G_t) ascends the policy gradient.
    const Tensor loss = -nn::mean(logp_a * rl::FloatVector(G), {0});
    grad_norms.push_back(static_cast<float>(adam.Minimize(loss)));
    monitor.AddLoss(loss.item<float>());
    ent = nn::mean(ac.entropy.detach(), {0}).item<float>();
    check::Require(std::isfinite(grad_norms.back()) && ent >= 0 && ent <= std::log(2.0) + 1e-5, "finite gradient, entropy in [0, log 2]");
    if (step >= next_log) {
      const float ev = next_log % cfg.eval_every == 0 ? rl::Evaluate(eval_env, greedy, cfg.eval_episodes, 1000 + step) : NAN;
      if (!std::isnan(ev)) best_eval = std::max(best_eval, ev);
      monitor.Log(step, ent, ev);
      next_log += cfg.log_every;
    }
  }
  double mean = 0, sq = 0;
  for (float g : grad_norms) mean += g, sq += double(g) * g;
  mean /= grad_norms.size();
  std::printf("  gradient norm over %zu updates: mean %.2f, std %.2f (before clipping at %.1f)\n", grad_norms.size(), mean,
              std::sqrt(std::max(0.0, sq / grad_norms.size() - mean * mean)), cfg.clip_norm);
  const float final_eval = rl::Evaluate(eval_env, greedy, 20, 777);
  std::printf("  final greedy evaluation over 20 episodes: %.1f\n", final_eval);
  // A random policy averages about 22; TF peaks at 500 on this seed (final 138.6).
  check::Require(std::max(best_eval, final_eval) >= 150, "REINFORCE learned to balance (best greedy evaluation >= 150)");
  return 0;
}
