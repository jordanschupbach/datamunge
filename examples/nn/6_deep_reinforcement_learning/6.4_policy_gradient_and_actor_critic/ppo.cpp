// CMD: just run-nn ppo
// Proximal policy optimization (PPO), clipped and KL-penalty variants.
//
// Port of tflow's ppo. PPO (Schulman et al. 2017) reuses each batch of
// experience for several epochs of minibatch updates while keeping the new
// policy close to the one that collected the data. With
// r_t = pi_new(a|s) / pi_old(a|s):
//   clipped:    L = E[ min(r_t A_t, clip(r_t, 1-eps, 1+eps) A_t) ]
//   KL penalty: L = E[ r_t A_t ] - beta KL(pi_old || pi_new), beta adapted
//               up or down so the KL stays near a target.
// Both variants are trained here on the same GAE-based batches.
//
// The TF graph (actor-critic, losses, Adam with global-norm clipping)
// becomes eager nn ops over a ParameterStore; minibatches are gathered with
// nn::take_rows instead of being copied into feed tensors. The vectorised
// CartPole, the action sampling and the minibatch shuffles use the same
// std::mt19937_64 streams as TF; the weight initialisation uses nn's
// generator, so results match TF statistically, not digit for digit. The
// vsim videos of the best evaluation episodes are omitted.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <datamunge/nn/nn.hpp>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "common/checks.hpp"
#include "common/rl/classic_control.hpp"
#include "common/rl/pg_common.hpp"
#include "common/rl/vec_env.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

float Train(bool clipped, const rl::PgConfig& cfg) {
  const float clip_eps = 0.2f, kl_target = 0.01f;
  const int epochs = 4, minibatch = 64;
  std::printf("\n== PPO-%s on CartPole: %d envs x %d steps per batch, %d epochs of minibatch %d, %d steps total\n",
              clipped ? "clip" : "penalty", cfg.n_envs, cfg.n_steps, epochs, minibatch, cfg.steps);
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

  struct Step {
    float loss, kl, clip_frac;
  };
  // One update on a minibatch; returns the loss, KL(pi_old || pi_new) and
  // the fraction of clipped ratios, all from before the update.
  auto update = [&](const Tensor& obs, const Tensor& act, const Tensor& adv, const Tensor& ret, const Tensor& logp_old,
                    const Tensor& probs_old, float beta) {
    const rl::ActorCritic ac = net(obs);
    const Tensor logp_a = rl::LogProbOf(ac.logp, act, A);
    const Tensor ratio = nn::exp(logp_a - logp_old);
    // KL(pi_old || pi_new) with the old probabilities.
    const Tensor kl = nn::mean(nn::sum(probs_old * (nn::log(nn::maximum(probs_old, nn::scalar(1e-8))) - ac.logp), {1}), {0});
    Tensor policy_loss;
    if (clipped) {
      const Tensor unclipped = ratio * adv;
      const Tensor clipped_r = nn::clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv;
      policy_loss = -nn::mean(nn::minimum(unclipped, clipped_r), {0});
    } else {
      policy_loss = -nn::mean(ratio * adv, {0}) + beta * kl;
    }
    const Tensor value_loss = nn::mean(nn::square(ret - ac.value), {0});
    const Tensor entropy = nn::mean(ac.entropy, {0});
    const Tensor loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy;
    adam.Minimize(loss);
    nn::NoGradGuard no_grad;
    const Tensor clip_frac = nn::mean(nn::cast(nn::greater(nn::abs(ratio.detach() - 1.0), clip_eps), nn::DType::Float32), {0});
    return Step{loss.item<float>(), kl.item<float>(), clip_frac.item<float>()};
  };

  rl::Monitor monitor(rl::MonitorConfig(cfg), "kl");
  auto policy = [&](const std::vector<rl::Obs>& o, std::vector<float>* probs, std::vector<float>* values) {
    nn::NoGradGuard no_grad;
    const rl::ActorCritic ac = net(rl::ObsMatrix(o));
    *probs = ac.probs.to_vector<float>();
    *values = ac.value.to_vector<float>();
  };
  auto greedy = [&](const rl::Obs& o) {  // the most probable action, for evaluation
    nn::NoGradGuard no_grad;
    return rl::ArgMax(net(rl::Row(o)).probs.to_vector<float>());
  };
  rl::RolloutCollector collector(venv, A, cfg.seed + 1);
  std::mt19937_64 rng(cfg.seed + 2);
  float beta_value = 1.f, last_kl = 0, last_clip = 0, best_eval = 0;
  long next_log = cfg.log_every;
  while (collector.total_steps() < cfg.steps) {
    rl::Rollout ro = collector.Collect(cfg.n_steps, policy, &monitor);
    // GAE per environment, then normalised advantages over the batch.
    std::vector<float> advantages, returns, logps, a_i, r_i;
    for (int i = 0; i < ro.n_envs; ++i) {
      rl::Gae(ro.rew[i], ro.val[i], ro.done[i], ro.last_val[i], cfg.gamma, cfg.lam, &a_i, &r_i);
      advantages.insert(advantages.end(), a_i.begin(), a_i.end());
      returns.insert(returns.end(), r_i.begin(), r_i.end());
      logps.insert(logps.end(), ro.logp[i].begin(), ro.logp[i].end());
    }
    const float mean = std::accumulate(advantages.begin(), advantages.end(), 0.f) / advantages.size();
    float var = 0;
    for (float a : advantages) var += (a - mean) * (a - mean);
    const float sd = std::sqrt(var / advantages.size()) + 1e-8f;
    for (float& a : advantages) a = (a - mean) / sd;
    const Tensor all_obs = rl::ObsMatrix(ro.FlatObs()), all_act = rl::IntVector(ro.FlatAct());
    const Tensor all_adv = rl::FloatVector(advantages), all_ret = rl::FloatVector(returns), all_logp = rl::FloatVector(logps);
    const int N = static_cast<int>(all_obs.size(0));
    // The old policy's full distributions, for the exact KL.
    Tensor old_probs;
    {
      nn::NoGradGuard no_grad;
      old_probs = net(all_obs).probs;
    }

    // Several epochs of shuffled minibatches on the same data.
    std::vector<int> order(N);
    std::iota(order.begin(), order.end(), 0);
    float kl_sum = 0, clip_sum = 0;
    int updates = 0;
    for (int epoch = 0; epoch < epochs; ++epoch) {
      std::shuffle(order.begin(), order.end(), rng);
      for (int start = 0; start + minibatch <= N; start += minibatch) {
        const std::vector<std::int64_t> rows(order.begin() + start, order.begin() + start + minibatch);
        const Step s = update(nn::take_rows(all_obs, rows), nn::take_rows(all_act, rows), nn::take_rows(all_adv, rows),
                              nn::take_rows(all_ret, rows), nn::take_rows(all_logp, rows), nn::take_rows(old_probs, rows),
                              beta_value);
        check::Require(std::isfinite(s.loss) && s.kl > -1e-5f, "finite loss and nonnegative KL");
        monitor.AddLoss(s.loss);
        kl_sum += s.kl;
        clip_sum += s.clip_frac;
        ++updates;
      }
    }
    last_kl = kl_sum / updates;
    last_clip = clip_sum / updates;
    // Adaptive penalty coefficient (KL-penalty variant only).
    if (!clipped) {
      if (last_kl > 1.5f * kl_target) beta_value *= 2;
      else if (last_kl < kl_target / 1.5f) beta_value /= 2;
    }
    if (collector.total_steps() >= next_log) {
      const float eval = next_log % cfg.eval_every == 0 ? rl::Evaluate(eval_env, greedy, cfg.eval_episodes, 1000 + next_log) : NAN;
      if (!std::isnan(eval)) best_eval = std::max(best_eval, eval);
      monitor.Log(static_cast<int>(collector.total_steps()), last_kl, eval);
      std::printf("           %s\n", clipped ? ("clip fraction " + std::to_string(last_clip).substr(0, 5)).c_str()
                                              : ("beta " + std::to_string(beta_value).substr(0, 6)).c_str());
      next_log += cfg.log_every;
    }
  }
  const float ev = rl::Evaluate(eval_env, greedy, 20, 777);
  std::printf("  final greedy evaluation over 20 episodes: %.1f\n", ev);
  // A random policy averages about 22; TF's best evaluation is 500 for both variants on this seed.
  check::TrainingQuality(std::max(best_eval, ev) >= 150, "PPO learned to balance (best greedy evaluation >= 150)");
  return ev;
}

}  // namespace

int main(int argc, char** argv) {
  rl::PgConfig cfg;
  cfg.n_steps = 32;
  cfg.lr = 3e-4f;
  cfg.steps = 100000;
  cfg.log_every = 20000;
  cfg.eval_every = 20000;
  rl::ParsePgConfig(argc, argv, &cfg);

  // GAE on a hand-checked trajectory: with lambda = 1 the advantages are
  // Monte Carlo returns minus the values.
  std::vector<float> adv, ret;
  rl::Gae({1, 1, 1}, {0.5f, 0.5f, 0.5f}, {0, 0, 0}, 2.f, 0.5f, 1.f, &adv, &ret);
  check::Near(ret[0], 1 + 0.5 + 0.25 + 0.125 * 2, 1e-6, "GAE(lambda=1) return = bootstrapped discounted sum");
  check::Near(adv[2], 1 + 0.5 * 2 - 0.5, 1e-6, "GAE last advantage = TD error");

  const float clip = Train(true, cfg), penalty = Train(false, cfg);
  std::printf("\n  final greedy return: clipped %.1f, KL penalty %.1f\n", clip, penalty);
  return 0;
}
