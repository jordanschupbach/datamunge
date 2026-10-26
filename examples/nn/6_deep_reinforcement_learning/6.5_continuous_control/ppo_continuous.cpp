// CMD: just run-nn ppo_continuous
// PPO for continuous actions (Gaussian policy).
//
// Port of tflow's ppo_continuous. The same clipped surrogate as the discrete
// PPO, with a diagonal Gaussian policy: the network outputs the mean, the log
// standard deviation is a free parameter vector, and actions are sampled on
// the host. Actions outside [-1, 1] are clipped when sent to the environment
// but the log-probability uses the unclipped sample, the usual convention.
// Pendulum rewards are scaled by 0.1 so the value regression is well
// conditioned.
//
// The TF graph becomes eager nn ops over one ParameterStore (Adam with
// global-norm clipping, rl::Adam). The vectorised Pendulum, the action
// sampling and the minibatch shuffles use the same std::mt19937_64 stream as
// TF; weights use nn's generator, so returns match TF statistically. The vsim
// video is replaced by PlayBestOfSeeds (same episode, printed, no video).

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <datamunge/nn/nn.hpp>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "common/checks.hpp"
#include "common/rl/classic_control.hpp"
#include "common/rl/continuous.hpp"
#include "common/rl/pg_common.hpp"
#include "common/rl/play_best.hpp"
#include "common/rl/vec_env.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

int main(int argc, char** argv) {
  rl::PgConfig cfg;
  cfg.steps = 400000;
  cfg.n_envs = 8;
  cfg.n_steps = 256;
  cfg.lr = 3e-4f;
  cfg.entropy_coef = 0.f;
  cfg.log_every = 50000;
  cfg.eval_every = 50000;
  rl::ParsePgConfig(argc, argv, &cfg);
  for (int i = 1; i < argc; ++i)  // threads=N (a TF hardware option) sets nn's thread count
    if (std::string(argv[i]).rfind("threads=", 0) == 0) nn::set_num_threads(std::atoi(argv[i] + 8));
  const float clip_eps = 0.2f, reward_scale = 0.1f;
  const int epochs = 10, minibatch = 64;
  rl::VecEnv venv([] { return std::make_unique<rl::Pendulum>(); }, cfg.n_envs, cfg.seed);
  rl::Pendulum eval_env;
  const rl::ActionSpace space = venv.action_space();
  const int obs_dim = venv.obs_size(), act_dim = space.dim;
  std::printf("== PPO (Gaussian) on Pendulum: %d envs x %d steps per batch, %d epochs of %d, %d steps total\n",
              cfg.n_envs, cfg.n_steps, epochs, minibatch, cfg.steps);

  nn::ParameterStore params(cfg.seed);
  struct Net {
    Tensor mean, log_std, value;
  };
  auto net = [&](const Tensor& obs) {
    Net n;
    const Tensor h = nn::mlp(params, "actor", obs, {cfg.hidden, cfg.hidden}, nn::Activation::Tanh);
    n.mean = nn::dense(params, "actor/mean", h, act_dim);
    n.log_std = params.get("actor/log_std", {act_dim}, nn::init::constant(-0.5));  // state independent
    const Tensor hv = nn::mlp(params, "critic", obs, {cfg.hidden, cfg.hidden}, nn::Activation::Tanh);
    n.value = nn::squeeze(nn::dense(params, "critic/value", hv, 1), {1});
    return n;
  };
  {
    nn::NoGradGuard no_grad;
    net(nn::zeros({1, obs_dim}));
  }
  rl::Adam adam(params, cfg.lr, cfg.clip_norm);
  const double half_log_2pi = 0.5 * std::log(2 * 3.14159265);

  rl::Monitor monitor(rl::MonitorConfig(cfg), "std");
  std::mt19937_64 rng(cfg.seed + 1);
  std::normal_distribution<float> normal(0.f, 1.f);
  auto act_mean = [&](const rl::Obs& o) {
    nn::NoGradGuard no_grad;
    return net(rl::Row(o)).mean.to_vector<float>();
  };

  std::vector<rl::Obs> cur = venv.reset();
  std::vector<float> ep_ret(cfg.n_envs, 0.f);
  long steps = 0, next_log = cfg.log_every;
  float std_value = std::exp(-0.5f), last_kl = 0, best_eval = -1e9f;
  while (steps < cfg.steps) {
    // Rollout: sample Gaussian actions on the host.
    std::vector<std::vector<rl::Obs>> r_obs(cfg.n_envs);
    std::vector<std::vector<std::vector<float>>> r_act(cfg.n_envs);
    std::vector<std::vector<float>> r_rew(cfg.n_envs), r_done(cfg.n_envs), r_val(cfg.n_envs), r_logp(cfg.n_envs);
    std::vector<float> last_val(cfg.n_envs);
    for (int t = 0; t <= cfg.n_steps; ++t) {
      std::vector<float> mean, log_std, value;
      {
        nn::NoGradGuard no_grad;
        const Net n = net(rl::ObsMatrix(cur));
        mean = n.mean.to_vector<float>(), log_std = n.log_std.to_vector<float>(), value = n.value.to_vector<float>();
      }
      if (t == cfg.n_steps) {
        for (int i = 0; i < cfg.n_envs; ++i) last_val[i] = value[i];
        break;
      }
      std::vector<rl::Action> actions(cfg.n_envs);
      for (int i = 0; i < cfg.n_envs; ++i) {
        std::vector<float> u(act_dim);
        float lp = 0;
        for (int j = 0; j < act_dim; ++j) {
          const float mu = mean[i * act_dim + j], ls = log_std[j], e = normal(rng);
          u[j] = mu + std::exp(ls) * e;
          lp += -0.5f * e * e - ls - 0.5f * std::log(2 * 3.14159265f);
        }
        r_obs[i].push_back(cur[i]), r_act[i].push_back(u), r_val[i].push_back(value[i]), r_logp[i].push_back(lp);
        actions[i] = rl::ScaleAction(u, space);
      }
      rl::VecStep sv = venv.step(actions);
      for (int i = 0; i < cfg.n_envs; ++i) {
        r_rew[i].push_back(sv.reward[i] * reward_scale);
        r_done[i].push_back(sv.terminated[i] ? 1.f : 0.f);
        ep_ret[i] += sv.reward[i];
        if (sv.done(i)) monitor.EndEpisode(ep_ret[i]), ep_ret[i] = 0;
      }
      cur = sv.obs;
      steps += cfg.n_envs;
    }
    std::vector<rl::Obs> all_obs;
    std::vector<float> advantages, returns, logps, a_i, r_i, flat_act;
    for (int i = 0; i < cfg.n_envs; ++i) {
      rl::Gae(r_rew[i], r_val[i], r_done[i], last_val[i], cfg.gamma, cfg.lam, &a_i, &r_i);
      advantages.insert(advantages.end(), a_i.begin(), a_i.end());
      returns.insert(returns.end(), r_i.begin(), r_i.end());
      logps.insert(logps.end(), r_logp[i].begin(), r_logp[i].end());
      all_obs.insert(all_obs.end(), r_obs[i].begin(), r_obs[i].end());
      for (auto& u : r_act[i]) flat_act.insert(flat_act.end(), u.begin(), u.end());
    }
    const int N = static_cast<int>(advantages.size());
    const float m = std::accumulate(advantages.begin(), advantages.end(), 0.f) / N;
    float var = 0;
    for (float x : advantages) var += (x - m) * (x - m);
    const float sd = std::sqrt(var / N) + 1e-8f;
    for (float& x : advantages) x = (x - m) / sd;
    const Tensor t_obs = rl::ObsMatrix(all_obs), t_act = nn::from_vector(flat_act, {N, act_dim});
    const Tensor t_adv = rl::FloatVector(advantages), t_ret = rl::FloatVector(returns), t_logp = rl::FloatVector(logps);
    std::vector<int> order(N);
    std::iota(order.begin(), order.end(), 0);
    bool checked_ratio = false;
    for (int epoch = 0; epoch < epochs; ++epoch) {
      std::shuffle(order.begin(), order.end(), rng);
      for (int start = 0; start + minibatch <= N; start += minibatch) {
        const std::vector<std::int64_t> rows(order.begin() + start, order.begin() + start + minibatch);
        const Tensor act = nn::take_rows(t_act, rows), adv = nn::take_rows(t_adv, rows), ret = nn::take_rows(t_ret, rows);
        const Tensor logp_old = nn::take_rows(t_logp, rows);
        const Net n = net(nn::take_rows(t_obs, rows));
        const Tensor logp = rl::GaussianLogProb(act, n.mean, n.log_std);
        const Tensor entropy = nn::sum(n.log_std + (0.5 + half_log_2pi), {0});
        const Tensor ratio = nn::exp(logp - logp_old);
        const Tensor policy_loss = -nn::mean(nn::minimum(ratio * adv, nn::clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv), {0});
        const Tensor value_loss = nn::mean(nn::square(ret - n.value), {0});
        const Tensor loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy;
        const Tensor approx_kl = nn::mean(logp_old - logp.detach(), {0});
        if (epoch == 0 && !checked_ratio) {  // the first minibatch is scored by the policy that collected it
          check::Near(nn::max(nn::abs(ratio.detach() - 1.0)).item<float>(), 0.0, 1e-3, "first-epoch probability ratio = 1");
          checked_ratio = true;
        }
        const float lv = loss.item<float>();
        check::Require(std::isfinite(lv), "finite PPO loss");
        last_kl = approx_kl.item<float>();
        std_value = std::exp(n.log_std.to_vector<float>()[0]);
        adam.Minimize(loss);
        monitor.AddLoss(lv);
      }
    }
    if (steps >= next_log) {
      const float ev = next_log % cfg.eval_every == 0 ? rl::EvaluateC(eval_env, act_mean, cfg.eval_episodes, 1000 + next_log) : NAN;
      if (!std::isnan(ev)) best_eval = std::max(best_eval, ev);
      monitor.Log(static_cast<int>(steps), std_value, ev);
      std::printf("           approx KL of the last minibatch %.4f\n", last_kl);
      next_log += cfg.log_every;
    }
  }
  std::printf("  ('std' is the learned action standard deviation; returns are unscaled)\n");
  const float final_eval = rl::EvaluateC(eval_env, act_mean, 20, 777);
  std::printf("  final evaluation over 20 episodes (mean action): %.1f\n", final_eval);
  // On film in TF: the mean action and the critic's V(s), the return it expects from here.
  rl::PlayBestOfSeeds(eval_env, [&](const rl::Obs& o) {
    nn::NoGradGuard no_grad;
    const Net n = net(rl::Row(o));
    return rl::Explain(rl::ScaleAction(n.mean.to_vector<float>(), space), n.value.item<float>(), "V(s) (critic)");
  }, "ppo_continuous_pendulum", 4242);
  // A random policy scores about -1200 per episode, a good one above -200 (TF seed 0: best -100.1, final -169.1).
  check::TrainingQuality(std::max(best_eval, final_eval) >= -400, "PPO learned to swing up (best mean-action evaluation >= -400)");
  return 0;
}
