// CMD: just run-nn mpo
// Maximum a posteriori policy optimization (MPO).
//
// Port of tflow's mpo. Abdolmaleki et al. 2018 alternate two steps on batches
// of replayed states:
//   E-step: sample K actions per state from the target policy, score them
//           with the critic, and form a non-parametric improved policy
//           q(a|s) ~ pi_target(a|s) exp(Q(s,a)/eta), where the temperature
//           eta is found by minimising a convex dual on the host so that
//           KL(q || pi_target) <= eps.
//   M-step: fit the parametric Gaussian policy to q by weighted maximum
//           likelihood, with separate KL trust regions on the mean and on
//           the covariance enforced by learned Lagrange multipliers.
// The critic is an ordinary TD(0) Q-network with a target copy.
//
// The TF graph becomes eager nn ops over five ParameterStores; the E-step
// (sampling, the golden-section search for eta, the weights) runs on the host
// exactly as in TF. The loop, Pendulum, replay sampling and every normal draw
// use the same std::mt19937_64 stream and order as TF; weights use nn's
// generator, so returns match TF statistically. The vsim video is replaced by
// PlayBestOfSeeds (same episode, printed, no video).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <datamunge/nn/nn.hpp>
#include <random>
#include <vector>

#include "common/checks.hpp"
#include "common/rl/classic_control.hpp"
#include "common/rl/continuous.hpp"
#include "common/rl/play_best.hpp"
#include "common/rl/replay.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

int main(int argc, char** argv) {
  rl::CConfig cfg;
  cfg.actor_lr = cfg.critic_lr = 3e-4f;
  rl::ParseCConfig(argc, argv, &cfg);
  const int K = 20, target_period = 100;
  const float eps_eta = 0.1f, eps_mean = 0.01f, eps_std = 1e-4f;
  rl::Pendulum env, eval_env;
  const rl::ActionSpace space = env.action_space();
  const int obs_dim = env.obs_size(), act_dim = space.dim;
  std::printf("== MPO on Pendulum: %d steps, %d action samples per state, eps_eta %.2f, eps_mean %.3f, eps_std %.0e\n",
              cfg.steps, K, eps_eta, eps_mean, eps_std);

  nn::ParameterStore actor(cfg.seed), actor_t(cfg.seed + 1), critic(cfg.seed + 2), critic_t(cfg.seed + 3), duals(cfg.seed + 4);
  auto policy = [&](nn::ParameterStore& p, const Tensor& s) { return rl::GaussianActor(p, "pi", s, act_dim, cfg.hidden); };
  auto q = [&](nn::ParameterStore& p, const Tensor& s, const Tensor& a) { return rl::QNetwork(p, "q", s, a, cfg.hidden); };
  const Tensor alpha_mean_raw = duals.get("alpha_mean_raw", {}, nn::init::constant(1.0));
  const Tensor alpha_std_raw = duals.get("alpha_std_raw", {}, nn::init::constant(1.0));
  {
    nn::NoGradGuard no_grad;
    const Tensor s0 = nn::zeros({1, obs_dim}), a0 = nn::zeros({1, act_dim});
    for (nn::ParameterStore* p : {&actor, &actor_t}) policy(*p, s0);
    for (nn::ParameterStore* p : {&critic, &critic_t}) q(*p, s0, a0);
  }
  rl::Adam critic_adam(critic, cfg.critic_lr), actor_adam(actor, cfg.actor_lr), dual_adam(duals, cfg.actor_lr);
  auto sync = [&] {
    actor_t.copy_from(actor, 1.0);
    critic_t.copy_from(critic, 1.0);
  };
  sync();

  rl::ReplayBuffer buffer(cfg.buffer, obs_dim);
  rl::Monitor monitor(rl::MonitorConfig(cfg), "eta");
  std::mt19937_64 rng(cfg.seed);
  std::normal_distribution<float> normal(0.f, 1.f);
  auto act_mean = [&](const rl::Obs& o) {
    nn::NoGradGuard no_grad;
    std::vector<float> u = policy(actor, rl::Row(o)).mean.to_vector<float>();
    for (float& x : u) x = std::max(-1.f, std::min(1.f, x));
    return u;
  };
  auto act_sample = [&](const rl::Obs& o) {
    nn::NoGradGuard no_grad;
    const rl::Gaussian g = policy(actor, rl::Row(o));
    const std::vector<float> mu = g.mean.to_vector<float>(), ls = g.log_std.to_vector<float>();
    std::vector<float> u(act_dim);
    for (int i = 0; i < act_dim; ++i) u[i] = std::max(-1.f, std::min(1.f, mu[i] + std::exp(ls[i]) * normal(rng)));
    return u;
  };

  const int BK = cfg.batch * K;
  std::vector<float> s_rep(static_cast<std::size_t>(BK) * obs_dim), a_samp(static_cast<std::size_t>(BK) * act_dim);
  std::vector<float> mean_rep(a_samp.size()), log_std_rep(a_samp.size()), w(BK);
  float eta = 1.f, kl_m = 0, kl_s = 0, best_eval = -1e9f;
  int updates = 0;
  bool dual_checked = false;
  rl::Obs obs = env.reset(cfg.seed);
  float ret = 0;
  for (int step = 1; step <= cfg.steps; ++step) {
    std::vector<float> u(act_dim);
    if (step <= cfg.warmup)
      for (float& x : u) x = std::uniform_real_distribution<float>(-1, 1)(rng);
    else
      u = act_sample(obs);
    rl::StepResult sr = env.step(rl::ScaleAction(u, space));
    buffer.Add({obs, 0, sr.reward, sr.obs, sr.terminated ? 1.f : 0.f, u});
    ret += sr.reward;
    obs = sr.obs;
    if (sr.done()) {
      monitor.EndEpisode(ret);
      ret = 0;
      obs = env.reset();
    }
    if (step > cfg.warmup) {
      const rl::Batch b = buffer.Sample(cfg.batch, rng);
      // Critic: TD(0) with an action sampled from the target policy at s'.
      const Tensor eps2 = rl::NormalNoise(cfg.batch, act_dim, rng);
      Tensor y;
      {
        nn::NoGradGuard no_grad;
        const rl::Gaussian gn = policy(actor_t, b.s2);
        const Tensor a_next = nn::clip(gn.mean + nn::exp(gn.log_std) * eps2, -1.0, 1.0);
        y = b.r + (cfg.gamma * (1.0 - b.done)) * q(critic_t, b.s2, a_next);
      }
      const Tensor critic_loss = nn::mean(nn::square(y - q(critic, b.s, b.act_c)), {0});
      critic_adam.Minimize(critic_loss);
      const float cl = critic_loss.item<float>();
      check::Require(std::isfinite(cl), "finite critic loss");
      monitor.AddLoss(cl);
      // E-step: K samples per state from the target policy, scored by the critic.
      std::vector<float> q_samples;
      {
        nn::NoGradGuard no_grad;
        const rl::Gaussian gt = policy(actor_t, b.s);
        const std::vector<float> mu = gt.mean.to_vector<float>(), ls = gt.log_std.to_vector<float>(), sv = b.s.to_vector<float>();
        for (int i = 0; i < cfg.batch; ++i)
          for (int k = 0; k < K; ++k) {
            const int row = i * K + k;
            for (int j = 0; j < obs_dim; ++j) s_rep[row * obs_dim + j] = sv[i * obs_dim + j];
            for (int j = 0; j < act_dim; ++j) {
              const float m = mu[i * act_dim + j], l = ls[i * act_dim + j];
              mean_rep[row * act_dim + j] = m;
              log_std_rep[row * act_dim + j] = l;
              a_samp[row * act_dim + j] = std::max(-1.f, std::min(1.f, m + std::exp(l) * normal(rng)));
            }
          }
        // The TF example scores the samples with the online critic (just updated).
        q_samples = q(critic, nn::from_vector(s_rep, {BK, obs_dim}), nn::from_vector(a_samp, {BK, act_dim})).to_vector<float>();
      }
      const float* qv = q_samples.data();
      // Dual for eta: g(eta) = eta eps + eta mean_s log mean_k exp(Q/eta), minimised by
      // a few steps of golden-section search in log space.
      auto dual = [&](float e) {
        double total = 0;
        for (int i = 0; i < cfg.batch; ++i) {
          float qmax = qv[i * K];
          for (int k = 1; k < K; ++k) qmax = std::max(qmax, qv[i * K + k]);
          double acc = 0;
          for (int k = 0; k < K; ++k) acc += std::exp((qv[i * K + k] - qmax) / e);
          total += qmax / e + std::log(acc / K);
        }
        return e * eps_eta + e * total / cfg.batch;
      };
      {
        float lo = std::log(1e-3f), hi = std::log(100.f);
        const float phi = 0.6180339887f;
        float x1 = hi - phi * (hi - lo), x2 = lo + phi * (hi - lo);
        for (int it = 0; it < 30; ++it) {
          if (dual(std::exp(x1)) < dual(std::exp(x2))) hi = x2, x2 = x1, x1 = hi - phi * (hi - lo);
          else lo = x1, x1 = x2, x2 = lo + phi * (hi - lo);
        }
        eta = std::exp(0.5f * (lo + hi));
      }
      if (!dual_checked && eta > 2e-3f && eta < 50.f) {  // an interior minimum of the convex dual
        const float g0 = dual(eta);
        check::Require(g0 <= dual(eta * 1.05f) + 1e-4f && g0 <= dual(eta / 1.05f) + 1e-4f, "golden-section search minimises the eta dual");
        dual_checked = true;
      }
      for (int i = 0; i < cfg.batch; ++i) {  // weights = softmax_k(Q/eta), normalised per state
        float qmax = qv[i * K];
        for (int k = 1; k < K; ++k) qmax = std::max(qmax, qv[i * K + k]);
        double acc = 0;
        for (int k = 0; k < K; ++k) acc += std::exp((qv[i * K + k] - qmax) / eta);
        for (int k = 0; k < K; ++k) w[i * K + k] = static_cast<float>(std::exp((qv[i * K + k] - qmax) / eta) / acc * K);
      }
      // M-step: weighted log-likelihood with decoupled KL penalties, and the dual updates.
      const Tensor mean_t = nn::from_vector(mean_rep, {BK, act_dim}), log_std_t = nn::from_vector(log_std_rep, {BK, act_dim});
      const Tensor a_t = nn::from_vector(a_samp, {BK, act_dim});
      const rl::Gaussian g = policy(actor, nn::from_vector(s_rep, {BK, obs_dim}));
      const Tensor nll = -nn::mean(nn::from_vector(w, {BK}) * rl::GaussianLogProb(a_t, g.mean, g.log_std), {0});
      const Tensor std_t = nn::exp(log_std_t);
      const Tensor kl_mean = nn::mean(nn::sum(0.5 * nn::square((g.mean - mean_t) / std_t), {1}), {0});
      const Tensor var_ratio = nn::exp(2.0 * (log_std_t - g.log_std));
      const Tensor kl_std = nn::mean(nn::sum((g.log_std - log_std_t) + 0.5 * var_ratio - 0.5, {1}), {0});
      const Tensor alpha_mean = nn::softplus(alpha_mean_raw) + 1e-8, alpha_std = nn::softplus(alpha_std_raw) + 1e-8;
      const Tensor policy_loss = nll + alpha_mean.detach() * kl_mean + alpha_std.detach() * kl_std;
      // The multipliers grow while their constraint is violated and shrink otherwise.
      const Tensor dual_loss = alpha_mean * (eps_mean - kl_mean.detach()) + alpha_std * (eps_std - kl_std.detach());
      kl_m = kl_mean.item<float>();
      kl_s = kl_std.item<float>();
      check::Require(std::isfinite(policy_loss.item<float>()) && kl_m >= -1e-6f && kl_s >= -1e-6f, "finite M-step loss, nonnegative KLs");
      actor_adam.Minimize(policy_loss);
      dual_adam.Minimize(dual_loss);
      if (++updates % target_period == 0) sync();
    }
    if (step % cfg.log_every == 0) {
      const float ev = step % cfg.eval_every == 0 ? rl::EvaluateC(eval_env, act_mean, cfg.eval_episodes, 1000 + step) : NAN;
      if (!std::isnan(ev)) best_eval = std::max(best_eval, ev);
      monitor.Log(step, eta, ev);
      std::printf("           KL(mean) %.4f (eps %.3f), KL(std) %.5f (eps %.0e)\n", kl_m, eps_mean, kl_s, eps_std);
    }
  }
  const float final_eval = rl::EvaluateC(eval_env, act_mean, 20, 777);
  std::printf("  final evaluation over 20 episodes: mean action %.1f\n", final_eval);
  // On film in TF: the action and the critic's estimate of the return from here.
  rl::PlayBestOfSeeds(eval_env, [&](const rl::Obs& o) {
    nn::NoGradGuard no_grad;
    const std::vector<float> u = act_mean(o);
    const float v = q(critic, rl::Row(o), nn::from_vector(u, {1, act_dim})).item<float>();
    return rl::Explain(rl::ScaleAction(u, space), v, "Q(s,a) (critic)");
  }, "mpo_pendulum", 4242);
  // A random policy scores about -1200 per episode, a good one above -200 (TF seed 0: final -169.5).
  check::TrainingQuality(std::max(best_eval, final_eval) >= -400, "MPO learned to swing up (best mean-action evaluation >= -400)");
  return 0;
}
