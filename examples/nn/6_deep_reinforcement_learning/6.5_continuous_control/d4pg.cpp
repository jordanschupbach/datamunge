// CMD: just run-nn d4pg
// Distributed distributional DDPG (D4PG).
//
// Port of tflow's d4pg. Barth-Maron et al. 2018 upgrade DDPG with the
// ingredients that worked for DQN: a categorical (C51-style) critic trained
// by cross-entropy on the projected n-step target distribution, prioritized
// replay, and n-step returns. The actor still follows the critic's gradient,
// now through the expected value of the predicted distribution. The
// "distributed" part, many actor processes feeding one replay, is the only
// piece not shown.
//
// The TF graph becomes eager nn ops over four ParameterStores; the target
// distribution is projected on the host with the shared ProjectDistribution
// (common/rl/dqn_common.hpp), exactly as in TF. The loop, Pendulum, the
// prioritized/n-step replay and the exploration noise use the same
// std::mt19937_64 stream as TF; weights use nn's generator, so returns match
// TF statistically. The vsim video is replaced by PlayBestOfSeeds (same
// episode, printed, no video).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <datamunge/nn/nn.hpp>
#include <random>
#include <utility>
#include <vector>

#include "common/checks.hpp"
#include "common/rl/classic_control.hpp"
#include "common/rl/continuous.hpp"
#include "common/rl/dqn_common.hpp"
#include "common/rl/play_best.hpp"
#include "common/rl/replay.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

int main(int argc, char** argv) {
  rl::CConfig cfg;
  rl::ParseCConfig(argc, argv, &cfg);
  const int atoms = 51, n_step = 5;
  const float vmin = -1000.f, vmax = 0.f, alpha_per = 0.6f, beta_start = 0.4f;
  const float gamma_n = std::pow(cfg.gamma, n_step);
  rl::Pendulum env, eval_env;
  const rl::ActionSpace space = env.action_space();
  const int obs_dim = env.obs_size(), act_dim = space.dim;
  std::printf("== D4PG on Pendulum: %d steps, %d atoms on [%.0f, %.0f], %d-step returns, prioritized replay\n", cfg.steps, atoms, vmin, vmax, n_step);

  std::vector<float> support(atoms);
  for (int i = 0; i < atoms; ++i) support[i] = vmin + i * (vmax - vmin) / (atoms - 1);
  const Tensor z = nn::from_vector(support, {atoms});
  nn::ParameterStore actor(cfg.seed), critic(cfg.seed + 1), actor_t(cfg.seed + 2), critic_t(cfg.seed + 3);
  // Distributional critic: logits over the atoms for (s, a).
  struct Dist {
    Tensor logits, probs, q;
  };
  auto dist_critic = [&](nn::ParameterStore& p, const Tensor& s, const Tensor& a) {
    const Tensor h = nn::mlp(p, "critic/mlp", nn::concat({s, a}, 1), {cfg.hidden, cfg.hidden});
    const Tensor logits = nn::dense(p, "critic/logits", h, atoms);
    const Tensor probs = nn::softmax(logits);
    return Dist{logits, probs, nn::sum(probs * z, {1})};
  };
  auto pi = [&](nn::ParameterStore& p, const Tensor& s) { return rl::DeterministicActor(p, "pi", s, act_dim, cfg.hidden); };
  {
    nn::NoGradGuard no_grad;
    const Tensor s0 = nn::zeros({1, obs_dim}), a0 = nn::zeros({1, act_dim});
    for (nn::ParameterStore* p : {&actor, &actor_t}) pi(*p, s0);
    for (nn::ParameterStore* p : {&critic, &critic_t}) dist_critic(*p, s0, a0);
  }
  rl::Adam critic_adam(critic, cfg.critic_lr), actor_adam(actor, cfg.actor_lr);
  actor_t.copy_from(actor, 1.0);
  critic_t.copy_from(critic, 1.0);

  {  // projection sanity: a terminal transition puts all mass on the atom(s) around r
    std::vector<float> probs(atoms, 1.f / atoms), out(atoms);
    const float r = -510.f, done = 1.f;
    rl::ProjectDistribution(probs.data(), &r, &done, 1, atoms, vmin, vmax, gamma_n, out.data());
    check::Near(out[24] + out[25], 1.0, 1e-5, "terminal projection: all mass on the two atoms around r");
    check::Near(out[24] * support[24] + out[25] * support[25], r, 1e-3, "terminal projection preserves the mean");
  }

  rl::PrioritizedReplay buffer(cfg.buffer, obs_dim, alpha_per);
  rl::NStepBuffer nstep(n_step, cfg.gamma);
  rl::Monitor monitor(rl::MonitorConfig(cfg), "noise");
  std::mt19937_64 rng(cfg.seed);
  std::normal_distribution<float> normal(0.f, 1.f);
  auto act = [&](const rl::Obs& o) {
    nn::NoGradGuard no_grad;
    return pi(actor, rl::Row(o)).to_vector<float>();
  };
  std::vector<float> target_dist(static_cast<std::size_t>(cfg.batch) * atoms);
  rl::Obs obs = env.reset(cfg.seed);
  float ret = 0, best_eval = -1e9f;
  for (int step = 1; step <= cfg.steps; ++step) {
    std::vector<float> u(act_dim);
    if (step <= cfg.warmup) {
      for (float& x : u) x = std::uniform_real_distribution<float>(-1, 1)(rng);
    } else {
      u = act(obs);
      for (float& x : u) x = std::max(-1.f, std::min(1.f, x + cfg.noise * normal(rng)));
    }
    rl::StepResult sr = env.step(rl::ScaleAction(u, space));
    for (rl::Experience& e : nstep.Push({obs, 0, sr.reward, sr.obs, sr.terminated ? 1.f : 0.f, u})) buffer.Add(std::move(e));
    if (sr.truncated)
      for (rl::Experience& e : nstep.Flush()) buffer.Add(std::move(e));
    ret += sr.reward;
    obs = sr.obs;
    if (sr.done()) {
      monitor.EndEpisode(ret);
      ret = 0;
      obs = env.reset();
    }
    if (step > cfg.warmup) {
      const float beta = beta_start + (1 - beta_start) * std::min(1.f, static_cast<float>(step) / cfg.steps);
      const rl::Batch b = buffer.Sample(cfg.batch, beta, rng);
      std::vector<float> next_probs;
      {
        nn::NoGradGuard no_grad;
        next_probs = dist_critic(critic_t, b.s2, pi(actor_t, b.s2)).probs.to_vector<float>();
      }
      const std::vector<float> r = b.r.to_vector<float>(), done = b.done.to_vector<float>();
      rl::ProjectDistribution(next_probs.data(), r.data(), done.data(), cfg.batch, atoms, vmin, vmax, gamma_n, target_dist.data());
      const Tensor m = nn::from_vector(target_dist, {cfg.batch, atoms});
      const Tensor sample_loss = -nn::sum(m * nn::log_softmax(dist_critic(critic, b.s, b.act_c).logits), {1});
      const Tensor critic_loss = nn::mean(b.weights * sample_loss, {0});
      critic_adam.Minimize(critic_loss);
      const float cl = critic_loss.item<float>();
      check::Require(std::isfinite(cl), "finite critic loss");
      monitor.AddLoss(cl);
      buffer.UpdatePriorities(b.indices, sample_loss.detach().to_vector<float>());
      actor_adam.Minimize(-nn::mean(dist_critic(critic, b.s, pi(actor, b.s)).q, {0}));
      actor_t.copy_from(actor, cfg.tau);
      critic_t.copy_from(critic, cfg.tau);
    }
    if (step % cfg.log_every == 0) {
      const float ev = step % cfg.eval_every == 0 ? rl::EvaluateC(eval_env, act, cfg.eval_episodes, 1000 + step) : NAN;
      if (!std::isnan(ev)) best_eval = std::max(best_eval, ev);
      monitor.Log(step, cfg.noise, ev);
    }
  }
  const float final_eval = rl::EvaluateC(eval_env, act, 20, 777);
  std::printf("  final deterministic evaluation over 20 episodes: %.1f\n", final_eval);
  // On film in TF: the action and the critic's estimate of the return from here.
  rl::PlayBestOfSeeds(eval_env, [&](const rl::Obs& o) {
    nn::NoGradGuard no_grad;
    const std::vector<float> u = act(o);
    const float v = dist_critic(critic, rl::Row(o), nn::from_vector(u, {1, act_dim})).q.item<float>();
    return rl::Explain(rl::ScaleAction(u, space), v, "Q(s,a) (mean of the critic's distribution)");
  }, "d4pg_pendulum", 4242);
  // The critic's return distribution for the upright state vs. hanging down.
  float q_up = 0, q_down = 0;
  for (auto [label, o] : std::vector<std::pair<const char*, rl::Obs>>{{"upright, still", {1, 0, 0}}, {"hanging down", {-1, 0, 0}}}) {
    nn::NoGradGuard no_grad;
    const std::vector<float> u = act(o);
    const Dist d = dist_critic(critic, rl::Row(o), nn::from_vector(u, {1, act_dim}));
    const std::vector<float> probs = d.probs.to_vector<float>();
    const float qv = d.q.item<float>();
    (o[0] > 0 ? q_up : q_down) = qv;
    std::printf("  %-16s expected return %.1f, mass per 200-wide bin from %.0f:", label, qv, vmin);
    for (int bin = 0; bin < 5; ++bin) {
      float mass = 0;
      for (int i = 0; i < atoms; ++i)
        if (support[i] >= vmin + bin * 200 && (support[i] < vmin + (bin + 1) * 200 || bin == 4)) mass += probs[i];
      std::printf(" %3.0f%%", 100 * mass);
    }
    std::printf("\n");
  }
  // TF seed 0: final -159.8; upright -0.0, hanging down -263.8.
  check::TrainingQuality(q_up > q_down, "the critic values the upright state above hanging down");
  check::TrainingQuality(std::max(best_eval, final_eval) >= -400, "D4PG learned to swing up (best deterministic evaluation >= -400)");
  return 0;
}
