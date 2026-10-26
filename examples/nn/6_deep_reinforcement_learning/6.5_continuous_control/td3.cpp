// CMD: just run-nn td3
// Twin delayed DDPG (TD3).
//
// Port of tflow's td3. Fujimoto et al. 2018 fix DDPG's overestimation with
// three changes:
//   1. twin critics, with the target using the minimum of the two,
//   2. target policy smoothing: clipped noise is added to the target action,
//      so the critic is regressed on a small neighbourhood instead of a point,
//   3. delayed updates: the actor and the target networks move only every
//      d critic updates.
//
// The TF graph becomes eager nn ops over four ParameterStores (the twin
// critics "q1"/"q2" share one store, as in TF); targets are computed under
// NoGradGuard and target updates use ParameterStore::copy_from. The loop,
// Pendulum, replay sampling and all noise use the same std::mt19937_64 stream
// as TF; weights use nn's generator, so returns match TF statistically. The
// vsim video is replaced by PlayBestOfSeeds (same episode, printed).

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
  rl::ParseCConfig(argc, argv, &cfg);
  const float target_noise = 0.2f, noise_clip = 0.5f;
  const int policy_delay = 2;
  rl::Pendulum env, eval_env;
  const rl::ActionSpace space = env.action_space();
  const int obs_dim = env.obs_size(), act_dim = space.dim;
  std::printf("== TD3 on Pendulum: %d steps, target noise %.1f clipped at %.1f, policy delay %d\n", cfg.steps, target_noise, noise_clip, policy_delay);

  nn::ParameterStore actor(cfg.seed), critic(cfg.seed + 1), actor_t(cfg.seed + 2), critic_t(cfg.seed + 3);
  auto pi = [&](nn::ParameterStore& p, const Tensor& s) { return rl::DeterministicActor(p, "pi", s, act_dim, cfg.hidden); };
  auto q = [&](nn::ParameterStore& p, const char* name, const Tensor& s, const Tensor& a) { return rl::QNetwork(p, name, s, a, cfg.hidden); };
  {
    nn::NoGradGuard no_grad;
    const Tensor s0 = nn::zeros({1, obs_dim}), a0 = nn::zeros({1, act_dim});
    for (nn::ParameterStore* p : {&actor, &actor_t}) pi(*p, s0);
    for (nn::ParameterStore* p : {&critic, &critic_t}) q(*p, "q1", s0, a0), q(*p, "q2", s0, a0);
  }
  rl::Adam critic_adam(critic, cfg.critic_lr), actor_adam(actor, cfg.actor_lr);
  actor_t.copy_from(actor, 1.0);
  critic_t.copy_from(critic, 1.0);
  check::Require(critic.count() == 2 * 17281, "twin critics: 2 x 17281 parameters");

  rl::ReplayBuffer buffer(cfg.buffer, obs_dim);
  rl::Monitor monitor(rl::MonitorConfig(cfg), "q1-min");
  std::mt19937_64 rng(cfg.seed);
  std::normal_distribution<float> normal(0.f, 1.f);
  auto act = [&](const rl::Obs& o) {
    nn::NoGradGuard no_grad;
    return pi(actor, rl::Row(o)).to_vector<float>();
  };
  rl::Obs obs = env.reset(cfg.seed);
  float ret = 0, gap = 0, best_eval = -1e9f;
  for (int step = 1; step <= cfg.steps; ++step) {
    std::vector<float> u(act_dim);
    if (step <= cfg.warmup) {
      for (float& x : u) x = std::uniform_real_distribution<float>(-1, 1)(rng);
    } else {
      u = act(obs);
      for (float& x : u) x = std::max(-1.f, std::min(1.f, x + cfg.noise * normal(rng)));
    }
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
      const Tensor eps = rl::NormalNoise(cfg.batch, act_dim, rng);  // [B, act_dim] smoothing noise
      Tensor y;
      {  // smoothed target action and the minimum of the twin target critics
        nn::NoGradGuard no_grad;
        const Tensor a_next = nn::clip(pi(actor_t, b.s2) + nn::clip(eps * target_noise, -noise_clip, noise_clip), -1.0, 1.0);
        y = b.r + (cfg.gamma * (1.0 - b.done)) * nn::minimum(q(critic_t, "q1", b.s2, a_next), q(critic_t, "q2", b.s2, a_next));
      }
      const Tensor q1 = q(critic, "q1", b.s, b.act_c), q2 = q(critic, "q2", b.s, b.act_c);
      const Tensor critic_loss = nn::mean(nn::square(y - q1), {0}) + nn::mean(nn::square(y - q2), {0});
      {  // overestimation diagnostic: Q1 on the batch vs. the twin minimum (before the update)
        nn::NoGradGuard no_grad;
        gap = nn::mean(q1.detach() - nn::minimum(q1.detach(), q2.detach()), {0}).item<float>();
      }
      critic_adam.Minimize(critic_loss);
      const float cl = critic_loss.item<float>();
      check::Require(std::isfinite(cl) && gap >= 0, "finite critic loss and nonnegative q1-min gap");
      monitor.AddLoss(cl);
      if (step % policy_delay == 0) {
        actor_adam.Minimize(-nn::mean(q(critic, "q1", b.s, pi(actor, b.s)), {0}));
        actor_t.copy_from(actor, cfg.tau);
        critic_t.copy_from(critic, cfg.tau);
      }
    }
    if (step % cfg.log_every == 0) {
      const float ev = step % cfg.eval_every == 0 ? rl::EvaluateC(eval_env, act, cfg.eval_episodes, 1000 + step) : NAN;
      if (!std::isnan(ev)) best_eval = std::max(best_eval, ev);
      monitor.Log(step, gap, ev);
    }
  }
  std::printf("  ('q1-min' is the mean gap between critic 1 and the twin minimum on a batch)\n");
  const float final_eval = rl::EvaluateC(eval_env, act, 20, 777);
  std::printf("  final deterministic evaluation over 20 episodes: %.1f\n", final_eval);
  // On film in TF: the action and the critic's estimate of the return from here.
  rl::PlayBestOfSeeds(eval_env, [&](const rl::Obs& o) {
    nn::NoGradGuard no_grad;
    const std::vector<float> u = act(o);
    const Tensor s = rl::Row(o), a = nn::from_vector(u, {1, act_dim});
    const float v = std::min(q(critic, "q1", s, a).item<float>(), q(critic, "q2", s, a).item<float>());
    return rl::Explain(rl::ScaleAction(u, space), v, "min(Q1,Q2)(s,a)");
  }, "td3_pendulum", 4242);
  // A random policy scores about -1200 per episode, a good one above -200 (TF seed 0: final -163.6).
  check::TrainingQuality(std::max(best_eval, final_eval) >= -400, "TD3 learned to swing up (best deterministic evaluation >= -400)");
  return 0;
}
