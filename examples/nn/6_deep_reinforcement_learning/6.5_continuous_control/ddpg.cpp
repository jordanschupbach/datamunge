// CMD: just run-nn ddpg
// Deep deterministic policy gradient (DDPG).
//
// Port of tflow's ddpg. Lillicrap et al. 2016 bring DQN's replay and target
// networks to continuous actions. A deterministic actor mu(s) is trained to
// maximise the critic, d Q(s, mu(s)) / d theta, while the critic regresses on
//   y = r + gamma (1 - done) Q'(s', mu'(s'))
// with slowly moving (Polyak) target copies of both. Exploration is Gaussian
// noise added to the actor's output. Pendulum: random policy about -1200,
// a good one above -200.
//
// The TF graph becomes eager nn ops over four ParameterStores: the target is
// computed under NoGradGuard (TF StopGradient), the actor's Adam only updates
// the actor store (gradients still flow through the critic), and the target
// updates are ParameterStore::copy_from. The loop, Pendulum, the replay
// sampling and the exploration noise use the same std::mt19937_64 stream as
// TF; the weight initialisation uses nn's generator, so returns match TF
// statistically, not digit for digit. The vsim video of the best evaluation
// episode is replaced by PlayBestOfSeeds (same episode, printed, no video).

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
  rl::Pendulum env, eval_env;
  const rl::ActionSpace space = env.action_space();
  const int obs_dim = env.obs_size(), act_dim = space.dim;
  std::printf("== DDPG on Pendulum: %d steps, batch %d, tau %.3f, exploration noise %.2f\n", cfg.steps, cfg.batch, cfg.tau, cfg.noise);

  nn::ParameterStore actor(cfg.seed), critic(cfg.seed + 1), actor_t(cfg.seed + 2), critic_t(cfg.seed + 3);
  auto pi = [&](nn::ParameterStore& p, const Tensor& s) { return rl::DeterministicActor(p, "pi", s, act_dim, cfg.hidden); };
  auto q = [&](nn::ParameterStore& p, const Tensor& s, const Tensor& a) { return rl::QNetwork(p, "q", s, a, cfg.hidden); };
  {
    nn::NoGradGuard no_grad;
    const Tensor s0 = nn::zeros({1, obs_dim}), a0 = nn::zeros({1, act_dim});
    for (nn::ParameterStore* p : {&actor, &actor_t}) pi(*p, s0);
    for (nn::ParameterStore* p : {&critic, &critic_t}) q(*p, s0, a0);
  }
  rl::Adam critic_adam(critic, cfg.critic_lr), actor_adam(actor, cfg.actor_lr);
  actor_t.copy_from(actor, 1.0);
  critic_t.copy_from(critic, 1.0);
  check::Require(actor.count() == 17153 && critic.count() == 17281, "parameter counts (actor 17153, critic 17281)");

  rl::ReplayBuffer buffer(cfg.buffer, obs_dim);
  rl::Monitor monitor(rl::MonitorConfig(cfg), "noise");
  std::mt19937_64 rng(cfg.seed);
  std::normal_distribution<float> normal(0.f, 1.f);
  auto act = [&](const rl::Obs& o) {
    nn::NoGradGuard no_grad;
    return pi(actor, rl::Row(o)).to_vector<float>();
  };

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
      Tensor y;
      {  // target: the target actor's action at s', scored by the target critic
        nn::NoGradGuard no_grad;
        y = b.r + (cfg.gamma * (1.0 - b.done)) * q(critic_t, b.s2, pi(actor_t, b.s2));
      }
      const Tensor critic_loss = nn::mean(nn::square(y - q(critic, b.s, b.act_c)), {0});
      critic_adam.Minimize(critic_loss);
      const float cl = critic_loss.item<float>();
      check::Require(std::isfinite(cl), "finite critic loss");
      monitor.AddLoss(cl);
      // Actor: climb the (just updated) critic; only the actor store moves.
      actor_adam.Minimize(-nn::mean(q(critic, b.s, pi(actor, b.s)), {0}));
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
    const float v = q(critic, rl::Row(o), nn::from_vector(u, {1, act_dim})).item<float>();
    return rl::Explain(rl::ScaleAction(u, space), v, "Q(s,a) (critic)");
  }, "ddpg_pendulum", 4242);
  // A random policy scores about -1200 per episode, a good one above -200.
  check::TrainingQuality(std::max(best_eval, final_eval) >= -400, "DDPG learned to swing up (best deterministic evaluation >= -400)");
  return 0;
}
