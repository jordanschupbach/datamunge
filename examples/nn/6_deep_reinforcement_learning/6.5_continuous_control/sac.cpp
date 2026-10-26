// CMD: just run-nn sac
// Soft actor-critic (SAC) with automatic temperature tuning.
//
// Port of tflow's sac. Haarnoja et al. 2018: maximum-entropy RL with a
// stochastic tanh-Gaussian actor. The critics regress on the soft target
//   y = r + gamma (1 - done) [ min(Q1', Q2')(s', a') - alpha log pi(a'|s') ],
// the actor minimises E[ alpha log pi(a|s) - min(Q1, Q2)(s, a) ] through the
// reparameterised sample, and the temperature alpha is itself learned so
// that the policy entropy stays near a target (-act_dim):
//   J(alpha) = -alpha (log pi(a|s) + H_target).
//
// The TF graph (and its prepared Callables) becomes eager nn ops over
// ParameterStores: the soft target is computed under NoGradGuard, the actor
// and temperature steps share one forward pass as TF's single actor_step run
// did, followed by the Polyak update of the target critics. The loop,
// Pendulum, replay sampling and the reparameterisation noise use the same
// std::mt19937_64 stream as TF; weights use nn's generator, so returns match
// TF statistically. As in TF, the periodic log lines are disabled (only the
// header and the final evaluation print). The vsim video is replaced by
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
  rl::Pendulum env, eval_env;
  const rl::ActionSpace space = env.action_space();
  const int obs_dim = env.obs_size(), act_dim = space.dim;
  const float target_entropy = -static_cast<float>(act_dim);
  std::printf("== SAC on Pendulum: %d steps, batch %d, target entropy %.1f, learned temperature\n", cfg.steps, cfg.batch, target_entropy);

  nn::ParameterStore actor(cfg.seed), critic(cfg.seed + 1), critic_t(cfg.seed + 2), temperature(cfg.seed + 3);
  const Tensor log_alpha = temperature.get("log_alpha", {}, nn::init::zeros());
  auto policy = [&](const Tensor& s) { return rl::GaussianActor(actor, "pi", s, act_dim, cfg.hidden); };
  auto q = [&](nn::ParameterStore& p, const char* name, const Tensor& s, const Tensor& a) { return rl::QNetwork(p, name, s, a, cfg.hidden); };
  {
    nn::NoGradGuard no_grad;
    const Tensor s0 = nn::zeros({1, obs_dim}), a0 = nn::zeros({1, act_dim});
    policy(s0);
    for (nn::ParameterStore* p : {&critic, &critic_t}) q(*p, "q1", s0, a0), q(*p, "q2", s0, a0);
  }
  rl::Adam critic_adam(critic, cfg.critic_lr), actor_adam(actor, cfg.actor_lr), alpha_adam(temperature, cfg.actor_lr);
  critic_t.copy_from(critic, 1.0);

  // log pi of a squashed sample, checked against the closed form for one row:
  // mean 0.3, log std -0.5, eps 0.7 -> u = 0.3 + e^-0.5 0.7.
  {
    nn::NoGradGuard no_grad;
    const rl::Squashed sq = rl::SquashedSample({nn::full({1, 1}, 0.3), nn::full({1, 1}, -0.5)}, nn::full({1, 1}, 0.7));
    const double u = 0.3 + std::exp(-0.5) * 0.7, a = std::tanh(u);
    const double expect = -0.5 * 0.49 + 0.5 - 0.5 * std::log(2 * M_PI) - std::log(1 - a * a + 1e-6);
    check::Near(sq.action.item<float>(), a, 1e-6, "squashed action = tanh(mean + std eps)");
    check::Near(sq.logp.item<float>(), expect, 1e-4, "squashed log-density with the tanh Jacobian");
  }

  rl::ReplayBuffer buffer(cfg.buffer, obs_dim);
  rl::Monitor monitor(rl::MonitorConfig(cfg), "alpha");
  std::mt19937_64 rng(cfg.seed);
  auto act_stochastic = [&](const rl::Obs& o) {
    nn::NoGradGuard no_grad;
    const rl::Gaussian g = policy(rl::Row(o));
    return rl::SquashedSample(g, rl::NormalNoise(1, act_dim, rng)).action.to_vector<float>();
  };
  auto act_mean = [&](const rl::Obs& o) {
    nn::NoGradGuard no_grad;
    return nn::tanh(policy(rl::Row(o)).mean).to_vector<float>();
  };
  rl::Obs obs = env.reset(cfg.seed);
  float ret = 0, alpha_value = 1, entropy = 0;
  for (int step = 1; step <= cfg.steps; ++step) {
    std::vector<float> u(act_dim);
    if (step <= cfg.warmup)
      for (float& x : u) x = std::uniform_real_distribution<float>(-1, 1)(rng);
    else
      u = act_stochastic(obs);
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
      const Tensor eps = rl::NormalNoise(cfg.batch, act_dim, rng);   // noise for a ~ pi(.|s)
      const Tensor eps2 = rl::NormalNoise(cfg.batch, act_dim, rng);  // noise for a' ~ pi(.|s')
      const Tensor alpha = nn::exp(log_alpha);
      Tensor y;
      {  // soft target with a' ~ pi(.|s') and the twin target critics
        nn::NoGradGuard no_grad;
        const rl::Squashed next = rl::SquashedSample(policy(b.s2), eps2);
        const Tensor soft_next = nn::minimum(q(critic_t, "q1", b.s2, next.action), q(critic_t, "q2", b.s2, next.action)) - alpha.detach() * next.logp;
        y = b.r + (cfg.gamma * (1.0 - b.done)) * soft_next;
      }
      const Tensor critic_loss = nn::mean(nn::square(y - q(critic, "q1", b.s, b.act_c)), {0}) +
                                 nn::mean(nn::square(y - q(critic, "q2", b.s, b.act_c)), {0});
      critic_adam.Minimize(critic_loss);
      const float cl = critic_loss.item<float>();
      check::Require(std::isfinite(cl), "finite critic loss");
      monitor.AddLoss(cl);
      // Actor: alpha log pi - min Q through the reparameterised action;
      // temperature: push alpha up when the entropy is below target, down otherwise.
      const rl::Squashed sample = rl::SquashedSample(policy(b.s), eps);
      const Tensor actor_loss = nn::mean(alpha.detach() * sample.logp - nn::minimum(q(critic, "q1", b.s, sample.action), q(critic, "q2", b.s, sample.action)), {0});
      const Tensor alpha_loss = -nn::mean(log_alpha * (sample.logp.detach() + target_entropy), {0});
      alpha_value = alpha.item<float>();
      entropy = -nn::mean(sample.logp.detach(), {0}).item<float>();
      actor_adam.Minimize(actor_loss);
      alpha_adam.Minimize(alpha_loss);
      critic_t.copy_from(critic, cfg.tau);
    }
    if (step % cfg.log_every == 0) {
      // monitor.Log(step, alpha_value, step % cfg.eval_every == 0 ? rl::EvaluateC(eval_env, act_mean, cfg.eval_episodes, 1000 + step) : NAN);
      // std::printf("           policy entropy estimate %.2f (target %.1f)\n", entropy, target_entropy);
    }
  }
  (void)entropy;
  const float mean_eval = rl::EvaluateC(eval_env, act_mean, 20, 777), sampled_eval = rl::EvaluateC(eval_env, act_stochastic, 20, 777);
  std::printf("  final evaluation over 20 episodes: mean action %.1f, sampled action %.1f\n", mean_eval, sampled_eval);
  std::printf("  final temperature alpha %.4f\n", alpha_value);
  // On film in TF: the action and the critic's estimate of the return from here.
  rl::PlayBestOfSeeds(eval_env, [&](const rl::Obs& o) {
    nn::NoGradGuard no_grad;
    const std::vector<float> u = act_mean(o);
    const Tensor s = rl::Row(o), a = nn::from_vector(u, {1, act_dim});
    const float v = std::min(q(critic, "q1", s, a).item<float>(), q(critic, "q2", s, a).item<float>());
    return rl::Explain(rl::ScaleAction(u, space), v, "min(Q1,Q2)(s,a)");
  }, "sac_pendulum", 4242);
  // A random policy scores about -1200 per episode, a good one above -200 (TF seed 0: -165.9 / -166.1).
  check::TrainingQuality(mean_eval >= -400, "SAC learned to swing up (mean-action evaluation >= -400)");
  return 0;
}
