// CMD: just run-nn actor_critic
// One-step actor-critic.
//
// Port of tflow's actor_critic. Replace the Monte Carlo return with a
// bootstrapped one-step estimate from a learned critic: the TD error
//   delta_t = r_t + gamma (1 - done_t) V(s_{t+1}) - V(s_t)
// is an estimate of the advantage of a_t and drives the actor, while the
// critic is trained to shrink delta. Updates happen after every single
// environment step, fully online, with no memory of the past (Sutton &
// Barto 13.5). This is the lowest-variance, highest-bias end of the family.
//
// The TF graph (two applications of the shared actor-critic, the
// stop-gradient TD target, Adam with global-norm clipping) becomes eager nn
// ops over a ParameterStore; the XLA hint is dropped. CartPole and the action
// sampling use the same std::mt19937_64 streams as TF; the weights use nn's
// generator, so results match TF statistically, not digit for digit. The vsim
// video is replaced by PlayBestOfSeeds (same episode, printed, no video).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <datamunge/nn/nn.hpp>
#include <random>
#include <vector>

#include "common/checks.hpp"
#include "common/rl/classic_control.hpp"
#include "common/rl/pg_common.hpp"
#include "common/rl/play_best.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

int main(int argc, char** argv) {
  rl::PgConfig cfg;
  cfg.steps = 60000;
  cfg.lr = 3e-4f;
  rl::ParsePgConfig(argc, argv, &cfg);
  rl::CartPole env, eval_env;
  const int obs_dim = env.obs_size(), A = env.action_space().n;
  std::printf("== One-step actor-critic on CartPole, %d steps, lr %.0e\n", cfg.steps, cfg.lr);

  nn::ParameterStore params(cfg.seed);
  auto net = [&](const Tensor& obs) { return rl::BuildActorCritic(params, obs, A, cfg.hidden); };
  {
    nn::NoGradGuard no_grad;
    net(nn::zeros({1, obs_dim}));
  }
  rl::Adam adam(params, cfg.lr, cfg.clip_norm);

  rl::Monitor monitor(rl::MonitorConfig(cfg), "entropy");
  std::mt19937_64 rng(cfg.seed);
  auto probs_of = [&](const rl::Obs& o) {
    nn::NoGradGuard no_grad;
    return net(rl::Row(o)).probs.to_vector<float>();
  };
  auto greedy = [&](const rl::Obs& o) { return rl::ArgMax(probs_of(o)); };
  auto explain = [&](const rl::Obs& o) {  // pi(.|s) and V(s), for the replayed episode
    nn::NoGradGuard no_grad;
    const rl::ActorCritic ac = net(rl::Row(o));
    return rl::GreedyPi(ac.probs.to_vector<float>(), ac.value.item<float>());
  };

  rl::Obs o = env.reset(cfg.seed);
  float ret = 0, ent = 0, best_eval = 0;
  bool checked_delta = false;
  for (int step = 1; step <= cfg.steps; ++step) {
    const std::vector<float> p = probs_of(o);
    const int a = rl::SampleAction(p.data(), A, rng);
    rl::StepResult sr = env.step(a);
    ret += sr.reward;
    // One update from this single transition. The next state's value goes
    // through the same weights but is a constant target (TF StopGradient).
    const float done = sr.terminated ? 1.f : 0.f;
    const rl::ActorCritic ac = net(rl::Row(o));
    const Tensor v_next = net(rl::Row(sr.obs)).value.detach();
    const Tensor target = sr.reward + cfg.gamma * (1.f - done) * v_next;
    const Tensor delta = target - ac.value;
    const Tensor logp_a = rl::LogProbOf(ac.logp, rl::IntVector({a}), A);
    const Tensor actor_loss = -nn::mean(logp_a * delta.detach(), {0});
    const Tensor critic_loss = nn::mean(nn::square(delta), {0});
    const Tensor loss = actor_loss + cfg.value_coef * critic_loss - cfg.entropy_coef * nn::mean(ac.entropy, {0});
    if (!checked_delta && sr.terminated) {  // a terminal transition's target is the reward alone
      check::Near(target.item<float>(), sr.reward, 1e-6, "TD target at a terminal step = r");
      checked_delta = true;
    }
    adam.Minimize(loss);
    monitor.AddLoss(loss.item<float>());
    ent = ac.entropy.to_vector<float>()[0];
    check::Require(std::isfinite(loss.item<float>()), "finite loss");
    o = sr.obs;
    if (sr.done()) {
      monitor.EndEpisode(ret);
      ret = 0;
      o = env.reset();
    }
    if (step % cfg.log_every == 0) {
      const float ev = step % cfg.eval_every == 0 ? rl::Evaluate(eval_env, greedy, cfg.eval_episodes, 1000 + step) : NAN;
      if (!std::isnan(ev)) best_eval = std::max(best_eval, ev);
      monitor.Log(step, ent, ev);
    }
  }
  const float final_eval = rl::Evaluate(eval_env, greedy, 20, 777);
  std::printf("  final greedy evaluation over 20 episodes: %.1f\n", final_eval);
  rl::PlayBestOfSeeds(eval_env, explain, "actor_critic_cartpole", 4242);
  check::Require(checked_delta, "saw a terminal transition");
  // An untrained greedy policy scores about 10, a random one about 22.
  // One-step actor-critic learns slowly and erratically: over seeds 0..9 the
  // best greedy evaluation is 65..414 (mean 255) for TF, 44..500 (mean 175)
  // for this port; seed 0 gives TF 266.8 (final 158.3), this port 245.2.
  check::TrainingQuality(std::max(best_eval, final_eval) >= 40, "actor-critic improved on random play (best greedy evaluation >= 40)");
  return 0;
}
