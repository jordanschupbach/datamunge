// CMD: just run-nn dqn
// Deep Q-network (DQN) with experience replay and a target network.
//
// Port of tflow's dqn. Q(s, a) is a small MLP trained to minimise the Huber
// loss of the TD error
//   r + gamma (1 - done) max_a' Q_target(s', a') - Q(s, a)
// on minibatches drawn from a replay buffer (Mnih et al. 2015). The target
// network is a slowly moving copy of the online network (Polyak averaging
// by default, or a hard copy every few hundred steps with tau=1), which
// keeps the regression target from chasing itself.
//
// In TF both networks, the loss and Adam were one graph run by a session;
// here they are eager nn ops over two ParameterStores, with the target
// computed under NoGradGuard (TF StopGradient) and the target update done
// by ParameterStore::copy_from. The C++ loop, the CartPole dynamics, the
// replay sampling and the epsilon draws use the same std::mt19937_64 streams
// as TF; the weight initialisation uses nn's generator, so learning curves
// match TF statistically rather than digit for digit. The vsim video of the
// best evaluation episode is omitted.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <datamunge/nn/nn.hpp>
#include <random>
#include <string>

#include "common/checks.hpp"
#include "common/rl/classic_control.hpp"
#include "common/rl/dqn_common.hpp"
#include "common/rl/replay.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

int main(int argc, char** argv) {
  rl::DqnConfig cfg;
  rl::ParseConfig(argc, argv, &cfg);
  rl::CartPole env, eval_env;
  const int obs_dim = env.obs_size(), A = env.action_space().n;
  const std::string target_update = cfg.tau < 1.f ? "Polyak tau=" + std::to_string(cfg.tau)
                                                  : "hard copy every " + std::to_string(cfg.target_sync) + " steps";
  std::printf("== DQN on CartPole: %d steps, batch %d, buffer %d, lr %.0e, target update: %s\n", cfg.steps, cfg.batch,
              cfg.buffer, cfg.lr, target_update.c_str());

  // ---- Networks
  nn::ParameterStore online(cfg.seed), target(cfg.seed + 1);
  auto q_net = [&](nn::ParameterStore& p, const Tensor& x) {
    const Tensor h = nn::mlp(p, "mlp", x, {cfg.hidden, cfg.hidden});
    return nn::dense(p, "q", h, A);  // [B, A]
  };
  {  // create the parameters of both networks
    nn::NoGradGuard no_grad;
    q_net(online, nn::zeros({1, obs_dim}));
    q_net(target, nn::zeros({1, obs_dim}));
  }
  std::printf("  online network: %lld parameters\n", static_cast<long long>(online.count()));
  check::Require(online.count() == 17410, "online parameter count (TF: 17410)");
  rl::Adam adam(online, cfg.lr, cfg.clip_norm);
  auto sync = [&] { target.copy_from(online, cfg.tau); };
  sync();

  struct TrainResult {
    float loss, q_scale;
  };
  auto train = [&](const rl::Batch& b) {
    Tensor y;
    {  // TD target with the target network, a constant for the gradient
      nn::NoGradGuard no_grad;
      y = b.r + (cfg.gamma * (1.0 - b.done)) * nn::max(q_net(target, b.s2), {1});
    }
    const Tensor q = q_net(online, b.s);
    const Tensor q_a = nn::sum(q * nn::one_hot(b.a, A), {1});
    const Tensor loss = nn::mean(nn::huber(y - q_a), {0});
    adam.Minimize(loss);
    nn::NoGradGuard no_grad;  // diagnostic: mean max_a Q over the batch
    return TrainResult{loss.item<float>(), nn::mean(nn::max(q.detach(), {1}), {0}).item<float>()};
  };

  // ---- Interaction
  rl::ReplayBuffer buffer(cfg.buffer, obs_dim);
  rl::Monitor monitor(cfg);
  std::mt19937_64 rng(cfg.seed);
  auto greedy = [&](const rl::Obs& o) {  // argmax_a Q(s, a) from the online network
    nn::NoGradGuard no_grad;
    return static_cast<int>(nn::argmax(q_net(online, rl::Row(o)), 1).item<std::int64_t>());
  };

  rl::Obs obs = env.reset(cfg.seed);
  float ret = 0;
  double q_sum = 0;
  int q_count = 0;
  float best_eval = 0;
  for (int step = 1; step <= cfg.steps; ++step) {
    const float eps = rl::Epsilon(cfg, step);
    const int act = std::uniform_real_distribution<float>(0, 1)(rng) < eps ? static_cast<int>(rng() % A) : greedy(obs);
    rl::StepResult sr = env.step(act);
    buffer.Add({obs, act, sr.reward, sr.obs, sr.terminated ? 1.f : 0.f, {}});
    ret += sr.reward;
    obs = sr.obs;
    if (sr.done()) {
      monitor.EndEpisode(ret);
      ret = 0;
      obs = env.reset();
    }
    if (step >= cfg.warmup && step % cfg.train_every == 0) {
      const TrainResult tr = train(buffer.Sample(cfg.batch, rng));
      check::Require(std::isfinite(tr.loss), "finite TD loss");
      monitor.AddLoss(tr.loss);
      q_sum += tr.q_scale;
      ++q_count;
    }
    if (cfg.tau < 1.f || step % cfg.target_sync == 0) sync();
    if (step % cfg.log_every == 0) {
      const float ev = step % cfg.eval_every == 0 ? rl::Evaluate(eval_env, greedy, cfg.eval_episodes, 1000 + step) : NAN;
      if (!std::isnan(ev)) best_eval = std::max(best_eval, ev);
      monitor.Log(step, eps, ev);
      std::printf("           mean max_a Q over training batches: %.1f\n", q_count ? q_sum / q_count : 0.0);
      q_sum = 0;
      q_count = 0;
    }
  }
  const float final_eval = rl::Evaluate(eval_env, greedy, 20, 777);
  std::printf("  final greedy evaluation over 20 episodes: %.1f (max 500)\n", final_eval);
  // A random policy averages about 22; TF reaches 500 on this seed, and
  // 68..500 (mean 354) over seeds 0..16, this port 54..500 (mean 282).
  check::TrainingQuality(std::max(best_eval, final_eval) >= 150, "DQN learned to balance (best greedy evaluation >= 150)");
  return 0;
}
