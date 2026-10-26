// CMD: just run-nn prioritized_replay
// Prioritized experience replay.
//
// Port of tflow's prioritized_replay. Instead of sampling the replay buffer
// uniformly, each transition is drawn with probability proportional to
// |TD error|^alpha (Schaul et al. 2016), so surprising transitions are
// revisited more often. The bias this introduces is corrected with
// importance-sampling weights (N P(i))^-beta, with beta annealed from 0.4 to
// 1. The buffer keeps priorities in a sum tree (common/rl/replay.hpp); the
// training step returns the per-sample TD errors so the priorities can be
// refreshed after every update.
//
// In TF the networks, loss and Adam were one graph; here they are eager nn
// ops over two ParameterStores (the target under NoGradGuard). CartPole,
// replay sampling and epsilon draws use the same std::mt19937_64 streams as
// TF; the weights use nn's initialiser, so the results match TF
// statistically. The vsim video of the best evaluation episode is omitted.
// Options: the rl::DqnConfig keys (steps=, lr=, seed=, ...).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "common/checks.hpp"
#include "common/rl/classic_control.hpp"
#include "common/rl/dqn_common.hpp"
#include "common/rl/replay.hpp"
#include "common/rl/utils.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

int main(int argc, char** argv) {
  try {
    rl::DqnConfig cfg;
    const float alpha = 0.6f, beta_start = 0.4f;
    rl::ParseConfig(argc, argv, &cfg);
    rl::CartPole env, eval_env;
    const int obs_dim = env.obs_size(), A = env.action_space().n;
    std::printf("== Double DQN + prioritized replay on CartPole, %d steps, alpha %.1f, beta %.1f -> 1\n", cfg.steps,
                alpha, beta_start);

    nn::ParameterStore online(cfg.seed), target(cfg.seed + 1);
    auto q_net = [&](nn::ParameterStore& p, const Tensor& x) {
      const Tensor h = nn::mlp(p, "mlp", x, {cfg.hidden, cfg.hidden});
      return nn::dense(p, "q", h, A);
    };
    {
      nn::NoGradGuard no_grad;
      q_net(online, nn::zeros({1, obs_dim}));
      q_net(target, nn::zeros({1, obs_dim}));
    }
    rl::Adam adam(online, cfg.lr, cfg.clip_norm);
    auto sync = [&] { target.copy_from(online, cfg.tau); };
    sync();

    // Per-sample TD errors y - Q(s, a) (Double DQN target).
    auto td_errors = [&](const rl::Batch& b) {
      Tensor y;
      {
        nn::NoGradGuard no_grad;
        const Tensor a_star = nn::argmax(q_net(online, b.s2), 1);
        const Tensor next_value = nn::sum(q_net(target, b.s2) * nn::one_hot(a_star, A), {1});
        y = b.r + (cfg.gamma * (1.0 - b.done)) * next_value;
      }
      const Tensor q_a = nn::sum(q_net(online, b.s) * nn::one_hot(b.a, A), {1});
      return y - q_a;
    };

    rl::PrioritizedReplay buffer(cfg.buffer, obs_dim, alpha);
    rl::Monitor monitor(cfg);
    std::mt19937_64 rng(cfg.seed);
    auto greedy = [&](const rl::Obs& o) {
      nn::NoGradGuard no_grad;
      return rl::ArgMax(q_net(online, rl::Row(o)).to_vector<float>());
    };

    rl::Obs obs = env.reset(cfg.seed);
    float ret = 0, best_eval = 0, min_w = 1, max_w = 0;
    double sampled_td = 0, uniform_td = 0;
    int td_count = 0;
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
      if (step >= cfg.warmup) {
        const float beta = beta_start + (1 - beta_start) * std::min(1.f, static_cast<float>(step) / cfg.steps);
        rl::Batch b = buffer.Sample(cfg.batch, beta, rng);
        for (float wi : b.weights.to_vector<float>()) min_w = std::min(min_w, wi), max_w = std::max(max_w, wi);
        const Tensor td = td_errors(b);
        // Importance weights scale each sample's loss.
        const Tensor loss = nn::mean(b.weights * nn::huber(td), {0});
        adam.Minimize(loss);
        const float l = loss.item<float>();
        check::Require(std::isfinite(l), "finite weighted TD loss");
        monitor.AddLoss(l);
        std::vector<float> td_abs = td.to_vector<float>();
        for (float& v : td_abs) v = std::abs(v);
        buffer.UpdatePriorities(b.indices, td_abs);
        // How much more "surprising" are prioritised samples than uniform ones?
        // Compare |TD| of this batch with a uniformly drawn one (forward pass only).
        if (step % 500 == 0) {
          for (float v : td_abs) sampled_td += v;
          rl::Batch u = buffer.SampleUniform(cfg.batch, rng);
          nn::NoGradGuard no_grad;
          for (float v : td_errors(u).to_vector<float>()) uniform_td += std::abs(v);
          td_count += cfg.batch;
        }
      }
      if (cfg.tau < 1.f || step % cfg.target_sync == 0) sync();
      if (step % cfg.log_every == 0) {
        const float ev = step % cfg.eval_every == 0 ? rl::Evaluate(eval_env, greedy, cfg.eval_episodes, 1000 + step) : NAN;
        if (!std::isnan(ev)) best_eval = std::max(best_eval, ev);
        monitor.Log(step, eps, ev);
      }
    }
    const float final_eval = rl::Evaluate(eval_env, greedy, 20, 777);
    std::printf("  final greedy evaluation over 20 episodes: %.1f\n", final_eval);
    const double mean_sampled = td_count ? sampled_td / td_count : 0.0, mean_uniform = td_count ? uniform_td / td_count : 0.0;
    std::printf("  mean |TD error|: prioritised batches %.4f vs uniform batches %.4f; max priority %.4f over %d transitions\n",
                mean_sampled, mean_uniform, buffer.max_priority(), buffer.size());
    std::printf("  importance weights seen: %.4f .. %.4f\n", min_w, max_w);

    check::Require(max_w <= 1.f + 1e-6f && min_w > 0.f, "importance weights lie in (0, 1]");
    check::TrainingQuality(mean_sampled > mean_uniform, "prioritised batches have larger |TD error| than uniform ones");
    check::TrainingQuality(std::max(best_eval, final_eval) >= 150, "PER Double DQN learned to balance (best greedy evaluation >= 150)");
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
