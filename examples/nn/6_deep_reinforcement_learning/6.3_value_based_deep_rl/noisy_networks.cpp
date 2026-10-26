// CMD: just run-nn noisy_networks
// Noisy networks for exploration.
//
// Port of tflow's noisy_networks. Instead of epsilon-greedy, every linear
// layer gets learnable noise: W = mu + sigma * eps with factorised Gaussian
// eps resampled on each forward pass (Fortunato et al. 2018; nn::noisy_dense).
// The agent acts greedily on its noisy Q-values, so exploration is
// state-dependent and the network can learn to switch it off by shrinking
// sigma. Epsilon is 0 throughout; the noise is disabled (noise_scale = 0)
// only for evaluation. The mean |sigma| is logged to show the noise being
// annealed by gradient descent rather than by a schedule.
//
// In TF the networks, loss and Adam were one graph; here they are eager nn
// ops over two ParameterStores (the target under NoGradGuard). CartPole and
// replay sampling use the same std::mt19937_64 streams as TF; the weights and
// the weight noise come from nn's generators (TF used its RandomNormal op),
// so the results match TF statistically. The vsim video of the best
// evaluation episode is omitted. Options: the rl::DqnConfig keys (steps=,
// lr=, seed=, ...).

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
    cfg.eps_start = cfg.eps_end = 0.f;  // exploration comes from the noise
    rl::ParseConfig(argc, argv, &cfg);
    rl::CartPole env, eval_env;
    const int obs_dim = env.obs_size(), A = env.action_space().n;
    std::printf("== NoisyNet Double DQN on CartPole, %d steps, epsilon 0\n", cfg.steps);

    nn::ParameterStore online(cfg.seed), target(cfg.seed + 1);
    nn::Generator noise(cfg.seed + 12345);  // the factorised weight noise
    auto q_net = [&](nn::ParameterStore& p, const Tensor& x, double noise_scale) {
      const Tensor h1 = nn::noisy_dense(p, "noisy1", x, cfg.hidden, noise, noise_scale, nn::Activation::Relu);
      const Tensor h2 = nn::noisy_dense(p, "noisy2", h1, cfg.hidden, noise, noise_scale, nn::Activation::Relu);
      return nn::noisy_dense(p, "noisy_q", h2, A, noise, noise_scale);
    };
    {
      nn::NoGradGuard no_grad;
      q_net(online, nn::zeros({1, obs_dim}), 1.0);
      q_net(target, nn::zeros({1, obs_dim}), 1.0);
    }
    rl::Adam adam(online, cfg.lr, cfg.clip_norm);
    auto sync = [&] { target.copy_from(online, cfg.tau); };
    sync();
    // Mean |sigma| over all noisy weights, for logging.
    auto mean_sigma = [&] {
      double total = 0;
      int parts = 0;
      for (const std::string& name : online.names())
        if (name.find("sigma") != std::string::npos) {
          total += nn::mean(nn::abs(online.at(name))).item<float>();
          ++parts;
        }
      return static_cast<float>(total / parts);
    };
    const float sigma_start = mean_sigma();

    rl::ReplayBuffer buffer(cfg.buffer, obs_dim);
    rl::Monitor monitor(cfg, "sigma");
    std::mt19937_64 rng(cfg.seed);
    auto act_with = [&](const rl::Obs& o, double noise_scale) {
      nn::NoGradGuard no_grad;
      return rl::ArgMax(q_net(online, rl::Row(o), noise_scale).to_vector<float>());
    };
    auto noisy_greedy = [&](const rl::Obs& o) { return act_with(o, 1.0); };
    auto quiet_greedy = [&](const rl::Obs& o) { return act_with(o, 0.0); };

    std::printf("  (eval uses noise_scale = 0; the 'sigma' column is the mean |sigma| of the noisy weights)\n");
    rl::Obs obs = env.reset(cfg.seed);
    float ret = 0, best_eval = 0;
    for (int step = 1; step <= cfg.steps; ++step) {
      const int act = noisy_greedy(obs);
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
        rl::Batch b = buffer.Sample(cfg.batch, rng);
        Tensor y;
        {  // Double DQN target, with noise in both networks as in TF's graph.
          nn::NoGradGuard no_grad;
          const Tensor a_star = nn::argmax(q_net(online, b.s2, 1.0), 1);
          const Tensor next_value = nn::sum(q_net(target, b.s2, 1.0) * nn::one_hot(a_star, A), {1});
          y = b.r + (cfg.gamma * (1.0 - b.done)) * next_value;
        }
        const Tensor q_a = nn::sum(q_net(online, b.s, 1.0) * nn::one_hot(b.a, A), {1});
        const Tensor loss = nn::mean(nn::huber(y - q_a), {0});
        adam.Minimize(loss);
        const float l = loss.item<float>();
        check::Require(std::isfinite(l), "finite TD loss");
        monitor.AddLoss(l);
      }
      if (cfg.tau < 1.f || step % cfg.target_sync == 0) sync();
      if (step % cfg.log_every == 0) {
        const float ev = step % cfg.eval_every == 0 ? rl::Evaluate(eval_env, quiet_greedy, cfg.eval_episodes, 1000 + step) : NAN;
        if (!std::isnan(ev)) best_eval = std::max(best_eval, ev);
        monitor.Log(step, mean_sigma(), ev);
      }
    }
    const float sigma_end = mean_sigma();
    const float quiet = rl::Evaluate(eval_env, quiet_greedy, 20, 777), noisy = rl::Evaluate(eval_env, noisy_greedy, 20, 777);
    std::printf("  final evaluation over 20 episodes: noise off %.1f, noise on %.1f\n", quiet, noisy);
    std::printf("  mean |sigma|: %.4f at the start, %.4f at the end\n", sigma_start, sigma_end);

    // A random policy averages about 22.
    check::Require(std::isfinite(sigma_end), "finite noise scales");
    check::TrainingQuality(std::max(best_eval, quiet) >= 150, "NoisyNet DQN learned to balance (best greedy evaluation >= 150)");
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
