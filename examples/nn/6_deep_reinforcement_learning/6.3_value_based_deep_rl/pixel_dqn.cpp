// CMD: just run-nn pixel_dqn
// DQN on pixel observations (Snake).
//
// Port of tflow's pixel_dqn. The same Double DQN as before, but the input is
// an image: Snake's three planes (head, body, food). A small convolutional
// network replaces the MLP, which is what the Atari agents do at a larger
// scale. It stays small enough to train quickly even on a CPU: a 6x6 board,
// two conv layers, 3-step returns and 40k steps are enough to see the agent
// go from dying immediately to eating several items.
//
// In TF the networks, loss and Adam were one graph; here they are eager nn
// ops (NHWC conv2d, SAME padding) over two ParameterStores (the target under
// NoGradGuard). Snake, replay sampling and epsilon draws use the same
// std::mt19937_64 streams as TF; the weights use nn's initialiser, so the
// results match TF statistically. The vsim video of the best greedy episode
// on the 3D board is omitted. Options: the rl::DqnConfig keys (steps=
// defaults to 40000 here).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "common/checks.hpp"
#include "common/rl/dqn_common.hpp"
#include "common/rl/replay.hpp"
#include "common/rl/snake.hpp"
#include "common/rl/utils.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

int main(int argc, char** argv) {
  try {
    rl::DqnConfig cfg;
    cfg.steps = 40000;
    cfg.train_every = 2;
    cfg.eps_decay_steps = 20000;
    cfg.eps_end = 0.02f;
    cfg.gamma = 0.97f;
    cfg.lr = 5e-4f;
    cfg.buffer = 20000;
    cfg.log_every = 4000;
    cfg.eval_every = 8000;
    cfg.eval_episodes = 10;
    const int n_step = 3;
    const float gamma_n = std::pow(cfg.gamma, n_step);
    rl::ParseConfig(argc, argv, &cfg);
    rl::Snake env(6, 6), eval_env(6, 6);
    const auto shape = env.obs_shape();  // [planes, H, W]
    const int planes = shape[0], Hh = shape[1], Ww = shape[2];
    const int obs_dim = env.obs_size(), A = env.action_space().n;
    std::printf("== Double DQN on Snake pixels: input %dx%dx%d, %d steps, %d-step returns\n", Hh, Ww, planes, cfg.steps,
                n_step);

    nn::ParameterStore online(cfg.seed), target(cfg.seed + 1);
    nn::ConvLayerOptions conv;
    conv.activation = nn::Activation::Relu;
    auto q_net = [&](nn::ParameterStore& p, const Tensor& x) {
      // [B, planes*H*W] -> [B, planes, H, W] -> NHWC [B, H, W, planes]
      const Tensor img = nn::transpose(nn::reshape(x, {-1, planes, Hh, Ww}), {0, 2, 3, 1});
      const Tensor c1 = nn::conv2d(p, "conv1", img, 16, 3, conv);
      const Tensor c2 = nn::conv2d(p, "conv2", c1, 32, 3, conv);
      const Tensor flat = nn::reshape(c2, {-1, Hh * Ww * 32});
      const Tensor h = nn::dense(p, "fc", flat, 128, nn::Activation::Relu);
      return nn::dense(p, "q", h, A);
    };
    {
      nn::NoGradGuard no_grad;
      q_net(online, nn::zeros({1, obs_dim}));
      q_net(target, nn::zeros({1, obs_dim}));
    }
    std::printf("  network: %lld parameters\n", static_cast<long long>(online.count()));
    rl::Adam adam(online, cfg.lr, cfg.clip_norm);
    auto sync = [&] { target.copy_from(online, cfg.tau); };
    sync();

    rl::ReplayBuffer buffer(cfg.buffer, obs_dim);
    rl::NStepBuffer nstep(n_step, cfg.gamma);
    rl::Monitor monitor(cfg);
    std::mt19937_64 rng(cfg.seed);
    auto greedy = [&](const rl::Obs& o) {
      nn::NoGradGuard no_grad;
      return rl::ArgMax(q_net(online, rl::Row(o)).to_vector<float>());
    };

    // Random-policy baseline on the evaluation seeds, for the check below.
    std::mt19937_64 rand_rng(4242);
    const float random_score = rl::Evaluate(eval_env, [&](const rl::Obs&) { return static_cast<int>(rand_rng() % A); }, 20, 777);

    rl::Obs obs = env.reset(cfg.seed);
    float ret = 0, best_eval = -1e9f;
    for (int step = 1; step <= cfg.steps; ++step) {
      const float eps = rl::Epsilon(cfg, step);
      const int act = std::uniform_real_distribution<float>(0, 1)(rng) < eps ? static_cast<int>(rng() % A) : greedy(obs);
      rl::StepResult sr = env.step(act);
      for (rl::Experience& t : nstep.Push({obs, act, sr.reward, sr.obs, sr.terminated ? 1.f : 0.f, {}}))
        buffer.Add(std::move(t));
      if (sr.truncated)
        for (rl::Experience& t : nstep.Flush()) buffer.Add(std::move(t));
      ret += sr.reward;
      obs = sr.obs;
      if (sr.done()) {
        monitor.EndEpisode(ret);
        ret = 0;
        obs = env.reset();
      }
      if (step >= cfg.warmup && step % cfg.train_every == 0) {
        rl::Batch b = buffer.Sample(cfg.batch, rng);
        Tensor y;
        {
          nn::NoGradGuard no_grad;
          const Tensor a_star = nn::argmax(q_net(online, b.s2), 1);
          const Tensor next_value = nn::sum(q_net(target, b.s2) * nn::one_hot(a_star, A), {1});
          y = b.r + (gamma_n * (1.0 - b.done)) * next_value;
        }
        const Tensor q_a = nn::sum(q_net(online, b.s) * nn::one_hot(b.a, A), {1});
        const Tensor loss = nn::mean(nn::huber(y - q_a), {0});
        adam.Minimize(loss);
        const float l = loss.item<float>();
        check::Require(std::isfinite(l), "finite TD loss");
        monitor.AddLoss(l);
      }
      if (cfg.tau < 1.f || step % cfg.target_sync == 0) sync();
      if (step % cfg.log_every == 0) {
        const float ev = step % cfg.eval_every == 0 ? rl::Evaluate(eval_env, greedy, cfg.eval_episodes, 1000 + step) : NAN;
        if (!std::isnan(ev)) best_eval = std::max(best_eval, ev);
        monitor.Log(step, eps, ev);
      }
    }
    const float final_eval = rl::Evaluate(eval_env, greedy, 20, 777);
    std::printf("  final greedy evaluation over 20 episodes: %.2f (food eaten minus deaths)\n", final_eval);
    std::printf("  random policy on the same episodes: %.2f\n", random_score);

    check::Require(online.count() == (3 * 3 * planes * 16 + 16) + (3 * 3 * 16 * 32 + 32) + (Hh * Ww * 32 * 128 + 128) + (128 * A + A),
                   "network size matches TF's conv-conv-fc-q layout");
    check::TrainingQuality(std::max(best_eval, final_eval) >= 2.f && final_eval > random_score + 1.f,
                   "the agent eats food (best greedy evaluation >= 2 and well above random)");
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
