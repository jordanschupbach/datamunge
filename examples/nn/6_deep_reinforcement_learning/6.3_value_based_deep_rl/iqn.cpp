// CMD: just run-nn iqn
// Implicit quantile networks (IQN).
//
// Port of tflow's iqn. QR-DQN predicts N fixed quantiles. IQN (Dabney et al.
// 2018b) instead learns the whole quantile function: it takes a quantile
// fraction tau as an *input*, embedded with cosine features
// phi(tau) = relu(W cos(pi i tau) + b) and multiplied into the state
// features, and outputs theta_tau(s, a). Each update samples fresh taus for
// the online and target networks; the loss is the same quantile Huber
// regression as QR-DQN with those sampled taus.
//
// In TF the networks, loss and Adam were one graph; here they are eager nn
// ops over two ParameterStores (the target under NoGradGuard). CartPole,
// replay sampling, epsilon draws and the sampled taus use the same
// std::mt19937_64 stream as TF, in the same order; the weights use nn's
// initialiser, so the results match TF statistically. The vsim video is
// omitted. Options: the rl::DqnConfig keys.

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
    const int N = 8, N_act = 32, embed = 64;
    rl::ParseConfig(argc, argv, &cfg);
    rl::CartPole env, eval_env;
    const int obs_dim = env.obs_size(), A = env.action_space().n;
    std::printf("== IQN on CartPole, %d steps, %d sampled taus per update, %d for acting\n", cfg.steps, N, N_act);

    std::vector<float> freqs(embed);
    for (int i = 0; i < embed; ++i) freqs[i] = rl::kPi * i;
    const Tensor pi_i = nn::from_vector(freqs, {1, 1, embed});

    struct Quantiles {
      Tensor theta, q;  // [B, N, A], [B, A]
    };
    nn::ParameterStore online(cfg.seed), target(cfg.seed + 1);
    auto iqn_net = [&](nn::ParameterStore& p, const Tensor& x, const Tensor& taus) {
      const Tensor psi = nn::mlp(p, "psi", x, {cfg.hidden});                    // [B, hidden]
      const Tensor cosines = nn::cos(nn::expand_dims(taus, 2) * pi_i);           // [B, N, embed]
      const Tensor phi = nn::dense(p, "phi", cosines, cfg.hidden, nn::Activation::Relu);  // [B, N, hidden]
      const Tensor zf = nn::expand_dims(psi, 1) * phi;                           // [B, N, hidden]
      const Tensor h = nn::dense(p, "head_hidden", zf, cfg.hidden, nn::Activation::Relu);
      const Tensor theta = nn::dense(p, "theta", h, A);                         // [B, N, A]
      return Quantiles{theta, nn::mean(theta, {1})};
    };
    {
      nn::NoGradGuard no_grad;
      iqn_net(online, nn::zeros({1, obs_dim}), nn::zeros({1, N}));
      iqn_net(target, nn::zeros({1, obs_dim}), nn::zeros({1, N}));
    }
    rl::Adam adam(online, cfg.lr, cfg.clip_norm);
    auto sync = [&] { target.copy_from(online, cfg.tau); };
    sync();

    rl::ReplayBuffer buffer(cfg.buffer, obs_dim);
    rl::Monitor monitor(cfg);
    std::mt19937_64 rng(cfg.seed);
    auto sample_taus = [&](int rows, int n) {
      std::vector<float> t(static_cast<std::size_t>(rows) * n);
      for (float& v : t) v = std::uniform_real_distribution<float>(0, 1)(rng);
      return nn::from_vector(t, {rows, n});
    };
    auto q_values = [&](const rl::Obs& o) {  // the mean over N_act sampled quantiles per action
      nn::NoGradGuard no_grad;
      return iqn_net(online, rl::Row(o), sample_taus(1, N_act)).q.to_vector<float>();
    };
    auto greedy = [&](const rl::Obs& o) { return rl::ArgMax(q_values(o)); };

    std::vector<float> target_q(cfg.batch * N);
    rl::Obs obs = env.reset(cfg.seed);
    float ret = 0, best_eval = 0;
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
        rl::Batch b = buffer.Sample(cfg.batch, rng);
        {  // target quantiles r + gamma (1 - done) theta'_tau'(s', a*), a* = argmax of the mean
          nn::NoGradGuard no_grad;
          const Quantiles next = iqn_net(target, b.s2, sample_taus(cfg.batch, N));
          const std::vector<float> th = next.theta.to_vector<float>(), qn = next.q.to_vector<float>();
          const std::vector<float> r = b.r.to_vector<float>(), done = b.done.to_vector<float>();
          for (int i = 0; i < cfg.batch; ++i) {
            const int best = static_cast<int>(std::max_element(qn.begin() + i * A, qn.begin() + (i + 1) * A) - (qn.begin() + i * A));
            for (int j = 0; j < N; ++j) target_q[i * N + j] = r[i] + cfg.gamma * (1.f - done[i]) * th[(i * N + j) * A + best];
          }
        }
        const Tensor T = nn::from_vector(target_q, {cfg.batch, N});
        const Tensor tau = sample_taus(cfg.batch, N);
        const Quantiles on = iqn_net(online, b.s, tau);
        const Tensor theta_a = nn::sum(on.theta * nn::expand_dims(nn::one_hot(b.a, A), 1), {2});  // [B, N]
        const Tensor u = nn::expand_dims(T, 1) - nn::expand_dims(theta_a, 2);                     // [B, N, N']
        const Tensor indicator = nn::cast(nn::less(u.detach(), 0.0), nn::DType::Float32);
        const Tensor weight = nn::abs(nn::expand_dims(tau, 2) - indicator);
        const Tensor loss = nn::mean(nn::sum(nn::mean(weight * nn::huber(u), {2}), {1}), {0});
        adam.Minimize(loss);
        const float l = loss.item<float>();
        check::Require(std::isfinite(l), "finite quantile Huber loss");
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
    std::printf("  final greedy evaluation over 20 episodes: %.1f\n", final_eval);

    // The quantile function can be queried at any tau: evaluate it on a grid.
    std::vector<float> grid(9);
    for (int i = 0; i < 9; ++i) grid[i] = 0.1f * (i + 1);
    std::vector<float> th, qv;
    {
      nn::NoGradGuard no_grad;
      const Quantiles g = iqn_net(online, rl::Row({0, 0, 0, 0}), nn::from_vector(grid, {1, 9}));
      th = g.theta.to_vector<float>();
      qv = g.q.to_vector<float>();
    }
    const int best = qv[1] > qv[0] ? 1 : 0;
    std::printf("  start state, action %d: theta(tau) for tau = 0.1..0.9 =", best);
    for (int i = 0; i < 9; ++i) std::printf(" %.1f", th[i * A + best]);
    std::printf("\n");
    // A quantile function is non-decreasing; IQN does not enforce it, so allow small crossings.
    check::TrainingQuality(th[8 * A + best] >= th[0 * A + best] - 0.5f, "theta(0.9) >= theta(0.1) at the start state");
    check::TrainingQuality(std::max(best_eval, final_eval) >= 150, "IQN learned to balance (best greedy evaluation >= 150)");
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
