// CMD: just run-nn qr_dqn
// Quantile regression DQN (QR-DQN).
//
// Port of tflow's qr_dqn. The return distribution is represented by N
// quantile values theta_i at fixed quantile fractions tau_i = (i + 0.5) / N
// (Dabney et al. 2018a). Unlike C51 there is no fixed support and no
// projection: the target quantiles r + gamma theta'_j are regressed with the
// quantile Huber loss
//   sum_i E_j[ |tau_i - 1{u_ij < 0}| * Huber(u_ij) ],   u_ij = T_j - theta_i,
// computed with tensor ops over the N x N' pairs.
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
    const int N = 32;
    rl::ParseConfig(argc, argv, &cfg);
    rl::CartPole env, eval_env;
    const int obs_dim = env.obs_size(), A = env.action_space().n;
    std::printf("== QR-DQN on CartPole, %d steps, %d quantiles\n", cfg.steps, N);

    struct Quantiles {
      Tensor theta, q;  // [B, A, N], [B, A]
    };
    nn::ParameterStore online(cfg.seed), target(cfg.seed + 1);
    auto quantile_net = [&](nn::ParameterStore& p, const Tensor& x) {
      const Tensor h = nn::mlp(p, "mlp", x, {cfg.hidden, cfg.hidden});
      const Tensor theta = nn::reshape(nn::dense(p, "theta", h, A * N), {-1, A, N});
      return Quantiles{theta, nn::mean(theta, {2})};
    };
    {
      nn::NoGradGuard no_grad;
      quantile_net(online, nn::zeros({1, obs_dim}));
      quantile_net(target, nn::zeros({1, obs_dim}));
    }
    rl::Adam adam(online, cfg.lr, cfg.clip_norm);
    auto sync = [&] { target.copy_from(online, cfg.tau); };
    sync();

    std::vector<float> taus(N);
    for (int i = 0; i < N; ++i) taus[i] = (i + 0.5f) / N;
    const Tensor tau = nn::from_vector(taus, {1, N, 1});

    rl::ReplayBuffer buffer(cfg.buffer, obs_dim);
    rl::Monitor monitor(cfg);
    std::mt19937_64 rng(cfg.seed);
    auto q_values = [&](const rl::Obs& o) {  // the mean of each action's return distribution
      nn::NoGradGuard no_grad;
      return quantile_net(online, rl::Row(o)).q.to_vector<float>();
    };
    auto greedy = [&](const rl::Obs& o) { return rl::ArgMax(q_values(o)); };

    std::vector<float> target_q(static_cast<size_t>(cfg.batch) * N);
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
        {  // Target quantiles: r + gamma (1 - done) theta'(s', a*) with a* the
           // target network's greedy action (by mean).
          nn::NoGradGuard no_grad;
          const Quantiles next = quantile_net(target, b.s2);
          const std::vector<float> th = next.theta.to_vector<float>(), qn = next.q.to_vector<float>();
          const std::vector<float> r = b.r.to_vector<float>(), done = b.done.to_vector<float>();
          for (int i = 0; i < cfg.batch; ++i) {
            const int best = static_cast<int>(std::max_element(qn.begin() + i * A, qn.begin() + (i + 1) * A) - (qn.begin() + i * A));
            const float cont = 1.f - done[i];
            for (int j = 0; j < N; ++j) target_q[i * N + j] = r[i] + cfg.gamma * cont * th[(i * A + best) * N + j];
          }
        }
        const Tensor T = nn::from_vector(target_q, {cfg.batch, N});
        const Quantiles on = quantile_net(online, b.s);
        const Tensor onehot = nn::expand_dims(nn::one_hot(b.a, A), 2);
        const Tensor theta_a = nn::sum(on.theta * onehot, {1});                       // [B, N]
        const Tensor u = nn::expand_dims(T, 1) - nn::expand_dims(theta_a, 2);         // [B, N, N']: T_j - theta_i
        const Tensor indicator = nn::cast(nn::less(u, 0.0), nn::DType::Float32);
        const Tensor weight = nn::abs(tau - indicator);
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

    // Quantiles of the return from the start state: the spread is the
    // network's own uncertainty about how long the episode will last.
    nn::NoGradGuard no_grad;
    const Quantiles start = quantile_net(online, rl::Row({0, 0, 0, 0}));
    const std::vector<float> th = start.theta.to_vector<float>(), qv = start.q.to_vector<float>();
    const int best = qv[1] > qv[0] ? 1 : 0;
    std::vector<float> raw(th.begin() + best * N, th.begin() + (best + 1) * N), sorted = raw;
    std::sort(sorted.begin(), sorted.end());
    std::printf("  start state, action %d: quantiles 10%%..90%% =", best);
    for (int pct = 10; pct <= 90; pct += 20) std::printf(" %.1f", sorted[pct * N / 100]);
    std::printf("  (mean %.1f)\n", qv[best]);
    // How far the learned quantiles are from monotone in tau (QR-DQN does not
    // enforce the ordering; training should approximately produce it).
    int inversions = 0;
    for (int i = 0; i + 1 < N; ++i) inversions += raw[i] > raw[i + 1] + 0.5f;
    std::printf("  quantile ordering: %d of %d neighbours out of order by more than 0.5\n", inversions, N - 1);

    // A random policy averages about 22.
    check::TrainingQuality(std::max(best_eval, final_eval) >= 150, "QR-DQN learned to balance (best greedy evaluation >= 150)");
    check::TrainingQuality(qv[best] > 0.f && qv[best] <= 1.f / (1.f - cfg.gamma) + 5.f, "start-state value within [0, 1/(1-gamma)]");
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
