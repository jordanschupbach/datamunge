// CMD: just run-nn rainbow
// Rainbow: all of the above combined.
//
// Port of tflow's rainbow. Hessel et al. 2018 stacked six improvements on
// DQN and found they are largely complementary. This agent combines, from the
// previous examples:
//   - Double DQN      : online network picks a*, target network scores it
//   - dueling head    : separate value and advantage streams (distributional)
//   - noisy layers    : exploration by parameter noise, epsilon = 0
//   - prioritized replay with importance weights
//   - n-step returns  : 3-step targets with gamma^3
//   - C51             : categorical return distributions, cross-entropy loss
// Priorities are the per-sample cross-entropy losses.
//
// In TF the networks, loss and Adam were one graph; here they are eager nn
// ops over two ParameterStores (the target under NoGradGuard). CartPole and
// replay sampling use the same std::mt19937_64 streams as TF; the weights and
// the weight noise come from nn's generators, so the results match TF
// statistically. The vsim video of the best evaluation episode is omitted.
// Options: the rl::DqnConfig keys (steps=, lr=, seed=, ...).

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
#include "common/rl/classic_control.hpp"
#include "common/rl/dqn_common.hpp"
#include "common/rl/replay.hpp"
#include "common/rl/utils.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

int main(int argc, char** argv) {
  try {
    rl::DqnConfig cfg;
    cfg.eps_start = cfg.eps_end = 0.f;
    const int atoms = 51, n_step = 3;
    const float vmin = 0.f, vmax = 1.f / (1.f - cfg.gamma), alpha = 0.6f, beta_start = 0.4f;
    const float gamma_n = std::pow(cfg.gamma, n_step);
    rl::ParseConfig(argc, argv, &cfg);
    rl::CartPole env, eval_env;
    const int obs_dim = env.obs_size(), A = env.action_space().n;
    std::printf("== Rainbow on CartPole, %d steps: double + dueling + noisy + PER + %d-step + C51(%d atoms)\n",
                cfg.steps, n_step, atoms);

    std::vector<float> support(atoms);
    for (int i = 0; i < atoms; ++i) support[i] = vmin + i * (vmax - vmin) / (atoms - 1);
    const Tensor z = nn::from_vector(support, {atoms});

    struct Dist {
      Tensor logits, probs, q;  // [B, A, atoms] x2, [B, A]
    };
    nn::ParameterStore online(cfg.seed), target(cfg.seed + 1);
    nn::Generator noise(cfg.seed + 12345);
    // Noisy dueling distributional head: value logits [B, 1, atoms] and
    // advantage logits [B, A, atoms], combined like the dueling architecture.
    auto rainbow_net = [&](nn::ParameterStore& p, const Tensor& x, double s) {
      const Tensor h = nn::mlp(p, "mlp", x, {cfg.hidden});
      const Tensor vh = nn::noisy_dense(p, "value_hidden", h, cfg.hidden, noise, s, nn::Activation::Relu);
      const Tensor ah = nn::noisy_dense(p, "adv_hidden", h, cfg.hidden, noise, s, nn::Activation::Relu);
      const Tensor value = nn::reshape(nn::noisy_dense(p, "value", vh, atoms, noise, s), {-1, 1, atoms});
      const Tensor adv = nn::reshape(nn::noisy_dense(p, "advantage", ah, A * atoms, noise, s), {-1, A, atoms});
      const Tensor logits = value + (adv - nn::mean(adv, {1}, /*keepdims=*/true));
      const Tensor probs = nn::softmax(logits, -1);
      return Dist{logits, probs, nn::sum(probs * z, {2})};
    };
    {
      nn::NoGradGuard no_grad;
      rainbow_net(online, nn::zeros({1, obs_dim}), 1.0);
      rainbow_net(target, nn::zeros({1, obs_dim}), 1.0);
    }
    rl::Adam adam(online, cfg.lr, cfg.clip_norm);
    auto sync = [&] { target.copy_from(online, cfg.tau); };
    sync();

    rl::PrioritizedReplay buffer(cfg.buffer, obs_dim, alpha);
    rl::NStepBuffer nstep(n_step, cfg.gamma);
    rl::Monitor monitor(cfg);
    std::mt19937_64 rng(cfg.seed);
    auto q_with = [&](const rl::Obs& o, double s) {  // Q(s, .) with the weight noise scaled
      nn::NoGradGuard no_grad;
      return rainbow_net(online, rl::Row(o), s).q.to_vector<float>();
    };
    auto noisy_greedy = [&](const rl::Obs& o) { return rl::ArgMax(q_with(o, 1.0)); };
    auto quiet_greedy = [&](const rl::Obs& o) { return rl::ArgMax(q_with(o, 0.0)); };

    std::vector<float> probs_next(cfg.batch * atoms), target_dist(cfg.batch * atoms);
    rl::Obs obs = env.reset(cfg.seed);
    float ret = 0, best_eval = 0;
    for (int step = 1; step <= cfg.steps; ++step) {
      const int act = noisy_greedy(obs);
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
      if (step >= cfg.warmup) {
        const float beta = beta_start + (1 - beta_start) * std::min(1.f, static_cast<float>(step) / cfg.steps);
        rl::Batch b = buffer.Sample(cfg.batch, beta, rng);
        {  // Double DQN: online network's greedy action at s', target's distribution for it.
          nn::NoGradGuard no_grad;
          const std::vector<float> qn = rainbow_net(online, b.s2, 1.0).q.to_vector<float>();
          const std::vector<float> pn = rainbow_net(target, b.s2, 1.0).probs.to_vector<float>();
          for (int i = 0; i < cfg.batch; ++i) {
            const int best = static_cast<int>(std::max_element(qn.begin() + i * A, qn.begin() + (i + 1) * A) - (qn.begin() + i * A));
            std::copy(pn.begin() + (i * A + best) * atoms, pn.begin() + (i * A + best + 1) * atoms, probs_next.begin() + i * atoms);
          }
        }
        const std::vector<float> r = b.r.to_vector<float>(), done = b.done.to_vector<float>();
        rl::ProjectDistribution(probs_next.data(), r.data(), done.data(), cfg.batch, atoms, vmin, vmax, gamma_n,
                                target_dist.data());
        const Tensor m = nn::from_vector(target_dist, {cfg.batch, atoms});
        const Dist on = rainbow_net(online, b.s, 1.0);
        const Tensor onehot = nn::expand_dims(nn::one_hot(b.a, A), 2);
        const Tensor log_p = nn::log_softmax(nn::sum(on.logits * onehot, {1}));
        const Tensor sample_loss = -nn::sum(m * log_p, {1});  // [B]
        const Tensor loss = nn::mean(b.weights * sample_loss, {0});
        adam.Minimize(loss);
        const float l = loss.item<float>();
        check::Require(std::isfinite(l), "finite weighted cross-entropy loss");
        monitor.AddLoss(l);
        buffer.UpdatePriorities(b.indices, sample_loss.to_vector<float>());
      }
      if (cfg.tau < 1.f || step % cfg.target_sync == 0) sync();
      if (step % cfg.log_every == 0) {
        const float ev = step % cfg.eval_every == 0 ? rl::Evaluate(eval_env, quiet_greedy, cfg.eval_episodes, 1000 + step) : NAN;
        if (!std::isnan(ev)) best_eval = std::max(best_eval, ev);
        monitor.Log(step, 0.f, ev);
      }
    }
    const float final_eval = rl::Evaluate(eval_env, quiet_greedy, 20, 777);
    std::printf("  final evaluation over 20 episodes (noise off): %.1f\n", final_eval);

    // A random policy averages about 22. With this small budget Rainbow often
    // plateaus near 110-140 (TF seeds 0..2: best 115.6, 198.6, 137.8; here
    // 234.6, 127.2, 269.6), so the bar is 100 rather than the 150 of c51.
    check::TrainingQuality(std::max(best_eval, final_eval) >= 100, "Rainbow learned to balance (best greedy evaluation >= 100)");
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
