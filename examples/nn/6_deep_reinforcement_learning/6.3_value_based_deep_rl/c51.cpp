// CMD: just run-nn c51
// Distributional RL: C51.
//
// Port of tflow's c51. Rather than the expected return, the network predicts
// a whole return distribution for each action as a categorical distribution
// over 51 fixed atoms z_i in [v_min, v_max] (Bellemare et al. 2017). The
// Bellman update shifts and shrinks the next state's distribution,
// r + gamma z, which lands between atoms, so it is projected back onto the
// support (done on the host in rl::ProjectDistribution). The loss is the
// cross-entropy between that projected target and the predicted distribution
// of the taken action.
//
// In TF the networks, loss and Adam were one graph; here they are eager nn
// ops over two ParameterStores (target network under NoGradGuard). The
// CartPole dynamics, replay sampling and epsilon draws use the same
// std::mt19937_64 streams as TF; the weights use nn's initialiser, so the
// learning curves match TF statistically, not digit for digit. The vsim video
// of the best evaluation episode is omitted. Options: the rl::DqnConfig keys
// (steps=, lr=, seed=, ...).

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
    const int atoms = 51;
    const float vmin = 0.f, vmax = 1.f / (1.f - cfg.gamma);  // CartPole returns lie in [0, 1/(1-gamma)]
    rl::ParseConfig(argc, argv, &cfg);
    rl::CartPole env, eval_env;
    const int obs_dim = env.obs_size(), A = env.action_space().n;
    std::printf("== C51 on CartPole, %d steps, %d atoms on [%.0f, %.0f]\n", cfg.steps, atoms, vmin, vmax);

    std::vector<float> support(atoms);
    for (int i = 0; i < atoms; ++i) support[i] = vmin + i * (vmax - vmin) / (atoms - 1);
    const Tensor z = nn::from_vector(support, {atoms});

    struct Dist {
      Tensor logits, probs, q;  // [B, A, atoms], [B, A, atoms], [B, A]
    };
    nn::ParameterStore online(cfg.seed), target(cfg.seed + 1);
    auto dist_net = [&](nn::ParameterStore& p, const Tensor& x) {
      const Tensor h = nn::mlp(p, "mlp", x, {cfg.hidden, cfg.hidden});
      const Tensor logits = nn::reshape(nn::dense(p, "logits", h, A * atoms), {-1, A, atoms});
      const Tensor probs = nn::softmax(logits, -1);
      return Dist{logits, probs, nn::sum(probs * z, {2})};  // expected return per action
    };
    {
      nn::NoGradGuard no_grad;
      dist_net(online, nn::zeros({1, obs_dim}));
      dist_net(target, nn::zeros({1, obs_dim}));
    }
    rl::Adam adam(online, cfg.lr, cfg.clip_norm);
    auto sync = [&] { target.copy_from(online, cfg.tau); };
    sync();

    rl::ReplayBuffer buffer(cfg.buffer, obs_dim);
    rl::Monitor monitor(cfg);
    std::mt19937_64 rng(cfg.seed);
    auto q_values = [&](const rl::Obs& o) {  // the mean of each action's return distribution
      nn::NoGradGuard no_grad;
      return dist_net(online, rl::Row(o)).q.to_vector<float>();
    };
    auto greedy = [&](const rl::Obs& o) { return rl::ArgMax(q_values(o)); };

    std::vector<float> probs_next(cfg.batch * atoms), target_dist(cfg.batch * atoms);
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
        {  // 1. Target network: next-state distributions and their means.
          nn::NoGradGuard no_grad;
          const Dist next = dist_net(target, b.s2);
          const std::vector<float> pn = next.probs.to_vector<float>(), qn = next.q.to_vector<float>();
          for (int i = 0; i < cfg.batch; ++i) {
            const int best = static_cast<int>(std::max_element(qn.begin() + i * A, qn.begin() + (i + 1) * A) - (qn.begin() + i * A));
            std::copy(pn.begin() + (i * A + best) * atoms, pn.begin() + (i * A + best + 1) * atoms, probs_next.begin() + i * atoms);
          }
        }
        // 2. Project r + gamma (1 - done) z onto the support.
        const std::vector<float> r = b.r.to_vector<float>(), done = b.done.to_vector<float>();
        rl::ProjectDistribution(probs_next.data(), r.data(), done.data(), cfg.batch, atoms, vmin, vmax, cfg.gamma,
                                target_dist.data());
        const Tensor m = nn::from_vector(target_dist, {cfg.batch, atoms});
        // 3. Cross-entropy between m and p(. | s, a).
        const Dist on = dist_net(online, b.s);
        const Tensor onehot = nn::expand_dims(nn::one_hot(b.a, A), 2);         // [B, A, 1]
        const Tensor log_p = nn::log_softmax(nn::sum(on.logits * onehot, {1}));  // [B, atoms]
        const Tensor loss = nn::mean(-nn::sum(m * log_p, {1}), {0});
        adam.Minimize(loss);
        const float l = loss.item<float>();
        check::Require(std::isfinite(l), "finite cross-entropy loss");
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

    // The learned return distribution for a balanced start state and for a
    // state about to fail.
    std::vector<float> q_states;
    for (const rl::Obs& o : std::vector<rl::Obs>{{0, 0, 0, 0}, {0, 0, 0.19f, 2.f}}) {
      nn::NoGradGuard no_grad;
      const Dist d = dist_net(online, rl::Row(o));
      const std::vector<float> p = d.probs.to_vector<float>(), qv = d.q.to_vector<float>();
      const int best = qv[1] > qv[0] ? 1 : 0;
      q_states.push_back(qv[best]);
      std::printf("\n  state %s, best action %d, Q = %.1f; return distribution (10 bins of %.0f):\n   ",
                  rl::FormatObs(o).c_str(), best, qv[best], (vmax - vmin) / 10);
      float total = 0;
      for (int bin = 0; bin < 10; ++bin) {
        float mass = 0;
        for (int i = 0; i < atoms; ++i)
          if (support[i] >= vmin + bin * (vmax - vmin) / 10 &&
              support[i] < vmin + (bin + 1) * (vmax - vmin) / 10 + (bin == 9 ? 1 : 0))
            mass += p[best * atoms + i];
        total += mass;
        std::printf(" %3.0f%%", 100 * mass);
      }
      std::printf("\n");
      check::Require(std::abs(total - 1.f) < 1e-3f, "the return distribution sums to 1");
    }
    // A random policy averages about 22; over seeds 0..2 the final evaluation was 323..500 in TF, 500 here.
    check::TrainingQuality(std::max(best_eval, final_eval) >= 150, "C51 learned to balance (best greedy evaluation >= 150)");
    check::TrainingQuality(q_states[0] > q_states[1], "the balanced state is worth more than the falling one");
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
