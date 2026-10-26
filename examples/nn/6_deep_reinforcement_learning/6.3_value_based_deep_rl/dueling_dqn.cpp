// CMD: just run-nn dueling_dqn
// Dueling DQN.
//
// Port of tflow's dueling_dqn. The dueling head (Wang et al. 2016) splits the
// network into a state value V(s) and action advantages A(s, a), recombined as
//   Q(s, a) = V(s) + A(s, a) - mean_a' A(s, a').
// Subtracting the mean advantage makes the decomposition identifiable. The
// value stream learns from every transition regardless of the action taken,
// which helps when many actions have similar consequences. Training uses the
// Double DQN target.
//
// In TF the networks, loss and Adam were one graph; here they are eager nn
// ops over two ParameterStores (the target under NoGradGuard). CartPole,
// replay sampling and epsilon draws use the same std::mt19937_64 streams as
// TF; the weights use nn's initialiser, so the results match TF
// statistically. The vsim video is omitted. Options: the rl::DqnConfig keys.

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
    rl::ParseConfig(argc, argv, &cfg);
    rl::CartPole env, eval_env;
    const int obs_dim = env.obs_size(), A = env.action_space().n;
    std::printf("== Dueling DQN on CartPole, %d steps\n", cfg.steps);

    struct Streams {
      Tensor q, value, advantage;
    };
    nn::ParameterStore online(cfg.seed), target(cfg.seed + 1);
    auto q_net = [&](nn::ParameterStore& p, const Tensor& x) {
      const Tensor h = nn::mlp(p, "mlp", x, {cfg.hidden});
      // Two streams from the shared features.
      const Tensor value = nn::dense(p, "value", nn::dense(p, "value_hidden", h, cfg.hidden, nn::Activation::Relu), 1);
      const Tensor adv = nn::dense(p, "advantage", nn::dense(p, "advantage_hidden", h, cfg.hidden, nn::Activation::Relu), A);
      const Tensor adv_centred = adv - nn::mean(adv, {1}, /*keepdims=*/true);
      return Streams{value + adv_centred, value, adv_centred};
    };
    {
      nn::NoGradGuard no_grad;
      q_net(online, nn::zeros({1, obs_dim}));
      q_net(target, nn::zeros({1, obs_dim}));
    }
    rl::Adam adam(online, cfg.lr, cfg.clip_norm);
    auto sync = [&] { target.copy_from(online, cfg.tau); };
    sync();

    auto train = [&](const rl::Batch& b) {
      Tensor y;
      {  // Double DQN target
        nn::NoGradGuard no_grad;
        const Tensor a_star = nn::argmax(q_net(online, b.s2).q, 1);
        const Tensor next_value = nn::sum(q_net(target, b.s2).q * nn::one_hot(a_star, A), {1});
        y = b.r + (cfg.gamma * (1.0 - b.done)) * next_value;
      }
      const Tensor q_a = nn::sum(q_net(online, b.s).q * nn::one_hot(b.a, A), {1});
      const Tensor loss = nn::mean(nn::huber(y - q_a), {0});
      adam.Minimize(loss);
      return loss.item<float>();
    };

    rl::ReplayBuffer buffer(cfg.buffer, obs_dim);
    rl::Monitor monitor(cfg);
    std::mt19937_64 rng(cfg.seed);
    auto q_values = [&](const rl::Obs& o) {  // Q(s, .) from the online network
      nn::NoGradGuard no_grad;
      return q_net(online, rl::Row(o)).q.to_vector<float>();
    };
    auto greedy = [&](const rl::Obs& o) { return rl::ArgMax(q_values(o)); };

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
        const float loss = train(buffer.Sample(cfg.batch, rng));
        check::Require(std::isfinite(loss), "finite TD loss");
        monitor.AddLoss(loss);
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

    // What the two streams say: on a balanced pole both actions are about
    // equally good (small advantages); leaning hard, the advantage gap opens.
    std::printf("\n  %-34s %8s %10s %10s\n", "state [x, x_dot, theta, theta_dot]", "V(s)", "A(s,left)", "A(s,right)");
    for (const rl::Obs& o : std::vector<rl::Obs>{{0, 0, 0, 0}, {0, 0, 0.15f, 1.f}, {0, 0, -0.15f, -1.f}, {1.5f, 1.f, 0, 0}}) {
      nn::NoGradGuard no_grad;
      const Streams st = q_net(online, rl::Row(o));
      const std::vector<float> adv = st.advantage.to_vector<float>();
      std::printf("  %-34s %8.2f %10.3f %10.3f\n", rl::FormatObs(o).c_str(), st.value.item<float>(), adv[0], adv[1]);
      check::Require(std::abs(adv[0] + adv[1]) < 1e-4f, "centred advantages sum to zero");
    }
    check::TrainingQuality(std::max(best_eval, final_eval) >= 150, "Dueling DQN learned to balance (best greedy evaluation >= 150)");
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
