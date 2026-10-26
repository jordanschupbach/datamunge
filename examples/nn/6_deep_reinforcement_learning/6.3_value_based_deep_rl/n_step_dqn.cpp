// CMD: just run-nn n_step_dqn
// n-step DQN.
//
// Port of tflow's n_step_dqn. Bootstrapping after n rewards instead of one,
//   y = r_t + gamma r_{t+1} + ... + gamma^{n-1} r_{t+n-1} + gamma^n max_a Q_target(s_{t+n}, a),
// propagates reward information n times faster at the cost of being slightly
// off-policy (the intermediate actions came from an older, exploring policy).
// rl::NStepBuffer assembles the n-step transitions as the episode unfolds.
// Three horizons are trained with the same budget and compared.
//
// In TF the networks, loss and Adam were one graph; here they are eager nn
// ops over two ParameterStores (the target under NoGradGuard). CartPole,
// replay sampling and epsilon draws use the same std::mt19937_64 streams as
// TF; the weights use nn's initialiser, so the results match TF
// statistically. The vsim videos are omitted. Options: the rl::DqnConfig keys
// (steps= defaults to 40000 here).

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

namespace {

struct Result {
  float final_eval, best_eval;
};

Result Train(int n, const rl::DqnConfig& cfg) {
  std::printf("\n== %d-step Double DQN on CartPole, %d steps\n", n, cfg.steps);
  rl::CartPole env, eval_env;
  const int obs_dim = env.obs_size(), A = env.action_space().n;
  const float gamma_n = std::pow(cfg.gamma, n);

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

  auto train = [&](const rl::Batch& b) {  // b.r is already the n-step sum
    Tensor y;
    {
      nn::NoGradGuard no_grad;
      const Tensor a_star = nn::argmax(q_net(online, b.s2), 1);
      const Tensor next_value = nn::sum(q_net(target, b.s2) * nn::one_hot(a_star, A), {1});
      y = b.r + (gamma_n * (1.0 - b.done)) * next_value;  // gamma^n because s2 is n steps ahead
    }
    const Tensor q_a = nn::sum(q_net(online, b.s) * nn::one_hot(b.a, A), {1});
    const Tensor loss = nn::mean(nn::huber(y - q_a), {0});
    adam.Minimize(loss);
    return loss.item<float>();
  };

  rl::ReplayBuffer buffer(cfg.buffer, obs_dim);
  rl::NStepBuffer nstep(n, cfg.gamma);
  rl::Monitor monitor(cfg);
  std::mt19937_64 rng(cfg.seed);
  auto greedy = [&](const rl::Obs& o) {
    nn::NoGradGuard no_grad;
    return rl::ArgMax(q_net(online, rl::Row(o)).to_vector<float>());
  };

  rl::Obs obs = env.reset(cfg.seed);
  float ret = 0, best_eval = 0;
  for (int step = 1; step <= cfg.steps; ++step) {
    const float eps = rl::Epsilon(cfg, step);
    const int act = std::uniform_real_distribution<float>(0, 1)(rng) < eps ? static_cast<int>(rng() % A) : greedy(obs);
    rl::StepResult sr = env.step(act);
    // The one-step transition goes into the accumulator; completed n-step
    // transitions come out (all remaining ones when the episode ends).
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
  return {final_eval, std::max(best_eval, final_eval)};
}

}  // namespace

int main(int argc, char** argv) {
  try {
    rl::DqnConfig cfg;
    cfg.steps = 40000;
    cfg.log_every = 8000;
    cfg.eval_every = 8000;
    rl::ParseConfig(argc, argv, &cfg);

    // NStepBuffer sanity: rewards 1, 1, 1 with n = 3 give 1 + g + g^2 and s_{t+3}.
    {
      rl::NStepBuffer nb(3, 0.5f);
      std::vector<rl::Experience> out;
      for (int t = 0; t < 4; ++t)
        for (auto& e : nb.Push({{static_cast<float>(t)}, 0, 1.f, {static_cast<float>(t + 1)}, 0.f, {}})) out.push_back(e);
      check::Require(out.size() == 2 && std::abs(out[0].r - 1.75f) < 1e-6f && out[0].s2[0] == 3.f,
                     "NStepBuffer combines n rewards and jumps n states ahead");
    }

    std::vector<std::pair<int, Result>> results;
    for (int n : {1, 3, 5}) results.push_back({n, Train(n, cfg)});
    std::printf("\n  final greedy return by horizon:");
    for (auto& [n, v] : results) std::printf("  n=%d: %.1f", n, v.final_eval);
    std::printf("\n");
    int learned = 0;
    for (auto& nr : results) learned += nr.second.best_eval >= 150;
    check::TrainingQuality(learned >= 2, "at least two horizons learned to balance (best greedy evaluation >= 150)");
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
