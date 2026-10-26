// CMD: just run-nn double_dqn
// Double DQN.
//
// Port of tflow's double_dqn. DQN's target max_a' Q_target(s', a') uses the
// same values to pick the best next action and to score it, so estimation
// noise is systematically selected for and Q is overestimated. Double DQN
// (van Hasselt et al. 2016) lets the online network choose the action and the
// target network score it:
//   y = r + gamma (1 - done) Q_target(s', argmax_a' Q_online(s', a')).
// Both variants are trained here and their value estimates at the start
// state are compared with the discounted return the greedy policy really
// achieves. On a task this small and with this little training the gap is
// dominated by under-converged values and run-to-run variance, so treat the
// numbers as a diagnostic rather than a verdict; the overestimation effect
// is clearest on larger problems and longer runs.
//
// In TF the networks, loss and Adam were one graph; here they are eager nn
// ops over two ParameterStores (the target under NoGradGuard, TF's
// StopGradient). CartPole, replay sampling and epsilon draws use the same
// std::mt19937_64 streams as TF; the weights use nn's initialiser, so the
// results match TF statistically. The vsim videos are omitted. Options: the
// rl::DqnConfig keys (steps= defaults to 40000 here, seed=, ...).

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

namespace {

struct Result {
  float eval_return, q_estimate, true_discounted, best_eval;
};

Result Train(bool double_q, const rl::DqnConfig& cfg) {
  std::printf("\n== %s DQN\n", double_q ? "Double" : "Plain");
  rl::CartPole env, eval_env;
  const int obs_dim = env.obs_size(), A = env.action_space().n;

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

  auto train = [&](const rl::Batch& b) {
    Tensor y;
    {
      nn::NoGradGuard no_grad;
      const Tensor q_next_target = q_net(target, b.s2);
      Tensor next_value;
      if (double_q) {  // online network selects, target network evaluates
        const Tensor a_star = nn::argmax(q_net(online, b.s2), 1);
        next_value = nn::sum(q_next_target * nn::one_hot(a_star, A), {1});
      } else {
        next_value = nn::max(q_next_target, {1});
      }
      y = b.r + (cfg.gamma * (1.0 - b.done)) * next_value;
    }
    const Tensor q_a = nn::sum(q_net(online, b.s) * nn::one_hot(b.a, A), {1});
    const Tensor loss = nn::mean(nn::huber(y - q_a), {0});
    adam.Minimize(loss);
    return loss.item<float>();
  };

  rl::ReplayBuffer buffer(cfg.buffer, obs_dim);
  rl::Monitor monitor(cfg);
  std::mt19937_64 rng(cfg.seed);
  auto q_values = [&](const rl::Obs& o) {
    nn::NoGradGuard no_grad;
    return q_net(online, rl::Row(o)).to_vector<float>();
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

  // Value estimate vs. reality: Q(s0, greedy) against the discounted return
  // actually collected from s0 by the greedy policy.
  Result res{rl::Evaluate(eval_env, greedy, 20, 777), 0.f, 0.f, 0.f};
  res.best_eval = std::max(best_eval, res.eval_return);
  for (int e = 0; e < 20; ++e) {
    rl::Obs o = eval_env.reset(5000 + e);
    std::vector<float> qv = q_values(o);
    res.q_estimate += *std::max_element(qv.begin(), qv.end()) / 20;
    float g = 0, discount = 1;
    while (true) {
      rl::StepResult sr = eval_env.step(greedy(o));
      g += discount * sr.reward;
      discount *= cfg.gamma;
      o = sr.obs;
      if (sr.done()) break;
    }
    res.true_discounted += g / 20;
  }
  std::printf("  greedy return %.1f; Q(s0) estimate %.2f vs true discounted return %.2f (gap %+.2f)\n",
              res.eval_return, res.q_estimate, res.true_discounted, res.q_estimate - res.true_discounted);
  return res;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    rl::DqnConfig cfg;
    cfg.steps = 40000;
    rl::ParseConfig(argc, argv, &cfg);
    Result plain = Train(false, cfg);
    Result dbl = Train(true, cfg);
    std::printf("\n  value estimation error (Q estimate - true discounted return): plain %+.2f, double %+.2f\n",
                plain.q_estimate - plain.true_discounted, dbl.q_estimate - dbl.true_discounted);
    // Q is bounded by 1 / (1 - gamma) = 50 on CartPole; a diverged network would blow past it.
    // A random policy scores about 22. With 40000 steps the best greedy evaluation of a
    // variant ranged 150..500 in TF and 138..500 in this port over seeds 0..2.
    for (const Result* r : {&plain, &dbl}) {
      check::Require(std::isfinite(r->q_estimate) && r->q_estimate < 1.5f / (1.f - cfg.gamma), "Q estimate stays bounded");
      check::TrainingQuality(r->best_eval >= 100, "both variants learned (best greedy evaluation >= 100)");
    }
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
