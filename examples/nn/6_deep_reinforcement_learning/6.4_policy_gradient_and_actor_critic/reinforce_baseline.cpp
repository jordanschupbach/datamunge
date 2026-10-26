// CMD: just run-nn reinforce_baseline
// REINFORCE with a learned baseline.
//
// Port of tflow's reinforce_baseline. Subtracting any state-dependent
// baseline b(s) from the return leaves the policy gradient unbiased but can
// shrink its variance enormously:
//   grad J = E[ grad log pi(a_t | s_t) (G_t - b(s_t)) ].
// The best simple choice is b(s) = V(s), fitted by regression on the same
// Monte Carlo returns. This program trains both variants with the same
// budget and reports the spread of the gradient norms side by side.
//
// The TF graph becomes eager nn ops over a ParameterStore; the norm of the
// policy-only gradient (TF's second AddSymbolicGradients over the "actor"
// variables) is one nn::grad over the actor parameters. CartPole and the
// action sampling use the same std::mt19937_64 streams as TF; the weights use
// nn's generator, so results match TF statistically, not digit for digit.
// The vsim videos are replaced by PlayBestOfSeeds (same episodes, printed).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <datamunge/nn/nn.hpp>
#include <numeric>
#include <random>
#include <vector>

#include "common/checks.hpp"
#include "common/rl/classic_control.hpp"
#include "common/rl/pg_common.hpp"
#include "common/rl/play_best.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

struct Result {
  float eval, best, grad_mean, grad_std;
};

Result Train(bool baseline, const rl::PgConfig& cfg) {
  std::printf("\n== REINFORCE %s on CartPole, %d steps\n",
              baseline ? "with a learned V(s) baseline" : "without baseline", cfg.steps);
  rl::CartPole env, eval_env;
  const int obs_dim = env.obs_size(), A = env.action_space().n;

  nn::ParameterStore params(cfg.seed);
  auto net = [&](const Tensor& obs) { return rl::BuildActorCritic(params, obs, A, cfg.hidden, /*with_value=*/baseline); };
  {
    nn::NoGradGuard no_grad;
    net(nn::zeros({1, obs_dim}));
  }
  rl::Adam adam(params, cfg.lr, cfg.clip_norm);
  // The actor's parameters, for the norm of the policy gradient only.
  std::vector<Tensor> actor_vars;
  for (const std::string& n : params.names())
    if (n.find("actor") != std::string::npos) actor_vars.push_back(params.at(n));

  rl::Monitor monitor(rl::MonitorConfig(cfg), "entropy");
  std::mt19937_64 rng(cfg.seed);
  auto explain = [&](const rl::Obs& o) {  // pi(.|s) and, with a baseline, V(s)
    nn::NoGradGuard no_grad;
    const rl::ActorCritic ac = net(rl::Row(o));
    return baseline ? rl::GreedyPi(ac.probs.to_vector<float>(), ac.value.item<float>())
                    : rl::GreedyPi(ac.probs.to_vector<float>());
  };
  auto probs_of = [&](const rl::Obs& o) {
    nn::NoGradGuard no_grad;
    return net(rl::Row(o)).probs.to_vector<float>();
  };
  auto greedy = [&](const rl::Obs& o) { return rl::ArgMax(probs_of(o)); };

  std::vector<rl::Obs> ep_obs;
  std::vector<int> ep_act;
  std::vector<float> ep_rew, ep_done, norms;
  int step = 0, next_log = cfg.log_every;
  float ent = 0, best = 0;
  while (step < cfg.steps) {
    ep_obs.clear(), ep_act.clear(), ep_rew.clear(), ep_done.clear();
    rl::Obs o = env.reset();
    while (true) {
      const int a = rl::SampleAction(probs_of(o).data(), A, rng);
      rl::StepResult sr = env.step(a);
      ep_obs.push_back(o), ep_act.push_back(a), ep_rew.push_back(sr.reward), ep_done.push_back(sr.terminated ? 1.f : 0.f);
      o = sr.obs;
      ++step;
      if (sr.done()) break;
    }
    monitor.EndEpisode(std::accumulate(ep_rew.begin(), ep_rew.end(), 0.f));
    const std::vector<float> G = rl::DiscountedReturns(ep_rew, ep_done, cfg.gamma);
    const Tensor ret = rl::FloatVector(G);
    const rl::ActorCritic ac = net(rl::ObsMatrix(ep_obs));
    const Tensor logp_a = rl::LogProbOf(ac.logp, rl::IntVector(ep_act), A);
    // The baseline is a constant for the policy term and is itself
    // regressed toward the returns.
    const Tensor advantage = baseline ? ret - nn::stop_gradient(ac.value) : ret;
    const Tensor policy_loss = -nn::mean(logp_a * advantage, {0});
    const Tensor loss =
        baseline ? policy_loss + cfg.value_coef * nn::mean(nn::square(ret - ac.value), {0}) : policy_loss;
    nn::GradOptions keep;
    keep.retain_graph = true;
    norms.push_back(static_cast<float>(nn::global_norm(nn::grad({policy_loss}, actor_vars, {}, keep))));
    adam.Minimize(loss);
    monitor.AddLoss(loss.item<float>());
    ent = nn::mean(ac.entropy.detach(), {0}).item<float>();
    check::Require(std::isfinite(loss.item<float>()) && std::isfinite(norms.back()) && ent >= 0 &&
                       ent <= std::log(2.0) + 1e-5,
                   "finite loss and policy-gradient norm, entropy in [0, log 2]");
    if (step >= next_log) {
      const float ev = next_log % cfg.eval_every == 0 ? rl::Evaluate(eval_env, greedy, cfg.eval_episodes, 1000 + step) : NAN;
      if (!std::isnan(ev)) best = std::max(best, ev);
      monitor.Log(step, ent, ev);
      next_log += cfg.log_every;
    }
  }
  Result res{rl::Evaluate(eval_env, greedy, 20, 777), 0, 0, 0};
  res.best = std::max(best, res.eval);
  rl::PlayBestOfSeeds(eval_env, explain, baseline ? "reinforce_baseline_cartpole" : "reinforce_no_baseline_cartpole", 4242);
  double mean = 0, sumsq = 0;
  // Only the second half, when both policies are competent and returns are large.
  for (size_t i = norms.size() / 2; i < norms.size(); ++i) mean += norms[i], sumsq += double(norms[i]) * norms[i];
  const double n = norms.size() - norms.size() / 2;
  res.grad_mean = static_cast<float>(mean / n);
  res.grad_std = static_cast<float>(std::sqrt(std::max(0.0, sumsq / n - (mean / n) * (mean / n))));
  std::printf("  final greedy evaluation over 20 episodes: %.1f; policy-gradient norm (2nd half): mean %.2f, std %.2f\n",
              res.eval, res.grad_mean, res.grad_std);
  return res;
}

}  // namespace

int main(int argc, char** argv) {
  rl::PgConfig cfg;
  cfg.steps = 100000;
  cfg.log_every = 20000;
  cfg.eval_every = 20000;
  rl::ParsePgConfig(argc, argv, &cfg);
  const Result without = Train(false, cfg);
  const Result with = Train(true, cfg);
  std::printf("\n  policy-gradient norm std: without baseline %.2f, with baseline %.2f\n", without.grad_std, with.grad_std);
  // tflow has no checks here; these follow the example's claim (see the report for seed sweeps).
  check::TrainingQuality(with.grad_std < without.grad_std, "the V(s) baseline shrinks the spread of the policy-gradient norm");
  check::TrainingQuality(with.best >= 150, "REINFORCE with a baseline learned to balance (best greedy evaluation >= 150)");
  return 0;
}
