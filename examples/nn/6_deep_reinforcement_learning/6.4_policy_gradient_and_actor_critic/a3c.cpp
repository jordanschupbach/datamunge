// CMD: just run-nn a3c
// Asynchronous advantage actor-critic (A3C), workers interleaved deterministically.
//
// Port of tflow's a3c. Mnih et al. 2016: several workers each run their own
// environment, collect a short rollout with the shared network, compute the
// actor-critic gradient and apply it straight to the shared parameters
// without any locking ("Hogwild"). The decorrelation across workers plays
// the role of a replay buffer.
//
// Deviation: tflow runs the workers as real threads sharing one
// tf::Session, so its run is not reproducible (the order in which workers
// read and update the shared weights depends on the scheduler, and the Adam
// updates race). This port keeps the algorithm's semantics but schedules the
// workers round-robin on one thread: worker 0 collects its rollout (up to
// n_steps, or to the end of its episode) with the current shared weights and
// applies its gradient, then worker 1, and so on. Each worker still owns its
// environment, its sampling stream (seed * 100 + id, as in TF) and its
// partial episode, and every update uses only that worker's rollout, so the
// result is A3C with zero-delay updates, and a given seed gives the same run
// every time. The main thread's logging/evaluation becomes a check after each
// update. The weights use nn's generator, so results match TF statistically.
// The vsim video is replaced by PlayBestOfSeeds (same episode, printed).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <datamunge/nn/nn.hpp>
#include <random>
#include <vector>

#include "common/checks.hpp"
#include "common/rl/classic_control.hpp"
#include "common/rl/pg_common.hpp"
#include "common/rl/play_best.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

// One A3C worker: its own environment, sampling stream and partial episode.
struct Worker {
  rl::CartPole env;
  std::mt19937_64 rng;
  rl::Obs o;
  float ep_ret = 0;
  Worker(uint64_t seed) : rng(seed) { o = env.reset(seed); }
};

}  // namespace

int main(int argc, char** argv) {
  rl::PgConfig cfg;
  cfg.n_envs = 4;  // reused as the number of workers
  cfg.lr = 5e-4f;
  rl::ParsePgConfig(argc, argv, &cfg);
  const int workers = cfg.n_envs;
  rl::CartPole eval_env;
  const int obs_dim = eval_env.obs_size(), A = eval_env.action_space().n;
  std::printf("== A3C on CartPole: %d workers (interleaved), %d-step rollouts, %d steps total\n", workers, cfg.n_steps,
              cfg.steps);

  nn::ParameterStore params(cfg.seed);  // the shared network
  auto net = [&](const Tensor& obs) { return rl::BuildActorCritic(params, obs, A, cfg.hidden); };
  {
    nn::NoGradGuard no_grad;
    net(nn::zeros({1, obs_dim}));
  }
  rl::Adam adam(params, cfg.lr, cfg.clip_norm);

  rl::Monitor monitor(rl::MonitorConfig(cfg), "entropy");
  auto explain = [&](const rl::Obs& o) {  // pi(.|s) and V(s): the most probable action, for evaluation and the replay
    nn::NoGradGuard no_grad;
    const rl::ActorCritic ac = net(rl::Row(o));
    return rl::GreedyPi(ac.probs.to_vector<float>(), ac.value.item<float>());
  };
  auto greedy = [&](const rl::Obs& o) { return explain(o).action.discrete; };

  std::vector<Worker> pool;
  pool.reserve(workers);
  for (int i = 0; i < workers; ++i) pool.emplace_back(cfg.seed * 100 + i);

  // One worker iteration: a rollout with the shared weights, then its update.
  long total_steps = 0;
  float last_entropy = 0;
  std::vector<rl::Obs> ro_obs;
  std::vector<int> ro_act;
  std::vector<float> ro_rew, ro_done, ro_val;
  auto run_worker = [&](Worker& w) {
    ro_obs.clear(), ro_act.clear(), ro_rew.clear(), ro_done.clear(), ro_val.clear();
    bool ended = false;
    for (int t = 0; t < cfg.n_steps && !ended; ++t) {
      std::vector<float> p;
      float v;
      {
        nn::NoGradGuard no_grad;
        const rl::ActorCritic ac = net(rl::Row(w.o));
        p = ac.probs.to_vector<float>();
        v = ac.value.item<float>();
      }
      const int a = rl::SampleAction(p.data(), A, w.rng);
      rl::StepResult sr = w.env.step(a);
      ro_obs.push_back(w.o), ro_act.push_back(a), ro_rew.push_back(sr.reward);
      ro_done.push_back(sr.terminated ? 1.f : 0.f), ro_val.push_back(v);
      w.ep_ret += sr.reward;
      w.o = sr.obs;
      ++total_steps;
      if (sr.done()) {
        monitor.EndEpisode(w.ep_ret);
        w.ep_ret = 0;
        w.o = w.env.reset();
        ended = true;
      }
    }
    // Bootstrap from the current state unless the episode just ended.
    float bootstrap = 0;
    if (!ended) {
      nn::NoGradGuard no_grad;
      bootstrap = net(rl::Row(w.o)).value.item<float>();
    }
    const std::vector<float> R = rl::DiscountedReturns(ro_rew, ro_done, cfg.gamma, bootstrap);
    std::vector<float> advantages(R.size());
    for (size_t t = 0; t < R.size(); ++t) advantages[t] = R[t] - ro_val[t];
    // The worker's gradient, applied to the shared parameters.
    const rl::ActorCritic ac = net(rl::ObsMatrix(ro_obs));
    const Tensor logp_a = rl::LogProbOf(ac.logp, rl::IntVector(ro_act), A);
    const Tensor loss = -nn::mean(logp_a * rl::FloatVector(advantages), {0}) +
                        cfg.value_coef * nn::mean(nn::square(rl::FloatVector(R) - ac.value), {0}) -
                        cfg.entropy_coef * nn::mean(ac.entropy, {0});
    adam.Minimize(loss);
    check::Require(std::isfinite(loss.item<float>()), "finite loss");
    monitor.AddLoss(loss.item<float>());
    last_entropy = ac.entropy.to_vector<float>()[0];
  };

  long next_log = cfg.log_every;
  float best_eval = 0;
  while (total_steps < cfg.steps) {
    for (Worker& w : pool) {
      run_worker(w);
      if (total_steps >= next_log) {  // TF's main thread logs as soon as the counter passes the mark
        const float ev =
            next_log % cfg.eval_every == 0 ? rl::Evaluate(eval_env, greedy, cfg.eval_episodes, 1000 + next_log) : NAN;
        if (!std::isnan(ev)) best_eval = std::max(best_eval, ev);
        monitor.Log(static_cast<int>(next_log), last_entropy, ev);
        next_log += cfg.log_every;
      }
    }
  }
  const float final_eval = rl::Evaluate(eval_env, greedy, 20, 777);
  std::printf("  final greedy evaluation over 20 episodes: %.1f\n", final_eval);
  rl::PlayBestOfSeeds(eval_env, explain, "a3c_cartpole", 4242);
  // A random policy averages about 22. Over seeds 0..4 the best greedy evaluation
  // averages 342 for TF's threaded run (final 148), 443 here (final 196).
  check::TrainingQuality(std::max(best_eval, final_eval) >= 100, "A3C learned (best greedy evaluation >= 100)");
  return 0;
}
