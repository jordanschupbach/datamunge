// CMD: just run-nn impala
// IMPALA with V-trace, actors and learner interleaved deterministically.
//
// Port of tflow's impala. Espeholt et al. 2018 decouple acting from learning:
// actors run their own (slightly stale) copies of the policy and stream whole
// trajectories to a learner, which trains the central network. Because the
// data came from an older policy mu, the learner corrects the value target
// with V-trace, truncated importance ratios rho_t = min(1, pi/mu):
//   v_s = V(s) + sum_t gamma^(t-s) (prod_{i<t} c_i) rho_t delta_t,
//   delta_t = r_t + gamma V(s_{t+1}) - V(s_t),
// and the policy gradient uses rho_s (r_s + gamma v_{s+1} - V(s_s)).
// Actors refresh their parameters every few learner updates.
//
// Deviation: tflow runs the actors as real threads that push trajectories
// into a queue (dropping them when more than 64 wait) while the main thread
// learns, so how stale the actors' weights are, and which trajectories are
// used, depends on the scheduler: no two runs agree. This port keeps every
// ingredient but fixes the schedule on one thread: in each round actors
// 0..n-1 each produce one trajectory of T steps with their own weight copy,
// then the learner takes batches of 8 trajectories from the FIFO queue
// (none are dropped) and updates, and every 4th update copies the learner's
// weights to all actors. The policy lag (up to 4 updates) and hence the
// off-policy correction remain; a given seed gives the same run every time.
// Each actor keeps its own CartPole, sampling stream (seed * 100 + k, as in
// TF) and partial episode. The weights use nn's generator, so results match
// TF statistically. The vsim video is replaced by PlayBestOfSeeds.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <datamunge/nn/nn.hpp>
#include <deque>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "common/checks.hpp"
#include "common/rl/classic_control.hpp"
#include "common/rl/pg_common.hpp"
#include "common/rl/play_best.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

struct Trajectory {
  std::vector<rl::Obs> obs;  // T + 1 (last one for bootstrapping)
  std::vector<int> act;
  std::vector<float> rew, done, logp_mu;
};

// V-trace targets vs[0..T] and policy-gradient advantages pg_adv[0..T-1] for
// one trajectory, given the learner's log pi(a_t|s_t), its values V(s_0..s_T)
// and the behaviour log-probs; c_bar = rho_bar = 1 as in tflow.
void VTrace(const float* logp_pi, const float* values, const Trajectory& tr, float gamma, std::vector<float>* vs,
            std::vector<float>* pg_adv) {
  const int T = static_cast<int>(tr.act.size());
  vs->assign(T + 1, 0.f);
  pg_adv->assign(T, 0.f);
  (*vs)[T] = values[T];
  for (int t = T - 1; t >= 0; --t) {
    const float rho = std::min(1.f, std::exp(logp_pi[t] - tr.logp_mu[t]));
    const float c = rho;
    const float cont = 1 - tr.done[t];
    const float delta = rho * (tr.rew[t] + gamma * cont * values[t + 1] - values[t]);
    (*vs)[t] = values[t] + delta + gamma * cont * c * ((*vs)[t + 1] - values[t + 1]);
  }
  for (int t = 0; t < T; ++t) {
    const float rho = std::min(1.f, std::exp(logp_pi[t] - tr.logp_mu[t]));
    const float cont = 1 - tr.done[t];
    (*pg_adv)[t] = rho * (tr.rew[t] + gamma * cont * (*vs)[t + 1] - values[t]);
  }
}

// One actor: its own environment, sampling stream, partial episode and weights.
struct Actor {
  rl::CartPole env;
  std::mt19937_64 rng;
  rl::Obs o;
  float ep_ret = 0;
  nn::ParameterStore params;
  Actor(uint64_t stream, uint64_t init_seed) : rng(stream), params(init_seed) { o = env.reset(stream); }
};

}  // namespace

int main(int argc, char** argv) {
  rl::PgConfig cfg;
  cfg.n_envs = 4;    // actors
  cfg.n_steps = 20;  // trajectory length
  cfg.lr = 5e-4f;
  rl::ParsePgConfig(argc, argv, &cfg);
  const int actors = cfg.n_envs, T = cfg.n_steps, batch_trajectories = 8, sync_every = 4;
  rl::CartPole eval_env;
  const int obs_dim = eval_env.obs_size(), A = eval_env.action_space().n;
  std::printf("== IMPALA on CartPole: %d actors (interleaved), trajectories of %d, learner batches of %d trajectories, "
              "actors resync every %d updates, %d steps total\n", actors, T, batch_trajectories, sync_every, cfg.steps);

  // V-trace on a hand-made trajectory: on-policy (pi = mu) it is the n-step
  // bootstrapped return; with pi far below mu (rho = c ~ 0) it keeps V.
  {
    Trajectory tr;
    tr.act = {0, 1, 0};
    tr.rew = {1, 2, 3};
    tr.done = {0, 1, 0};
    tr.logp_mu = {-0.7f, -0.7f, -0.7f};
    const float values[4] = {0.5f, 1.f, -1.f, 4.f}, on[3] = {-0.7f, -0.7f, -0.7f}, off[3] = {-40, -40, -40};
    std::vector<float> vs, adv;
    VTrace(on, values, tr, 0.9f, &vs, &adv);
    const std::vector<float> R = rl::DiscountedReturns(tr.rew, tr.done, 0.9f, values[3]);
    for (int t = 0; t < 3; ++t) check::Near(vs[t], R[t], 1e-5, "on-policy V-trace target = n-step return");
    check::Near(adv[0], 1 + 0.9 * vs[1] - values[0], 1e-5, "on-policy V-trace advantage = r + gamma v_{s+1} - V(s)");
    VTrace(off, values, tr, 0.9f, &vs, &adv);
    for (int t = 0; t < 3; ++t) check::Near(vs[t], values[t], 1e-5, "with rho = c = 0 the V-trace target stays V(s)");
  }

  nn::ParameterStore learner(cfg.seed);
  auto net = [&](nn::ParameterStore& p, const Tensor& obs, bool with_value) {
    return rl::BuildActorCritic(p, obs, A, cfg.hidden, with_value);
  };
  {
    nn::NoGradGuard no_grad;
    net(learner, nn::zeros({1, obs_dim}), true);
  }
  rl::Adam adam(learner, cfg.lr, cfg.clip_norm);
  std::vector<std::unique_ptr<Actor>> pool;
  auto sync = [&]() {
    for (auto& a : pool)
      for (const std::string& n : a->params.names()) a->params.set(n, learner.at(n));
  };
  for (int k = 0; k < actors; ++k) {
    pool.push_back(std::make_unique<Actor>(cfg.seed * 100 + k, cfg.seed + 10 + k));
    nn::NoGradGuard no_grad;
    net(pool.back()->params, nn::zeros({1, obs_dim}), false);  // policy only
  }
  sync();

  rl::Monitor monitor(rl::MonitorConfig(cfg), "mean rho");
  long env_steps = 0;
  auto act_once = [&](Actor& w) {
    Trajectory tr;
    for (int t = 0; t < T; ++t) {
      std::vector<float> p;
      {
        nn::NoGradGuard no_grad;
        p = net(w.params, rl::Row(w.o), false).probs.to_vector<float>();
      }
      const int a = rl::SampleAction(p.data(), A, w.rng);
      rl::StepResult sr = w.env.step(a);
      tr.obs.push_back(w.o), tr.act.push_back(a), tr.rew.push_back(sr.reward);
      tr.done.push_back(sr.terminated ? 1.f : 0.f), tr.logp_mu.push_back(std::log(std::max(p[a], 1e-8f)));
      w.ep_ret += sr.reward;
      w.o = sr.obs;
      ++env_steps;
      if (sr.done()) {
        monitor.EndEpisode(w.ep_ret);
        w.ep_ret = 0;
        // A terminal step inside the trajectory is handled by `done`; the
        // observation after it is the new episode's first state.
        w.o = w.env.reset();
      }
    }
    tr.obs.push_back(w.o);
    return tr;
  };

  auto explain = [&](const rl::Obs& o) {  // pi(.|s) and V(s) from the learner's network
    nn::NoGradGuard no_grad;
    const rl::ActorCritic ac = net(learner, rl::Row(o), true);
    return rl::GreedyPi(ac.probs.to_vector<float>(), ac.value.item<float>());
  };
  auto greedy = [&](const rl::Obs& o) { return explain(o).action.discrete; };
  std::deque<Trajectory> queue;
  int updates = 0;
  long next_log = cfg.log_every;
  double rho_sum = 0;
  long rho_count = 0;
  float best_eval = 0;
  while (env_steps < cfg.steps) {
    for (auto& w : pool) queue.push_back(act_once(*w));
    while (queue.size() >= static_cast<size_t>(batch_trajectories)) {
      std::vector<Trajectory> batch;
      for (int i = 0; i < batch_trajectories; ++i) {
        batch.push_back(std::move(queue.front()));
        queue.pop_front();
      }
      // Current policy's log-probs and values on the batch (T + 1 rows per trajectory).
      std::vector<rl::Obs> all_obs;
      std::vector<int> all_act;
      for (const Trajectory& tr : batch) {
        all_obs.insert(all_obs.end(), tr.obs.begin(), tr.obs.end());
        all_act.insert(all_act.end(), tr.act.begin(), tr.act.end());
        all_act.push_back(0);  // the bootstrap row has no action
      }
      std::vector<float> logp_pi, values;
      {
        nn::NoGradGuard no_grad;
        const rl::ActorCritic ac = net(learner, rl::ObsMatrix(all_obs), true);
        logp_pi = rl::LogProbOf(ac.logp, rl::IntVector(all_act), A).to_vector<float>();
        values = ac.value.to_vector<float>();
      }
      std::vector<rl::Obs> train_obs;
      std::vector<int> train_act;
      std::vector<float> pg_adv, vs_all, vs, adv;
      for (size_t b = 0; b < batch.size(); ++b) {
        const Trajectory& tr = batch[b];
        const int base = static_cast<int>(b) * (T + 1);
        VTrace(&logp_pi[base], &values[base], tr, cfg.gamma, &vs, &adv);
        for (int t = 0; t < T; ++t) {
          rho_sum += std::exp(logp_pi[base + t] - tr.logp_mu[t]);
          ++rho_count;
          train_obs.push_back(tr.obs[t]);
          train_act.push_back(tr.act[t]);
          pg_adv.push_back(adv[t]);
          vs_all.push_back(vs[t]);
        }
      }
      const rl::ActorCritic ac = net(learner, rl::ObsMatrix(train_obs), true);
      const Tensor logp_a = rl::LogProbOf(ac.logp, rl::IntVector(train_act), A);
      const Tensor loss = -nn::mean(logp_a * rl::FloatVector(pg_adv), {0}) +
                          cfg.value_coef * nn::mean(nn::square(rl::FloatVector(vs_all) - ac.value), {0}) -
                          cfg.entropy_coef * nn::mean(ac.entropy, {0});
      adam.Minimize(loss);
      check::Require(std::isfinite(loss.item<float>()), "finite loss");
      monitor.AddLoss(loss.item<float>());
      if (++updates % sync_every == 0) sync();
      if (env_steps >= next_log) {
        const float ev =
            next_log % cfg.eval_every == 0 ? rl::Evaluate(eval_env, greedy, cfg.eval_episodes, 1000 + next_log) : NAN;
        if (!std::isnan(ev)) best_eval = std::max(best_eval, ev);
        const double mean_rho = rho_sum / std::max(1L, rho_count);
        check::Require(std::isfinite(mean_rho) && std::abs(mean_rho - 1) < 0.1,
                       "mean importance ratio pi/mu stays near 1 (E_mu[pi/mu] = 1)");
        monitor.Log(static_cast<int>(next_log), static_cast<float>(mean_rho), ev);
        rho_sum = 0, rho_count = 0;
        next_log += cfg.log_every;
      }
    }
  }
  std::printf("  (the 'mean rho' column is the average untruncated importance ratio pi/mu: 1 would be on-policy)\n");
  std::printf("  %d learner updates\n", updates);
  const float final_eval = rl::Evaluate(eval_env, greedy, 20, 777);
  std::printf("  final greedy evaluation over 20 episodes: %.1f\n", final_eval);
  rl::PlayBestOfSeeds(eval_env, explain, "impala_cartpole", 4242);
  // tflow has no checks. TF seed 0: best 214.0, final 156.7; random play ~22.
  check::TrainingQuality(std::max(best_eval, final_eval) >= 100, "IMPALA learned (best greedy evaluation >= 100)");
  return 0;
}
