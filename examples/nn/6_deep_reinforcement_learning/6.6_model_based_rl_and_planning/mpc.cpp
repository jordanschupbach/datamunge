// CMD: just run-nn mpc
// Model-predictive control with a learned model.
//
// Port of tflow's mpc.
// No policy at all: a dynamics model is fit to data, and at every step the
// controller searches for the action sequence whose predicted return over a
// short horizon is highest, executes its first action, and re-plans. Two
// optimisers are compared: random shooting (sample N sequences, keep the
// best) and the cross-entropy method (iteratively refit a Gaussian to the
// elite sequences). Data collected by the controller is added to the
// training set so the model improves where it is used (Nagabandi et al. 2018).
//
// The model is common/rl/dynamics_model.hpp (eager port of tflow's
// dynamics_model.h, one member). Pendulum, the planners and every random draw
// use the same std::mt19937_64 stream as TF; weights use nn's generator, so
// returns match TF statistically. The vsim video of CEM is replaced by
// PlayEpisode (same episode, printed, no video). Options: threads=N, seed=0
// (model weights and random stream; TF fixes 0). The port
// adds checks (exit 1 on failure).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <datamunge/nn/nn.hpp>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "common/checks.hpp"
#include "common/rl/classic_control.hpp"
#include "common/rl/continuous.hpp"
#include "common/rl/dynamics_model.hpp"
#include "common/rl/hud.hpp"
#include "common/rl/playback.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

struct Planner {
  rl::DynamicsEnsemble* model;
  int horizon, candidates, cem_iters, elites;
  int act_dim;
  std::mt19937_64* rng;
  float last_return = 0;  // the model's return of the best sequence of the last Plan()
  // Returns the first action of the best sequence found from `obs`.
  std::vector<float> Plan(const rl::Obs& obs, bool use_cem) {
    const int H = horizon, N = candidates;
    std::vector<float> mean(H * act_dim, 0.f), stdv(H * act_dim, 1.f);
    std::vector<float> best_seq(H * act_dim, 0.f);
    float best_return = -1e30f;
    std::normal_distribution<float> normal(0.f, 1.f);
    for (int it = 0; it < (use_cem ? cem_iters : 1); ++it) {
      // Sample N action sequences (uniform for shooting, Gaussian for CEM).
      std::vector<std::vector<float>> seqs(N, std::vector<float>(H * act_dim));
      for (auto& s : seqs)
        for (int j = 0; j < H * act_dim; ++j)
          s[j] = std::max(-1.f, std::min(1.f, use_cem ? mean[j] + stdv[j] * normal(*rng) : std::uniform_real_distribution<float>(-1, 1)(*rng)));
      // Roll all candidates through the model together.
      const int obs_dim = static_cast<int>(obs.size());
      std::vector<float> s0(static_cast<size_t>(N) * obs_dim);
      for (int i = 0; i < N; ++i)
        for (int j = 0; j < obs_dim; ++j) s0[i * obs_dim + j] = obs[j];
      Tensor S = nn::from_vector(s0, {N, obs_dim});
      std::vector<float> returns(N, 0.f);
      for (int t = 0; t < H; ++t) {
        std::vector<float> a(static_cast<size_t>(N) * act_dim);
        for (int i = 0; i < N; ++i)
          for (int d = 0; d < act_dim; ++d) a[i * act_dim + d] = seqs[i][t * act_dim + d] * 2.f;  // Pendulum torque range
        rl::DynamicsEnsemble::Prediction p = model->Predict(0, S, nn::from_vector(a, {N, act_dim}), false, *rng);
        const std::vector<float> r = p.r.to_vector<float>();
        for (int i = 0; i < N; ++i) returns[i] += r[i];
        S = p.next_s;
      }
      // Elites refit the sampling distribution; remember the best sequence.
      std::vector<int> order(N);
      for (int i = 0; i < N; ++i) order[i] = i;
      std::partial_sort(order.begin(), order.begin() + elites, order.end(), [&](int x, int y) { return returns[x] > returns[y]; });
      if (returns[order[0]] > best_return) best_return = returns[order[0]], best_seq = seqs[order[0]];
      for (int j = 0; j < H * act_dim; ++j) {
        float m = 0, v = 0;
        for (int k = 0; k < elites; ++k) m += seqs[order[k]][j] / elites;
        for (int k = 0; k < elites; ++k) v += (seqs[order[k]][j] - m) * (seqs[order[k]][j] - m) / elites;
        mean[j] = m;
        stdv[j] = std::sqrt(v) + 0.05f;
      }
    }
    last_return = best_return;
    return std::vector<float>(best_seq.begin(), best_seq.begin() + act_dim);
  }
};

}  // namespace

int main(int argc, char** argv) {
  try {
  uint64_t seed = 0;  // model weights and the planner's random stream (TF: 0)
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg.rfind("threads=", 0) == 0) nn::set_num_threads(std::atoi(arg.c_str() + 8));
    if (arg.rfind("seed=", 0) == 0) seed = std::strtoull(arg.c_str() + 5, nullptr, 10);
  }
  rl::Pendulum env, eval_env;
  const rl::ActionSpace space = env.action_space();
  const int obs_dim = env.obs_size(), act_dim = space.dim;
  const int iterations = example_runtime::Limit(6, 2), model_steps = example_runtime::Limit(400, 8), batch = example_runtime::Limit(128, 8);
  std::printf("== MPC on Pendulum with a learned model: %d rounds of (collect one episode, refit the model)\n", iterations);

  rl::DynamicsEnsemble model(/*members=*/1, obs_dim, act_dim, 128, 1e-3f, seed);

  std::mt19937_64 rng(seed);
  rl::TransitionSet data;
  Planner shooting{&model, 15, 500, 1, 50, act_dim, &rng};
  Planner cem{&model, 15, 200, 4, 20, act_dim, &rng};
  // Environment actions are stored in the model's units (the real torque).
  auto run_episode = [&](const std::function<std::vector<float>(const rl::Obs&)>& policy, bool record, uint64_t seed) {
    rl::Obs obs = env.reset(seed);
    float ret = 0;
    while (true) {
      std::vector<float> u = policy(obs);
      rl::Action a = rl::ScaleAction(u, space);
      rl::StepResult sr = env.step(a);
      if (record) data.Add(obs, a.continuous, sr.obs, sr.reward);
      ret += sr.reward;
      obs = sr.obs;
      if (sr.done()) break;
    }
    return ret;
  };
  float first_mse = -1;
  auto train_model = [&]() {
    float mse = 0;
    for (int i = 0; i < model_steps; ++i) {
      Tensor S, A, S2, R;
      data.Sample(batch, rng, &S, &A, &S2, &R);
      mse = model.Train(S, A, S2, R).second;
      if (first_mse < 0) first_mse = mse;
    }
    return mse;
  };

  std::printf("  %5s %10s %12s %16s %12s\n", "round", "data", "model MSE", "random shooting", "CEM");
  auto random_policy = [&](const rl::Obs&) { return std::vector<float>{std::uniform_real_distribution<float>(-1, 1)(rng)}; };
  run_episode(random_policy, true, 1);
  run_episode(random_policy, true, 2);
  float last_mse = 0, best_shoot = -1e30f, best_cem = -1e30f;
  for (int round = 1; round <= iterations; ++round) {
    const float mse = train_model();
    const float r_shoot = run_episode([&](const rl::Obs& o) { return shooting.Plan(o, false); }, /*record=*/true, 100 + round);
    const float r_cem = run_episode([&](const rl::Obs& o) { return cem.Plan(o, true); }, /*record=*/true, 200 + round);
    std::printf("  %5d %10d %12.4f %16.1f %12.1f\n", round, data.size(), mse, r_shoot, r_cem);
    std::fflush(stdout);
    last_mse = mse, best_shoot = std::max(best_shoot, r_shoot), best_cem = std::max(best_cem, r_cem);
  }
  std::printf("  (random policy about -1200; a good controller is above -200)\n");
  // On film in TF: CEM re-plans at every step; the overlay shows the torque and
  // the model's return for the plan it was taken from.
  const rl::Playback pb = rl::PlayEpisode(env, [&](const rl::Obs& o) {
    const std::vector<float> u = cem.Plan(o, true);
    return rl::Explain(rl::ScaleAction(u, space), cem.last_return, "planned return (15 steps)", "CEM: 200 candidates, 4 iterations, learned model");
  }, "mpc_cem_pendulum", 777);

  // Checks added by the port (tflow prints the table only).
  check::Require(std::isfinite(last_mse) && std::isfinite(first_mse), "finite model fitting losses in both runtime profiles");
  check::TrainingQuality(last_mse < first_mse, "the model's error falls as it is refit");
  check::TrainingQuality(best_cem > -400, "CEM with the learned model swings the pendulum up in some round (return > -400)");
  check::TrainingQuality(best_shoot > -600, "random shooting does well in some round (return > -600)");
  check::TrainingQuality(pb.ret > -600, "the CEM episode played at the end does well (return > -600)");
  std::printf("\nchecks passed (vsim video omitted)\n");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "CHECK FAILED: %s\n", e.what());
    return 1;
  }
  return 0;
}
