// CMD: just run-nn pets
// PETS: probabilistic ensembles with trajectory sampling.
//
// Port of tflow's pets.
// Chua et al. 2018 make model-predictive control robust to model error by
// (1) using an ensemble of probabilistic networks, which captures both the
// noise in the dynamics (the predicted variance) and the model's own
// uncertainty (disagreement between members), and (2) planning over
// *particles*: each candidate action sequence is rolled out several times,
// each particle following a randomly chosen member and sampling from its
// Gaussian (TS1), and the candidate is scored by its mean return. CEM picks
// the sequence; data collected is added to the model's training set.
//
// The ensemble is common/rl/dynamics_model.hpp (eager port of tflow's
// dynamics_model.h). TF predicts every row with every member and keeps the
// rows assigned to that member; the port evaluates each member only on its own
// rows (5x less work) but draws the same Gaussian noise for all rows, so the
// random stream is the same as TF's. Pendulum, CEM and every random draw use
// the same std::mt19937_64 as TF; weights use nn's generator, so returns match
// TF statistically. The vsim video is replaced by PlayEpisode (same episode,
// printed, no video). Options: threads=N, seed=0 (model weights and random
// stream; TF fixes 0). The port adds checks (exit 1 on failure).

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

int main(int argc, char** argv) {
  try {
  uint64_t seed = 0;  // model weights and the planner's random stream (TF: 0)
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg.rfind("threads=", 0) == 0) nn::set_num_threads(std::atoi(arg.c_str() + 8));
    if (arg.rfind("seed=", 0) == 0) seed = std::strtoull(arg.c_str() + 5, nullptr, 10);
  }
  rl::Pendulum env;
  const rl::ActionSpace space = env.action_space();
  const int obs_dim = env.obs_size(), act_dim = space.dim;
  const int members = 5, particles = example_runtime::Limit(10, 2), candidates = example_runtime::Limit(100, 16),
            elites = example_runtime::Limit(10, 4), horizon = example_runtime::Limit(15, 5),
            cem_iters = example_runtime::Limit(4, 2);
  const int iterations = example_runtime::Limit(6, 2), model_steps = example_runtime::Limit(400, 8), batch = example_runtime::Limit(128, 8);
  std::printf("== PETS on Pendulum: %d-member ensemble, %d particles per candidate, CEM with %d candidates over %d steps\n",
              members, particles, candidates, horizon);

  rl::DynamicsEnsemble model(members, obs_dim, act_dim, 128, 1e-3f, seed);
  std::mt19937_64 rng(seed);
  rl::TransitionSet data;

  // CEM with trajectory sampling. Rows are (candidate, particle) pairs; at
  // every step each row is advanced by a random ensemble member.
  float last_plan_return = 0;  // the ensemble's return of the best sequence of the last plan()
  auto plan = [&](const rl::Obs& obs, bool trajectory_sampling) {
    const int H = horizon, N = candidates, P = trajectory_sampling ? particles : 1;
    std::vector<float> mean(H * act_dim, 0.f), stdv(H * act_dim, 1.f), best(act_dim, 0.f);
    float best_return = -1e30f;
    std::normal_distribution<float> normal(0.f, 1.f);
    for (int it = 0; it < cem_iters; ++it) {
      std::vector<std::vector<float>> seqs(N, std::vector<float>(H * act_dim));
      for (auto& s : seqs)
        for (int j = 0; j < H * act_dim; ++j) s[j] = std::max(-1.f, std::min(1.f, mean[j] + stdv[j] * normal(rng)));
      const int rows = N * P;
      std::vector<float> S(static_cast<size_t>(rows) * obs_dim);  // current state of every row
      for (int i = 0; i < rows; ++i)
        for (int j = 0; j < obs_dim; ++j) S[i * obs_dim + j] = obs[j];
      std::vector<float> returns(N, 0.f);
      for (int t = 0; t < H; ++t) {
        // Each row follows one randomly chosen member for this step (TS1).
        std::vector<int> member_of(rows);
        for (int& m : member_of) m = trajectory_sampling ? static_cast<int>(rng() % members) : 0;
        for (int m = 0; m < members; ++m) {
          // TF's Predict(m, all rows) draws [rows, obs + 1] standard normals when sampling.
          std::vector<float> eps;
          if (trajectory_sampling) {
            eps.resize(static_cast<size_t>(rows) * (obs_dim + 1));
            std::normal_distribution<float> noise(0.f, 1.f);
            for (float& x : eps) x = noise(rng);
          }
          std::vector<int> idx;
          for (int i = 0; i < rows; ++i)
            if (member_of[i] == m) idx.push_back(i);
          if (idx.empty()) continue;
          const int n = static_cast<int>(idx.size());
          std::vector<float> s(static_cast<size_t>(n) * obs_dim), a(static_cast<size_t>(n) * act_dim);
          for (int k = 0; k < n; ++k) {
            for (int j = 0; j < obs_dim; ++j) s[k * obs_dim + j] = S[idx[k] * obs_dim + j];
            for (int d = 0; d < act_dim; ++d) a[k * act_dim + d] = seqs[idx[k] / P][t * act_dim + d] * 2.f;
          }
          std::vector<float> out;
          {
            nn::NoGradGuard no_grad;
            const rl::DynamicsEnsemble::Output o = model.Forward(m, nn::from_vector(s, {n, obs_dim}), nn::from_vector(a, {n, act_dim}));
            Tensor y = o.mean;
            if (trajectory_sampling) {
              std::vector<float> e(static_cast<size_t>(n) * (obs_dim + 1));
              for (int k = 0; k < n; ++k)
                for (int j = 0; j <= obs_dim; ++j) e[k * (obs_dim + 1) + j] = eps[idx[k] * (obs_dim + 1) + j];
              y = o.mean + nn::exp(0.5 * o.logvar) * nn::from_vector(e, {n, obs_dim + 1});
            }
            out = y.to_vector<float>();
          }
          for (int k = 0; k < n; ++k) {
            for (int j = 0; j < obs_dim; ++j) S[idx[k] * obs_dim + j] += out[k * (obs_dim + 1) + j];
            returns[idx[k] / P] += out[k * (obs_dim + 1) + obs_dim] / P;
          }
        }
      }
      std::vector<int> order(N);
      for (int i = 0; i < N; ++i) order[i] = i;
      std::partial_sort(order.begin(), order.begin() + elites, order.end(), [&](int x, int y) { return returns[x] > returns[y]; });
      if (returns[order[0]] > best_return) {
        best_return = returns[order[0]];
        best.assign(seqs[order[0]].begin(), seqs[order[0]].begin() + act_dim);
      }
      for (int j = 0; j < H * act_dim; ++j) {
        float m = 0, v = 0;
        for (int k = 0; k < elites; ++k) m += seqs[order[k]][j] / elites;
        for (int k = 0; k < elites; ++k) v += (seqs[order[k]][j] - m) * (seqs[order[k]][j] - m) / elites;
        mean[j] = m, stdv[j] = std::sqrt(v) + 0.05f;
      }
    }
    last_plan_return = best_return;
    return best;
  };
  auto run_episode = [&](const std::function<std::vector<float>(const rl::Obs&)>& policy, uint64_t seed) {
    rl::Obs obs = env.reset(seed);
    float ret = 0;
    while (true) {
      rl::Action a = rl::ScaleAction(policy(obs), space);
      rl::StepResult sr = env.step(a);
      data.Add(obs, a.continuous, sr.obs, sr.reward);
      ret += sr.reward;
      obs = sr.obs;
      if (sr.done()) break;
    }
    return ret;
  };

  std::printf("  %5s %10s %12s %12s %20s %20s\n", "round", "data", "model NLL", "model MSE", "PETS (ensemble+TS)", "single det. model");
  run_episode([&](const rl::Obs&) { return std::vector<float>{std::uniform_real_distribution<float>(-1, 1)(rng)}; }, 1);
  run_episode([&](const rl::Obs&) { return std::vector<float>{std::uniform_real_distribution<float>(-1, 1)(rng)}; }, 2);
  float first_nll = 1e30f, last_nll = 0, best_pets = -1e30f;
  for (int round = 1; round <= iterations; ++round) {
    std::pair<float, float> stats{0, 0};
    for (int i = 0; i < model_steps; ++i) {
      Tensor S, A, S2, R;
      data.Sample(batch, rng, &S, &A, &S2, &R);
      stats = model.Train(S, A, S2, R);
      if (first_nll > 1e29f) first_nll = stats.first;
    }
    const float r_pets = run_episode([&](const rl::Obs& o) { return plan(o, true); }, 100 + round);
    const float r_single = run_episode([&](const rl::Obs& o) { return plan(o, false); }, 200 + round);
    std::printf("  %5d %10d %12.3f %12.4f %20.1f %20.1f\n", round, data.size(), stats.first, stats.second, r_pets, r_single);
    std::fflush(stdout);
    last_nll = stats.first, best_pets = std::max(best_pets, r_pets);
  }
  // On film in TF: the overlay shows the torque and the ensemble's return for
  // the plan it was taken from (averaged over the particles).
  const rl::Playback pb = rl::PlayEpisode(env, [&](const rl::Obs& o) {
    const std::vector<float> u = plan(o, true);
    return rl::Explain(rl::ScaleAction(u, space), last_plan_return, "planned return (15 steps)", "CEM, 5-member ensemble, 10 particles per candidate");
  }, "pets_pendulum", 777);
  // Ensemble disagreement as an uncertainty signal: far from the data it grows.
  const Tensor S = nn::from_vector(std::vector<float>{1, 0, 0, 1, 0, 8}, {2, obs_dim});  // upright at rest (seen often) vs. upright spinning at max speed (rare)
  const Tensor A = nn::zeros({2, act_dim});
  std::vector<std::vector<float>> preds(2);
  for (int m = 0; m < members; ++m) {
    const std::vector<float> next = model.Predict(m, S, A, false, rng).next_s.to_vector<float>();
    for (int i = 0; i < 2; ++i) preds[i].push_back(next[i * obs_dim + 2]);
  }
  float spread[2];
  for (int i = 0; i < 2; ++i) {
    float m = 0, v = 0;
    for (float x : preds[i]) m += x / members;
    for (float x : preds[i]) v += (x - m) * (x - m) / members;
    spread[i] = std::sqrt(v);
    std::printf("  predicted next angular velocity, %s: members' std %.4f\n", i == 0 ? "common state" : "rare state  ", std::sqrt(v));
  }

  // Checks added by the port (tflow prints the table only).
  check::Require(std::isfinite(last_nll) && std::isfinite(first_nll), "finite model fitting losses in both runtime profiles");
  check::TrainingQuality(last_nll < first_nll, "the ensemble's NLL falls as it is refit");
  check::TrainingQuality(best_pets > -400, "PETS swings the pendulum up in some round (return > -400)");
  check::TrainingQuality(pb.ret > -600, "the PETS episode played at the end does well (return > -600)");
  check::TrainingQuality(spread[1] > spread[0], "the members disagree more on the rare state than on the common one");
  std::printf("\nchecks passed (vsim video omitted)\n");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "CHECK FAILED: %s\n", e.what());
    return 1;
  }
  return 0;
}
