// CMD: just run-nn planet
// PlaNet: latent dynamics and CEM planning.
//
// Port of tflow's planet.
// Hafner et al. 2019 learn a recurrent state-space model (common/rl/rssm.hpp)
// from replayed sequences, by reconstructing observations and rewards and
// keeping the posterior close to the prior (a variational objective), and
// then act by planning in the latent space: from the filtered latent state,
// the cross-entropy method searches action sequences whose imagined reward
// under the model's prior is highest. The original works from pixels; here
// the observations are Pendulum's three numbers, which keeps it CPU-sized.
//
// The TF graphs (training unroll, one-step filter, N x H imagination) become
// eager calls on the RSSM port; Adam clips the global norm at 100 as tflow's
// nn::Adam. Pendulum, replay windows, CEM and every noise draw use the same
// std::mt19937_64 and the single shared normal_distribution in TF's order
// (the imagination noise is drawn although planning uses the prior mean, as
// in TF); weights use nn's generator, so returns match TF statistically. The
// vsim video of the last episode is replaced by playing it and printing its
// return. Options: rounds=30 model_steps=200 threads=N. The port adds checks
// (exit 1 on failure).

#include <algorithm>
#include <array>
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
#include "common/rl/dqn_common.hpp"
#include "common/rl/rssm.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

struct Episode {
  std::vector<rl::Obs> obs;
  std::vector<std::vector<float>> act;
  std::vector<float> rew;
};

// Column k of a row-major [N, K, D] buffer as an [N, D] tensor (TF's Unstack on axis 1).
std::vector<Tensor> Unstack1(const std::vector<float>& v, int N, int K, int D) {
  std::vector<Tensor> out;
  for (int k = 0; k < K; ++k) {
    std::vector<float> s(static_cast<size_t>(N) * D);
    for (int n = 0; n < N; ++n)
      for (int d = 0; d < D; ++d) s[static_cast<size_t>(n) * D + d] = v[(static_cast<size_t>(n) * K + k) * D + d];
    out.push_back(nn::from_vector(s, {N, D}));
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int rounds = 30, model_steps = 200;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      const size_t eq = arg.find('=');
      if (eq == std::string::npos) continue;
      const std::string key = arg.substr(0, eq);
      const int v = std::atoi(arg.c_str() + eq + 1);
      if (key == "threads") nn::set_num_threads(v);
      else if (key == "rounds") rounds = v;
      else if (key == "model_steps") model_steps = v;
      else if (!rl::IsHardwareOption(key)) std::fprintf(stderr, "unknown option %s\n", arg.c_str());
    }
    rounds = example_runtime::Limit(rounds, 2);
    model_steps = example_runtime::Limit(model_steps, 8);
    rl::Pendulum env;
    const rl::ActionSpace space = env.action_space();
    const int obs_dim = env.obs_size(), act_dim = space.dim;
    const int T = 16, B = example_runtime::Limit(32, 4), H = example_runtime::Limit(20, 8), N = example_runtime::Limit(200, 32), elites = example_runtime::Limit(20, 4), cem_iters = example_runtime::Limit(4, 2);
    const float reward_scale = 0.1f;  // the model predicts r / 10; CEM ranking is unaffected
    const int kActionRepeat = 2;      // each decision is held for this many environment steps
    const bool kPlanWithMean = true;  // imagine with the prior mean instead of samples
    rl::RssmConfig cfg;
    cfg.obs_dim = obs_dim, cfg.act_dim = act_dim;
    std::printf("== PlaNet on Pendulum: RSSM (det %d, stoch %d), sequences of %d, CEM with %d candidates over %d steps, %d rounds (rewards scaled by 0.1 in the model)\n",
                cfg.det, cfg.stoch, T, N, H, rounds);

    nn::ParameterStore params(0);
    rl::Rssm rssm(params, cfg);
    {  // build every weight once so Adam sees them
      nn::NoGradGuard no_grad;
      const rl::RssmState st = rssm.Zero(1);
      rssm.Observe(st, {nn::zeros({1, obs_dim}), nn::zeros({1, obs_dim})}, {nn::zeros({1, act_dim}), nn::zeros({1, act_dim})},
                   {nn::zeros({1, cfg.stoch}), nn::zeros({1, cfg.stoch})});
      rssm.Prior(st.h);
    }
    rl::Adam adam(params, 1e-3, 100.0);

    std::mt19937_64 rng(0);
    std::normal_distribution<float> normal(0.f, 1.f);
    std::vector<Episode> episodes;
    // One training step on B random windows of length T.
    auto train_step = [&]() {
      std::vector<float> O(static_cast<size_t>(B) * T * obs_dim), Av(static_cast<size_t>(B) * T * act_dim), R(static_cast<size_t>(B) * T),
          E(static_cast<size_t>(B) * T * cfg.stoch);
      for (int b = 0; b < B; ++b) {
        const Episode& ep = episodes[rng() % episodes.size()];
        const int start = static_cast<int>(rng() % (ep.rew.size() - T + 1));
        for (int t = 0; t < T; ++t) {
          for (int j = 0; j < obs_dim; ++j) O[(b * T + t) * obs_dim + j] = ep.obs[start + t][j];
          for (int j = 0; j < act_dim; ++j) Av[(b * T + t) * act_dim + j] = ep.act[start + t][j];
          R[b * T + t] = ep.rew[start + t];
          for (int j = 0; j < cfg.stoch; ++j) E[(b * T + t) * cfg.stoch + j] = normal(rng);
        }
      }
      const std::vector<Tensor> obs_t = Unstack1(O, B, T, obs_dim), act_t = Unstack1(Av, B, T, act_dim), rew_t = Unstack1(R, B, T, 1),
                                eps_t = Unstack1(E, B, T, cfg.stoch);
      const rl::Rssm::Unrolled un = rssm.Observe(rssm.Zero(B), obs_t, act_t, eps_t);
      Tensor recon_sum, rew_sum, kl_sum;
      for (int t = 0; t < T; ++t) {
        const Tensor rec = nn::mean(nn::sum(nn::square(un.recon[t] - obs_t[t]), {1}), {0});
        const Tensor rw = nn::mean(nn::square(un.reward[t] - nn::squeeze(rew_t[t], {1})), {0});
        const Tensor kl = nn::maximum(nn::mean(un.kl[t], {0}), nn::full({}, cfg.free_nats));
        recon_sum = t ? recon_sum + rec : rec;
        rew_sum = t ? rew_sum + rw : rw;
        kl_sum = t ? kl_sum + kl : kl;
      }
      const Tensor recon_loss = recon_sum / static_cast<double>(T), rew_loss = rew_sum / static_cast<double>(T),
                   kl_loss = kl_sum / static_cast<double>(T);
      const Tensor loss = (recon_loss + rew_loss) + kl_loss;
      const std::array<float, 4> l{loss.item<float>(), recon_loss.item<float>(), rew_loss.item<float>(), kl_loss.item<float>()};
      adam.Minimize(loss);
      return l;
    };
    // CEM over H-step action sequences in latent space from the filtered state.
    float last_plan_return = 0;  // the model's return (in r / 10 units) of the best candidate of the last plan()
    auto plan = [&](const rl::RssmState& filtered) {
      nn::NoGradGuard no_grad;
      std::vector<float> mean(H * act_dim, 0.f), stdv(H * act_dim, 1.f);
      const rl::RssmState start{nn::tile(filtered.h, {N, 1}), nn::tile(filtered.c, {N, 1}), nn::tile(filtered.z, {N, 1})};
      for (int it = 0; it < cem_iters; ++it) {
        std::vector<float> Av(static_cast<size_t>(N) * H * act_dim), E(static_cast<size_t>(N) * H * cfg.stoch);
        for (int i = 0; i < N; ++i)
          for (int t = 0; t < H; ++t) {
            for (int d = 0; d < act_dim; ++d)
              Av[(i * H + t) * act_dim + d] = std::max(-1.f, std::min(1.f, mean[t * act_dim + d] + stdv[t * act_dim + d] * normal(rng)));
            for (int j = 0; j < cfg.stoch; ++j) E[(i * H + t) * cfg.stoch + j] = normal(rng);
          }
        const std::vector<Tensor> pa_t = Unstack1(Av, N, H, act_dim);
        const std::vector<Tensor> pe_t = kPlanWithMean ? std::vector<Tensor>() : Unstack1(E, N, H, cfg.stoch);
        rl::RssmState st = start;
        Tensor plan_return;
        for (int t = 0; t < H; ++t) {
          const nn::LstmState next = rssm.Transition(st, pa_t[t]);
          st.h = next.h, st.c = next.c;
          const rl::RssmGauss prior = rssm.Prior(st.h);
          st.z = kPlanWithMean ? prior.mean : rssm.Sample(prior, pe_t[t]);
          const Tensor r = rssm.DecodeReward(rssm.Features(st));
          plan_return = t ? plan_return + r : r;
        }
        const std::vector<float> ret = plan_return.to_vector<float>();
        std::vector<int> order(N);
        for (int i = 0; i < N; ++i) order[i] = i;
        std::partial_sort(order.begin(), order.begin() + elites, order.end(), [&](int x, int y) { return ret[x] > ret[y]; });
        last_plan_return = ret[order[0]];
        for (int j = 0; j < H * act_dim; ++j) {
          const int t = j / act_dim, d = j % act_dim;
          float m = 0, v = 0;
          for (int k = 0; k < elites; ++k) m += Av[(order[k] * H + t) * act_dim + d] / elites;
          for (int k = 0; k < elites; ++k) {
            const float x = Av[(order[k] * H + t) * act_dim + d];
            v += (x - m) * (x - m) / elites;
          }
          mean[j] = m, stdv[j] = std::sqrt(v) + 0.02f;
        }
      }
      return std::vector<float>(mean.begin(), mean.begin() + act_dim);
    };
    // One episode with the planner (or random actions), filtering the latent state online.
    // `on_step(u, sr)` is called after every environment step with the action held and the step's result.
    auto run_episode = [&](bool use_planner, uint64_t seed, const std::function<void(const std::vector<float>&, const rl::StepResult&)>& on_step = {}) {
      Episode ep;
      rl::Obs o = env.reset(seed);
      rl::RssmState st = rssm.Zero(1);
      Tensor a_last = nn::zeros({1, act_dim});
      float ret = 0;
      bool first_step = true;
      while (true) {
        {
          nn::NoGradGuard no_grad;
          if (!first_step) {  // TF blends with first = 1 at the episode start (no transition)
            const nn::LstmState hc = rssm.Transition(st, a_last);
            st.h = hc.h, st.c = hc.c;
          }
          st.z = rssm.Posterior(st.h, rssm.Embed(rl::Row(o))).mean;  // filter with the posterior mean
        }
        first_step = false;
        std::vector<float> u(act_dim);
        if (use_planner) u = plan(st);
        else
          for (float& x : u) x = std::uniform_real_distribution<float>(-1, 1)(rng);
        rl::StepResult sr;
        float r_sum = 0;
        for (int k = 0; k < kActionRepeat; ++k) {
          sr = env.step(rl::ScaleAction(u, space));
          r_sum += sr.reward;
          if (on_step) on_step(u, sr);
          if (sr.done()) break;
        }
        ep.obs.push_back(o), ep.act.push_back(u), ep.rew.push_back(r_sum * reward_scale);
        a_last = nn::from_vector(u, {1, act_dim});
        ret += r_sum;
        o = sr.obs;
        if (sr.done()) break;
      }
      episodes.push_back(std::move(ep));
      return ret;
    };

    std::printf("  %5s %10s %10s %10s %10s %14s\n", "round", "loss", "recon", "reward", "KL", "planner return");
    for (int i = 0; i < 5; ++i) run_episode(false, i);
    float first_loss = -1;
    std::array<float, 4> l{};
    std::vector<float> returns;
    for (int round = 1; round <= rounds; ++round) {
      for (int i = 0; i < model_steps; ++i) {
        l = train_step();
        check::Require(std::isfinite(l[0]), "finite PlaNet loss");
        if (first_loss < 0) first_loss = l[0];
      }
      const float ret = run_episode(true, 100 + round);
      returns.push_back(ret);
      std::printf("  %5d %10.3f %10.3f %10.3f %10.3f %14.1f\n", round, l[0], l[1], l[2], l[3], ret);
      std::fflush(stdout);
    }
    std::printf("  (random policy about -1200; planning in the learned latent space should climb toward -200)\n");
    // On film in TF, with the torque and the latent model's planned return as the overlay.
    const float filmed = run_episode(true, 777);
    std::printf("  played planet_pendulum (video omitted): return %.1f (last planned return %.1f)\n", filmed, last_plan_return / reward_scale);

    // Checks added by the port (tflow prints the table only).
    check::TrainingQuality(l[0] < first_loss, "the world-model loss falls");
    if (rounds >= 30) {
      const float last5 = *std::max_element(returns.end() - 5, returns.end());
      check::TrainingQuality(last5 > -600, "planning in the learned latent space swings the pendulum up (best of the last 5 rounds > -600)");
    }
    std::printf("\nchecks passed (vsim video omitted)\n");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "CHECK FAILED: %s\n", e.what());
    return 1;
  }
  return 0;
}
