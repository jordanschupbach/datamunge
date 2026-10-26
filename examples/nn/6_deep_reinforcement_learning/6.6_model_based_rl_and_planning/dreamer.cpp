// CMD: just run-nn dreamer
// Dreamer (v1): actor-critic learned entirely inside the latent model.
//
// Port of tflow's dreamer. Hafner et al. 2020 keep PlaNet's world model but
// replace planning with a learned actor and critic trained on *imagined*
// trajectories: from latent states of replayed sequences, the actor's actions
// are rolled through the model's prior for H steps, the critic estimates
// values along the way, and the actor maximises lambda-returns by
// back-propagating through the model's dynamics and reward decoder. v2 and v3
// refine this (discrete latents, KL balancing, symlog targets, REINFORCE for
// the actor); the structure is the same.
//
// The TF graph becomes eager nn code: the RSSM is common/rl/rssm.hpp (port of
// tflow's rssm.h). As in TF's single session run, one training step builds
// the world-model, actor and critic losses from the same parameters, takes all
// three gradients, and only then applies the three Adam updates (each clipped
// at global norm 100). The loop, Pendulum, sequence sampling and every noise
// tensor use the same std::mt19937_64 stream and the same single
// normal_distribution as TF; weights use nn's generator, so returns match TF
// statistically. tflow films one evaluation episode on vsim; the episode is
// played here (same seed) and its return printed, without the video.
// Options: rounds=N (default 30), steps_per_round=N (250), seed=N (weight
// seed, 0), threads=N. The port adds checks (exit 1 on failure) when run with
// the default rounds.

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
#include "common/rl/rssm.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

struct Episode {
  std::vector<rl::Obs> obs;
  std::vector<std::vector<float>> act;
  std::vector<float> rew;
};

// Splits a row-major [N, K, D] buffer into K tensors [N, D] (TF's Unstack on axis 1).
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

// Adam step with gradients computed beforehand (so several losses can share one forward pass).
struct ClippedAdam {
  ClippedAdam(const nn::ParameterStore& store, double lr, double clip) : opt(store, lr, {nn::OptimizerKind::Adam}), clip(clip) {}
  std::vector<Tensor> Grads(const Tensor& loss, bool retain) {
    nn::GradOptions o;
    o.retain_graph = retain;
    return nn::clip_by_global_norm(nn::grad(loss, opt.parameters(), o), clip).first;
  }
  nn::Optimizer opt;
  double clip;
};

}  // namespace

int main(int argc, char** argv) {
  int rounds = 30, steps_per_round = 250;
  std::uint64_t seed = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const size_t eq = arg.find('=');
    if (eq == std::string::npos) continue;
    const std::string key = arg.substr(0, eq), val = arg.substr(eq + 1);
    if (key == "rounds") rounds = std::atoi(val.c_str());
    else if (key == "steps_per_round") steps_per_round = std::atoi(val.c_str());
    else if (key == "seed") seed = std::strtoull(val.c_str(), nullptr, 10);
    else if (key == "threads") nn::set_num_threads(std::atoi(val.c_str()));
    else if (!rl::IsHardwareOption(key)) std::fprintf(stderr, "unknown option %s\n", arg.c_str());
  }
  rounds = example_runtime::Limit(rounds, 2);
  steps_per_round = example_runtime::Limit(steps_per_round, 8);
  try {
    rl::Pendulum env, eval_env;
    const rl::ActionSpace space = env.action_space();
    const int obs_dim = env.obs_size(), act_dim = space.dim;
    const int T = 16, B = example_runtime::Limit(32, 4), H = example_runtime::Limit(15, 8);
    const float reward_scale = 0.1f;  // the model and critic work with r / 10
    const int kActionRepeat = 2;      // each decision is held for this many environment steps
    const float gamma = 0.99f, lambda = 0.95f;
    rl::RssmConfig cfg;
    cfg.obs_dim = obs_dim, cfg.act_dim = act_dim;
    std::printf("== Dreamer on Pendulum: RSSM (det %d, stoch %d), imagination horizon %d, %d rounds of %d updates\n",
                cfg.det, cfg.stoch, H, rounds, steps_per_round);

    nn::ParameterStore model_p(seed), actor_p(seed + 1), critic_p(seed + 2);
    rl::Rssm rssm(model_p, cfg);
    auto actor = [&](const Tensor& features, const Tensor& eps) {
      const rl::Gaussian g = rl::GaussianActor(actor_p, "pi", features, act_dim, 64);
      return rl::SquashedSample(g, eps).action;
    };
    auto critic = [&](const Tensor& features) {
      const Tensor h = nn::dense(critic_p, "v/h", features, 64, nn::Activation::Relu);
      return nn::squeeze(nn::dense(critic_p, "v/out", h, 1), {1});
    };
    {  // build every weight once so the optimisers see them
      nn::NoGradGuard no_grad;
      const rl::RssmState st = rssm.Zero(1);
      const nn::LstmState hc = rssm.Transition(st, nn::zeros({1, act_dim}));
      rssm.Observe(st, {nn::zeros({1, obs_dim})}, {nn::zeros({1, act_dim})}, {nn::zeros({1, cfg.stoch})});
      rssm.Prior(hc.h);
      actor(rssm.Features(st), nn::zeros({1, act_dim}));
      critic(rssm.Features(st));
    }
    ClippedAdam model_adam(model_p, 1e-3, 100), actor_adam(actor_p, 1e-4, 100), critic_adam(critic_p, 1e-4, 100);

    std::mt19937_64 rng(0);
    std::normal_distribution<float> normal(0.f, 1.f);
    std::vector<Episode> episodes;
    auto noise3 = [&](int a, int b, int c) {
      std::vector<float> v(static_cast<size_t>(a) * b * c);
      for (float& x : v) x = normal(rng);
      return v;
    };
    auto train_step = [&]() {
      std::vector<float> O(static_cast<size_t>(B) * T * obs_dim), Av(static_cast<size_t>(B) * T * act_dim), R(static_cast<size_t>(B) * T);
      for (int b = 0; b < B; ++b) {
        const Episode& ep = episodes[rng() % episodes.size()];
        const int start = static_cast<int>(rng() % (ep.rew.size() - T + 1));
        for (int t = 0; t < T; ++t) {
          for (int j = 0; j < obs_dim; ++j) O[(b * T + t) * obs_dim + j] = ep.obs[start + t][j];
          for (int j = 0; j < act_dim; ++j) Av[(b * T + t) * act_dim + j] = ep.act[start + t][j];
          R[b * T + t] = ep.rew[start + t];
        }
      }
      // Noise in TF's feed order: posterior samples, actor noise, imagined prior samples.
      const std::vector<float> eps_seq = noise3(B, T, cfg.stoch), act_eps = noise3(B * T, H, act_dim), im_eps = noise3(B * T, H, cfg.stoch);
      const std::vector<Tensor> obs_t = Unstack1(O, B, T, obs_dim), act_t = Unstack1(Av, B, T, act_dim), rew_t = Unstack1(R, B, T, 1),
                                eps_t = Unstack1(eps_seq, B, T, cfg.stoch), ae_t = Unstack1(act_eps, B * T, H, act_dim),
                                ie_t = Unstack1(im_eps, B * T, H, cfg.stoch);
      // ---- World model (as in planet).
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
      const Tensor model_loss = (recon_sum + rew_sum + kl_sum) / static_cast<double>(T);

      // ---- Imagination from every posterior state of the batch (B*T starts),
      // with the actor choosing actions and the critic valuing features.
      // Imagination must not update the world model: start from detached states.
      std::vector<Tensor> hs, cs, zs;
      for (const rl::RssmState& st : un.states) hs.push_back(st.h), cs.push_back(st.c), zs.push_back(st.z);
      rl::RssmState st{nn::concat(hs, 0).detach(), nn::concat(cs, 0).detach(), nn::concat(zs, 0).detach()};
      std::vector<Tensor> im_reward, im_value;
      im_value.push_back(critic(rssm.Features(st)));
      for (int t = 0; t < H; ++t) {
        const Tensor f = rssm.Features(st);
        const nn::LstmState next = rssm.Transition(st, actor(f, ae_t[t]));
        st.h = next.h, st.c = next.c;
        st.z = rssm.Sample(rssm.Prior(st.h), ie_t[t]);
        const Tensor f2 = rssm.Features(st);
        im_reward.push_back(rssm.DecodeReward(f2));
        im_value.push_back(critic(f2));
      }
      // lambda-returns backwards: R_t = r_t + gamma [(1 - lambda) V_{t+1} + lambda R_{t+1}], R_H = V_H.
      std::vector<Tensor> lambda_ret(H + 1);
      lambda_ret[H] = im_value[H];
      for (int t = H - 1; t >= 0; --t)
        lambda_ret[t] = im_reward[t] + gamma * ((1 - lambda) * im_value[t + 1] + lambda * lambda_ret[t + 1]);
      Tensor actor_sum, critic_sum;
      for (int t = 0; t < H; ++t) {
        const Tensor a_term = nn::mean(lambda_ret[t], {0});
        const Tensor c_term = nn::mean(nn::square(lambda_ret[t].detach() - im_value[t]), {0});
        actor_sum = t ? actor_sum + a_term : a_term;
        critic_sum = t ? critic_sum + c_term : c_term;
      }
      const Tensor actor_loss = -(actor_sum / static_cast<double>(H));
      const Tensor critic_loss = critic_sum / static_cast<double>(H);
      // All gradients from the same parameters, then the three updates.
      const std::vector<Tensor> gm = model_adam.Grads(model_loss, false);
      const std::vector<Tensor> ga = actor_adam.Grads(actor_loss, true);
      const std::vector<Tensor> gc = critic_adam.Grads(critic_loss, false);
      model_adam.opt.step(gm);
      actor_adam.opt.step(ga);
      critic_adam.opt.step(gc);
      return std::array<float, 3>{model_loss.item<float>(), actor_loss.item<float>(), critic_loss.item<float>()};
    };

    // ---- Acting: filter the latent state and query the actor.
    // `on_step(u, sr)` is called after every environment step with the action held and the step's result.
    float last_value = 0;  // the critic's value (in r / 10 units) of the latest filtered state
    auto run_episode = [&](rl::Env& e, bool random_actions, float explore_noise, bool record, uint64_t ep_seed,
                           const std::function<void(const std::vector<float>&, const rl::StepResult&)>& on_step = {}) {
      nn::NoGradGuard no_grad;
      Episode ep;
      rl::Obs o = e.reset(ep_seed);
      rl::RssmState s = rssm.Zero(1);
      Tensor a_last = nn::zeros({1, act_dim});
      float ret = 0;
      bool first_step = true;
      while (true) {
        std::vector<float> an(act_dim);
        for (int d = 0; d < act_dim; ++d) an[d] = explore_noise * normal(rng);
        rl::RssmState f = s;
        if (!first_step) {
          const nn::LstmState hc = rssm.Transition(s, a_last);
          f.h = hc.h, f.c = hc.c;
        }
        f.z = rssm.Posterior(f.h, rssm.Embed(rl::Row(o))).mean;
        const std::vector<float> a_f = actor(rssm.Features(f), nn::from_vector(an, {1, act_dim})).to_vector<float>();
        last_value = critic(rssm.Features(f)).item<float>();
        s = f;
        first_step = false;
        std::vector<float> u(act_dim);
        for (int d = 0; d < act_dim; ++d) u[d] = random_actions ? std::uniform_real_distribution<float>(-1, 1)(rng) : a_f[d];
        rl::StepResult sr;
        float r_sum = 0;
        for (int k = 0; k < kActionRepeat; ++k) {
          sr = e.step(rl::ScaleAction(u, space));
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
      if (record) episodes.push_back(std::move(ep));
      return ret;
    };

    std::printf("  %5s %12s %12s %12s %14s %14s\n", "round", "model loss", "actor loss", "critic loss", "explore return", "eval (mean)");
    for (int i = 0; i < 5; ++i) run_episode(env, true, 0.f, true, i);
    float first_model_loss = 0, last_model_loss = 0;
    std::vector<float> evals;
    for (int round = 1; round <= rounds; ++round) {
      std::array<float, 3> l{};
      for (int i = 0; i < steps_per_round; ++i) {
        l = train_step();
        if (round == 1 && i == 0) first_model_loss = l[0];
        check::Require(std::isfinite(l[0]) && std::isfinite(l[1]) && std::isfinite(l[2]), "finite losses");
      }
      last_model_loss = l[0];
      const float ret = run_episode(env, false, 0.3f, true, 100 + round);
      const float ev = run_episode(eval_env, false, 0.f, false, 500 + round);
      evals.push_back(ev);
      std::printf("  %5d %12.3f %12.2f %12.3f %14.1f %14.1f\n", round, l[0], l[1], l[2], ret, ev);
      std::fflush(stdout);
    }
    std::printf("  (the actor never sees a real gradient: it is trained only on imagined latent trajectories)\n");
    // In tflow, on film: the actor's torque and the critic's value of the filtered latent state.
    int steps = 0;
    const float film_ret = run_episode(eval_env, false, 0.f, false, 777, [&](const std::vector<float>&, const rl::StepResult&) { ++steps; });
    std::printf("  played dreamer_pendulum (video omitted): %d steps, return %.1f, final V(s) %.1f\n", steps, film_ret,
                last_value / reward_scale);

    check::TrainingQuality(last_model_loss < first_model_loss, "the world-model loss falls below its first-step value");
    if (rounds == 30) {
      float best_last5 = -1e9f;
      for (int i = rounds - 5; i < rounds; ++i) best_last5 = std::max(best_last5, evals[i]);
      check::TrainingQuality(best_last5 > -800, "Dreamer improves on a random policy (best eval of the last 5 rounds > -800)");
    }
    std::printf("\nchecks passed (vsim video omitted)\n");
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "CHECK FAILED: %s\n", ex.what());
    return 1;
  }
  return 0;
}
