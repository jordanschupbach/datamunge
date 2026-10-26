// CMD: just run-nn r2d2
// Recurrent replay (R2D2-style) for partial observability.
//
// Port of tflow's r2d2. CartPole with the velocities hidden: the agent only
// sees [x, theta], so a single observation does not determine the state and
// a feedforward Q-network cannot do well. An LSTM Q-network integrates
// observations over time. As in R2D2 (Kapturowski et al. 2019) the replay
// buffer stores fixed-length sequences; training unrolls the online and
// target LSTMs over each sequence (from a zero state, the "zero start"
// variant) and applies the Double DQN loss at every step, masked past the
// episode's end. Acting runs the same LSTM one step at a time, carrying its
// state across the episode.
//
// In TF the acting graph and the unrolled training graph shared variables;
// here both are eager nn::lstm_step calls on the same ParameterStore (the
// target under NoGradGuard). CartPole, sequence sampling and epsilon draws
// use the same std::mt19937_64 streams as TF; the weights use nn's
// initialiser, so the results match TF statistically. The vsim video of an
// evaluation episode is omitted. Options: the rl::DqnConfig keys (steps=,
// lr=, seed=, ...).

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

// CartPole observed through [x, theta] only.
class MaskedCartPole : public rl::Env {
 public:
  std::string name() const override { return "MaskedCartPole"; }
  int obs_size() const override { return 2; }
  const rl::CartPole& inner() const { return inner_; }
  rl::ActionSpace action_space() const override { return inner_.action_space(); }
  std::vector<std::string> action_names() const override { return inner_.action_names(); }
  void Seed(uint64_t seed) override { inner_.Seed(seed); }
  int max_steps() const override { return inner_.max_steps(); }
  rl::StepResult step(const rl::Action& a) override {
    rl::StepResult r = inner_.step(a);
    r.obs = Mask(r.obs);
    return r;
  }

 protected:
  rl::Obs Reset() override { return Mask(inner_.reset()); }

 private:
  static rl::Obs Mask(const rl::Obs& o) { return {o[0], o[2]}; }
  rl::CartPole inner_;
};

}  // namespace

int main(int argc, char** argv) {
  try {
    rl::DqnConfig cfg;
    cfg.train_every = 2;
    cfg.batch = 32;
    const int L = 10, H = 64;
    rl::ParseConfig(argc, argv, &cfg);
    MaskedCartPole env, eval_env;
    const int obs_dim = env.obs_size(), A = env.action_space().n;
    std::printf("== R2D2-style LSTM Double DQN on CartPole without velocities, %d steps, sequences of %d\n", cfg.steps, L);

    nn::ParameterStore online(cfg.seed), target(cfg.seed + 1);
    // One recurrent step: embed -> LSTM -> Q.
    struct StepOut {
      nn::LstmState state;
      Tensor q;
    };
    auto net_step = [&](nn::ParameterStore& p, const Tensor& x, const nn::LstmState& st) {
      const Tensor e = nn::dense(p, "embed", x, H, nn::Activation::Relu);
      nn::LstmState next = nn::lstm_step(p, "lstm", e, st);
      return StepOut{next, nn::dense(p, "q", next.h, A)};
    };
    // Unroll over seq [B, L+1, obs] from a zero state; returns L+1 tensors [B, A].
    auto unroll = [&](nn::ParameterStore& p, const std::vector<Tensor>& steps) {
      std::vector<Tensor> qs;
      nn::LstmState st = nn::lstm_zero_state(steps[0].shape()[0], H);
      for (const Tensor& x : steps) {
        StepOut o = net_step(p, x, st);
        st = o.state;
        qs.push_back(o.q);
      }
      return qs;
    };
    {
      nn::NoGradGuard no_grad;
      net_step(online, nn::zeros({1, obs_dim}), nn::lstm_zero_state(1, H));
      net_step(target, nn::zeros({1, obs_dim}), nn::lstm_zero_state(1, H));
    }
    rl::Adam adam(online, cfg.lr, cfg.clip_norm);
    auto sync = [&] { target.copy_from(online, cfg.tau); };
    sync();

    rl::SequenceReplay buffer(cfg.buffer / L, L, obs_dim);
    rl::Monitor monitor(cfg);
    std::mt19937_64 rng(cfg.seed);

    // Recurrent acting: the LSTM state persists for the whole episode.
    struct Actor {
      nn::LstmState st;
      void Reset(int units) { st = nn::lstm_zero_state(1, units); }
    };
    auto step_actor = [&](Actor& actor, const rl::Obs& o) {  // advances the state, returns argmax Q
      nn::NoGradGuard no_grad;
      StepOut out = net_step(online, rl::Row(o), actor.st);
      actor.st = out.state;
      return rl::ArgMax(out.q.to_vector<float>());
    };
    auto evaluate = [&](int episodes, uint64_t seed) {
      double total = 0;
      for (int e = 0; e < episodes; ++e) {
        Actor actor;
        actor.Reset(H);
        rl::Obs o = eval_env.reset(seed + e);
        while (true) {
          rl::StepResult sr = eval_env.step(step_actor(actor, o));
          total += sr.reward;
          o = sr.obs;
          if (sr.done()) break;
        }
      }
      return static_cast<float>(total / episodes);
    };

    auto train = [&](const rl::SeqBatch& b) {
      const std::vector<Tensor> steps = nn::unstack(b.obs, 1);
      const std::vector<Tensor> q_on = unroll(online, steps);
      std::vector<Tensor> q_tg;
      {
        nn::NoGradGuard no_grad;
        q_tg = unroll(target, steps);
      }
      const std::vector<Tensor> a_cols = nn::unstack(b.a, 1), r_cols = nn::unstack(b.r, 1), d_cols = nn::unstack(b.done, 1),
                                m_cols = nn::unstack(b.mask, 1);
      Tensor total;
      for (int t = 0; t < L; ++t) {
        Tensor y;
        {
          nn::NoGradGuard no_grad;
          const Tensor a_star = nn::argmax(nn::stop_gradient(q_on[t + 1]), 1);
          const Tensor next_value = nn::sum(q_tg[t + 1] * nn::one_hot(a_star, A), {1});
          y = r_cols[t] + (cfg.gamma * (1.0 - d_cols[t])) * next_value;
        }
        const Tensor q_a = nn::sum(q_on[t] * nn::one_hot(a_cols[t], A), {1});
        const Tensor l = m_cols[t] * nn::huber(y - q_a);
        total = t == 0 ? l : total + l;
      }
      const Tensor loss = nn::sum(total, {0}) / nn::sum(b.mask);
      adam.Minimize(loss);
      return loss.item<float>();
    };

    Actor actor;
    actor.Reset(H);
    rl::Obs obs = env.reset(cfg.seed);
    std::vector<rl::Obs> ep_obs{obs};
    std::vector<int> ep_a;
    std::vector<float> ep_r, ep_done;
    float ret = 0, best_eval = 0;
    for (int step = 1; step <= cfg.steps; ++step) {
      const float eps = rl::Epsilon(cfg, step);
      const int greedy_a = step_actor(actor, obs);  // always step the LSTM so its state stays current
      const int act = std::uniform_real_distribution<float>(0, 1)(rng) < eps ? static_cast<int>(rng() % A) : greedy_a;
      rl::StepResult sr = env.step(act);
      ep_a.push_back(act);
      ep_r.push_back(sr.reward);
      ep_done.push_back(sr.terminated ? 1.f : 0.f);
      ep_obs.push_back(sr.obs);
      ret += sr.reward;
      obs = sr.obs;
      if (sr.done()) {
        buffer.AddEpisode(ep_obs, ep_a, ep_r, ep_done);
        monitor.EndEpisode(ret);
        ret = 0;
        obs = env.reset();
        ep_obs = {obs};
        ep_a.clear();
        ep_r.clear();
        ep_done.clear();
        actor.Reset(H);
      }
      if (step >= cfg.warmup && step % cfg.train_every == 0 && buffer.size() > 0) {
        const float l = train(buffer.Sample(cfg.batch, rng));
        check::Require(std::isfinite(l), "finite masked TD loss");
        monitor.AddLoss(l);
      }
      if (cfg.tau < 1.f || step % cfg.target_sync == 0) sync();
      if (step % cfg.log_every == 0) {
        const float ev = step % cfg.eval_every == 0 ? evaluate(cfg.eval_episodes, 1000 + step) : NAN;
        if (!std::isnan(ev)) best_eval = std::max(best_eval, ev);
        monitor.Log(step, eps, ev);
      }
    }
    const float final_eval = evaluate(20, 777);
    std::printf("  final evaluation over 20 episodes: %.1f (a feedforward net on [x, theta] alone typically stays below 100)\n",
                final_eval);

    // The unrolled training path and the one-step acting path are the same network.
    {
      nn::NoGradGuard no_grad;
      rl::SeqBatch b = buffer.Sample(1, rng);
      const std::vector<Tensor> steps = nn::unstack(b.obs, 1);
      const std::vector<Tensor> qs = unroll(online, steps);
      Actor a;
      a.Reset(H);
      float max_diff = 0;
      for (int t = 0; t <= L; ++t) {
        StepOut o = net_step(online, steps[t], a.st);
        a.st = o.state;
        max_diff = std::max(max_diff, nn::max(nn::abs(o.q - qs[t])).item<float>());
      }
      check::Require(max_diff < 1e-5f, "unrolled and step-by-step LSTM Q-values agree");
    }
    // TF's bound for a feedforward net on [x, theta] is "below 100"; TF seeds
    // 0..2 reached best 500, 382, 224.8, here 500, 144, 272.2.
    check::TrainingQuality(std::max(best_eval, final_eval) >= 100,
                   "the LSTM agent beats the feedforward bound without velocities (best evaluation >= 100)");
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
