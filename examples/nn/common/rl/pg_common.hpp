// datamunge::nn port of tflow_examples src/common/rl/pg_common.h.
// Pieces shared by the policy-gradient and actor-critic examples: config,
// a softmax actor + value critic built from nn layers, action sampling,
// discounted returns and GAE on the host, and batched rollouts from a VecEnv.
// Monitor, Evaluate and Adam are reused from dqn_common.hpp.
//
// Changes from the TF version: BuildActorCritic runs the network on a
// tensor (eager) instead of adding graph nodes, ObsMatrix/IntVector return
// nn::Tensor (actions as int64), and TF hardware options are ignored.
#pragma once
#include "common/runtime.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <datamunge/nn/nn.hpp>
#include <random>
#include <string>
#include <vector>

#include "common/rl/dqn_common.hpp"
#include "common/rl/env.hpp"
#include "common/rl/utils.hpp"
#include "common/rl/vec_env.hpp"

namespace rl {

struct PgConfig {
  int steps = 100000;        // environment steps in total (all envs)
  float gamma = 0.99f;
  float lam = 0.95f;         // GAE lambda
  float lr = 1e-3f;
  float clip_norm = 0.5f;
  float entropy_coef = 0.01f;
  float value_coef = 0.5f;
  int hidden = 64;
  int n_envs = 8;            // parallel environments for batched rollouts
  int n_steps = 5;           // rollout length per update
  int log_every = 10000;
  int eval_every = 10000;
  int eval_episodes = 5;
  uint64_t seed = 0;
};

// `just run-nn a2c steps=200000 lr=5e-4 n_envs=16`
inline void ParsePgConfig(int argc, char** argv, PgConfig* c) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const size_t eq = arg.find('=');
    if (eq == std::string::npos) continue;
    const std::string key = arg.substr(0, eq);
    const float v = std::strtof(arg.c_str() + eq + 1, nullptr);
    if (IsHardwareOption(key)) continue;
    if (key == "steps") c->steps = static_cast<int>(v);
    else if (key == "gamma") c->gamma = v;
    else if (key == "lam") c->lam = v;
    else if (key == "lr") c->lr = v;
    else if (key == "clip_norm") c->clip_norm = v;
    else if (key == "entropy") c->entropy_coef = v;
    else if (key == "value_coef") c->value_coef = v;
    else if (key == "hidden") c->hidden = static_cast<int>(v);
    else if (key == "n_envs") c->n_envs = static_cast<int>(v);
    else if (key == "n_steps") c->n_steps = static_cast<int>(v);
    else if (key == "log_every") c->log_every = static_cast<int>(v);
    else if (key == "eval_every") c->eval_every = static_cast<int>(v);
    else if (key == "seed") c->seed = static_cast<uint64_t>(v);
    else if (key != "task" && key != "game") std::fprintf(stderr, "unknown option %s\n", arg.c_str());
  }

  if (!example_runtime::Full()) {
    c->steps = std::min(c->steps, 512);
    c->n_envs = std::min(c->n_envs, 4);
    c->log_every = c->eval_every = std::max(1, c->steps);
    c->eval_episodes = 1;
    example_runtime::Describe();
  }
}

// A DqnConfig carrying the logging fields, for the shared Monitor.
inline DqnConfig MonitorConfig(const PgConfig& c) {
  DqnConfig d;
  d.log_every = c.log_every;
  d.eval_every = c.eval_every;
  d.eval_episodes = c.eval_episodes;
  return d;
}

// Softmax policy over discrete actions plus (optionally) a state-value head
// on its own trunk. All outputs are per row of `obs` [B, obs_dim].
struct ActorCritic {
  nn::Tensor logits, logp, probs, entropy, value;  // [B,A] [B,A] [B,A] [B] [B]
};
inline ActorCritic BuildActorCritic(nn::ParameterStore& p, const nn::Tensor& obs, int n_actions, int hidden,
                                    bool with_value = true, nn::Activation act = nn::Activation::Tanh) {
  ActorCritic ac;
  const nn::Tensor h = nn::mlp(p, "actor", obs, {hidden, hidden}, act);
  ac.logits = nn::dense(p, "actor/logits", h, n_actions);
  ac.logp = nn::log_softmax(ac.logits);
  ac.probs = nn::exp(ac.logp);
  ac.entropy = -nn::sum(ac.probs * ac.logp, {1});
  if (with_value) {
    const nn::Tensor hv = nn::mlp(p, "critic", obs, {hidden, hidden}, act);
    ac.value = nn::squeeze(nn::dense(p, "critic/value", hv, 1), {1});
  }
  return ac;
}

// log pi(a | s) for the taken actions: [B]
inline nn::Tensor LogProbOf(const nn::Tensor& logp, const nn::Tensor& actions, int n_actions) {
  return nn::sum(logp * nn::one_hot(actions, n_actions), {1});
}

inline int SampleAction(const float* probs, int n, std::mt19937_64& rng) {
  float u = std::uniform_real_distribution<float>(0, 1)(rng), acc = 0;
  for (int a = 0; a < n; ++a) {
    acc += probs[a];
    if (u < acc) return a;
  }
  return n - 1;
}

// Discounted returns G_t = r_t + gamma (1 - done_t) G_{t+1}, bootstrapped
// from `bootstrap` after the last step (0 for a complete episode).
inline std::vector<float> DiscountedReturns(const std::vector<float>& r, const std::vector<float>& done, float gamma,
                                            float bootstrap = 0.f) {
  std::vector<float> g(r.size());
  float next = bootstrap;
  for (int t = static_cast<int>(r.size()) - 1; t >= 0; --t) {
    next = r[t] + gamma * (1 - done[t]) * next;
    g[t] = next;
  }
  return g;
}

// Generalized advantage estimation (Schulman et al. 2016) for one
// trajectory: delta_t = r_t + gamma (1 - done_t) V(s_{t+1}) - V(s_t),
// A_t = delta_t + gamma lambda (1 - done_t) A_{t+1}; returns = A + V.
inline void Gae(const std::vector<float>& r, const std::vector<float>& v, const std::vector<float>& done, float next_v,
                float gamma, float lam, std::vector<float>* adv, std::vector<float>* ret) {
  const int T = static_cast<int>(r.size());
  adv->assign(T, 0.f);
  ret->assign(T, 0.f);
  float next_adv = 0;
  for (int t = T - 1; t >= 0; --t) {
    const float v_next = t + 1 < T ? v[t + 1] : next_v;
    const float delta = r[t] + gamma * (1 - done[t]) * v_next - v[t];
    next_adv = delta + gamma * lam * (1 - done[t]) * next_adv;
    (*adv)[t] = next_adv;
    (*ret)[t] = next_adv + v[t];
  }
}

// Rows of observations as a float [n, obs_dim] tensor.
inline nn::Tensor ObsMatrix(const std::vector<Obs>& rows) {
  const std::int64_t n = static_cast<std::int64_t>(rows.size()), d = n ? static_cast<std::int64_t>(rows[0].size()) : 0;
  std::vector<float> flat;
  flat.reserve(static_cast<std::size_t>(n * d));
  for (const Obs& o : rows) flat.insert(flat.end(), o.begin(), o.end());
  return nn::from_vector(flat, {n, d});
}
// Actions as an int64 [n] tensor.
inline nn::Tensor IntVector(const std::vector<int>& v) {
  return nn::from_vector(std::vector<std::int64_t>(v.begin(), v.end()), {static_cast<std::int64_t>(v.size())});
}
inline nn::Tensor FloatVector(const std::vector<float>& v) {
  return nn::from_vector(v, {static_cast<std::int64_t>(v.size())});
}

// n_steps x n_envs rollouts from a VecEnv, stored env-major so each
// environment's sub-trajectory is contiguous for GAE.
struct Rollout {
  int n_envs = 0, n_steps = 0;
  std::vector<std::vector<Obs>> obs;      // [env][t]
  std::vector<std::vector<int>> act;      // [env][t]
  std::vector<std::vector<float>> rew, done, val, logp;
  std::vector<Obs> last_obs;              // [env], the observation after the last step
  std::vector<float> last_val;            // [env]
  // Flatten in env-major order.
  std::vector<Obs> FlatObs() const {
    std::vector<Obs> o;
    for (const auto& e : obs) o.insert(o.end(), e.begin(), e.end());
    return o;
  }
  std::vector<int> FlatAct() const {
    std::vector<int> a;
    for (const auto& e : act) a.insert(a.end(), e.begin(), e.end());
    return a;
  }
};

// Collects a rollout. `policy(obs_batch, &probs [n_envs * A], &values [n_envs])`
// runs the network; episode returns are reported to the monitor.
class RolloutCollector {
 public:
  RolloutCollector(VecEnv& venv, int n_actions, uint64_t seed)
      : venv_(venv), A_(n_actions), rng_(seed), obs_(venv.reset()), ep_return_(venv.size(), 0.f) {}

  template <typename PolicyFn>
  Rollout Collect(int n_steps, PolicyFn policy, Monitor* monitor) {
    const int n = venv_.size();
    Rollout ro;
    ro.n_envs = n;
    ro.n_steps = n_steps;
    ro.obs.assign(n, {});
    ro.act.assign(n, {});
    ro.rew.assign(n, {});
    ro.done.assign(n, {});
    ro.val.assign(n, {});
    ro.logp.assign(n, {});
    std::vector<float> probs, values;
    std::vector<Action> actions(n);
    for (int t = 0; t < n_steps; ++t) {
      policy(obs_, &probs, &values);
      for (int i = 0; i < n; ++i) {
        const int a = SampleAction(&probs[i * A_], A_, rng_);
        actions[i] = Action(a);
        ro.obs[i].push_back(obs_[i]);
        ro.act[i].push_back(a);
        ro.val[i].push_back(values[i]);
        ro.logp[i].push_back(std::log(std::max(probs[i * A_ + a], 1e-8f)));
      }
      VecStep s = venv_.step(actions);
      for (int i = 0; i < n; ++i) {
        ro.rew[i].push_back(s.reward[i]);
        ro.done[i].push_back(s.terminated[i] ? 1.f : 0.f);  // truncation still bootstraps
        ep_return_[i] += s.reward[i];
        if (s.done(i)) {
          if (monitor) monitor->EndEpisode(ep_return_[i]);
          ep_return_[i] = 0;
        }
        // On truncation the next observation is a fresh episode; bootstrap
        // from the final observation instead would be more exact, but the
        // reset observation is what we have in batch form.
      }
      obs_ = s.obs;
      total_steps_ += n;
    }
    policy(obs_, &probs, &values);
    ro.last_obs = obs_;
    ro.last_val = values;
    return ro;
  }
  long total_steps() const { return total_steps_; }

 private:
  VecEnv& venv_;
  int A_;
  std::mt19937_64 rng_;
  std::vector<Obs> obs_;
  std::vector<float> ep_return_;
  long total_steps_ = 0;
};

}  // namespace rl
