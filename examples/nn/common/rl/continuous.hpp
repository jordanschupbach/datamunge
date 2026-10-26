// datamunge::nn port of tflow_examples src/common/rl/continuous.h.
// Pieces shared by the continuous-control examples: config, deterministic
// and Gaussian actors, tanh-squashed sampling with its log-density
// correction, state-action critics, and deterministic evaluation. Actions
// inside the networks live in [-1, 1]; the environment's bounds are applied
// at the boundary with ScaleAction.
//
// Changes from the TF version: the networks are eager functions of a
// ParameterStore and a tensor (calling one twice with the same store and
// name shares its weights, as nn::Params did), NormalNoise returns an
// nn::Tensor, and the TF hardware options are accepted and ignored except
// threads=N, which sets nn's thread count.
#pragma once
#include "common/runtime.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <datamunge/nn/nn.hpp>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "common/rl/dqn_common.hpp"
#include "common/rl/env.hpp"
#include "common/rl/replay.hpp"

namespace rl {

struct CConfig {
  int steps = 200000;
  int warmup = 1000;       // random actions before training
  int batch = 128;
  int buffer = 100000;
  float gamma = 0.99f;
  float tau = 0.005f;      // Polyak factor for target networks
  float actor_lr = 1e-3f, critic_lr = 1e-3f;
  int hidden = 128;
  float noise = 0.1f;      // exploration noise std (fraction of the action range)
  int log_every = 2000;
  int eval_every = 4000;
  int eval_episodes = 5;
  uint64_t seed = 0;
};

inline void ParseCConfig(int argc, char** argv, CConfig* c) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const size_t eq = arg.find('=');
    if (eq == std::string::npos) continue;
    const std::string key = arg.substr(0, eq);
    const float v = std::strtof(arg.c_str() + eq + 1, nullptr);
    if (key == "threads") {
      datamunge::nn::set_num_threads(static_cast<int>(v));
      continue;
    }
    if (IsHardwareOption(key)) continue;
    if (key == "steps") c->steps = static_cast<int>(v);
    else if (key == "warmup") c->warmup = static_cast<int>(v);
    else if (key == "batch") c->batch = static_cast<int>(v);
    else if (key == "buffer") c->buffer = static_cast<int>(v);
    else if (key == "gamma") c->gamma = v;
    else if (key == "tau") c->tau = v;
    else if (key == "actor_lr") c->actor_lr = v;
    else if (key == "critic_lr") c->critic_lr = v;
    else if (key == "lr") c->actor_lr = c->critic_lr = v;
    else if (key == "hidden") c->hidden = static_cast<int>(v);
    else if (key == "noise") c->noise = v;
    else if (key == "log_every") c->log_every = static_cast<int>(v);
    else if (key == "eval_every") c->eval_every = static_cast<int>(v);
    else if (key == "seed") c->seed = static_cast<uint64_t>(v);
    else if (key != "task" && key != "game") std::fprintf(stderr, "unknown option %s\n", arg.c_str());
  }

  if (!example_runtime::Full()) {
    c->steps = std::min(c->steps, 256);
    c->warmup = std::min(c->warmup, 32);
    c->batch = std::min(c->batch, 8);
    c->log_every = c->eval_every = std::max(1, c->steps);
    c->eval_episodes = 1;
    example_runtime::Describe();
  }
}

inline DqnConfig MonitorConfig(const CConfig& c) {
  DqnConfig d;
  d.log_every = c.log_every;
  d.eval_every = c.eval_every;
  d.eval_episodes = c.eval_episodes;
  return d;
}

// [-1, 1]^dim -> the environment's box.
inline Action ScaleAction(const std::vector<float>& unit, const ActionSpace& space) {
  std::vector<float> a(unit.size());
  for (size_t i = 0; i < a.size(); ++i) {
    const float u = std::max(-1.f, std::min(1.f, unit[i]));
    a[i] = space.low[i] + (u + 1) * 0.5f * (space.high[i] - space.low[i]);
  }
  return Action(a);
}

// Deterministic actor: tanh(MLP(s)) in [-1, 1]^act_dim.
inline nn::Tensor DeterministicActor(nn::ParameterStore& p, const std::string& name, const nn::Tensor& obs, int act_dim,
                                     int hidden) {
  const nn::Tensor h = nn::mlp(p, name + "/mlp", obs, {hidden, hidden});
  return nn::tanh(nn::dense(p, name + "/out", h, act_dim));
}

// Gaussian actor: mean and (clipped) log-std heads on a shared trunk.
struct Gaussian {
  nn::Tensor mean, log_std;  // [B, act_dim] each
};
inline Gaussian GaussianActor(nn::ParameterStore& p, const std::string& name, const nn::Tensor& obs, int act_dim,
                              int hidden, float log_std_min = -5.f, float log_std_max = 2.f) {
  const nn::Tensor h = nn::mlp(p, name + "/mlp", obs, {hidden, hidden});
  Gaussian g;
  g.mean = nn::dense(p, name + "/mean", h, act_dim);
  g.log_std = nn::clip(nn::dense(p, name + "/log_std", h, act_dim), log_std_min, log_std_max);
  return g;
}

// Reparameterised sample u = mean + std * eps, squashed to a = tanh(u), with
//   log pi(a|s) = log N(u; mean, std) - sum_i log(1 - tanh(u_i)^2)
// (Haarnoja et al. 2018, appendix C). `eps` is [B, act_dim] standard normal.
struct Squashed {
  nn::Tensor action, logp;  // [B, act_dim] in [-1, 1], [B]
};
inline Squashed SquashedSample(const Gaussian& g, const nn::Tensor& eps) {
  const nn::Tensor u = g.mean + nn::exp(g.log_std) * eps;
  Squashed out;
  out.action = nn::tanh(u);
  const nn::Tensor gauss_logp = nn::sum(-0.5 * nn::square(eps) - (g.log_std + 0.5 * std::log(2 * 3.14159265)), {1});
  const nn::Tensor correction = nn::sum(nn::log(1.0 - nn::square(out.action) + 1e-6), {1});
  out.logp = gauss_logp - correction;
  return out;
}

// log N(a; mean, exp(log_std)) summed over action dimensions, rowwise: [B].
inline nn::Tensor GaussianLogProb(const nn::Tensor& a, const nn::Tensor& mean, const nn::Tensor& log_std) {
  const nn::Tensor z = (a - mean) / nn::exp(log_std);
  return nn::sum(-0.5 * nn::square(z) - (log_std + 0.5 * std::log(2 * 3.14159265)), {1});
}

// Q(s, a) as an MLP on the concatenated state and action: [B].
inline nn::Tensor QNetwork(nn::ParameterStore& p, const std::string& name, const nn::Tensor& obs, const nn::Tensor& act,
                           int hidden) {
  const nn::Tensor x = nn::concat({obs, act}, 1);
  const nn::Tensor h = nn::mlp(p, name + "/mlp", x, {hidden, hidden});
  return nn::squeeze(nn::dense(p, name + "/out", h, 1), {1});
}

// [rows, cols] standard normal noise, drawn with a fresh distribution per call as in TF.
inline nn::Tensor NormalNoise(int rows, int cols, std::mt19937_64& rng) {
  std::vector<float> v(static_cast<std::size_t>(rows) * cols);
  std::normal_distribution<float> n(0.f, 1.f);
  for (float& x : v) x = n(rng);
  return nn::from_vector(v, {rows, cols});
}

// Evaluation with a deterministic policy returning actions in [-1, 1].
inline float EvaluateC(Env& env, const std::function<std::vector<float>(const Obs&)>& act, int episodes, uint64_t seed) {
  const ActionSpace space = env.action_space();
  double total = 0;
  for (int e = 0; e < episodes; ++e) {
    Obs obs = env.reset(seed + e);
    while (true) {
      StepResult r = env.step(ScaleAction(act(obs), space));
      total += r.reward;
      obs = r.obs;
      if (r.done()) break;
    }
  }
  return static_cast<float>(total / episodes);
}

}  // namespace rl
