// datamunge::nn port of tflow_examples src/common/rl/dqn_common.h.
// Pieces shared by the DQN-family examples: hyper-parameters, the epsilon
// schedule, a training monitor, Q-value heads (plain / dueling), greedy
// evaluation, the C51 distribution projection, and Adam with global-norm
// clipping (tflow's nn::Adam).
//
// Changes from the TF version: the TF hardware options (device=, threads=,
// xla=) are accepted and ignored, since nn runs eagerly on the CPU; QHead
// builds on a ParameterStore and returns a tensor instead of adding graph
// nodes; the noisy-network head (nn::NoisyDense) is not included, because
// datamunge::nn has no noisy layer yet.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <datamunge/nn/nn.hpp>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

#include "common/rl/env.hpp"
#include "common/rl/utils.hpp"

namespace rl {

namespace nn = datamunge::nn;

struct DqnConfig {
  int steps = 50000;          // environment steps
  int warmup = 1000;          // steps before training starts
  int batch = 64;
  int buffer = 50000;
  float gamma = 0.98f;        // 0.99 is the usual choice; 0.98 bounds Q by 50 and makes CartPole far more stable
  float lr = 5e-4f;
  float clip_norm = 10.f;
  int target_sync = 500;      // hard target update period (steps), used when tau == 1
  float tau = 0.005f;         // target <- tau * online + (1 - tau) * target every step; tau = 1 means a hard copy every target_sync steps
  float eps_start = 1.f, eps_end = 0.05f;
  int eps_decay_steps = 10000;
  int hidden = 128;
  int train_every = 2;
  int log_every = 5000;
  int eval_every = 5000;
  int eval_episodes = 5;
  uint64_t seed = 0;
};

// TF-only options that the ports accept and ignore.
inline bool IsHardwareOption(const std::string& key) {
  return key == "device" || key == "threads" || key == "intra_threads" || key == "inter_threads" || key == "xla";
}

// Overrides from the command line, e.g. `just run-nn dqn steps=50000 lr=5e-4`.
inline void ParseConfig(int argc, char** argv, DqnConfig* c) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const size_t eq = arg.find('=');
    if (eq == std::string::npos) continue;
    const std::string key = arg.substr(0, eq);
    const float v = std::strtof(arg.c_str() + eq + 1, nullptr);
    if (IsHardwareOption(key)) continue;
    if (key == "steps") c->steps = static_cast<int>(v);
    else if (key == "warmup") c->warmup = static_cast<int>(v);
    else if (key == "batch") c->batch = static_cast<int>(v);
    else if (key == "buffer") c->buffer = static_cast<int>(v);
    else if (key == "gamma") c->gamma = v;
    else if (key == "lr") c->lr = v;
    else if (key == "clip_norm") c->clip_norm = v;
    else if (key == "target_sync") c->target_sync = static_cast<int>(v);
    else if (key == "tau") c->tau = v;
    else if (key == "eps_start") c->eps_start = v;
    else if (key == "eps_end") c->eps_end = v;
    else if (key == "eps_decay_steps") c->eps_decay_steps = static_cast<int>(v);
    else if (key == "hidden") c->hidden = static_cast<int>(v);
    else if (key == "train_every") c->train_every = static_cast<int>(v);
    else if (key == "log_every") c->log_every = static_cast<int>(v);
    else if (key == "eval_every") c->eval_every = static_cast<int>(v);
    else if (key == "seed") c->seed = static_cast<uint64_t>(v);
    else if (key != "task" && key != "game") std::fprintf(stderr, "unknown option %s\n", arg.c_str());
  }
}

inline float Epsilon(const DqnConfig& c, int step) {
  const float frac = std::min(1.f, static_cast<float>(step) / c.eps_decay_steps);
  return c.eps_start + frac * (c.eps_end - c.eps_start);
}

// Greedy evaluation with any action function.
inline float Evaluate(Env& env, const std::function<int(const Obs&)>& act, int episodes, uint64_t seed) {
  double total = 0;
  for (int e = 0; e < episodes; ++e) {
    Obs obs = env.reset(seed + e);
    while (true) {
      StepResult r = env.step(act(obs));
      total += r.reward;
      obs = r.obs;
      if (r.done()) break;
    }
  }
  return static_cast<float>(total / episodes);
}

// Tracks episode returns and losses and prints one line per log period.
class Monitor {
 public:
  explicit Monitor(const DqnConfig& cfg, const char* third_column = "eps") : cfg_(cfg) {
    std::printf("  %8s %8s %12s %7s %10s %10s\n", "step", "episodes", "return(20)", third_column, "loss", "eval");
  }
  void EndEpisode(float ret) { returns_.push_back(ret); }
  void AddLoss(float loss) { losses_.push_back(loss); }
  int episodes() const { return static_cast<int>(returns_.size()); }
  float RecentReturn(int n = 20) const {
    if (returns_.empty()) return 0.f;
    const size_t from = returns_.size() > static_cast<size_t>(n) ? returns_.size() - n : 0;
    return std::accumulate(returns_.begin() + from, returns_.end(), 0.f) / (returns_.size() - from);
  }
  void Log(int step, float eps, float eval_return) {
    const float loss = losses_.empty() ? 0.f : std::accumulate(losses_.begin(), losses_.end(), 0.f) / losses_.size();
    losses_.clear();
    char eval[16];
    if (std::isnan(eval_return)) std::snprintf(eval, sizeof eval, "%10s", "");
    else std::snprintf(eval, sizeof eval, "%10.1f", eval_return);
    std::printf("  %8d %8d %12.1f %7.3f %10.4f %s\n", step, episodes(), RecentReturn(), eps, loss, eval);
    std::fflush(stdout);
  }

 private:
  DqnConfig cfg_;
  std::vector<float> returns_, losses_;
};

// Q-value head on top of features h [B, in]. Dueling splits into a state
// value and advantages (Wang et al. 2016).
inline nn::Tensor QHead(nn::ParameterStore& p, const std::string& name, const nn::Tensor& h, int n_actions,
                        bool dueling) {
  if (!dueling) return nn::dense(p, name + "/q", h, n_actions);
  const nn::Tensor value = nn::dense(p, name + "/value", h, 1);              // [B, 1]
  const nn::Tensor adv = nn::dense(p, name + "/advantage", h, n_actions);   // [B, A]
  return value + (adv - nn::mean(adv, {1}, /*keepdims=*/true));
}

// C51 projection (Bellemare et al. 2017): the target distribution
// r + gamma (1 - done) z, with z the fixed support, is redistributed onto
// the support's atoms by linear interpolation. probs_next is [B, atoms] for
// the chosen next actions; out receives [B, atoms].
inline void ProjectDistribution(const float* probs_next, const float* r, const float* done, int B, int atoms,
                                float vmin, float vmax, float gamma, float* out) {
  const float dz = (vmax - vmin) / (atoms - 1);
  std::fill(out, out + B * atoms, 0.f);
  for (int i = 0; i < B; ++i)
    for (int j = 0; j < atoms; ++j) {
      const float z = vmin + j * dz;
      const float tz = std::min(vmax, std::max(vmin, r[i] + gamma * (1 - done[i]) * z));
      const float b = (tz - vmin) / dz;
      const int l = static_cast<int>(std::floor(b)), u = static_cast<int>(std::ceil(b));
      const float p = probs_next[i * atoms + j];
      if (l == u) {
        out[i * atoms + l] += p;
      } else {
        out[i * atoms + l] += p * (u - b);
        out[i * atoms + u] += p * (b - l);
      }
    }
}

// Adam over every parameter of a store with optional clipping by global
// norm (tflow's nn::Adam(scope, params, loss, lr, clip_norm)). Build the
// network once before constructing it, so the store holds its parameters.
class Adam {
 public:
  Adam(const nn::ParameterStore& store, double lr, double clip_norm = 0.0)
      : opt_(store, lr, {nn::OptimizerKind::Adam}), clip_(clip_norm) {}
  // One update from the gradients of `loss`; returns the global gradient
  // norm before clipping (tflow's adam.global_norm).
  double Minimize(const nn::Tensor& loss) {
    std::vector<nn::Tensor> grads = nn::grad(loss, opt_.parameters());
    double norm;
    if (clip_ > 0) {
      auto clipped = nn::clip_by_global_norm(grads, clip_);
      grads = std::move(clipped.first);
      norm = clipped.second;
    } else {
      norm = nn::global_norm(grads);
    }
    opt_.step(grads);
    return norm;
  }
  nn::Optimizer& optimizer() { return opt_; }

 private:
  nn::Optimizer opt_;
  double clip_;
};

// One observation as a [1, obs_dim] tensor.
inline nn::Tensor Row(const Obs& o) { return nn::from_vector(o, {1, static_cast<std::int64_t>(o.size())}); }

}  // namespace rl
