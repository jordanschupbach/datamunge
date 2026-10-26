// datamunge::nn port: copied from tflow_examples src/common/rl/bandit.h (TF-free code, unchanged).
// Bandit environments: k-armed (stateless) and contextual.
//
// Both are one-step episodes: reset() gives the (possibly empty) context,
// step(arm) gives the reward and terminates.
#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "common/rl/env.hpp"

namespace rl {

inline std::vector<std::string> ArmNames(int arms) {
  std::vector<std::string> names;
  for (int a = 0; a < arms; ++a) names.push_back("arm " + std::to_string(a));
  return names;
}

class MultiArmedBandit : public EnvBase {
 public:
  enum Kind { kGaussian, kBernoulli };

  // Gaussian arms: reward ~ N(mean[arm], stddev).
  MultiArmedBandit(std::vector<float> means, float stddev)
      : kind_(kGaussian), means_(std::move(means)), stddev_(stddev) {}
  // Bernoulli arms: reward is 1 with probability p[arm], else 0.
  static MultiArmedBandit Bernoulli(std::vector<float> probs) {
    MultiArmedBandit b(std::move(probs), 0.f);
    b.kind_ = kBernoulli;
    return b;
  }
  // Sutton & Barto's 10-armed testbed: means drawn from N(0, 1), unit noise.
  static MultiArmedBandit Testbed(int k, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> n(0.f, 1.f);
    std::vector<float> means(k);
    for (float& m : means) m = n(rng);
    return MultiArmedBandit(means, 1.f);
  }

  std::string name() const override {
    return std::string(kind_ == kGaussian ? "Gaussian" : "Bernoulli") + "Bandit(" +
           std::to_string(num_arms()) + ")";
  }
  int num_arms() const { return static_cast<int>(means_.size()); }
  int obs_size() const override { return 0; }
  ActionSpace action_space() const override { return ActionSpace::Discrete(num_arms()); }
  std::vector<std::string> action_names() const override { return ArmNames(num_arms()); }
  float mean(int arm) const { return means_[arm]; }
  int best_arm() const {
    return static_cast<int>(std::max_element(means_.begin(), means_.end()) - means_.begin());
  }
  float best_mean() const { return means_[best_arm()]; }

  Obs Reset() override {
    BeginEpisode();
    return {};
  }
  StepResult step(const Action& action) override {
    StepResult r;
    r.reward = Pull(action.discrete);
    r.terminated = true;
    return r;
  }
  float Pull(int arm) {
    if (kind_ == kGaussian) return Normal(means_[arm], stddev_);
    return Uniform(0, 1) < means_[arm] ? 1.f : 0.f;
  }

 private:
  Kind kind_;
  std::vector<float> means_;
  float stddev_;
};

// Contextual bandit: a context vector x ~ N(0, I) is drawn each round and
// arm a pays 1 with probability sigmoid(w_a . x + b_a). The weights are
// random but fixed per instance, so the optimal arm depends on the context.
class ContextualBandit : public EnvBase {
 public:
  ContextualBandit(int context_dim, int num_arms, uint64_t seed)
      : dim_(context_dim), arms_(num_arms), w_(num_arms, std::vector<float>(context_dim)),
        b_(num_arms) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> n(0.f, 1.f);
    for (auto& row : w_)
      for (float& v : row) v = n(rng);
    for (float& v : b_) v = 0.5f * n(rng);
  }
  std::string name() const override { return "ContextualBandit"; }
  int obs_size() const override { return dim_; }
  ActionSpace action_space() const override { return ActionSpace::Discrete(arms_); }
  std::vector<std::string> action_names() const override { return ArmNames(arms_); }
  int num_arms() const { return arms_; }

  Obs Reset() override {
    BeginEpisode();
    context_.resize(dim_);
    for (float& v : context_) v = Normal(0, 1);
    return context_;
  }
  const Obs& context() const { return context_; }
  // P(reward = 1 | context, arm): the quantity a learner is trying to find.
  float ExpectedReward(int arm) const {
    float z = b_[arm];
    for (int i = 0; i < dim_; ++i) z += w_[arm][i] * context_[i];
    return 1.f / (1.f + std::exp(-z));
  }
  int BestArm() const {
    int best = 0;
    for (int a = 1; a < arms_; ++a)
      if (ExpectedReward(a) > ExpectedReward(best)) best = a;
    return best;
  }
  StepResult step(const Action& action) override {
    StepResult r;
    r.reward = Uniform(0, 1) < ExpectedReward(action.discrete) ? 1.f : 0.f;
    r.terminated = true;
    return r;
  }

 private:
  int dim_, arms_;
  std::vector<std::vector<float>> w_;
  std::vector<float> b_;
  Obs context_;
};

}  // namespace rl

