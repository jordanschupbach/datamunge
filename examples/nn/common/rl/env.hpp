// datamunge::nn port: copied from tflow_examples src/common/rl/env.h (TF-free code, unchanged).
// Environment interface shared by the reinforcement learning examples.
//
// Modelled on the Gymnasium API: reset() returns the first observation,
// step() returns the next observation, a reward, and two end-of-episode
// flags: `terminated` (the MDP reached a terminal state) and `truncated`
// (a time limit cut the episode short; bootstrapping is still valid).
#pragma once

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace rl {

using Obs = std::vector<float>;

// Either a discrete action (an index) or a continuous one (a vector).
struct Action {
  int discrete = -1;
  std::vector<float> continuous;
  Action() = default;
  Action(int a) : discrete(a) {}                                      // NOLINT: implicit by design
  Action(std::vector<float> a) : continuous(std::move(a)) {}          // NOLINT: implicit by design
  Action(std::initializer_list<float> a) : continuous(a) {}           // NOLINT: implicit by design
};

struct ActionSpace {
  enum Kind { kDiscrete, kContinuous };
  Kind kind = kDiscrete;
  int n = 0;    // number of actions (discrete)
  int dim = 0;  // action vector length (continuous)
  std::vector<float> low, high;  // per-dimension bounds (continuous)

  static ActionSpace Discrete(int n) {
    ActionSpace s;
    s.kind = kDiscrete;
    s.n = n;
    return s;
  }
  static ActionSpace Continuous(std::vector<float> low, std::vector<float> high) {
    ActionSpace s;
    s.kind = kContinuous;
    s.dim = static_cast<int>(low.size());
    s.low = std::move(low);
    s.high = std::move(high);
    return s;
  }
  // A uniformly random action from this space.
  Action Sample(std::mt19937_64& rng) const {
    if (kind == kDiscrete) return Action(static_cast<int>(rng() % n));
    std::vector<float> a(dim);
    for (int i = 0; i < dim; ++i)
      a[i] = std::uniform_real_distribution<float>(low[i], high[i])(rng);
    return Action(a);
  }
  std::string ToString() const {
    if (kind == kDiscrete) return "Discrete(" + std::to_string(n) + ")";
    std::string s = "Box(" + std::to_string(dim) + ") [";
    for (int i = 0; i < dim; ++i)
      s += (i ? ", " : "") + std::to_string(low[i]) + ".." + std::to_string(high[i]);
    return s + "]";
  }
};

struct StepResult {
  Obs obs;
  float reward = 0.f;
  bool terminated = false;
  bool truncated = false;
  bool done() const { return terminated || truncated; }
};

class Env {
 public:
  virtual ~Env() = default;
  virtual std::string name() const = 0;
  virtual int obs_size() const = 0;
  virtual ActionSpace action_space() const = 0;
  // Start a new episode. The seeded overload reseeds the environment's RNG
  // first, which makes the whole episode reproducible. Environments implement
  // the virtual Reset(); the public reset() wrappers are non-virtual so that
  // overriding does not hide the seeded overload.
  Obs reset() { return Reset(); }
  Obs reset(uint64_t seed) {
    Seed(seed);
    return Reset();
  }
  virtual StepResult step(const Action& action) = 0;
  // Human-readable picture of the current state (optional).
  virtual std::string render() const { return ""; }
  // A name per discrete action, or per dimension of a continuous one
  // (optional; the video overlay labels the moves with them).
  virtual std::vector<std::string> action_names() const { return {}; }
  virtual void Seed(uint64_t seed) = 0;
  // 0 means no time limit.
  virtual int max_steps() const { return 0; }

 protected:
  virtual Obs Reset() = 0;
};

// Common plumbing: an RNG, a step counter and time-limit truncation.
class EnvBase : public Env {
 public:
  explicit EnvBase(int max_steps = 0) : max_steps_(max_steps) {}
  void Seed(uint64_t seed) override { rng_.seed(seed); }
  int max_steps() const override { return max_steps_; }
  int steps() const { return steps_; }
  std::mt19937_64& rng() { return rng_; }

 protected:
  // Call at the start of reset()/step() implementations.
  void BeginEpisode() { steps_ = 0; }
  // Returns true if the time limit is reached after this step.
  bool CountStep() { return max_steps_ > 0 && ++steps_ >= max_steps_; }
  float Uniform(float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(rng_); }
  float Normal(float mean, float stddev) {
    return std::normal_distribution<float>(mean, stddev)(rng_);
  }

 private:
  std::mt19937_64 rng_{0};
  int max_steps_ = 0;
  int steps_ = 0;
};

// Outcome of taking action a in state s, for environments that expose their
// model (needed by dynamic programming).
struct Transition {
  float prob;
  int next_state;
  float reward;
  bool terminal;
};

// A finite MDP: states are indices, observations are the index as a float.
class TabularEnv : public EnvBase {
 public:
  using EnvBase::EnvBase;
  virtual int num_states() const = 0;
  virtual int state() const = 0;
  // The full transition model p(s', r | s, a), for planning methods.
  virtual std::vector<Transition> transitions(int s, int a) const = 0;
  int obs_size() const override { return 1; }
  Obs StateObs() const { return {static_cast<float>(state())}; }
};

}  // namespace rl

