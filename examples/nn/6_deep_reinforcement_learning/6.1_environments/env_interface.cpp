// CMD: just run-nn env_interface
// Environment interface: reset, step, render, seeding.
//
// Port of tflow's env_interface. All RL examples talk to environments through
// rl::Env (common/rl/env.hpp, copied from tflow), a small Gymnasium-style
// interface:
//   reset()            -> first observation (optionally with a seed)
//   step(action)       -> {obs, reward, terminated, truncated}
//   render()           -> text picture of the state
//   action_space()     -> Discrete(n) or Continuous(low, high)
// This program writes a tiny environment against that interface and then
// drives it, and a built-in one, through the same generic loop. The vsim video
// of a random CartPole episode is omitted; the port adds checks that seeding
// reproduces an episode and that every corridor episode ends one of the three
// ways.

#include <cstdio>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "common/checks.hpp"
#include "common/rl/classic_control.hpp"
#include "common/rl/env.hpp"
#include "common/rl/utils.hpp"

// Experiment workflow from examples/nn/common/rl/utils.hpp.
namespace rl {
using namespace ::rl;
inline EpisodeStats RunEpisodes(Env& env, int episodes, uint64_t seed,
                                const std::function<Action(const Obs&)>& policy) {
  EpisodeStats stats;
  for (int e = 0; e < episodes; ++e) {
    Obs obs = env.reset(seed + e);
    float ret = 0;
    int len = 0;
    while (true) {
      StepResult r = env.step(policy(obs));
      ret += r.reward;
      ++len;
      obs = r.obs;
      if (r.done()) break;
    }
    stats.returns.push_back(ret);
    stats.lengths.push_back(len);
  }
  return stats;
}
}  // namespace rl

// Experiment workflow from examples/nn/common/rl/utils.hpp.
namespace rl {
using namespace ::rl;
inline EpisodeStats RunRandom(Env& env, int episodes, uint64_t seed) {
  std::mt19937_64 rng(seed);
  const ActionSpace space = env.action_space();
  return RunEpisodes(env, episodes, seed, [&](const Obs&) { return space.Sample(rng); });
}
}  // namespace rl



namespace {

// A 1-D corridor: the agent starts in the middle of `length` cells and must
// reach the right end (+1) while avoiding the left end (-1). Each step the
// chosen move succeeds with probability 0.8, otherwise the agent slips the
// other way, so the environment needs its RNG and seeding matters.
class Corridor : public rl::EnvBase {
 public:
  explicit Corridor(int length = 7, int max_steps = 20) : rl::EnvBase(max_steps), length_(length) {}
  std::string name() const override { return "Corridor"; }
  int obs_size() const override { return 1; }
  rl::ActionSpace action_space() const override { return rl::ActionSpace::Discrete(2); }

  rl::Obs Reset() override {
    BeginEpisode();  // resets the step counter used for truncation
    pos_ = length_ / 2;
    return {static_cast<float>(pos_)};
  }
  rl::StepResult step(const rl::Action& action) override {
    int dir = action.discrete == 1 ? 1 : -1;
    if (Uniform(0, 1) > 0.8f) dir = -dir;  // slip
    pos_ += dir;
    rl::StepResult r;
    r.obs = {static_cast<float>(pos_)};
    if (pos_ == length_ - 1) {
      r.reward = 1.f;
      r.terminated = true;
    } else if (pos_ == 0) {
      r.reward = -1.f;
      r.terminated = true;
    } else {
      r.truncated = CountStep();  // time limit: not a terminal state
    }
    return r;
  }
  std::string render() const override {
    std::string s(length_, '.');
    s[0] = 'x';
    s[length_ - 1] = 'G';
    s[pos_] = 'A';
    return s;
  }

 private:
  int length_;
  int pos_ = 0;
};

// One episode through the generic interface, printing every step. Returns the
// sequence of observations and rewards as text (for the seeding check).
std::string Trace(rl::Env& env, uint64_t seed, int max_print, bool show_render = true) {
  std::ostringstream log;
  std::mt19937_64 rng(seed);
  const rl::ActionSpace space = env.action_space();
  rl::Obs obs = env.reset(seed);
  std::cout << "  t=0  obs=" << rl::FormatObs(obs) << "  " << (show_render ? env.render() : "") << "\n";
  log << rl::FormatObs(obs);
  for (int t = 1;; ++t) {
    rl::Action a = space.Sample(rng);
    rl::StepResult r = env.step(a);
    log << " " << a.discrete << ":" << r.reward << ":" << rl::FormatObs(r.obs);
    if (t <= max_print)
      std::printf("  t=%-2d a=%d  r=%+.2f  obs=%s  %s%s%s\n", t, a.discrete, r.reward,
                  rl::FormatObs(r.obs).c_str(), show_render ? env.render().c_str() : "",
                  r.terminated ? "  TERMINATED" : "", r.truncated ? "  TRUNCATED" : "");
    if (r.done()) {
      if (t > max_print) std::printf("  ... episode ended at t=%d\n", t);
      break;
    }
  }
  return log.str();
}

}  // namespace

int main() {
  try {
    Corridor corridor;
    std::cout << "== " << corridor.name() << ": obs_size=" << corridor.obs_size()
              << " action_space=" << corridor.action_space().ToString()
              << " max_steps=" << corridor.max_steps() << "\n";
    std::cout << "\n-- A random episode (seed 1)\n";
    Trace(corridor, 1, 25);

    std::cout << "\n-- Seeding: the same seed reproduces the episode exactly\n";
    const std::string a = Trace(corridor, 7, 4);
    const std::string b = Trace(corridor, 7, 4);
    std::cout << "-- A different seed does not\n";
    const std::string c = Trace(corridor, 8, 4);
    check::Require(a == b, "the same seed should reproduce the episode");
    check::Require(a != c, "a different seed should give a different episode");

    std::cout << "\n-- terminated vs truncated over 1000 random episodes\n";
    {
      std::mt19937_64 rng(0);
      int goal = 0, pit = 0, timeout = 0;
      for (int e = 0; e < 1000; ++e) {
        corridor.reset(100 + e);
        while (true) {
          rl::StepResult r = corridor.step(corridor.action_space().Sample(rng));
          if (r.terminated) (r.reward > 0 ? goal : pit)++;
          if (r.truncated) ++timeout;
          if (r.done()) break;
        }
      }
      std::printf("  reached goal %d, fell in pit %d, hit the %d-step limit %d\n", goal, pit,
                  corridor.max_steps(), timeout);
      check::Require(goal + pit + timeout == 1000, "every episode ends exactly one way");
      check::Require(goal > 300 && pit > 300 && timeout > 0, "a symmetric random walk reaches both ends about equally often");
    }

    std::cout << "\n-- The same loop works on any Env: CartPole through the base class\n";
    std::unique_ptr<rl::Env> env = std::make_unique<rl::CartPole>();
    std::cout << "  " << env->name() << ": obs_size=" << env->obs_size()
              << " action_space=" << env->action_space().ToString() << "\n";
    Trace(*env, 3, 3, /*show_render=*/false);
    rl::PrintStats("random policy, 20 episodes", rl::RunRandom(*env, 20, 0));

    std::cout << "\n-- Continuous action spaces sample vectors\n";
    rl::Pendulum pendulum;
    std::mt19937_64 rng(0);
    const rl::Action sample = pendulum.action_space().Sample(rng);
    std::cout << "  " << pendulum.name() << " action_space=" << pendulum.action_space().ToString()
              << "  sample=" << rl::FormatObs(sample.continuous) << "\n";
    check::Require(sample.continuous.size() == 1 && sample.continuous[0] >= -2.f && sample.continuous[0] <= 2.f,
                   "a Pendulum action is one torque in [-2, 2]");
    std::cout << "\nchecks passed (vsim video omitted)\n";
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
