// CMD: just run-nn mountain_car
// MountainCar, discrete and continuous.
//
// Port of tflow's mountain_car. The car's engine is too weak to climb
// directly, so it has to build momentum. Random actions never reach the flag
// within the step limit; a one-line energy-pumping controller (accelerate in
// the direction you are already moving) does it in about 100 steps. The
// environments are common/rl/classic_control.hpp (copied from tflow,
// TF-free). The vsim videos are omitted: the filmed episodes are played
// without a video (common/rl/playback.hpp) and checks are added.

#include <algorithm>
#include <cstdio>
#include <functional>
#include <iostream>
#include <string>

#include "common/checks.hpp"
#include "common/rl/classic_control.hpp"
#include "common/rl/hud.hpp"
#include "common/rl/playback.hpp"
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

// A one-line "chart" of the car's position over an episode.
std::string PositionTrace(rl::MountainCar& env, const std::function<rl::Action(const rl::Obs&)>& pi,
                          uint64_t seed) {
  std::string trace;
  rl::Obs obs = env.reset(seed);
  for (int t = 0; t < 400; ++t) {
    rl::StepResult r = env.step(pi(obs));
    obs = r.obs;
    if (t % 5 == 0) trace += "._-=^"[std::min(4, std::max(0, static_cast<int>((obs[0] + 1.2f) / 1.8f * 5)))];
    if (r.done()) break;
  }
  return trace;
}

}  // namespace

int main() {
  try {
    std::cout << "== MountainCar-v0: obs [position, velocity], Discrete(3) = push left/none/right, "
                 "-1 per step, flag at x=0.5, 200-step limit\n";
    rl::MountainCar env;
    const rl::EpisodeStats random = rl::RunRandom(env, 50, 0);
    rl::PrintStats("random, 50 episodes", random);
    auto pump = [](const rl::Obs& o) { return rl::Note(rl::Action(o[1] >= 0 ? 2 : 0), "velocity " + rl::HudSigned(o[1]) + ": push along it"); };
    const rl::EpisodeStats pumped = rl::RunEpisodes(env, 50, 0, pump);
    rl::PrintStats("pump energy (push along velocity)", pumped);
    std::cout << "  position over time, low ('.') to high ('^'):\n";
    std::mt19937_64 rng(0);
    std::cout << "    random: " << PositionTrace(env, [&](const rl::Obs&) { return env.action_space().Sample(rng); }, 1) << "\n";
    const std::string pump_trace = PositionTrace(env, pump, 1);
    std::cout << "    pump:   " << pump_trace << "\n";
    std::cout << "  the pump controller on the hill (tflow films it in 3D; video omitted):\n";
    const rl::Playback pb = rl::PlayEpisode(env, pump, "mountain_car_pump", 1);

    std::cout << "\n== MountainCarContinuous-v0: action in [-1, 1], reward -0.1 a^2 per step, "
                 "+100 at x=0.45, 999-step limit\n";
    rl::MountainCarContinuous cenv;
    const rl::EpisodeStats crandom = rl::RunRandom(cenv, 20, 0);
    rl::PrintStats("random, 20 episodes", crandom);
    auto cpump = [](const rl::Obs& o) { return rl::Action({o[1] >= 0 ? 1.f : -1.f}); };
    const rl::EpisodeStats full = rl::RunEpisodes(cenv, 20, 0, cpump);
    rl::PrintStats("pump energy (full throttle)", full);
    // The continuous task rewards fuel economy: half throttle still makes it
    // and pays a quarter of the action cost.
    auto gentle = [](const rl::Obs& o) { return rl::Note(rl::Action({o[1] >= 0 ? 0.5f : -0.5f}), "half throttle along the velocity"); };
    const rl::EpisodeStats half = rl::RunEpisodes(cenv, 20, 0, gentle);
    rl::PrintStats("pump energy (half throttle)", half);
    std::cout << "    pump:   " << PositionTrace(cenv, cpump, 1) << "\n";
    std::cout << "  half throttle (video omitted):\n";
    const rl::Playback cpb = rl::PlayEpisode(cenv, gentle, "mountain_car_continuous_half_throttle", 1);

    check::Require(random.max_return() == -200.f, "random actions should never reach the flag in 200 steps");
    check::Require(pumped.min_return() > -200.f && pumped.mean_length() < 150.f,
                   "the pump controller should reach the flag in every episode");
    check::Require(!pump_trace.empty() && pump_trace.back() == '^', "the pump trace should end at the top");
    check::Require(pb.steps < 200 && pb.ret == -static_cast<float>(pb.steps), "the played pump episode should reach the flag");
    check::Require(crandom.max_return() < 0.f && crandom.mean_length() == 999.f,
                   "random throttle should time out in the continuous task");
    check::Require(full.min_return() > 80.f && half.min_return() > 80.f, "both pump throttles should reach the flag");
    check::Require(half.mean_return() > full.mean_return(), "half throttle should pay less fuel than full throttle");
    check::Require(cpb.ret > 90.f, "the played half-throttle episode should reach the flag");
    std::cout << "\nchecks passed (vsim videos omitted)\n";
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
