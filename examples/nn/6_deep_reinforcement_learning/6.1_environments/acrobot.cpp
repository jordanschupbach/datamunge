// CMD: just run-nn acrobot
// Acrobot: a two-link underactuated swing-up.
//
// Port of tflow's acrobot. Gymnasium's Acrobot-v1, including its RK4
// integrator. Only the elbow is actuated, with torque -1, 0 or +1; the goal is
// to swing the tip above a height of one link length. Reward is -1 per step,
// so shorter is better. The environment is common/rl/classic_control.hpp
// (copied from tflow, TF-free). The vsim video of the pump controller is
// omitted; the port adds checks that the pump reaches the goal in every
// episode and random actions in none.

#include <cmath>
#include <cstdio>
#include <iostream>

#include "common/checks.hpp"
#include "common/rl/classic_control.hpp"
#include "common/rl/hud.hpp"
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

// Pump energy with the elbow torque opposing the first link's swing: the
// reaction on the first link then pushes along its motion, so each swing
// gains amplitude. Crude, but it reaches the goal in about 80 steps where
// random actions never do.
rl::Decision Pump(const rl::Obs& o) {
  const float dtheta1 = o[4];
  return rl::Note(rl::Action(dtheta1 > 0 ? 0 : (dtheta1 < 0 ? 2 : 1)), "theta1_dot " + rl::HudSigned(dtheta1) + ": torque against it");
}

}  // namespace

int main() {
  try {
    rl::Acrobot env;
    std::cout << "== " << env.name()
              << ": obs [cos t1, sin t1, cos t2, sin t2, t1_dot, t2_dot], Discrete(3) torque -1/0/+1, "
                 "-1 per step, goal: tip height > 1, max_steps="
              << env.max_steps() << "\n";
    const rl::EpisodeStats random = rl::RunRandom(env, 30, 0);
    rl::PrintStats("random, 30 episodes", random);
    const rl::EpisodeStats pump = rl::RunEpisodes(env, 30, 0, Pump);
    rl::PrintStats("pump (torque against t1_dot)", pump);

    std::cout << "\n-- Tip height over one episode under the pump controller\n";
    rl::Obs obs = env.reset(1);
    bool goal = false;
    for (int t = 1; t <= env.max_steps(); ++t) {
      rl::StepResult r = env.step(Pump(obs));
      obs = r.obs;
      if (t % 20 == 0 || r.done()) {
        std::printf("  t=%-3d height=%+.2f  t1=%+.2f t2=%+.2f%s\n", t, env.tip_height(),
                    std::atan2(obs[1], obs[0]), std::atan2(obs[3], obs[2]),
                    r.terminated ? "  GOAL" : "");
      }
      if (r.terminated) goal = env.tip_height() > 1.f;
      if (r.done()) break;
    }

    check::Require(pump.min_return() > -static_cast<float>(env.max_steps()) && pump.mean_length() < 150.f,
                   "the pump controller should reach the goal quickly in every episode");
    check::Require(pump.mean_length() < random.mean_length(), "the pump should beat random actions");
    check::Require(goal, "the traced pump episode should end at the goal with the tip above 1");
    std::cout << "\nchecks passed (vsim video omitted)\n";
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
