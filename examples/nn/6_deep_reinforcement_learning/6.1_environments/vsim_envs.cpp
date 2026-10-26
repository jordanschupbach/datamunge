// CMD: just run-nn vsim_envs
// vsim tasks through the binding: InvertedPendulum and Hopper.
//
// Port of tflow's vsim_envs -- PARTIAL. In tflow, rl::VsimEnv loads the
// articulated scenes in data/vsim/ (inverted_pendulum.json, hopper.json),
// steps them with vsim's Jolt-based PhysicsWorld (compiled C++20 library
// tflow_rl_vsim) and films them with vsim's renderer. The vsim engine and its
// renderer are out of scope for the datamunge ports (plan: "RL envs that need
// ALE/vsim binaries"), so neither task can be simulated here and none of
// tflow's statistics (random 10.65 / feedback 1000.0 on InvertedPendulum,
// random 10.11 / zero-torque 136.02 on Hopper) are reproduced.
//
// What remains testable without the engine is the agent-side contract the
// example uses: policies emit unit-box actions in [-1, 1] that ScaleAction
// maps onto the task's action box (InvertedPendulum: cart force / 100 in
// [-3, 3]; Hopper: three joint controls in [-1, 1]), including the linear
// feedback controller's clipping. This program checks that mapping and says
// that the simulation part is skipped; it never prints simulated numbers.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

#include "common/checks.hpp"
#include "common/rl/env.hpp"

namespace {

// tflow's rl::ScaleAction (src/common/rl/continuous.h, a TF-bound header not ported here):
// clip each unit control to [-1, 1] and map it affinely onto [low, high].
rl::Action ScaleAction(const std::vector<float>& unit, const rl::ActionSpace& space) {
  std::vector<float> a(unit.size());
  for (size_t i = 0; i < a.size(); ++i) {
    const float u = std::max(-1.f, std::min(1.f, unit[i]));
    a[i] = space.low[i] + (u + 1) * 0.5f * (space.high[i] - space.low[i]);
  }
  return rl::Action(a);
}

}  // namespace

int main() {
  try {
    std::printf("== InvertedPendulum: obs [x, angle, x_dot, angle_dot], action = cart force / 100 in [-3, 3], +1 per step up to 1000\n");
    std::printf("  SKIPPED: needs vsim's physics engine and scene data/vsim/inverted_pendulum.json (out of scope)\n");
    const rl::ActionSpace pend_space = rl::ActionSpace::Continuous({-3.f}, {3.f});
    // A linear feedback controller: push against the angle and its rate.
    auto feedback = [](const rl::Obs& o) { return std::vector<float>{std::max(-1.f, std::min(1.f, 5.f * o[1] + 1.f * o[3] + 0.2f * o[0] + 0.3f * o[2]))}; };
    std::printf("  feedback controller -> cart action (unit control scaled to %s):\n", pend_space.ToString().c_str());
    bool pend_ok = true;
    for (const rl::Obs& o : std::vector<rl::Obs>{{0, 0, 0, 0}, {0, 0.05f, 0, 0.1f}, {0.1f, -0.02f, -0.2f, 0}, {0, 0.5f, 0, 1.f}}) {
      const float u = feedback(o)[0];
      const float a = ScaleAction(feedback(o), pend_space).continuous[0];
      std::printf("    obs [%+.2f %+.2f %+.2f %+.2f]  u=%+.3f  action=%+.3f\n", o[0], o[1], o[2], o[3], u, a);
      pend_ok &= std::abs(a - 3.f * u) < 1e-5f && std::abs(a) <= 3.f;
    }

    std::printf("\n== Hopper: obs [height, pitch, 3 joint angles, 6 rates], 3 joint torques = gear * u with u in [-1, 1], reward = forward speed + 1 - 1e-3 |u|^2\n");
    std::printf("  SKIPPED: needs vsim's physics engine and scene data/vsim/hopper.json (out of scope)\n");
    const rl::ActionSpace hop_space = rl::ActionSpace::Continuous({-1.f, -1.f, -1.f}, {1.f, 1.f, 1.f});
    const rl::Action zero = ScaleAction({0.f, 0.f, 0.f}, hop_space);
    const rl::Action clipped = ScaleAction({2.f, -5.f, 0.25f}, hop_space);
    std::printf("  zero torques -> [%+.2f %+.2f %+.2f]; unit [+2 -5 +0.25] clipped -> [%+.2f %+.2f %+.2f]\n",
                zero.continuous[0], zero.continuous[1], zero.continuous[2], clipped.continuous[0],
                clipped.continuous[1], clipped.continuous[2]);

    check::Require(pend_ok, "ScaleAction should map the unit control onto the [-3, 3] cart box");
    check::Require(zero.continuous == std::vector<float>({0.f, 0.f, 0.f}), "zero controls should stay zero on the symmetric Hopper box");
    check::Require(clipped.continuous == std::vector<float>({1.f, -1.f, 0.25f}), "out-of-range controls should be clipped to the box");
    std::cout << "\nchecks passed (simulation and vsim videos skipped: no vsim engine)\n";
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
