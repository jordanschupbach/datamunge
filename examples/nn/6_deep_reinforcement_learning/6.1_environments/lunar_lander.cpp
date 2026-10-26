// CMD: just run-nn lunar_lander
// LunarLander-style 2D physics (simplified).
//
// Port of tflow's lunar_lander. A rigid body with a main engine and two side
// thrusters has to touch down softly on the pad. The physics is a hand-rolled
// point mass with rotation (no Box2D), but the observation, actions and reward
// shaping follow Gymnasium's LunarLander so later agents transfer with little
// change. The environment is common/rl/lunar_lander.hpp (copied from tflow,
// TF-free). The vsim video of the heuristic landing is omitted: the port plays
// the same episode without filming (common/rl/playback.hpp) and adds checks
// (the heuristic lands every episode, random crashes, the replay matches).

#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>

#include "common/checks.hpp"
#include "common/rl/classic_control.hpp"
#include "common/rl/hud.hpp"
#include "common/rl/lunar_lander.hpp"
#include "common/rl/playback.hpp"
#include "common/rl/utils.hpp"

namespace {

// A proportional controller in the spirit of Gymnasium's heuristic: tilt
// toward the pad, cancel horizontal speed, descend slowly, stay upright.
rl::Decision Heuristic(const rl::Obs& o) {
  const float x = o[0], y = o[1], vx = o[2], vy = o[3], angle = o[4], ang_vel = o[5];
  float angle_target = rl::Clamp(0.6f * x + 1.0f * vx, -0.4f, 0.4f);
  float angle_todo = (angle_target - angle) * 2.0f - ang_vel * 1.0f;
  float hover_target = 0.5f * std::abs(x);  // come down once over the pad
  float hover_todo = (hover_target - y) * 0.5f - vy * 1.0f;
  const std::string note = "hover " + rl::HudSigned(hover_todo) + "   angle " + rl::HudSigned(angle_todo);
  if (hover_todo > std::abs(angle_todo) && hover_todo > 0.05f) return rl::Note(rl::Action(2), note);  // main
  if (angle_todo < -0.05f) return rl::Note(rl::Action(3), note);  // right thruster: rotate clockwise
  if (angle_todo > 0.05f) return rl::Note(rl::Action(1), note);   // left thruster: rotate counter-clockwise
  return rl::Note(rl::Action(0), note);
}

struct Outcome {
  int landed = 0, crashed = 0, timeout = 0;
  float mean = 0;
};

Outcome Outcomes(rl::LunarLander& env, int episodes, uint64_t seed,
                 const std::function<rl::Action(const rl::Obs&)>& pi, const char* label) {
  Outcome out;
  float total = 0;
  for (int e = 0; e < episodes; ++e) {
    rl::Obs obs = env.reset(seed + e);
    while (true) {
      rl::StepResult r = env.step(pi(obs));
      total += r.reward;
      obs = r.obs;
      if (r.done()) {
        if (env.state().landed) ++out.landed;
        else if (env.state().crashed) ++out.crashed;
        else ++out.timeout;
        break;
      }
    }
  }
  out.mean = total / episodes;
  std::printf("  %-22s mean return %8.1f   landed %3d  crashed %3d  timed out %3d  (of %d)\n",
              label, out.mean, out.landed, out.crashed, out.timeout, episodes);
  return out;
}

}  // namespace

int main() {
  try {
    rl::LunarLander env;
    std::cout << "== " << env.name()
              << ": obs [x, y, vx, vy, angle, ang_vel, leg_l, leg_r], Discrete(4) = noop/left/main/right\n"
                 "   +100 soft landing on the pad, -100 crash, shaping toward slow+upright+close, fuel cost\n";
    std::mt19937_64 rng(0);
    const Outcome random = Outcomes(env, 100, 0, [&](const rl::Obs&) { return env.action_space().Sample(rng); }, "random");
    const Outcome heuristic = Outcomes(env, 100, 0, Heuristic, "heuristic controller");

    std::cout << "\n-- One heuristic landing\n";
    rl::Obs obs = env.reset(4);
    float ret = 0;
    int steps = 0;
    bool landed = false;
    for (int t = 1;; ++t) {
      rl::StepResult r = env.step(Heuristic(obs));
      ret += r.reward;
      obs = r.obs;
      steps = t;
      if (t % 60 == 0 || r.done()) {
        std::printf("  t=%-4d x=%+.2f y=%.2f vx=%+.2f vy=%+.2f angle=%+.2f  return so far %.1f%s\n",
                    t, obs[0], obs[1], obs[2], obs[3], obs[4], ret,
                    env.state().landed ? "  LANDED" : env.state().crashed ? "  CRASHED" : "");
      }
      if (r.done()) {
        landed = env.state().landed;
        break;
      }
    }
    std::cout << "  the same landing replayed (tflow films it in the 3D world; video omitted):\n";
    const rl::Playback pb = rl::PlayEpisode(env, Heuristic, "lunar_lander_pilot", 4);

    check::Require(heuristic.landed == 100, "the heuristic controller should land in every episode");
    check::Require(random.landed == 0 && random.crashed > 90, "random actions should (almost) always crash");
    check::Require(heuristic.mean > 200.f && heuristic.mean > random.mean + 200.f,
                   "the heuristic should score > 200 on average and far above random");
    check::Require(landed, "the traced episode should end with a landing");
    check::Require(pb.steps == steps && std::abs(pb.ret - ret) < 1e-3f,
                   "replaying seed 4 should reproduce the traced episode exactly");
    std::cout << "\nchecks passed (vsim video omitted)\n";
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
