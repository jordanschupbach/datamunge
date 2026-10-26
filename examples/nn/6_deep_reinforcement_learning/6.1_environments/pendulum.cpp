// CMD: just run-nn pendulum
// Pendulum: continuous-control swing-up.
//
// Port of tflow's pendulum. Gymnasium's Pendulum-v1: torque in [-2, 2] is
// not enough to lift the pendulum directly, so a controller must pump energy
// by pushing along the current rotation, then catch it at the top. Reward is
// minus a quadratic cost, so the best possible episode return is 0 and a
// random policy scores around -1200. The environment is
// common/rl/classic_control.hpp (copied from tflow, TF-free). The vsim video is
// omitted: the filmed episode is played without a video
// (common/rl/playback.hpp) and checks are added.

#include <cmath>
#include <cstdio>
#include <iostream>

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

// Energy-based swing-up plus a PD catch near the top. With g=10, m=l=1 and
// the rod's inertia ml^2/3, the energy relative to the top is
//   E = theta_dot^2 / 6 + 5 (cos theta - 1)        (0 at the top, -10 at rest)
rl::Decision SwingUp(const rl::Obs& o) {
  const float cos_t = o[0], sin_t = o[1], thdot = o[2];
  const float theta = std::atan2(sin_t, cos_t);
  if (std::abs(theta) < 0.5f && std::abs(thdot) < 3.f) {
    return rl::Note(rl::Action({rl::Clamp(-8.f * theta - 2.f * thdot, -2.f, 2.f)}), "PD regulator near the top");
  }
  const float energy = thdot * thdot / 6.f + 5.f * (cos_t - 1.f);
  const float dir = thdot >= 0 ? 1.f : -1.f;
  return rl::Note(rl::Action({energy < 0 ? 2.f * dir : -2.f * dir}), "energy " + rl::HudSigned(energy) + (energy < 0 ? ": pump along the swing" : ": brake"));
}

}  // namespace

int main() {
  try {
    rl::Pendulum env;
    std::cout << "== " << env.name() << ": obs [cos theta, sin theta, theta_dot], "
              << env.action_space().ToString() << ", 200 steps, reward = -(theta^2 + 0.1 theta_dot^2 + 0.001 u^2)\n";
    const rl::EpisodeStats random = rl::RunRandom(env, 50, 0);
    rl::PrintStats("random, 50 episodes", random);
    const rl::EpisodeStats ctrl = rl::RunEpisodes(env, 50, 0, SwingUp);
    rl::PrintStats("energy swing-up + PD", ctrl);

    std::cout << "\n-- One episode under the controller (theta=0 is upright)\n";
    rl::Obs obs = env.reset(2);
    float ret = 0;
    for (int t = 0; t < 200; ++t) {
      rl::StepResult r = env.step(SwingUp(obs));
      ret += r.reward;
      obs = r.obs;
      if (t % 25 == 0 || t == 199) {
        std::printf("  t=%-3d theta=%+.2f theta_dot=%+.2f reward=%+.3f\n", t,
                    std::atan2(obs[1], obs[0]), obs[2], r.reward);
      }
    }
    std::printf("  return %.1f\n", ret);
    const float final_theta = std::atan2(obs[1], obs[0]);
    std::cout << "\n-- The same episode replayed (tflow films it in the pendulum's 3D world; video omitted)\n";
    const rl::Playback pb = rl::PlayEpisode(env, SwingUp, "pendulum_energy_swing_up", 2);

    check::Require(random.mean_return() < -900.f, "a random policy should score far below zero (about -1200)");
    check::Require(ctrl.mean_return() > -300.f && ctrl.max_return() > -1.f,
                   "the swing-up controller should average above -300 and balance perfectly from some starts");
    check::Require(std::abs(final_theta) < 0.01f, "the traced episode should end upright");
    check::Require(pb.steps == 200 && std::abs(pb.ret - ret) < 1e-3f, "the replay should reproduce the traced episode");
    std::cout << "\nchecks passed (vsim video omitted)\n";
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
