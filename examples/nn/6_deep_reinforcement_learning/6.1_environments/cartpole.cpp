// CMD: just run-nn cartpole
// CartPole: the balancing benchmark.
//
// Port of tflow's cartpole. Same physics and constants as Gymnasium's
// CartPole-v1 (common/rl/classic_control.hpp, copied from tflow). A random
// policy lasts about 20 steps; a two-line hand-written controller holds the
// pole for the full 500, which is the ceiling later learning algorithms aim
// for. The vsim videos of both policies are omitted; the port adds checks on
// the returns and on the always-right failure.

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



int main() {
  try {
    rl::CartPole env;
    std::cout << "== " << env.name() << ": obs [x, x_dot, theta, theta_dot], "
              << env.action_space().ToString() << " (0 push left, 1 push right), reward +1/step, "
              << "max_steps=" << env.max_steps() << "\n";

    std::cout << "\n-- Random policy\n";
    const rl::EpisodeStats random = rl::RunRandom(env, 100, 0);
    rl::PrintStats("random, 100 episodes", random);

    // Push toward the side the pole is falling to; the angular velocity term
    // adds damping. Good enough to balance indefinitely from most starts.
    auto controller = [](const rl::Obs& o) {
      const float lean = o[2] + 0.5f * o[3];  // positive: the pole is falling to the right
      return rl::Note(rl::Action(lean > 0 ? 1 : 0), "theta + 0.5 theta_dot = " + rl::HudSigned(lean));
    };
    std::cout << "\n-- Hand-written controller: action = (theta + 0.5 theta_dot > 0)\n";
    const rl::EpisodeStats ctrl = rl::RunEpisodes(env, 100, 0, controller);
    rl::PrintStats("controller, 100 episodes", ctrl);

    std::cout << "\n-- Failure: always pushing right\n";
    env.reset(5);
    float theta = 0;
    int fell_at = 0;
    for (int t = 1;; ++t) {
      rl::StepResult r = env.step(1);
      if (r.done()) {
        std::printf("  episode over at t=%d (theta=%+.3f rad, limit is +-0.209)\n", t, r.obs[2]);
        theta = r.obs[2];
        fell_at = t;
        break;
      }
    }

    check::Require(random.mean_length() > 10 && random.mean_length() < 40, "a random policy should last about 20 steps");
    check::Require(ctrl.mean_return() > 450, "the hand-written controller should balance (nearly) every episode to 500");
    check::Require(fell_at < 50 && std::abs(theta) > 0.2f, "always pushing right should topple the pole past 0.209 rad");
    std::cout << "\nchecks passed (vsim videos omitted)\n";
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
