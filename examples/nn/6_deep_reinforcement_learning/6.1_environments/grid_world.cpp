// CMD: just run-nn grid_world
// Grid world and cliff walking.
//
// Port of tflow's grid_world. Two small finite MDPs used by the tabular
// examples in 6.2 (common/rl/grid_world.hpp, copied from tflow). Both expose
// their transition model (TabularEnv::transitions) so dynamic programming can
// plan on them, and both render as text. The vsim videos of the policies are
// omitted; the port adds checks that the transition models are distributions
// and that the cliff paths have Sutton & Barto's returns (-13 and -17).

#include <cmath>
#include <cstdio>
#include <iostream>

#include "common/checks.hpp"
#include "common/rl/grid_world.hpp"
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

void Describe(rl::GridBase& env) {
  std::cout << "== " << env.name() << ": " << env.height() << "x" << env.width() << ", "
            << env.num_states() << " states, " << env.action_space().ToString()
            << ", max_steps=" << env.max_steps() << "\n";
  env.reset(0);
  std::cout << env.render();
  std::cout << "  (A agent, S start, G goal, # wall, ~ cliff)\n";
}

void ShowModel(const rl::GridBase& env, int s) {
  auto [r, c] = env.Coords(s);
  std::printf("  model from state %d (row %d, col %d):\n", s, r, c);
  for (int a = 0; a < 4; ++a) {
    std::printf("    action %s:", rl::GridBase::kArrows[a]);
    for (const rl::Transition& t : env.transitions(s, a))
      std::printf("  p=%.2f -> s'=%d r=%+.0f%s", t.prob, t.next_state, t.reward,
                  t.terminal ? " (terminal)" : "");
    std::printf("\n");
  }
}

// Every (s, a) model sums to probability 1 (walls and terminal states have no model).
void CheckModel(const rl::GridBase& env) {
  for (int s = 0; s < env.num_states(); ++s)
    for (int a = 0; a < 4; ++a) {
      double p = 0;
      if (env.transitions(s, a).empty()) continue;
      for (const rl::Transition& t : env.transitions(s, a)) p += t.prob;
      check::Near(p, 1.0, 1e-5, env.name() + ": transition probabilities from state " + std::to_string(s));
    }
}

}  // namespace

int main() {
  try {
    rl::GridWorld grid(/*slip_prob=*/0.2f);
    Describe(grid);
    ShowModel(grid, grid.Index(0, 2));
    CheckModel(grid);
    rl::PrintStats("random policy, 200 episodes", rl::RunRandom(grid, 200, 0));

    // A hand-written policy: go down the left side, along the bottom, to G.
    auto manual = [&](const rl::Obs& o) {
      auto [r, c] = grid.Coords(static_cast<int>(o[0]));
      return rl::Action(r < 4 ? rl::GridBase::kDown : rl::GridBase::kRight);
    };
    rl::PrintStats("manual policy, slip 0.2", rl::RunEpisodes(grid, 200, 0, manual));
    rl::GridWorld det(/*slip_prob=*/0.f);
    const rl::EpisodeStats det_stats = rl::RunEpisodes(det, 10, 0, manual);
    rl::PrintStats("manual policy, no slip", det_stats);
    check::Require(det_stats.min_return() == det_stats.max_return(), "without slip the manual policy is deterministic");

    std::cout << "\n";
    rl::CliffWalking cliff;
    Describe(cliff);
    ShowModel(cliff, cliff.Index(2, 1));
    CheckModel(cliff);
    rl::PrintStats("random policy, 100 episodes", rl::RunRandom(cliff, 100, 0));
    // The two policies from Sutton & Barto's figure: hug the cliff, or go up
    // and around. Both reach the goal; the safe one takes 4 extra steps.
    auto risky = [&](const rl::Obs& o) {
      auto [r, c] = cliff.Coords(static_cast<int>(o[0]));
      if (r == 3 && c == 0) return rl::Action(rl::GridBase::kUp);
      if (r == 2 && c < 11) return rl::Action(rl::GridBase::kRight);
      return rl::Action(rl::GridBase::kDown);
    };
    auto safe = [&](const rl::Obs& o) {
      auto [r, c] = cliff.Coords(static_cast<int>(o[0]));
      if (c == 0 && r > 0) return rl::Action(rl::GridBase::kUp);
      if (r == 0 && c < 11) return rl::Action(rl::GridBase::kRight);
      return rl::Action(rl::GridBase::kDown);
    };
    const rl::EpisodeStats risky_stats = rl::RunEpisodes(cliff, 5, 0, risky);
    const rl::EpisodeStats safe_stats = rl::RunEpisodes(cliff, 5, 0, safe);
    rl::PrintStats("risky path (along the cliff)", risky_stats);
    rl::PrintStats("safe path (top row)", safe_stats);
    check::Near(risky_stats.mean_return(), -13, 1e-6, "risky cliff path return");
    check::Near(safe_stats.mean_return(), -17, 1e-6, "safe cliff path return");

    std::cout << "\n-- Walking into the cliff: -100 and back to the start, episode continues\n";
    cliff.reset(0);
    cliff.step(rl::GridBase::kUp);
    rl::StepResult r = cliff.step(rl::GridBase::kRight);
    r = cliff.step(rl::GridBase::kDown);
    std::printf("  reward %+.0f terminated=%d, now at state %d\n", r.reward, r.terminated,
                cliff.state());
    check::Require(r.reward == -100.f && !r.terminated && cliff.state() == cliff.Index(3, 0),
                   "the cliff costs -100 and sends the agent back to the start");
    std::cout << "\nchecks passed (vsim videos omitted)\n";
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
