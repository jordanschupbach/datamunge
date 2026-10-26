// CMD: just run-nn dyna_q
// Dyna-Q: learning and planning from a learned tabular model.
//
// Port of tflow's dyna_q (TF-free apart from the Q-table, so the numbers match
// tflow exactly).
// Sutton & Barto chapter 8: every real step updates Q by Q-learning *and*
// records (s, a) -> (r, s') in a model; then n extra "planning" updates
// replay random remembered pairs through the same Q-learning rule. The
// planning steps squeeze far more learning out of each real interaction.
// Dyna-Q+ adds an exploration bonus kappa sqrt(tau) for pairs not tried for
// tau steps, which lets it notice when the world changes (the blocking maze).
//
// The vsim video of Dyna-Q+'s greedy policy is replaced by PlayEpisode (the
// same episode, printed, no video). The port adds checks (exit 1 on failure).

#include <cmath>
#include <cstdio>
#include <iostream>
#include <iterator>
#include <map>
#include <random>
#include <vector>

#include "common/checks.hpp"
#include "common/rl/grid_world.hpp"
#include "common/rl/hud.hpp"
#include "common/rl/playback.hpp"
#include "common/rl/tabular.hpp"

namespace {

// A grid with a configurable wall row that can be changed mid-run.
class BlockingMaze : public rl::GridBase {
 public:
  BlockingMaze() : rl::GridBase(6, 9, {5, 3}, {0, 8}, 0.f, 3000) { SetWall(/*gap_at_left=*/false); }
  std::string name() const override { return "BlockingMaze"; }
  // A wall across row 3 with a gap at one end: the short path is on the
  // right at first, then only the left is open.
  void SetWall(bool gap_at_left) {
    walls_.clear();
    for (int c = 0; c < 9; ++c)
      if (!(gap_at_left ? c == 0 : c == 8)) AddWall(3, c);
  }
  Outcome Apply(int r, int c, int a) const override {
    Outcome o = rl::GridBase::Apply(r, c, a);
    o.reward = o.terminal ? 1.f : 0.f;  // +1 at the goal, 0 elsewhere
    return o;
  }
};

struct Result {
  std::vector<float> steps_per_episode;
  std::vector<int> cumulative_reward_per_step;
};

// Dyna-Q(+) on `env` for `total_steps` real steps. `change_at` flips the maze.
Result Run(rl::GridBase& env, int planning_steps, float kappa, int total_steps, int change_at, uint64_t seed,
           rl::QTable* q_out = nullptr) {
  const int S = env.num_states(), A = env.action_space().n;
  rl::QTable Q(S, A);
  std::map<std::pair<int, int>, std::pair<float, int>> model;  // (s, a) -> (r, s')
  std::vector<int> last_tried(S * A, 0);
  std::mt19937_64 rng(seed);
  const float alpha = 0.1f, gamma = 0.95f, eps = 0.1f;
  Result res;
  int cumulative = 0, ep_steps = 0;
  int s = static_cast<int>(env.reset(seed)[0]);
  for (int step = 1; step <= total_steps; ++step) {
    if (step == change_at) dynamic_cast<BlockingMaze&>(env).SetWall(true);
    const int a = Q.EpsilonGreedy(s, eps, rng);
    rl::StepResult r = env.step(a);
    const int s2 = static_cast<int>(r.obs[0]);
    // (a) direct RL
    Q(s, a) += alpha * (r.reward + (r.terminated ? 0.f : gamma * Q.Max(s2)) - Q(s, a));
    // (b) model learning (deterministic environment: remember the last outcome)
    model[{s, a}] = {r.reward, r.terminated ? -1 : s2};
    last_tried[s * A + a] = step;
    // (c) planning: replay random remembered pairs. Dyna-Q+ also tries
    // untried actions from visited states, assuming they lead back to s.
    for (int n = 0; n < planning_steps && !model.empty(); ++n) {
      auto it = model.begin();
      std::advance(it, rng() % model.size());
      int ps = it->first.first, pa = it->first.second;
      float pr = it->second.first;
      int ps2 = it->second.second;
      if (kappa > 0) {
        pa = static_cast<int>(rng() % A);
        auto found = model.find({ps, pa});
        if (found == model.end()) pr = 0, ps2 = ps;
        else pr = found->second.first, ps2 = found->second.second;
        pr += kappa * std::sqrt(static_cast<float>(step - last_tried[ps * A + pa]));
      }
      Q(ps, pa) += alpha * (pr + (ps2 < 0 ? 0.f : gamma * Q.Max(ps2)) - Q(ps, pa));
    }
    cumulative += r.reward > 0.5f;
    res.cumulative_reward_per_step.push_back(cumulative);
    ++ep_steps;
    s = s2;
    if (r.done()) {
      res.steps_per_episode.push_back(static_cast<float>(ep_steps));
      ep_steps = 0;
      s = static_cast<int>(env.reset()[0]);
    }
  }
  if (q_out) *q_out = Q;
  return res;
}

}  // namespace

int main() {
  try {
    std::printf("== Dyna-Q on the blocking maze (reward 1 at the goal, 0 elsewhere)\n");
    BlockingMaze maze;
    maze.reset(0);
    std::cout << maze.render();
    std::printf("\n-- Planning steps vs. learning speed: steps per episode (averaged over 10 runs), first 10 episodes\n");
    std::printf("  %-12s", "n");
    for (int e = 1; e <= 10; ++e) std::printf(" %6d", e);
    std::printf("\n");
    std::map<int, std::vector<double>> curves;
    for (int n : {0, 5, 50}) {
      std::vector<double> avg(10, 0);
      for (int run = 0; run < 10; ++run) {
        BlockingMaze m;
        Result r = Run(m, n, 0.f, 6000, /*change_at=*/-1, run + 1);
        for (int e = 0; e < 10 && e < static_cast<int>(r.steps_per_episode.size()); ++e) avg[e] += r.steps_per_episode[e] / 10;
      }
      std::printf("  n = %-8d", n);
      for (int e = 0; e < 10; ++e) std::printf(" %6.0f", avg[e]);
      std::printf("\n");
      curves[n] = avg;
    }
    // Planning speeds learning up: by episode 3, n = 50 needs a fraction of the
    // steps n = 0 does, and n = 50 reaches the shortest path (10 steps) quickly.
    check::Require(curves[50][2] < 0.5 * curves[0][2] && curves[5][2] < curves[0][2], "more planning steps learn faster (episode 3)");
    check::Require(curves[50][9] < 20, "Dyna-Q (n = 50) is near the 10-step shortest path by episode 10");

    std::printf("\n-- The world changes at step 1000 (the gap moves from the right to the left end of the wall)\n");
    std::printf("   cumulative goals reached, averaged over 10 runs, Dyna-Q vs Dyna-Q+ (kappa 1e-4), n = 50\n");
    std::printf("  %-10s", "step");
    for (int t = 500; t <= 3000; t += 500) std::printf(" %7d", t);
    std::printf("\n");
    std::vector<double> final_goals;
    for (float kappa : {0.f, 1e-4f}) {
      std::vector<double> avg(3000, 0);
      for (int run = 0; run < 10; ++run) {
        BlockingMaze m;
        Result r = Run(m, 50, kappa, 3000, 1000, 100 + run);
        for (int t = 0; t < 3000; ++t) avg[t] += r.cumulative_reward_per_step[t] / 10.0;
      }
      std::printf("  %-10s", kappa > 0 ? "Dyna-Q+" : "Dyna-Q");
      for (int t = 500; t <= 3000; t += 500) std::printf(" %7.1f", avg[t - 1]);
      std::printf("\n");
      final_goals.push_back(avg[2999]);
    }
    std::cout << "  After the change, Dyna-Q keeps planning with the stale model until exploration happens to find the\n"
                 "  new gap; Dyna-Q+'s bonus for long-untried actions makes it re-explore and recover sooner.\n";
    check::Require(final_goals[1] > final_goals[0], "Dyna-Q+ reaches the goal more often than Dyna-Q after the change");

    // In tflow, on film: Dyna-Q+'s greedy policy on the changed maze (the gap
    // is now on the left). Here the same episode is played and printed.
    std::cout << "\n-- Playing Dyna-Q+'s greedy policy after the wall moved (tflow films it)\n";
    BlockingMaze film_maze;
    rl::QTable Q(1, 1);
    Run(film_maze, 50, 1e-4f, 3000, 1000, 100, &Q);
    const rl::Playback pb = rl::PlayEpisode(film_maze, [&](const rl::Obs& o) { return rl::GreedyQ(Q.Row(static_cast<int>(o[0])), Q.A); },
                                            "dyna_q_plus_blocking_maze", 0);
    check::Require(pb.ret == 1.f && pb.steps <= 20, "the greedy Dyna-Q+ policy goes round the moved wall to the goal");
    std::printf("\nchecks passed (vsim video omitted)\n");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "CHECK FAILED: %s\n", e.what());
    return 1;
  }
  return 0;
}
