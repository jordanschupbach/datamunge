// CMD: just run-nn count_exploration
// Count-based exploration and pseudo-counts.
//
// Port of tflow's count_exploration. The example is TF-free (tabular Q-learning,
// common/rl/count_bonus.hpp is a verbatim copy of tflow's count_bonus.h), so every
// number matches tflow exactly. The vsim videos are replaced by PlayEpisode (the
// same greedy episode, printed, no video); video=0 skips them as in tflow. The
// port adds result checks after the table (exit 1 on failure).
//
// Compare epsilon-greedy Q-learning, an optimistic epsilon-greedy control,
// exact state-visit bonuses and density pseudo-count bonuses in a sparse-
// reward four-room maze. Only reaching the goal pays +1; all other transitions
// pay zero. The same step budget, seeds,
// epsilon, learning rate and discount are used for all four methods.
//
// A state arrival earns beta / sqrt(N(s') + 1), using the count BEFORE the
// arrival. Pseudo-counts replace N with rho(1-rho')/(rho'-rho): rho is the
// density before learning s', and rho' is its probability after one update.
// Here rho is a product of smoothed row and column frequencies, so novelty
// generalizes between cells. It can also mistake unseen combinations of
// familiar features for familiar states; the opening checks demonstrate it.
// This is a small density-model example, not a pixel-density Atari agent.
//
// The behaviour Q-table learns task reward plus the novelty bonus. A second
// off-policy Q-table learns task reward alone from exactly the same data.
// Evaluation and the video use the latter, with no exploration or bonuses.
// Counts include reset observations and persist across episodes. A real
// terminal stops bootstrapping; a time limit only starts a fresh rollout.
// Bonus rewards change as the counts grow, so augmented returns are never
// reported as task performance. Positive bonuses may also reward loitering.
// To avoid preferring visited cells just because their values have already
// risen above zero, both bonus agents start with optimistic behaviour values
// beta/(1-gamma) = 2. The optimistic epsilon-only control uses the same
// initialization, isolating its effect. Every task-only evaluation table
// starts at zero. All settings and both controls are printed in the results.
//
// Options: steps=40000 runs=8 seed=1 video=1 (video=0 skips rendering).
// The numeric identity and edge-case checks run before every experiment.
// Reference: Bellemare et al. (2016), https://arxiv.org/abs/1606.01868.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/rl/count_bonus.hpp"
#include "common/rl/dqn_common.hpp"
#include "common/rl/hud.hpp"
#include "common/rl/playback.hpp"
#include "common/rl/tabular.hpp"

namespace {

class SparseRooms : public rl::GridBase {
 public:
  SparseRooms() : GridBase(9, 9, {0, 0}, {8, 8}, 0.f, 250) {
    for (int i = 0; i < 9; ++i) {
      if (i != 1 && i != 7) {
        AddWall(4, i);
        AddWall(i, 4);
      }
    }
  }
  std::string name() const override { return "SparseFourRooms"; }
  Outcome Apply(int r, int c, int a) const override {
    auto o = GridBase::Apply(r, c, a);
    o.reward = o.terminal ? 1.f : 0.f;
    return o;
  }
};

void Require(bool condition, const char* what) {
  if (!condition) throw std::runtime_error(what);
}

template <typename Fn>
void Rejects(Fn fn, const char* what) {
  bool rejected = false;
  try { fn(); } catch (const std::invalid_argument&) { rejected = true; }
  Require(rejected, what);
}

void CheckCounts() {
  std::cout << "== Pseudo-count identities and edge cases\n";
  // For an empirical density rho=N/n, recoding gives (N+1)/(n+1).
  double max_error = 0;
  for (int total : {10, 100, 10000}) {
    for (int count : {0, 1, total / 2, total - 1}) {
      const double n = rl::PseudoCount(static_cast<double>(count) / total,
                                      static_cast<double>(count + 1) / (total + 1));
      max_error = std::max(max_error, std::abs(n - count));
      Require(std::abs(n - count) < 1e-7 * (count + 1), "empirical density must recover N");
    }
  }
  std::printf("  empirical density recovers visit counts: max error %.3g\n", max_error);

  rl::FactorizedDensity joint({4});
  int visits[4] = {};
  for (int s : {0, 1, 0, 3, 0, 1, 2}) {
    const double expected = visits[s]++ + 0.5;
    Require(std::abs(joint.Observe({s}) - expected) < 1e-10, "Dirichlet count must be N + prior");
  }
  Require(rl::PseudoCount(0, 0) == 0 && rl::PseudoCount(0, 0.2) == 0, "unseen event count");
  Require(std::isinf(rl::PseudoCount(0.2, 0.2)), "unchanged density implies infinite count");
  Require(rl::CountBonus(std::numeric_limits<double>::infinity(), 1) == 0, "no gain means no bonus");
  Require(rl::CountBonus(0, 0.1) == 0.1 && rl::CountBonus(99, 1) == 0.1, "inverse square-root bonus");
  Rejects([] { rl::PseudoCount(0.3, 0.2); }, "decreasing density must be rejected");
  Rejects([] { rl::PseudoCount(0.3, 1); }, "inconsistent recoding must be rejected");
  Rejects([] { rl::PseudoCount(0, 1); }, "certain recoding of unseen event must be rejected");
  Rejects([] { rl::CountBonus(-1, 1); }, "negative count must be rejected");
  Rejects([] { rl::FactorizedDensity d({4}, 0); }, "zero prior must be rejected");
  Rejects([] { rl::PseudoCount(std::numeric_limits<double>::quiet_NaN(), 0.5); }, "NaN density must be rejected");
  const double before = joint.Probability({0});
  bool invalid_feature = false;
  try { joint.Observe({4}); } catch (const std::out_of_range&) { invalid_feature = true; }
  Require(invalid_feature && joint.Probability({0}) == before, "invalid observation must not mutate density");
  std::cout << "  smoothed categorical density recovers N + 0.5; edge cases passed\n";

  // Only (0,1) and (1,0) have ever occurred. Both features of (0,0) are
  // familiar, even though their combination has never been seen.
  rl::FactorizedDensity factored({3, 3}), categorical({9});
  for (int i = 0; i < 100; ++i) {
    factored.Observe({0, 1});
    factored.Observe({1, 0});
    categorical.Observe({1});
    categorical.Observe({3});
  }
  const double generalized = factored.Observe({0, 0});
  const double unseen = categorical.Observe({0});
  Require(generalized > unseen && std::abs(unseen - 0.5) < 1e-10, "feature sharing demonstration");
  std::printf("  unseen (0,0): exact visits 0, joint pseudo-count %.2f, factored %.2f\n", unseen, generalized);
  std::cout << "  Shared features reduce novelty, even for an unseen combination.\n";
}

enum class Method { kEpsilon, kOptimistic, kCount, kPseudo };
const char* Name(Method method) {
  if (method == Method::kCount) return "state counts";
  if (method == Method::kPseudo) return "pseudo-counts";
  if (method == Method::kOptimistic) return "optimistic eps";
  return "epsilon only";
}

struct Result {
  explicit Result(int states) : task_q(states, 4) {}
  rl::QTable task_q;
  int first_goal = 0, goals = 0, visited = 0, eval_steps = 0;
  float eval_return = 0;
};

Result Train(Method method, int steps, uint64_t seed) {
  SparseRooms env;
  const int states = env.num_states();
  Result result(states);
  constexpr float alpha = 0.3f, gamma = 0.99f, epsilon = 0.1f, beta = 0.02f;
  const bool use_bonus = method == Method::kCount || method == Method::kPseudo;
  rl::QTable behaviour_q(states, 4, method == Method::kEpsilon ? 0.f : beta / (1 - gamma));
  std::vector<int> counts(states, 0);
  rl::FactorizedDensity density({env.height(), env.width()});
  std::mt19937_64 rng(seed);
  auto observe = [&](int state) {
    const int previous = counts[state]++;
    if (method == Method::kPseudo) {
      const auto [r, c] = env.Coords(state);
      return density.Observe({r, c});
    }
    return static_cast<double>(previous);
  };
  int s = static_cast<int>(env.reset(seed)[0]);
  observe(s);  // reset observations count, but do not earn a reward
  for (int t = 0; t < steps; ++t) {
    const int a = behaviour_q.EpsilonGreedy(s, epsilon, rng);
    const auto r = env.step(a);
    const int next = static_cast<int>(r.obs[0]);
    const double count = observe(next);
    const float bonus = use_bonus ? static_cast<float>(rl::CountBonus(count, beta)) : 0.f;
    const float bootstrap = r.terminated ? 0.f : gamma;
    behaviour_q(s, a) += alpha * (r.reward + bonus + bootstrap * behaviour_q.Max(next) - behaviour_q(s, a));
    result.task_q(s, a) += alpha * (r.reward + bootstrap * result.task_q.Max(next) - result.task_q(s, a));
    if (r.terminated) {
      ++result.goals;
      if (!result.first_goal) result.first_goal = t + 1;
    }
    s = next;
    if (r.done() && t + 1 < steps) {
      s = static_cast<int>(env.reset()[0]);
      observe(s);
    }
  }
  result.visited = std::count_if(counts.begin(), counts.end(), [](int n) { return n > 0; });
  // Separate environment, no novelty-model updates, no epsilon, task reward only.
  SparseRooms eval;
  s = static_cast<int>(eval.reset(seed)[0]);
  while (true) {
    const auto r = eval.step(result.task_q.Greedy(s));
    ++result.eval_steps;
    result.eval_return += r.reward;
    s = static_cast<int>(r.obs[0]);
    if (r.done()) break;
  }
  for (int state = 0; state < states; ++state)
    for (int a = 0; a < 4; ++a)
      Require(std::isfinite(behaviour_q(state, a)) && std::isfinite(result.task_q(state, a)), "Q values must stay finite");
  return result;
}

int ParseInt(const std::string& value) {
  size_t used = 0;
  const int n = std::stoi(value, &used);
  if (used != value.size()) throw std::invalid_argument("expected an integer: " + value);
  return n;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int steps = 40000, runs = 8, seed = 1, video = 1;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--help") {
        std::cout << "count_exploration [steps=40000] [runs=8] [seed=1] [video=0|1]\n";
        return 0;
      }
      const auto sep = arg.find('=');
      if (sep == std::string::npos) throw std::invalid_argument("expected key=value: " + arg);
      const auto key = arg.substr(0, sep), value = arg.substr(sep + 1);
      if (rl::IsHardwareOption(key)) continue;
      if (key == "steps") steps = ParseInt(value);
      else if (key == "runs") runs = ParseInt(value);
      else if (key == "seed") seed = ParseInt(value);
      else if (key == "video") video = ParseInt(value);
      else throw std::invalid_argument("unknown option: " + key);
    }
    if (steps < 1 || runs < 1 || seed < 0 || (video != 0 && video != 1))
      throw std::invalid_argument("steps/runs must be positive, seed nonnegative, video 0 or 1");
    CheckCounts();
    SparseRooms maze;
    maze.reset();
    int cells = 0;
    for (int r = 0; r < maze.height(); ++r)
      for (int c = 0; c < maze.width(); ++c) cells += !maze.IsWall(r, c);
    std::cout << "\n== Sparse four-room maze (+1 at the goal, zero elsewhere)\n" << maze.render();
    std::printf("  %d steps/run, %d runs, seeds %d..%llu; epsilon .1, alpha .3, gamma .99, beta .02\n",
                steps, runs, seed, static_cast<unsigned long long>(static_cast<uint64_t>(seed) + runs - 1));
    std::printf("  %d reachable cells; horizon 250; first-goal mean excludes runs with no goal\n", cells);
    std::cout << "  Behaviour Q starts at 0 for epsilon only, 2 for the other methods; task Q starts at 0.\n";
    std::printf("  %-15s %11s %12s %12s %12s %12s\n", "method", "found goal", "first step", "goals/run", "coverage %", "greedy win %");
    struct Summary {
      int found;
      double first, goals, coverage, win;
    };
    std::vector<Summary> summary;  // epsilon, optimistic, counts, pseudo-counts
    for (Method method : {Method::kEpsilon, Method::kOptimistic, Method::kCount, Method::kPseudo}) {
      double goals = 0, first = 0, coverage = 0, wins = 0;
      int found = 0;
      Result film(maze.num_states());
      for (int run = 0; run < runs; ++run) {
        auto result = Train(method, steps, static_cast<uint64_t>(seed) + run);
        goals += result.goals;
        coverage += result.visited;
        wins += result.eval_return;
        if (result.first_goal) ++found, first += result.first_goal;
        // Always film the first seed, without selecting the best run.
        if (run == 0) film = result;
      }
      const std::string first_text = found ? std::to_string(static_cast<int>(std::lround(first / found))) : "n/a";
      std::printf("  %-15s %5d/%-5d %12s %12.1f %12.1f %12.1f\n", Name(method), found, runs,
                  first_text.c_str(), goals / runs, 100 * coverage / (static_cast<double>(runs) * cells), 100 * wins / runs);
      std::printf("    seed %d greedy task return %.0f, %d steps\n", seed, film.eval_return, film.eval_steps);
      if (video) {
        const std::string name = method == Method::kEpsilon ? "epsilon" : method == Method::kOptimistic ? "optimistic" : method == Method::kCount ? "counts" : "pseudo_counts";
        rl::PlayEpisode(maze, [&](const rl::Obs& o) {
          return rl::GreedyQ(film.task_q.Row(static_cast<int>(o[0])), film.task_q.A);
        }, "count_exploration_" + name, seed);
      }
      summary.push_back({found, found ? first / found : 1e30, goals / runs, 100 * coverage / (static_cast<double>(runs) * cells), 100 * wins / runs});
    }
    std::cout << "\n  Greedy evaluation uses only task Q-values; curiosity is used only during collection.\n"
                 "  Coverage includes resets and goal cells; walls are excluded.\n"
                 "  Feature sharing changes novelty, so pseudo-counts need not beat exact counts.\n";
    // Checks added by the port (tflow prints the table only). At the default budget the
    // bonus agents reach the goal in every run, much sooner than epsilon-greedy, and every
    // method's task-only greedy policy wins (tflow: first goal 1763 / 416 / 225 / 261 steps).
    if (steps >= 40000 && runs >= 8) {
      Require(summary[2].found == runs && summary[3].found == runs, "both bonus methods reach the goal in every run");
      Require(summary[2].first < 0.5 * summary[0].first && summary[3].first < 0.5 * summary[0].first,
              "count and pseudo-count bonuses find the goal at least twice as fast as epsilon-greedy");
      for (const Summary& m : summary) Require(m.win >= 75, "the task-only greedy policy reaches the goal in most runs");
    }
    std::cout << "\nchecks passed (vsim videos omitted)\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "count_exploration: " << e.what() << "\n";
    return 1;
  }
}
