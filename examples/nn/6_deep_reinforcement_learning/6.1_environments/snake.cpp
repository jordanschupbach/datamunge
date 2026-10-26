// CMD: just run-nn snake
// Snake: a simple Atari-like pixel game.
//
// Port of tflow's snake. The observation is an image: three planes (head,
// body, food) over the grid, flattened to a float vector. That makes Snake the
// stand-in for Atari in the pixel-based examples later on (DQN on pixels,
// 6.3). The environment is common/rl/snake.hpp (copied from tflow, TF-free).
// tflow films the greedy episode on a vsim 3D board; the port replays the same
// episode without vsim/ffmpeg and instead writes its frames, drawn from the
// board (common/rl/snake_frames.hpp), as one PNG filmstrip to
// output/nn/frames/snake_greedy_policy.png. Checks are added.

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "common/checks.hpp"
#include "common/paths.hpp"
#include "common/rl/playback.hpp"
#include "common/rl/snake.hpp"
#include "common/rl/snake_frames.hpp"
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

// Head for the food along whichever axis is further off, but never into a
// wall or the body if another safe move exists.
class GreedyPolicy {
 public:
  explicit GreedyPolicy(const rl::Snake& env) : env_(env) {}
  rl::Action operator()(const rl::Obs&) {
    auto [hr, hc] = env_.head();
    auto [fr, fc] = env_.food();
    int prefer[6];
    int n = 0;
    if (std::abs(fr - hr) >= std::abs(fc - hc)) {
      prefer[n++] = fr < hr ? rl::Snake::kUp : rl::Snake::kDown;
      prefer[n++] = fc < hc ? rl::Snake::kLeft : rl::Snake::kRight;
    } else {
      prefer[n++] = fc < hc ? rl::Snake::kLeft : rl::Snake::kRight;
      prefer[n++] = fr < hr ? rl::Snake::kUp : rl::Snake::kDown;
    }
    for (int d = 0; d < 4; ++d) prefer[n++] = d;
    for (int i = 0; i < n; ++i)
      if (Safe(prefer[i])) return rl::Action(prefer[i]);
    return rl::Action(prefer[0]);
  }

 private:
  bool Safe(int dir) const {
    static const int dr[4] = {-1, 0, 1, 0}, dc[4] = {0, 1, 0, -1};
    auto [hr, hc] = env_.head();
    const int r = hr + dr[dir], c = hc + dc[dir];
    if (r < 0 || r >= env_.height() || c < 0 || c >= env_.width()) return false;
    return !env_.Occupied(r, c);
  }
  const rl::Snake& env_;
};

}  // namespace

int main() {
  try {
    rl::Snake env(8, 8);
    auto shape = env.obs_shape();
    std::printf("== %s: observation [planes=%d, height=%d, width=%d] = %d floats, %s, "
                "+1 food, -1 death\n",
                env.name().c_str(), shape[0], shape[1], shape[2], env.obs_size(),
                env.action_space().ToString().c_str());

    const rl::EpisodeStats random = rl::RunRandom(env, 200, 0);
    rl::PrintStats("random, 200 episodes", random);
    GreedyPolicy greedy(env);
    const rl::EpisodeStats greedy_stats = rl::RunEpisodes(env, 200, 0, greedy);
    rl::PrintStats("greedy toward food, 200 episodes", greedy_stats);

    std::cout << "\n-- A greedy episode\n";
    rl::Obs obs = env.reset(3);
    int steps = 0;
    bool game_over = false;
    for (int t = 0;; ++t) {
      rl::StepResult r = env.step(greedy(obs));
      obs = r.obs;
      steps = t + 1;
      if (t % 15 == 0 || r.done()) {
        std::printf("  t=%d length=%d reward=%+.0f%s\n", t, env.length(), r.reward,
                    r.terminated ? "  GAME OVER" : "");
      }
      if (r.done()) {
        game_over = r.terminated;
        break;
      }
    }

    std::cout << "\n-- The observation planes for that final state\n";
    const char* names[3] = {"head", "body", "food"};
    int counts[3] = {0, 0, 0};
    for (int p = 0; p < 3; ++p) {
      std::cout << "  " << names[p] << ":\n";
      for (int r = 0; r < shape[1]; ++r) {
        std::cout << "    ";
        for (int c = 0; c < shape[2]; ++c) {
          const bool on = obs[p * shape[1] * shape[2] + r * shape[2] + c] > 0;
          counts[p] += on;
          std::cout << (on ? '1' : '.');
        }
        std::cout << "\n";
      }
    }

    std::cout << "\n-- The greedy episode replayed as frames (tflow films it on a 3D board; video omitted)\n";
    rl::SnakeFilmstrip strip;
    const rl::Playback pb = rl::PlayEpisode(env, greedy, "snake_greedy_policy", 3,
                                            [&](const rl::Snake& e, const rl::Hud&) { strip.Add(e); });
    const std::string png = (paths::OutputDir("frames") / "snake_greedy_policy.png").string();
    strip.Save(png);
    std::printf("  wrote %s (%d frames, one per step plus the reset)\n", png.c_str(), strip.frames());

    check::Require(greedy_stats.mean_return() > 10.f && greedy_stats.min_return() > 0.f,
                   "the greedy policy should eat food in every episode, > 10 on average");
    check::Require(random.mean_return() < 0.f, "random moves should mostly die before eating");
    check::Require(game_over, "the traced greedy episode should end in a collision");
    check::Require(counts[0] == 1 && counts[2] == 1 && counts[1] == env.length() - 1,
                   "the planes should hold one head, one food and length-1 body cells");
    check::Require(pb.steps == steps && pb.ret == static_cast<float>(env.length() - 2),
                   "the replay should reproduce the traced episode (food eaten - 1 = final length - 2, the fatal step drops the tail)");
    check::Require(strip.frames() == steps + 1, "the filmstrip should have one frame per step plus the reset");
    std::cout << "\nchecks passed (vsim video replaced by a PNG filmstrip)\n";
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
