// CMD: just run-nn ale_envs
// Atari through ALE with DQN preprocessing.
//
// Port of tflow's ale_envs. What a DQN agent sees: every agent step repeats
// the action for 4 emulator frames, max-pools the last two to remove flicker,
// converts to grayscale, resizes to 84x84 and stacks the last 4 frames so
// velocities are visible. This program shows the pipeline on the games
// available (Tetris by default, anything in data/roms/), measures throughput,
// and checks that motion is visible in the stack.
//
// Deviations: ALE is not a datamunge dependency (the plan keeps emulators out
// of scope), so the emulator part only runs when the example is compiled with
// ALE on the include path (common/rl/ale_env.hpp, DATAMUNGE_EXAMPLES_HAVE_ALE)
// and linked with libale; `just run-nn` builds without it. Without ALE the
// program lists the ROMs and runs the same preprocessing code
// (rl::AtariFrameStack) on a synthetic 160x210 test screen with a flickering,
// moving sprite instead. The video (raw screen next to the 4 stacked frames)
// is omitted in both cases. Options: game=<name>.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <random>

#include "common/checks.hpp"
#include "common/rl/ale_env.hpp"
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

// Mean |newest - oldest| over the 84x84 pixels of a stacked observation.
double StackDifference(const rl::Obs& obs) {
  constexpr int P = 84 * 84;
  double s = 0;
  for (int i = 0; i < P; ++i) s += std::abs(obs[3 * P + i] - obs[i]);
  return s / P;
}

#if !DATAMUNGE_EXAMPLES_HAVE_ALE
// A 160x210 grayscale "screen": dark background, a bright 8x8 sprite moving 3 px
// per emulator frame, drawn only on even frames (Atari-style flicker).
std::vector<unsigned char> TestScreen(int frame) {
  std::vector<unsigned char> s(160 * 210, 20);
  if (frame % 2 == 0) {
    const int x0 = (10 + 3 * frame) % 150, y0 = 100;
    for (int y = y0; y < y0 + 8; ++y)
      for (int x = x0; x < x0 + 8; ++x) s[y * 160 + x] = 230;
  }
  return s;
}

void SyntheticPipeline() {
  std::printf("== ALE is not built in: the emulator part is skipped (see the header).\n"
              "   The DQN preprocessing runs on a synthetic 160x210 screen instead\n"
              "   (an 8x8 sprite moving 3 px per frame, visible only on even frames).\n");
  rl::AtariFrameStack stack(160, 210);
  for (int i = 0; i < rl::AtariFrameStack::kStack; ++i) stack.PushFrame(TestScreen(0));
  rl::Obs obs = stack.StackedObs();
  int frame = 0;
  double min_peak_pooled = 1, min_peak_plain = 1;
  for (int step = 0; step < 8; ++step) {
    std::vector<unsigned char> prev, cur;
    for (int k = 0; k < 4; ++k, ++frame)  // frame skip 4
      if (k == 2) prev = TestScreen(frame);
    cur = TestScreen(frame - 1);  // the last emulated frame (odd: the sprite is not drawn)
    const std::vector<unsigned char> plain = cur;
    rl::AtariFrameStack::MaxPool(cur, prev);
    stack.PushFrame(cur);
    obs = stack.StackedObs();
    auto peak = [](const std::vector<unsigned char>& f) { return *std::max_element(f.begin(), f.end()) / 255.0; };
    rl::AtariFrameStack single(160, 210);
    single.PushFrame(plain);
    min_peak_pooled = std::min(min_peak_pooled, peak(stack.last_frame()));
    min_peak_plain = std::min(min_peak_plain, peak(single.last_frame()));
  }
  std::printf("  obs: %zu floats = %d frames of 84x84, values in [%.3f, %.3f]\n", obs.size(), rl::AtariFrameStack::kStack,
              *std::min_element(obs.begin(), obs.end()), *std::max_element(obs.begin(), obs.end()));
  std::printf("  brightest pixel of the last frame alone: %.3f, after max-pooling the last two frames: %.3f -> flicker removed\n",
              min_peak_plain, min_peak_pooled);
  const double diff = StackDifference(obs);
  std::printf("  frame stack difference (newest - oldest, mean |diff| over pixels): %.4f -> motion is visible to the network\n", diff);
  check::Require(static_cast<int>(obs.size()) == 4 * 84 * 84, "the observation is 4 stacked 84x84 frames");
  check::Require(min_peak_plain < 0.1 && min_peak_pooled > 0.5, "max-pooling the last two frames should keep the flickering sprite");
  check::Require(diff > 0.001, "the stack should show the sprite's motion");
}
#endif

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string game = "tetris";
    for (int i = 1; i < argc; ++i) if (std::string(argv[i]).rfind("game=", 0) == 0) game = std::string(argv[i]).substr(5);
    std::printf("== available ROMs:");
    for (const std::string& dir : rl::RomDirs())
      if (std::filesystem::exists(dir))
        for (const auto& e : std::filesystem::directory_iterator(dir))
          if (e.path().extension() == ".bin") std::printf(" %s", e.path().stem().string().c_str());
    std::printf("   (pass game=<name> to pick one)\n");

#if DATAMUNGE_EXAMPLES_HAVE_ALE
    rl::AtariEnv::Options opt;
    opt.clip_rewards = false;
    rl::AtariEnv env(game, opt);
    std::printf("== %s: %s, %d actions:", env.name().c_str(), env.action_space().ToString().c_str(), env.action_space().n);
    for (auto& a : env.action_names()) std::printf(" %s", a.c_str());
    std::printf("\n");

    std::mt19937_64 rng(0);
    auto t0 = std::chrono::steady_clock::now();
    rl::EpisodeStats stats = rl::RunRandom(env, 5, 0);
    auto t1 = std::chrono::steady_clock::now();
    rl::PrintStats("random policy, 5 episodes", stats);
    const double secs = std::chrono::duration<double>(t1 - t0).count();
    const double steps = stats.mean_length() * 5;
    std::printf("  throughput: %.0f agent steps/s = %.0f emulator frames/s (with preprocessing)\n", steps / secs, 4 * steps / secs);

    // tflow films these 400 steps (raw screen and the four stacked frames); the video is omitted.
    rl::Obs obs = env.reset(1);
    for (int t = 0; t < 400; ++t) {
      rl::StepResult r = env.step(env.action_space().Sample(rng));
      obs = r.obs;
      if (r.done()) break;
    }
    const double diff = StackDifference(obs);
    std::printf("  frame stack difference (newest - oldest, mean |diff| over pixels): %.4f -> motion is visible to the network\n", diff);
    check::Require(static_cast<int>(obs.size()) == env.obs_size(), "the observation is 4 stacked 84x84 frames");
    check::Require(diff > 0, "the stack should show motion");
#else
    (void)game;
    SyntheticPipeline();
#endif
    std::cout << "\nchecks passed (video omitted)\n";
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
