// CMD: just run-nn external_envs
// Binding to external environments: ALE (Atari 2600) and vsim.
//
// Port of tflow's external_envs. tflow reaches two real simulators through
// their C++ APIs:
//   - the Arcade Learning Environment, wrapped as rl::AtariEnv with DQN-style
//     preprocessing (common/rl/ale_env.hpp);
//   - vsim, the game engine whose scene files and Jolt-based PhysicsWorld the
//     articulated tasks in tflow's common/rl/vsim_env.h are built on.
// Atari ROMs are not redistributable; the free Tetris homebrew ships with ALE
// and other ROMs can be dropped into data/roms/.
//
// Deviations: ALE and vsim are not datamunge dependencies (the plan keeps
// emulators, vsim and video recording out of scope). The Atari demo runs only
// when the example is compiled with ALE on the include path and linked with
// libale (DATAMUNGE_EXAMPLES_HAVE_ALE); `just run-nn` builds without it and
// then only reports where the ROMs are. The vsim demo (a ball knocking over a
// crate stack, stepped at 120 Hz) and both videos are omitted.

#include <cstdio>
#include <filesystem>
#include <iostream>
#include <random>

#include "common/checks.hpp"
#include "common/rl/ale_env.hpp"
#include "common/rl/env.hpp"

namespace {

void AtariDemo() {
  std::printf("#### ALE\n");
#if DATAMUNGE_EXAMPLES_HAVE_ALE
  rl::AtariEnv env("tetris");
  std::printf("  %s: raw screen %dx%d, %d-frame stack of 84x84 -> obs of %d floats, %s\n", env.name().c_str(), env.raw_width(),
              env.raw_height(), rl::AtariEnv::kStack, env.obs_size(), env.action_space().ToString().c_str());
  std::printf("  minimal action set:");
  for (const std::string& a : env.action_names()) std::printf(" %s", a.c_str());
  std::printf("\n");
  std::mt19937_64 rng(0);
  env.reset(0);
  float ret = 0;
  int steps = 0;
  while (true) {
    rl::StepResult r = env.step(env.action_space().Sample(rng));
    ret += r.reward;
    ++steps;
    if (r.done()) break;
  }
  std::printf("  random agent: %d agent steps (%d emulator frames), score %.0f, game over = %d\n", steps, steps * 4, ret, env.lives() == 0);
  // The preprocessed frame the agent actually sees.
  std::printf("  last 84x84 frame, downsampled 4x for display (. < 64, : < 128, + < 192, # otherwise):\n");
  const auto& f = env.last_frame();
  for (int y = 0; y < 84; y += 4) {
    std::printf("    ");
    for (int x = 0; x < 84; x += 2) {
      const int v = f[y * 84 + x];
      std::printf("%c", v < 64 ? '.' : v < 128 ? ':' : v < 192 ? '+' : '#');
    }
    std::printf("\n");
  }
  check::Require(env.raw_width() == 160 && env.raw_height() == 210, "the Atari screen is 160x210");
  check::Require(steps > 0 && env.obs_size() == 4 * 84 * 84, "a random Tetris game runs to the end");
#else
  std::printf("  ALE is not built in (see the header): the Atari demo is skipped.\n");
  try {
    std::printf("  tetris ROM: %s\n", rl::FindRom("tetris").c_str());
  } catch (const std::exception& e) {
    std::printf("  %s\n", e.what());
  }
#endif
}

void VsimDemo() {
  std::printf("\n#### vsim\n");
  std::printf("  vsim is not available to the datamunge ports (out of scope): the box-stack scene demo is skipped.\n");
}

}  // namespace

int main() {
  try {
    AtariDemo();
    VsimDemo();
    std::printf("\n  no videos are written by the port\n");
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
