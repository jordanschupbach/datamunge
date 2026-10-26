// datamunge::nn port helper standing in for tflow's vsim_studio.h RecordBest.
//
// RecordBest plays `candidates` seeded episodes (seeds seed .. seed+candidates-1) without
// filming, then films the one with the highest return. PlayBestOfSeeds runs the same candidate
// episodes, prints the same "best of N" line, and replays the best one with PlayEpisode
// (common/rl/playback.hpp), which prints its steps and return instead of writing a video.
#pragma once

#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <utility>

#include "common/rl/playback.hpp"

namespace rl {

template <typename Env, typename PolicyFn>
Playback PlayBestOfSeeds(Env& env, PolicyFn&& policy, const std::string& name, uint64_t seed, int candidates = 5) {
  uint64_t best_seed = seed;
  float    best = -std::numeric_limits<float>::infinity();
  for (int c = 0; c < candidates; ++c) {
    float ret = 0;
    Obs   obs = env.reset(seed + c);
    while (true) {
      const StepResult r = env.step(Decide(policy, obs).action);
      ret += r.reward;
      obs = r.obs;
      if (r.done()) break;
    }
    if (ret > best) best = ret, best_seed = seed + c;
  }
  std::printf("  best of %d episodes (return %.1f):\n", candidates, best);
  return PlayEpisode(env, std::forward<PolicyFn>(policy), name, best_seed);
}

}  // namespace rl
