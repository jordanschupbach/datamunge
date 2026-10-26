// datamunge::nn port helper standing in for tflow's src/common/rl/vsim_studio.h recorders.
//
// tflow films episodes with vsim's renderer and ffmpeg (RecordEpisode); both are out of scope
// for the datamunge ports. PlayEpisode runs exactly the episode RecordEpisode would film (same
// reset seed, the same Decide/Hud bookkeeping, one env step per loop) without a video and prints
// the same summary line, so the steps and return can be compared with tflow's "recorded ..."
// line. tflow's recorder also stops when the film reaches `max_frames`; frames are not steps
// (they depend on the world's frame rate), so PlayEpisode ignores that cap: for every episode
// the 6.1 examples film, the episode ends before the film is full.
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <type_traits>
#include <utility>

#include "common/rl/env.hpp"
#include "common/rl/hud.hpp"

namespace rl {

// The policy's decision for `obs`, as in vsim_studio.h: a Decision passes through, a bare
// action (an int, a vector or an Action) is wrapped.
template <typename PolicyFn>
Decision Decide(PolicyFn& policy, const Obs& obs) {
  if constexpr (std::is_same_v<std::decay_t<decltype(policy(obs))>, Decision>) return policy(obs);
  else return Decision(Action(policy(obs)));
}

struct Playback {
  int steps = 0;
  float ret = 0;
  Hud hud;  // the overlay state after the last step (decision, reward, return)
};

// Plays one episode of `policy` from env.reset(seed); `after_step(env, hud)` is called after
// the reset (step 0) and after every step, e.g. to snapshot frames.
template <typename Env, typename PolicyFn, typename AfterStep>
Playback PlayEpisode(Env& env, PolicyFn&& policy, const std::string& name, uint64_t seed, AfterStep after_step) {
  Playback pb;
  pb.hud.Reset(env.action_space(), env.action_names());
  Obs obs = env.reset(seed);
  after_step(static_cast<const Env&>(env), static_cast<const Hud&>(pb.hud));
  while (true) {
    pb.hud.Choose(Decide(policy, obs));
    const StepResult r = env.step(pb.hud.decision.action);
    pb.hud.Observe(r);
    pb.ret += r.reward;
    ++pb.steps;
    obs = r.obs;
    after_step(static_cast<const Env&>(env), static_cast<const Hud&>(pb.hud));
    if (r.done()) break;
  }
  std::printf("  played %s (video omitted): %d steps, return %.1f\n", name.c_str(), pb.steps, pb.ret);
  return pb;
}

template <typename Env, typename PolicyFn>
Playback PlayEpisode(Env& env, PolicyFn&& policy, const std::string& name, uint64_t seed) {
  return PlayEpisode(env, std::forward<PolicyFn>(policy), name, seed, [](const Env&, const Hud&) {});
}

}  // namespace rl
