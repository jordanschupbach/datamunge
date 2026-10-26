// datamunge::nn port of tflow_examples src/common/rl/hud.h: Decision, its helpers and the Hud
// state are copied unchanged; DrawHud (drawing onto a video::Canvas) is dropped, since video
// recording is out of scope for the ports. Policies written for tflow's recorders still compile.
// The overlay on the videos: the move the agent makes and what it expects.
//
// A Decision is one move as the agent explains it: the action, the estimate
// it holds for every candidate action (Q-values, action probabilities, visit
// counts, planned returns, ...), which candidate it took, and the return it
// expects from the current state. The recorders in vsim_studio.h accept
// policies that return a Decision instead of a bare Action and draw it on
// every frame together with the step count, the last reward and the return
// so far (the Hud); a plain policy still gets the move, reward and return
// shown. DrawHud draws onto a video::Canvas, so the videos drawn without the
// 3D renderer (Atari) carry the same overlay.
//
// The helpers cover the usual cases: GreedyQ for value-based agents (the
// argmax, the Q row and max_a Q as the expected return), GreedyPi for a
// policy's probabilities with the critic's V(s), and Explain for an action
// with one estimate attached (a continuous torque with its critic value, a
// planner's first action with the planned return).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "common/rl/env.hpp"

namespace rl {

struct Decision {
  Action action;
  std::string label;                // the move in words ("right", "torque +1.3"); from the action space when empty
  std::string kind;                 // what `values` are: "Q(s,a)", "pi(a|s)", "visits", "planned return"
  std::vector<std::string> labels;  // one per candidate; the environment's action names when empty
  std::vector<float> values;        // the agent's estimate per candidate action (empty when it has none)
  int chosen = -1;                  // the candidate taken (-1: the action is not one of them)
  float value = NAN;                // the return the agent expects from here (V(s), max_a Q, root value, ...)
  std::string value_name = "expected return";
  float lo = NAN, hi = NAN;         // the bar scale for `values`; NaN: from the values themselves (and 0)
  int decimals = -1;                // digits after the point for `values` (-1: by magnitude)
  std::string note;                 // one free line, e.g. "CEM over 15 steps" or "epsilon 0.05"

  Decision() = default;
  explicit Decision(Action a) : action(std::move(a)) {}
  // A Decision stands in for its action, so one policy serves both the
  // recorders and the plain episode runners (RunEpisodes, env.step).
  operator const Action&() const { return action; }  // NOLINT: implicit by design
};

// Greedy over Q(s, .): the argmax, the whole row, and max_a Q as the expected return.
inline Decision GreedyQ(const float* q, int n, const std::string& kind = "Q(s,a)") {
  Decision d;
  d.values.assign(q, q + n);
  d.chosen = static_cast<int>(std::max_element(q, q + n) - q);
  d.action = Action(d.chosen);
  d.kind = kind;
  d.value = q[d.chosen];
  return d;
}
inline Decision GreedyQ(const std::vector<float>& q, const std::string& kind = "Q(s,a)") {
  return GreedyQ(q.data(), static_cast<int>(q.size()), kind);
}

// The most probable action of a stochastic policy, with the probabilities and
// (when the agent has a critic) V(s).
inline Decision GreedyPi(const std::vector<float>& probs, float v = NAN, const std::string& kind = "pi(a|s)") {
  Decision d;
  d.values = probs;
  d.chosen = static_cast<int>(std::max_element(probs.begin(), probs.end()) - probs.begin());
  d.action = Action(d.chosen);
  d.kind = kind;
  d.lo = 0.f, d.hi = 1.f;
  d.decimals = 2;
  d.value = v;
  d.value_name = "V(s)";
  return d;
}

// An action with one estimate attached: a torque and its critic value, a
// planner's action and the return of its plan.
inline Decision Explain(Action action, float value, const std::string& value_name = "expected return", const std::string& note = "") {
  Decision d(std::move(action));
  d.value = value;
  d.value_name = value_name;
  d.note = note;
  return d;
}

// An action with a remark for the overlay: what a hand-written rule looked at.
inline Decision Note(Action action, const std::string& note) {
  Decision d(std::move(action));
  d.note = note;
  return d;
}

// A number with as many decimals as its size warrants.
inline std::string HudNum(float v, int decimals = -1) {
  if (std::isnan(v)) return "-";
  char buf[32];
  if (decimals < 0) decimals = std::abs(v) >= 100 ? 0 : std::abs(v) >= 10 ? 1 : 2;
  std::snprintf(buf, sizeof buf, "%.*f", decimals, v);
  return buf;
}
inline std::string HudSigned(float v) {
  if (std::isnan(v)) return "-";
  char buf[32];
  std::snprintf(buf, sizeof buf, "%+.2f", v);
  return buf;
}

// Everything the overlay shows: the episode so far and the current decision.
// The recorders fill it; Complete() derives the labels from the action space.
struct Hud {
  std::string title;  // optional top line
  int step = 0;
  float reward = NAN;  // the last reward (NaN before the first step)
  float ret = 0;       // the return so far
  bool done = false, truncated = false;
  bool has_decision = false;
  Decision decision;
  ActionSpace space;
  std::vector<std::string> names;  // the environment's action names (discrete) or action dimensions (continuous)

  void Reset(const ActionSpace& s, std::vector<std::string> action_names) {
    step = 0, reward = NAN, ret = 0, done = truncated = false, has_decision = false;
    decision = Decision();
    space = s;
    names = std::move(action_names);
  }
  // The move taken; the step that follows reports its reward through Observe().
  void Choose(Decision d) {
    decision = std::move(d);
    has_decision = true;
    Complete();
  }
  void Observe(const StepResult& r) {
    ++step;
    reward = r.reward;
    ret += r.reward;
    done = r.done();
    truncated = r.truncated;
  }
  std::string NameOf(int i, const char* prefix) const {
    if (i >= 0 && i < static_cast<int>(names.size())) return names[i];
    return prefix + std::to_string(i);
  }
  // Fills the action label and the candidate labels the policy left empty.
  void Complete() {
    Decision& d = decision;
    if (d.label.empty()) {
      if (space.kind == ActionSpace::kDiscrete) {
        d.label = NameOf(d.action.discrete, "action ");
      } else {
        for (size_t i = 0; i < d.action.continuous.size(); ++i) {
          if (i) d.label += "  ";
          if (i < names.size()) d.label += names[i] + " ";
          d.label += HudSigned(d.action.continuous[i]);
        }
      }
    }
    if (d.labels.size() != d.values.size()) {
      d.labels.clear();
      const bool per_action = space.kind == ActionSpace::kDiscrete && static_cast<int>(d.values.size()) == space.n;
      for (size_t i = 0; i < d.values.size(); ++i) d.labels.push_back(per_action ? NameOf(static_cast<int>(i), "action ") : std::to_string(i));
    }
  }
};

}  // namespace rl
