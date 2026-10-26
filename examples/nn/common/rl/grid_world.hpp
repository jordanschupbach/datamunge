// datamunge::nn port: copied from tflow_examples src/common/rl/grid_world.h (TF-free code, unchanged).
// Grid-world environments: a configurable GridWorld and the classic
// CliffWalking task from Sutton & Barto (example 6.6).
#pragma once

#include <set>
#include <string>
#include <utility>
#include <vector>

#include "common/rl/env.hpp"

namespace rl {

// Shared grid mechanics: 4 actions (up, right, down, left), walls block
// movement, a goal cell ends the episode, and optional "slip" noise replaces
// the chosen action with a perpendicular one.
class GridBase : public TabularEnv {
 public:
  enum Move { kUp = 0, kRight = 1, kDown = 2, kLeft = 3 };
  static constexpr const char* kArrows[4] = {"^", ">", "v", "<"};

  GridBase(int height, int width, std::pair<int, int> start, std::pair<int, int> goal,
           float slip_prob, int max_steps)
      : TabularEnv(max_steps),
        height_(height),
        width_(width),
        start_(start),
        goal_(goal),
        slip_prob_(slip_prob) {}

  int height() const { return height_; }
  int width() const { return width_; }
  int num_states() const override { return height_ * width_; }
  int state() const override { return Index(row_, col_); }
  ActionSpace action_space() const override { return ActionSpace::Discrete(4); }
  std::vector<std::string> action_names() const override { return {"up", "right", "down", "left"}; }
  int Index(int r, int c) const { return r * width_ + c; }
  std::pair<int, int> Coords(int s) const { return {s / width_, s % width_}; }
  bool IsWall(int r, int c) const { return walls_.count({r, c}) > 0; }
  bool IsGoal(int r, int c) const { return std::make_pair(r, c) == goal_; }
  void AddWall(int r, int c) { walls_.insert({r, c}); }

  Obs Reset() override {
    BeginEpisode();
    row_ = start_.first;
    col_ = start_.second;
    return StateObs();
  }

  StepResult step(const Action& action) override {
    int a = action.discrete;
    if (slip_prob_ > 0 && Uniform(0, 1) < slip_prob_) a = (a + (rng()() % 2 ? 1 : 3)) % 4;
    Outcome o = Apply(row_, col_, a);
    row_ = o.row;
    col_ = o.col;
    StepResult r;
    r.obs = StateObs();
    r.reward = o.reward;
    r.terminated = o.terminal;
    r.truncated = !o.terminal && CountStep();
    return r;
  }

  // Deterministic outcome of action a from (r, c); subclasses customise.
  struct Outcome {
    int row, col;
    float reward;
    bool terminal;
  };
  virtual Outcome Apply(int r, int c, int a) const {
    static const int dr[4] = {-1, 0, 1, 0}, dc[4] = {0, 1, 0, -1};
    int nr = r + dr[a], nc = c + dc[a];
    if (nr < 0 || nr >= height_ || nc < 0 || nc >= width_ || IsWall(nr, nc)) {
      nr = r;
      nc = c;
    }
    return {nr, nc, -1.f, IsGoal(nr, nc)};
  }

  std::vector<Transition> transitions(int s, int a) const override {
    // Plain ints rather than a structured binding: the lambda below captures
    // them, which C++17 does not allow for structured bindings.
    const int r = Coords(s).first, c = Coords(s).second;
    std::vector<Transition> out;
    if (IsGoal(r, c) || IsWall(r, c)) return out;  // absorbing / unreachable
    auto add = [&](int act, float p) {
      Outcome o = Apply(r, c, act);
      out.push_back({p, Index(o.row, o.col), o.reward, o.terminal});
    };
    if (slip_prob_ > 0) {
      add(a, 1 - slip_prob_);
      add((a + 1) % 4, slip_prob_ / 2);
      add((a + 3) % 4, slip_prob_ / 2);
    } else {
      add(a, 1.f);
    }
    return out;
  }

  // Draw the grid; `overlay(r, c)` may return a one-character string to show
  // in a cell (e.g. a policy arrow) or "" for the default glyph.
  template <typename Overlay>
  std::string Render(Overlay overlay) const {
    std::string s;
    for (int r = 0; r < height_; ++r) {
      for (int c = 0; c < width_; ++c) {
        std::string glyph = overlay(r, c);
        if (glyph.empty()) {
          if (r == row_ && c == col_) glyph = "A";
          else if (IsWall(r, c)) glyph = "#";
          else if (IsGoal(r, c)) glyph = "G";
          else if (std::make_pair(r, c) == start_) glyph = "S";
          else glyph = CellGlyph(r, c);
        }
        s += glyph + " ";
      }
      s += "\n";
    }
    return s;
  }
  std::string render() const override {
    return Render([](int, int) { return std::string(); });
  }
  virtual std::string CellGlyph(int, int) const { return "."; }

 protected:
  int height_, width_;
  std::pair<int, int> start_, goal_;
  float slip_prob_;
  std::set<std::pair<int, int>> walls_;
  int row_ = 0, col_ = 0;
};

// A 5x6 room with a wall in the middle. Reward -1 per step until the goal.
class GridWorld : public GridBase {
 public:
  explicit GridWorld(float slip_prob = 0.f, int max_steps = 100)
      : GridBase(5, 6, {0, 0}, {4, 5}, slip_prob, max_steps) {
    for (int r = 0; r < 4; ++r) AddWall(r, 3);
  }
  std::string name() const override { return "GridWorld"; }
};

// Sutton & Barto's cliff: 4x12, start bottom-left, goal bottom-right, and a
// cliff along the bottom row. Stepping into the cliff costs -100 and sends
// the agent back to the start without ending the episode.
class CliffWalking : public GridBase {
 public:
  explicit CliffWalking(int max_steps = 500) : GridBase(4, 12, {3, 0}, {3, 11}, 0.f, max_steps) {}
  std::string name() const override { return "CliffWalking"; }
  bool IsCliff(int r, int c) const { return r == 3 && c >= 1 && c <= 10; }
  Outcome Apply(int r, int c, int a) const override {
    Outcome o = GridBase::Apply(r, c, a);
    if (IsCliff(o.row, o.col)) return {start_.first, start_.second, -100.f, false};
    return o;
  }
  std::string CellGlyph(int r, int c) const override { return IsCliff(r, c) ? "~" : "."; }
};

}  // namespace rl

