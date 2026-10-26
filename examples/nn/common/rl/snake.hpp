// datamunge::nn port: copied from tflow_examples src/common/rl/snake.h (TF-free code, unchanged).
// Snake on a small grid, with a pixel-style observation: three planes
// (head, body, food) of height x width, flattened to a float vector. Eating
// food gives +1, dying gives -1, every other step 0. The episode is truncated
// if the snake goes too long without eating.
#pragma once

#include <deque>
#include <string>
#include <utility>
#include <vector>

#include "common/rl/env.hpp"

namespace rl {

class Snake : public EnvBase {
 public:
  enum Dir { kUp = 0, kRight = 1, kDown = 2, kLeft = 3 };
  Snake(int height = 8, int width = 8) : EnvBase(0), h_(height), w_(width) {}
  std::string name() const override { return "Snake"; }
  int obs_size() const override { return 3 * h_ * w_; }
  // Observation shape as [planes, height, width], for reshaping into an image.
  std::vector<int> obs_shape() const { return {3, h_, w_}; }
  ActionSpace action_space() const override { return ActionSpace::Discrete(4); }
  std::vector<std::string> action_names() const override { return {"up", "right", "down", "left"}; }
  int length() const { return static_cast<int>(body_.size()); }
  std::pair<int, int> head() const { return body_.front(); }
  // Body cells from the head to the tail.
  const std::deque<std::pair<int, int>>& body() const { return body_; }
  std::pair<int, int> food() const { return food_; }
  int height() const { return h_; }
  int width() const { return w_; }
  bool Occupied(int r, int c) const {
    for (const auto& seg : body_)
      if (seg == std::make_pair(r, c)) return true;
    return false;
  }

  Obs Reset() override {
    BeginEpisode();
    body_.clear();
    body_.push_back({h_ / 2, w_ / 2});
    body_.push_back({h_ / 2, w_ / 2 - 1});
    dir_ = kRight;
    since_food_ = 0;
    PlaceFood();
    return ObsNow();
  }

  StepResult step(const Action& action) override {
    // Reversing into the body is ignored; the snake keeps its direction.
    const int a = action.discrete;
    if ((a + 2) % 4 != dir_) dir_ = static_cast<Dir>(a);
    static const int dr[4] = {-1, 0, 1, 0}, dc[4] = {0, 1, 0, -1};
    std::pair<int, int> next = {body_.front().first + dr[dir_], body_.front().second + dc[dir_]};
    StepResult r;
    const bool ate = next == food_;
    if (!ate) body_.pop_back();  // the tail moves away unless we grow
    const bool hit_wall = next.first < 0 || next.first >= h_ || next.second < 0 || next.second >= w_;
    bool hit_self = false;
    for (const auto& seg : body_) hit_self |= seg == next;
    if (hit_wall || hit_self) {
      r.reward = -1.f;
      r.terminated = true;
      r.obs = ObsNow();
      return r;
    }
    body_.push_front(next);
    if (ate) {
      r.reward = 1.f;
      since_food_ = 0;
      if (length() == h_ * w_) r.terminated = true;  // the board is full: won
      else PlaceFood();
    } else if (++since_food_ > 4 * h_ * w_) {
      r.truncated = true;
    }
    r.obs = ObsNow();
    return r;
  }

  std::string render() const override {
    std::vector<std::string> g(h_, std::string(w_, '.'));
    for (const auto& seg : body_) g[seg.first][seg.second] = 'o';
    g[body_.front().first][body_.front().second] = '@';
    g[food_.first][food_.second] = '*';
    std::string s;
    for (auto& row : g) s += row + "\n";
    return s;
  }

 private:
  Obs ObsNow() const {
    Obs o(3 * h_ * w_, 0.f);
    const int plane = h_ * w_;
    o[0 * plane + body_.front().first * w_ + body_.front().second] = 1.f;
    for (size_t i = 1; i < body_.size(); ++i)
      o[1 * plane + body_[i].first * w_ + body_[i].second] = 1.f;
    o[2 * plane + food_.first * w_ + food_.second] = 1.f;
    return o;
  }
  void PlaceFood() {
    std::vector<std::pair<int, int>> free;
    for (int r = 0; r < h_; ++r)
      for (int c = 0; c < w_; ++c) {
        bool occupied = false;
        for (const auto& seg : body_) occupied |= seg == std::make_pair(r, c);
        if (!occupied) free.push_back({r, c});
      }
    food_ = free[rng()() % free.size()];
  }
  int h_, w_;
  std::deque<std::pair<int, int>> body_;
  Dir dir_ = kRight;
  std::pair<int, int> food_{0, 0};
  int since_food_ = 0;
};

}  // namespace rl

