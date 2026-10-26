// datamunge::nn port: copied from tflow_examples src/common/rl/lunar_lander.h (TF-free code, unchanged).
// A simplified LunarLander: a rigid body with a main engine and two side
// thrusters must touch down gently on a pad. No Box2D; the body is a point
// mass with orientation, integrated with semi-implicit Euler.
//
// Observation (8): [x, y, vx, vy, angle, angular_velocity, left_leg_contact,
// right_leg_contact]. Actions: 0 noop, 1 fire left thruster, 2 fire main
// engine, 3 fire right thruster. Rewards follow the Gymnasium shaping: closer,
// slower and more upright is better; fuel costs a little; landing on the pad
// with low speed gives +100, crashing or leaving the screen gives -100.
#pragma once

#include <cmath>
#include <string>
#include <vector>

#include "common/rl/env.hpp"

namespace rl {

class LunarLander : public EnvBase {
 public:
  explicit LunarLander(int max_steps = 1000) : EnvBase(max_steps) {}
  std::string name() const override { return "LunarLander"; }
  int obs_size() const override { return 8; }
  ActionSpace action_space() const override { return ActionSpace::Discrete(4); }
  std::vector<std::string> action_names() const override { return {"no-op", "left thruster", "main engine", "right thruster"}; }

  struct State {
    float x, y, vx, vy, angle, angular_velocity;
    bool landed = false, crashed = false;
  };
  const State& state() const { return s_; }
  // The action of the last step (0 noop, 1 left thruster, 2 main engine, 3 right thruster).
  int last_action() const { return last_action_; }

  Obs Reset() override {
    BeginEpisode();
    s_ = State{Uniform(-0.3f, 0.3f), 1.4f, Uniform(-0.3f, 0.3f), Uniform(-0.3f, 0.f),
               Uniform(-0.2f, 0.2f), 0.f};
    last_action_ = 0;
    prev_shaping_ = Shaping();
    return ObsNow();
  }

  StepResult step(const Action& action) override {
    const float dt = 0.02f, gravity = 1.6f, main_thrust = 4.0f, side_thrust = 0.4f,
                side_torque = 3.0f, pad_half_width = 0.25f;
    const int a = action.discrete;
    last_action_ = a;
    float fx = 0, fy = -gravity, torque = 0, fuel = 0;
    if (a == 2) {  // main engine pushes along the body's up axis
      fx += -std::sin(s_.angle) * main_thrust;
      fy += std::cos(s_.angle) * main_thrust;
      fuel = 0.3f;
    } else if (a == 1 || a == 3) {
      const float dir = a == 1 ? 1.f : -1.f;  // left thruster pushes right / rotates ccw
      fx += dir * std::cos(s_.angle) * side_thrust;
      fy += dir * std::sin(s_.angle) * side_thrust;
      torque += dir * side_torque;
      fuel = 0.03f;
    }
    s_.vx += fx * dt;
    s_.vy += fy * dt;
    s_.angular_velocity += torque * dt;
    s_.x += s_.vx * dt;
    s_.y += s_.vy * dt;
    s_.angle += s_.angular_velocity * dt;

    StepResult r;
    if (s_.y <= 0.f) {  // touched the ground
      s_.y = 0.f;
      const bool gentle = std::abs(s_.vy) < 0.5f && std::abs(s_.vx) < 0.5f &&
                          std::abs(s_.angle) < 0.3f;
      const bool on_pad = std::abs(s_.x) <= pad_half_width;
      if (gentle && on_pad) s_.landed = true;
      else s_.crashed = true;
      s_.vx = s_.vy = s_.angular_velocity = 0;
    }
    if (std::abs(s_.x) > 1.f || s_.y > 2.f) s_.crashed = true;

    const float shaping = Shaping();
    r.reward = shaping - prev_shaping_ - fuel;
    prev_shaping_ = shaping;
    if (s_.landed) r.reward += 100.f;
    if (s_.crashed) r.reward -= 100.f;
    r.terminated = s_.landed || s_.crashed;
    r.truncated = !r.terminated && CountStep();
    r.obs = ObsNow();
    return r;
  }

  std::string render() const override {
    // 21 columns x 11 rows; the pad is drawn on the bottom row.
    std::vector<std::string> g(11, std::string(21, ' '));
    for (int c = 0; c < 21; ++c) g[10][c] = std::abs(c - 10) <= 2 ? '=' : '_';
    int c = static_cast<int>(std::round(10 + s_.x * 10));
    int rr = static_cast<int>(std::round(9 - s_.y / 1.5f * 9));
    if (c >= 0 && c < 21 && rr >= 0 && rr < 11) g[rr][c] = s_.crashed ? '*' : 'V';
    std::string s;
    for (auto& row : g) s += row + "\n";
    return s;
  }

 private:
  Obs ObsNow() const {
    const bool contact = s_.y <= 0.f;
    return {s_.x, s_.y, s_.vx, s_.vy, s_.angle, s_.angular_velocity, contact ? 1.f : 0.f,
            contact ? 1.f : 0.f};
  }
  // Potential-based shaping: higher when close, slow and upright.
  float Shaping() const {
    return -100.f * std::sqrt(s_.x * s_.x + s_.y * s_.y) -
           100.f * std::sqrt(s_.vx * s_.vx + s_.vy * s_.vy) - 100.f * std::abs(s_.angle);
  }
  State s_{};
  int last_action_ = 0;
  float prev_shaping_ = 0;
};

}  // namespace rl

