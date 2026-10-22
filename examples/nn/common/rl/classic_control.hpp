// datamunge::nn port: copied from tflow_examples src/common/rl/classic_control.h (TF-free code, unchanged).
// Classic control environments, with the same dynamics and constants as the
// Gymnasium versions: CartPole-v1, MountainCar-v0, MountainCarContinuous-v0,
// Pendulum-v1 and Acrobot-v1.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <vector>

#include "common/rl/env.hpp"

namespace rl {

constexpr float kPi = 3.14159265358979323846f;

inline float Clamp(float v, float lo, float hi) { return std::max(lo, std::min(hi, v)); }
inline float WrapAngle(float a) {  // to [-pi, pi)
  a = std::fmod(a + kPi, 2 * kPi);
  if (a < 0) a += 2 * kPi;
  return a - kPi;
}

// ---------------------------------------------------------------------------
// CartPole: balance a pole on a cart by pushing left (0) or right (1).
// Observation [x, x_dot, theta, theta_dot]; +1 per step; episode ends when
// |theta| > 12 degrees or |x| > 2.4, or after 500 steps.
// ---------------------------------------------------------------------------
class CartPole : public EnvBase {
 public:
  explicit CartPole(int max_steps = 500) : EnvBase(max_steps) {}
  std::string name() const override { return "CartPole"; }
  int obs_size() const override { return 4; }
  ActionSpace action_space() const override { return ActionSpace::Discrete(2); }
  std::vector<std::string> action_names() const override { return {"push left", "push right"}; }
  // [x, x_dot, theta, theta_dot]; theta > 0 leans the pole toward +x.
  const std::array<float, 4>& state() const { return s_; }

  Obs Reset() override {
    BeginEpisode();
    for (float& v : s_) v = Uniform(-0.05f, 0.05f);
    return {s_.begin(), s_.end()};
  }
  StepResult step(const Action& action) override {
    const float gravity = 9.8f, masscart = 1.0f, masspole = 0.1f, total_mass = masscart + masspole,
                length = 0.5f, polemass_length = masspole * length, force_mag = 10.0f, tau = 0.02f;
    float x = s_[0], x_dot = s_[1], theta = s_[2], theta_dot = s_[3];
    const float force = action.discrete == 1 ? force_mag : -force_mag;
    const float costheta = std::cos(theta), sintheta = std::sin(theta);
    const float temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass;
    const float thetaacc = (gravity * sintheta - costheta * temp) /
                           (length * (4.0f / 3.0f - masspole * costheta * costheta / total_mass));
    const float xacc = temp - polemass_length * thetaacc * costheta / total_mass;
    x += tau * x_dot;
    x_dot += tau * xacc;
    theta += tau * theta_dot;
    theta_dot += tau * thetaacc;
    s_ = {x, x_dot, theta, theta_dot};
    StepResult r;
    r.obs = {s_.begin(), s_.end()};
    r.reward = 1.f;
    r.terminated = std::abs(x) > 2.4f || std::abs(theta) > 12 * 2 * kPi / 360;
    r.truncated = !r.terminated && CountStep();
    return r;
  }
  std::string render() const override {
    // A 41-column strip; the cart is at the centre when x = 0.
    std::string line(41, '-');
    int cart = static_cast<int>(std::round(20 + s_[0] / 2.4f * 20));
    cart = std::max(0, std::min(40, cart));
    line[cart] = '=';
    int tip = static_cast<int>(std::round(cart + std::sin(s_[2]) * 20));
    std::string pole(41, ' ');
    pole[std::max(0, std::min(40, tip))] = '|';
    return pole + "\n" + line + "\n";
  }

 private:
  std::array<float, 4> s_{};
};

// ---------------------------------------------------------------------------
// MountainCar: an underpowered car must rock back and forth to reach the
// flag at x = 0.5. Discrete actions push left/none/right; -1 per step.
// ---------------------------------------------------------------------------
class MountainCar : public EnvBase {
 public:
  explicit MountainCar(int max_steps = 200) : EnvBase(max_steps) {}
  std::string name() const override { return "MountainCar"; }
  int obs_size() const override { return 2; }
  ActionSpace action_space() const override { return ActionSpace::Discrete(3); }
  std::vector<std::string> action_names() const override { return {"push left", "coast", "push right"}; }
  float position() const { return pos_; }
  float velocity() const { return vel_; }

  Obs Reset() override {
    BeginEpisode();
    pos_ = Uniform(-0.6f, -0.4f);
    vel_ = 0.f;
    return {pos_, vel_};
  }
  StepResult step(const Action& action) override {
    const float force = (action.discrete - 1) * 0.001f;
    Integrate(force);
    StepResult r;
    r.obs = {pos_, vel_};
    r.reward = -1.f;
    r.terminated = pos_ >= 0.5f;
    r.truncated = !r.terminated && CountStep();
    return r;
  }

 protected:
  void Integrate(float force) {
    vel_ = Clamp(vel_ + force + std::cos(3 * pos_) * -0.0025f, -0.07f, 0.07f);
    pos_ = Clamp(pos_ + vel_, -1.2f, 0.6f);
    if (pos_ <= -1.2f && vel_ < 0) vel_ = 0;
  }
  float pos_ = 0, vel_ = 0;
};

// Continuous variant: action in [-1, 1] scaled by 0.0015; reward -0.1 a^2
// per step and +100 on reaching x = 0.45.
class MountainCarContinuous : public MountainCar {
 public:
  explicit MountainCarContinuous(int max_steps = 999) : MountainCar(max_steps) {}
  std::string name() const override { return "MountainCarContinuous"; }
  ActionSpace action_space() const override { return ActionSpace::Continuous({-1.f}, {1.f}); }
  std::vector<std::string> action_names() const override { return {"throttle"}; }
  StepResult step(const Action& action) override {
    const float a = Clamp(action.continuous[0], -1.f, 1.f);
    Integrate(a * 0.0015f);
    StepResult r;
    r.obs = {pos_, vel_};
    r.terminated = pos_ >= 0.45f;
    r.reward = -0.1f * a * a + (r.terminated ? 100.f : 0.f);
    r.truncated = !r.terminated && CountStep();
    return r;
  }
};

// ---------------------------------------------------------------------------
// Pendulum: swing a pendulum upright with a torque in [-2, 2]. Observation
// [cos theta, sin theta, theta_dot]; reward -(theta^2 + 0.1 theta_dot^2 +
// 0.001 u^2), so 0 is the best possible. Episodes are 200 steps.
// ---------------------------------------------------------------------------
class Pendulum : public EnvBase {
 public:
  explicit Pendulum(int max_steps = 200) : EnvBase(max_steps) {}
  std::string name() const override { return "Pendulum"; }
  int obs_size() const override { return 3; }
  ActionSpace action_space() const override { return ActionSpace::Continuous({-2.f}, {2.f}); }
  std::vector<std::string> action_names() const override { return {"torque"}; }
  float theta() const { return th_; }
  float theta_dot() const { return thdot_; }

  Obs Reset() override {
    BeginEpisode();
    th_ = Uniform(-kPi, kPi);
    thdot_ = Uniform(-1.f, 1.f);
    return ObsNow();
  }
  StepResult step(const Action& action) override {
    const float g = 10.f, m = 1.f, l = 1.f, dt = 0.05f, max_speed = 8.f, max_torque = 2.f;
    const float u = Clamp(action.continuous[0], -max_torque, max_torque);
    const float cost = WrapAngle(th_) * WrapAngle(th_) + 0.1f * thdot_ * thdot_ + 0.001f * u * u;
    thdot_ = Clamp(thdot_ + (3 * g / (2 * l) * std::sin(th_) + 3.f / (m * l * l) * u) * dt,
                   -max_speed, max_speed);
    th_ += thdot_ * dt;
    StepResult r;
    r.obs = ObsNow();
    r.reward = -cost;
    r.truncated = CountStep();
    return r;
  }
  std::string render() const override {
    // 9x9 picture of the rod; theta = 0 is upright.
    std::vector<std::string> g(9, std::string(9, ' '));
    g[4][4] = 'o';
    for (int k = 1; k <= 4; ++k) {
      int r = static_cast<int>(std::round(4 - k * std::cos(th_)));
      int c = static_cast<int>(std::round(4 + k * std::sin(th_)));
      if (r >= 0 && r < 9 && c >= 0 && c < 9) g[r][c] = '#';
    }
    std::string s;
    for (auto& row : g) s += row + "\n";
    return s;
  }

 private:
  Obs ObsNow() const { return {std::cos(th_), std::sin(th_), thdot_}; }
  float th_ = 0, thdot_ = 0;
};

// ---------------------------------------------------------------------------
// Acrobot: a two-link pendulum actuated only at the elbow; swing the tip
// above a height of 1. Actions apply torque -1, 0, +1; reward -1 per step.
// Observation [cos t1, sin t1, cos t2, sin t2, t1_dot, t2_dot].
// ---------------------------------------------------------------------------
class Acrobot : public EnvBase {
 public:
  explicit Acrobot(int max_steps = 500) : EnvBase(max_steps) {}
  std::string name() const override { return "Acrobot"; }
  int obs_size() const override { return 6; }
  ActionSpace action_space() const override { return ActionSpace::Discrete(3); }
  std::vector<std::string> action_names() const override { return {"torque -1", "torque 0", "torque +1"}; }
  float tip_height() const { return -std::cos(s_[0]) - std::cos(s_[1] + s_[0]); }
  // Joint angles from hanging straight down: the first link's and the elbow's.
  float theta1() const { return s_[0]; }
  float theta2() const { return s_[1]; }

  Obs Reset() override {
    BeginEpisode();
    for (float& v : s_) v = Uniform(-0.1f, 0.1f);
    return ObsNow();
  }
  StepResult step(const Action& action) override {
    const float torque = static_cast<float>(action.discrete - 1);
    Rk4(torque, 0.2f);
    s_[0] = WrapAngle(s_[0]);
    s_[1] = WrapAngle(s_[1]);
    s_[2] = Clamp(s_[2], -4 * kPi, 4 * kPi);
    s_[3] = Clamp(s_[3], -9 * kPi, 9 * kPi);
    StepResult r;
    r.obs = ObsNow();
    r.terminated = tip_height() > 1.f;
    r.reward = r.terminated ? 0.f : -1.f;
    r.truncated = !r.terminated && CountStep();
    return r;
  }

 private:
  using State = std::array<float, 4>;  // theta1, theta2, dtheta1, dtheta2
  Obs ObsNow() const {
    return {std::cos(s_[0]), std::sin(s_[0]), std::cos(s_[1]), std::sin(s_[1]), s_[2], s_[3]};
  }
  static State Derivs(const State& s, float torque) {
    const float m1 = 1, m2 = 1, l1 = 1, lc1 = 0.5f, lc2 = 0.5f, I1 = 1, I2 = 1, g = 9.8f;
    const float t1 = s[0], t2 = s[1], dt1 = s[2], dt2 = s[3];
    const float d1 = m1 * lc1 * lc1 + m2 * (l1 * l1 + lc2 * lc2 + 2 * l1 * lc2 * std::cos(t2)) + I1 + I2;
    const float d2 = m2 * (lc2 * lc2 + l1 * lc2 * std::cos(t2)) + I2;
    const float phi2 = m2 * lc2 * g * std::cos(t1 + t2 - kPi / 2);
    const float phi1 = -m2 * l1 * lc2 * dt2 * dt2 * std::sin(t2) -
                       2 * m2 * l1 * lc2 * dt2 * dt1 * std::sin(t2) +
                       (m1 * lc1 + m2 * l1) * g * std::cos(t1 - kPi / 2) + phi2;
    // "book" dynamics (Sutton & Barto), as in Gymnasium's default.
    const float ddt2 = (torque + d2 / d1 * phi1 - m2 * l1 * lc2 * dt1 * dt1 * std::sin(t2) - phi2) /
                       (m2 * lc2 * lc2 + I2 - d2 * d2 / d1);
    const float ddt1 = -(d2 * ddt2 + phi1) / d1;
    return {dt1, dt2, ddt1, ddt2};
  }
  void Rk4(float torque, float dt) {
    auto axpy = [](const State& a, float h, const State& k) {
      State r;
      for (int i = 0; i < 4; ++i) r[i] = a[i] + h * k[i];
      return r;
    };
    State k1 = Derivs(s_, torque);
    State k2 = Derivs(axpy(s_, dt / 2, k1), torque);
    State k3 = Derivs(axpy(s_, dt / 2, k2), torque);
    State k4 = Derivs(axpy(s_, dt, k3), torque);
    for (int i = 0; i < 4; ++i) s_[i] += dt / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
  }
  State s_{};
};

}  // namespace rl

