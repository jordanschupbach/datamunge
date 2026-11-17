#pragma once
/// @file lunar_lander.hpp
/// @brief Source point-mass lunar landing with semi-implicit Euler and explicit reward shaping.
#include <cmath>
#include <datamunge/rl/environments/interface.hpp>
#include <string>
#include <vector>
namespace datamunge::rl::environments {
/// @brief Source simplified lunar-landing environment with point-mass rotational physics.
///
/// Eight Float32 observation coordinates are x,y,vx,vy,angle,angular velocity and two identical
/// ground contacts. Semi-implicit Euler uses dt.02, gravity1.6 and source thrust/fuel constants.
/// This is not Box2D/Gym physics. Copies own state/RNG; callers synchronize mutation. No
/// episode/controller loop is owned.
///
/// @par Example
/// @code
/// dm::rl::environments::LunarLander env;auto start=env.reset(0);auto next=env.step(0);
/// @endcode
/// @par References
/// - Gymnasium LunarLander reward/interface reference,
/// https://gymnasium.farama.org/environments/box2d/lunar_lander/ ; source hand-written point-mass
/// protocol retained.
/// @see EnvBase, State
class LunarLander : public EnvBase {
 public:
  /// @brief Initialize a native landing task and optional episode cap.
  ///
  /// @param max_steps Nonnegative nonterminal step limit, default1000;0 disables the limit.
  /// @throws std::invalid_argument The episode limit is negative.
  /// @par Example
  /// @code
  /// dm::rl::environments::LunarLander env;auto start=env.reset(0);auto next=env.step(0);
  /// @endcode
  /// @see EnvBase
  explicit LunarLander(int max_steps = 1000) : EnvBase(max_steps) {}
  /// @brief Read the source task name.
  ///
  /// @return LunarLander.
  /// @see LunarLander
  std::string name() const override { return "LunarLander"; }
  /// @brief Read the flattened observation width.
  ///
  /// @return Eight scalar coordinates.
  /// @see LunarLander
  int obs_size() const override { return 8; }
  /// @brief Read the discrete thruster action descriptor.
  ///
  /// @return Four actions: no-op,left,main,right.
  /// @see LunarLander
  ActionSpace action_space() const override { return ActionSpace::Discrete(4); }
  /// @brief Read thruster names in index order.
  ///
  /// @return Owned four-name vector.
  /// @see LunarLander
  std::vector<std::string> action_names() const override {
    return {"no-op", "left thruster", "main engine", "right thruster"};
  }

  /// @brief Owned Float32 physical coordinates and source terminal flags.
  ///
  /// @par Example
  /// @code
  /// dm::rl::environments::LunarLander env;env.reset(0);const auto& state=env.state();
  /// @endcode
  /// @see state()
  struct State {
    float x;                ///< Horizontal position, pad centered at0.
    float y;                ///< Height above ground.
    float vx;               ///< Horizontal velocity.
    float vy;               ///< Vertical velocity.
    float angle;            ///< Orientation in radians.
    float angular_velocity; ///< Angular velocity in radians per source time unit.
    bool  landed  = false;  ///< True after a gentle pad touchdown.
    bool  crashed = false;  ///< True after unsafe touchdown or leaving source bounds.
  };
  /// @brief Borrow the current physical state.
  ///
  /// @return Reference valid for owner lifetime; changes on reset/step.
  /// @see LunarLander
  const State& state() const { return s_; }
  // The action of the last step (0 noop, 1 left thruster, 2 main engine, 3 right thruster).
  /// @brief Read the most recently applied action.
  ///
  /// @return Discrete action0..3;0 after reset.
  /// @see LunarLander
  int last_action() const { return last_action_; }

  /// @brief Reset with source uniform draws and initialize shaping potential.
  ///
  /// @return Owning eight-coordinate observation; consumes the owned continuing RNG.
  /// @par Example
  /// @code
  /// dm::rl::environments::LunarLander env;auto start=env.reset(0);auto next=env.step(0);
  /// @endcode
  /// @see LunarLander
  Obs Reset() override {
    BeginEpisode();
    s_ = State{
        Uniform(-0.3f, 0.3f),
        1.4f,
        Uniform(-0.3f, 0.3f),
        Uniform(-0.3f, 0.f),
        Uniform(-0.2f, 0.2f),
        0.f};
    last_action_  = 0;
    prev_shaping_ = Shaping();
    return ObsNow();
  }

  /// @brief Advance source physics, shaping and independent episode-end flags.
  ///
  /// A gentle touchdown has |vy|<.5, |vx|<.5 and |angle|<.3, and requires |x|<=.25 for landing.
  /// Landing adds100; crash subtracts100. Terminal steps short-circuit CountStep as in source. No
  /// automatic reset is performed.
  ///
  /// @param action Discrete thruster index0..3; validated before mutation.
  /// @return Owning next observation, shaped reward minus fuel, terminal/truncated flags.
  /// @throws std::invalid_argument Action descriptor is incompatible or out of range.
  /// @par Example
  /// @code
  /// dm::rl::environments::LunarLander env;auto start=env.reset(0);auto next=env.step(0);
  /// @endcode
  /// @see EnvBase::CountStep()
  StepResult step(const Action& action) override {
    ValidateAction(action_space(), action);
    const float dt = 0.02f, gravity = 1.6f, main_thrust = 4.0f, side_thrust = 0.4f,
                side_torque = 3.0f, pad_half_width = 0.25f;
    const int a  = action.discrete;
    last_action_ = a;
    float fx = 0, fy = -gravity, torque = 0, fuel = 0;
    if (a == 2) { // main engine pushes along the body's up axis
      fx += -std::sin(s_.angle) * main_thrust;
      fy += std::cos(s_.angle) * main_thrust;
      fuel = 0.3f;
    } else if (a == 1 || a == 3) {
      const float dir = a == 1 ? 1.f : -1.f; // left thruster pushes right / rotates ccw
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
    if (s_.y <= 0.f) { // touched the ground
      s_.y = 0.f;
      const bool gentle =
          std::abs(s_.vy) < 0.5f && std::abs(s_.vx) < 0.5f && std::abs(s_.angle) < 0.3f;
      const bool on_pad = std::abs(s_.x) <= pad_half_width;
      if (gentle && on_pad)
        s_.landed = true;
      else
        s_.crashed = true;
      s_.vx = s_.vy = s_.angular_velocity = 0;
    }
    if (std::abs(s_.x) > 1.f || s_.y > 2.f)
      s_.crashed = true;

    const float shaping = Shaping();
    r.reward            = shaping - prev_shaping_ - fuel;
    prev_shaping_       = shaping;
    if (s_.landed)
      r.reward += 100.f;
    if (s_.crashed)
      r.reward -= 100.f;
    r.terminated = s_.landed || s_.crashed;
    r.truncated  = !r.terminated && CountStep();
    r.obs        = ObsNow();
    return r;
  }

  /// @brief Rasterize current state into a compact ASCII landing view.
  ///
  /// @return Owned21-column by11-row string with a newline per row.
  /// @par Example
  /// @code
  /// dm::rl::environments::LunarLander env;auto start=env.reset(0);auto next=env.step(0);
  /// @endcode
  /// @see LunarLander
  std::string render() const override {
    // 21 columns x 11 rows; the pad is drawn on the bottom row.
    std::vector<std::string> g(11, std::string(21, ' '));
    for (int c = 0; c < 21; ++c)
      g[10][c] = std::abs(c - 10) <= 2 ? '=' : '_';
    int c  = static_cast<int>(std::round(10 + s_.x * 10));
    int rr = static_cast<int>(std::round(9 - s_.y / 1.5f * 9));
    if (c >= 0 && c < 21 && rr >= 0 && rr < 11)
      g[rr][c] = s_.crashed ? '*' : 'V';
    std::string s;
    for (auto& row : g)
      s += row + "\n";
    return s;
  }

 private:
  /// @brief Pack the source observation from current physical state.
  ///
  /// @return Owning eight Float32 coordinates with identical leg contacts.
  /// @see LunarLander
  Obs ObsNow() const {
    const bool contact = s_.y <= 0.f;
    return {
        s_.x,
        s_.y,
        s_.vx,
        s_.vy,
        s_.angle,
        s_.angular_velocity,
        contact ? 1.f : 0.f,
        contact ? 1.f : 0.f};
  }
  // Potential-based shaping: higher when close, slow and upright.
  /// @brief Compute the source distance/speed/upright shaping potential.
  ///
  /// @return Negative100 times each distance/speed/absolute angle sum.
  /// @see LunarLander
  float Shaping() const {
    return -100.f * std::sqrt(s_.x * s_.x + s_.y * s_.y) -
           100.f * std::sqrt(s_.vx * s_.vx + s_.vy * s_.vy) - 100.f * std::abs(s_.angle);
  }
  State s_{};              ///< Owned physical/terminal state.
  int   last_action_  = 0; ///< Last source thruster index.
  float prev_shaping_ = 0; ///< Previous potential used for reward differences.
};

} // namespace datamunge::rl::environments
