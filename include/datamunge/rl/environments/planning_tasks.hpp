#pragma once
/// @file planning_tasks.hpp
/// @brief Source mutable blocking maze and 12x24 binary CartPole observations.
#include <datamunge/rl/environments/classic_control.hpp>
#include <datamunge/rl/environments/grid_world.hpp>
namespace datamunge::rl::environments {
/// @brief Mutable source blocking maze for Dyna-Q adaptation experiments.
///
/// Six rows, nine columns; start(5,3), goal(0,8), episode limit3000. Copies own grid state. Calls
/// mutating one instance require synchronization.
///
/// @par Example
/// @code
/// dm::rl::environments::BlockingMaze maze;auto obs=maze.reset(0);maze.SetWall(true);
/// @endcode
/// @see GridBase
class BlockingMaze : public GridBase {
 public:
  /// @brief Construct the source maze with the initial right-hand wall gap.
  ///
  /// @par Example
  /// @code
  /// dm::rl::environments::BlockingMaze maze;
  /// @endcode
  /// @see SetWall()
  BlockingMaze() : GridBase(6, 9, {5, 3}, {0, 8}, 0.f, 3000) { SetWall(/*gap_at_left=*/false); }
  /// @brief Return the source task name.
  ///
  /// @return BlockingMaze.
  /// @see GridBase::name()
  std::string name() const override { return "BlockingMaze"; }
  /// @brief Replace all walls with a row3 barrier and one end gap.
  ///
  /// Does not reset position, step count or random state. Moving the wall can close a gap under the
  /// current agent, matching the source experiment.
  ///
  /// @param gap_at_left True opens column0; false opens column8.
  /// @par Example
  /// @code
  /// dm::rl::environments::BlockingMaze maze;maze.SetWall(true);
  /// @endcode
  /// @see Apply()
  void SetWall(bool gap_at_left) {
    walls_.clear();
    for (int c = 0; c < 9; ++c)
      if (!(gap_at_left ? c == 0 : c == 8))
        AddWall(3, c);
  }
  /// @brief Evaluate the native grid transition and replace task reward.
  ///
  /// @param r Source row.
  /// @param c Source column.
  /// @param a Discrete cardinal action as accepted by GridBase.
  /// @return Grid transition; reward1 on a terminal goal and0 otherwise.
  /// @par Example
  /// @code
  /// dm::rl::environments::BlockingMaze maze;auto transition=maze.Apply(5,3,0);
  /// @endcode
  /// @see GridBase::Apply()
  Outcome Apply(int r, int c, int a) const override {
    Outcome o = GridBase::Apply(r, c, a);
    o.reward  = o.terminal ? 1.f : 0.f; // +1 at the goal, 0 elsewhere
    return o;
  }
};

/// @brief Native CartPole observed through the source binary12-by24 renderer.
///
/// Physics, rewards, termination and random stream are delegated to owned CartPole. Rendering
/// creates owning row-major observations; copies own physics. Mutating calls require
/// synchronization.
///
/// @par Example
/// @code
/// dm::rl::environments::WorldPixelCartPole env;auto frame=env.reset(0);
/// @endcode
/// @see CartPole, Render()
class WorldPixelCartPole : public Env {
 public:
  static constexpr int kH      = 12;      ///< Number of image rows.
  static constexpr int kW      = 24;      ///< Number of image columns.
  static constexpr int kPixels = kH * kW; ///< Flattened observation width288.
                                          /// @brief Return the source pixel task name.
                                          ///
                                          /// @return PixelCartPole.
                                          /// @see CartPole
  std::string name() const override { return "PixelCartPole"; }
  /// @brief Return flattened image width.
  ///
  /// @return 288 pixels.
  /// @see CartPole
  int obs_size() const override { return kPixels; }
  /// @brief Read the native CartPole action space.
  ///
  /// @return Owned discrete action descriptor with two actions.
  /// @see CartPole
  ActionSpace action_space() const override { return inner_.action_space(); }
  /// @brief Seed native physics for a later reset.
  ///
  /// @param seed Caller seed forwarded unchanged.
  /// @see CartPole
  void Seed(uint64_t seed) override { inner_.Seed(seed); }
  /// @brief Read the native CartPole time limit.
  ///
  /// @return Native time-limit steps.
  /// @see CartPole
  int max_steps() const override { return inner_.max_steps(); }
  /// @brief Advance native physics and replace only its observation with pixels.
  ///
  /// @param a Native CartPole discrete action.
  /// @return Owning pixels and unchanged reward/termination/truncation flags.
  /// @see CartPole
  StepResult step(const Action& a) override {
    StepResult r = inner_.step(a);
    r.obs        = Render(r.obs);
    return r;
  }
  /// @brief Borrow immutable underlying physics.
  ///
  /// @return Reference valid for this wrapper lifetime.
  /// @see CartPole
  const CartPole& inner() const { return inner_; }
  /// @brief Read native discrete action names.
  ///
  /// @return Owned names in action-index order.
  /// @see CartPole
  std::vector<std::string> action_names() const override { return inner_.action_names(); }
  /// @brief Rasterize a finite four-coordinate source CartPole observation.
  ///
  /// Three cart pixels occupy the bottom row; ten pole samples project sin(angle). Rounded cart
  /// coordinate is clamped before integer conversion. Invalid width/nonfinite coordinates throw
  /// std::invalid_argument.
  ///
  /// @param state Finite[x,velocity,angle,angular_velocity]; only x and angle affect pixels.
  /// @return Owning288-element vector with values0 or1, row-major.
  /// @par Example
  /// @code
  /// auto frame=dm::rl::environments::WorldPixelCartPole::Render({0,0,0,0});
  /// @endcode
  /// @see Ascii()
  static Obs Render(const Obs& state) {
    if (state.size() != 4)
      throw std::invalid_argument("source CartPole observation width4 required");
    for (float v : state)
      if (!std::isfinite(v))
        throw std::invalid_argument("finite CartPole observation required");
    Obs         img(kPixels, 0.f);
    const float x = state[0], theta = state[2];
    const int   cx =
        static_cast<int>(std::clamp(std::round((x + 2.4f) / 4.8f * (kW - 1)), 1.f, float(kW - 2)));
    for (int c = cx - 1; c <= cx + 1; ++c)
      img[(kH - 1) * kW + c] = 1.f;
    for (int k = 1; k <= kH - 2; ++k) {
      const int r = kH - 1 - k, c = static_cast<int>(std::round(cx + std::sin(theta) * k * 2.f));
      if (c >= 0 && c < kW)
        img[r * kW + c] = 1.f;
    }
    return img;
  }
  /// @brief Render one flattened source frame as twelve indented ASCII rows.
  ///
  /// Throws std::invalid_argument for the wrong width.
  ///
  /// @param img Exactly288 values; each value>.5 becomes#, all others become a dot.
  /// @return Owned string with six-space row prefixes and trailing newlines.
  /// @par Example
  /// @code
  /// auto frame=dm::rl::environments::WorldPixelCartPole::Render({0,0,0,0});
  /// auto text=dm::rl::environments::WorldPixelCartPole::Ascii(frame);
  /// @endcode
  /// @see Render()
  static std::string Ascii(const std::vector<float>& img) {
    if (img.size() != kPixels)
      throw std::invalid_argument("source frame width288 required");
    std::string s;
    for (int r = 0; r < kH; ++r) {
      s += "      ";
      for (int c = 0; c < kW; ++c)
        s += img[r * kW + c] > 0.5f ? '#' : '.';
      s += '\n';
    }
    return s;
  }

 protected:
  /// @brief Reset native physics and render its observation.
  ///
  /// @return Owning binary pixels using previously configured seed.
  /// @see CartPole
  Obs Reset() override { return Render(inner_.reset()); }

 private:
  CartPole inner_; ///< Owned native physics and random state.
};

} // namespace datamunge::rl::environments
