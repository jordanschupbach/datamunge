// datamunge::nn port: copied from tflow_examples src/common/rl/grid_world.h (TF-free dynamics;
// checked native public inputs). Grid-world environments: a configurable GridWorld and the classic
// CliffWalking task from Sutton & Barto (example 6.6).
#pragma once
/// @file grid_world.hpp
/// @brief Native grid world simulation components for explicit caller-owned rollouts.

#include <datamunge/rl/environments/interface.hpp>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace datamunge::rl::environments {

// Shared grid mechanics: 4 actions (up, right, down, left), walls block
// movement, a goal cell ends the episode, and optional "slip" noise replaces
// the chosen action with a perpendicular one.
/// @brief Own the GridBase simulation state or interface.
///
/// Instances own mutable state and a local RNG; concurrent access to one instance requires external
/// synchronization. Returned observations own their values. Reset before stepping and after done.
/// VecEnv owns its children; client calls remain serialized, while workers access distinct
/// children.
/// @par Example
/// @code
/// namespace e=dm::rl::environments;e::GridWorld concrete;e::GridBase& obj=concrete;obj.reset(1);
/// @endcode
/// @par References
/// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/ ;
/// https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
/// - Sutton and Barto (2018), Reinforcement Learning, second edition,
/// http://incompleteideas.net/book/the-book-2nd.html .
/// @see Env, VecEnv
class GridBase : public TabularEnv {
 public:
  /// @brief Source action-direction or reward-family enumeration.
  /// @par Example
  /// @code
  /// auto choice=dm::rl::environments::GridBase::kUp;
  /// @endcode
  enum Move {
    kUp    = 0, ///< Move up (decrease row).
    kRight = 1, ///< Move right (increase column).
    kDown  = 2, ///< Move down (increase row).
    kLeft  = 3, ///< Move left (decrease column).
  };
  /// @brief Fixed up/right/down/left ASCII glyphs.
  static constexpr const char* kArrows[4] = {"^", ">", "v", "<"};

  /// @brief Construct GridBase with independent owned simulation state.
  /// @param height Board row count; source geometry must be positive and is bounded1024.
  /// @param width Board column count; source geometry must be positive and is bounded1024.
  /// @param start Valid start row/column.
  /// @param goal Valid absorbing-goal row/column.
  /// @param slip_prob Finite probability in[0,1] of a perpendicular replacement move.
  /// @param max_steps Nonnegative source time limit; zero means unlimited.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @throws std::invalid_argument Invalid checked bounds, geometry or action.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::GridWorld concrete;e::GridBase& obj=concrete;obj.reset(1);
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  GridBase(
      int                 height,
      int                 width,
      std::pair<int, int> start,
      std::pair<int, int> goal,
      float               slip_prob,
      int                 max_steps)
      : TabularEnv(max_steps),
        height_(height),
        width_(width),
        start_(start),
        goal_(goal),
        slip_prob_(slip_prob) {
    CheckedCount(height, 1024);
    CheckedCount(width, 1024);
    if (!std::isfinite(slip_prob) || slip_prob < 0 || slip_prob > 1 || start.first < 0 ||
        start.first >= height || start.second < 0 || start.second >= width || goal.first < 0 ||
        goal.first >= height || goal.second < 0 || goal.second >= width)
      throw std::invalid_argument("grid geometry/slip");
  }

  /// @brief Return the number of board rows.
  /// @return Owning value or borrowed reference exactly as indicated by the return type;
  /// observation/state units follow the owning class.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::GridWorld concrete;e::GridBase&
  /// obj=concrete;obj.reset(1);auto value=obj.height();
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  int height() const { return height_; }
  /// @brief Return the number of board columns.
  /// @return Owning value or borrowed reference exactly as indicated by the return type;
  /// observation/state units follow the owning class.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::GridWorld concrete;e::GridBase&
  /// obj=concrete;obj.reset(1);auto value=obj.width();
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  int width() const { return width_; }
  /// @brief Return the number of finite MDP state indices.
  /// @return Owning value or borrowed reference exactly as indicated by the return type;
  /// observation/state units follow the owning class.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::GridWorld concrete;e::GridBase&
  /// obj=concrete;obj.reset(1);auto value=obj.num_states();
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  int num_states() const override { return height_ * width_; }
  /// @brief Return the current flattened finite-state index or borrow the physical state as
  /// specified by its type.
  /// @return Current state value, or const borrowed physical-state array valid until mutation.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::GridWorld concrete;e::GridBase&
  /// obj=concrete;obj.reset(1);auto value=obj.state();
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  int state() const override { return Index(row_, col_); }
  /// @brief Return an owning copy of the discrete or continuous action space.
  /// @return Owning value or borrowed reference exactly as indicated by the return type;
  /// observation/state units follow the owning class.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::GridWorld concrete;e::GridBase&
  /// obj=concrete;obj.reset(1);auto value=obj.action_space();
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  ActionSpace action_space() const override { return ActionSpace::Discrete(4); }
  /// @brief Return owning labels for discrete actions or continuous dimensions.
  /// @return Owning value or borrowed reference exactly as indicated by the return type;
  /// observation/state units follow the owning class.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::GridWorld concrete;e::GridBase&
  /// obj=concrete;obj.reset(1);auto value=obj.action_names();
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  std::vector<std::string> action_names() const override { return {"up", "right", "down", "left"}; }
  /// @brief Flatten a row and column in row-major order.
  /// @param r Board row within the owning grid.
  /// @param c Board column within the owning grid.
  /// @return Owning value or borrowed reference exactly as indicated by the return type;
  /// observation/state units follow the owning class.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @throws std::out_of_range The checked cell, state or row index is outside its owning storage.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::GridWorld concrete;e::GridBase&
  /// obj=concrete;obj.reset(1);auto index=obj.Index(0,0);
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  int Index(int r, int c) const {
    if (r < 0 || r >= height_ || c < 0 || c >= width_)
      throw std::out_of_range("grid cell");
    return r * width_ + c;
  }
  /// @brief Unflatten a row-major state index into row and column.
  /// @param s Finite MDP state index in0..num_states()-1.
  /// @return Owning value or borrowed reference exactly as indicated by the return type;
  /// observation/state units follow the owning class.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @throws std::out_of_range The checked cell, state or row index is outside its owning storage.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::GridWorld concrete;e::GridBase&
  /// obj=concrete;obj.reset(1);auto cell=obj.Coords(0);
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  std::pair<int, int> Coords(int s) const {
    if (s < 0 || s >= num_states())
      throw std::out_of_range("grid state");
    return {s / width_, s % width_};
  }
  /// @brief Test whether the cell belongs to the explicit wall set.
  /// @param r Board row within the owning grid.
  /// @param c Board column within the owning grid.
  /// @return Owning value or borrowed reference exactly as indicated by the return type;
  /// observation/state units follow the owning class.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::GridWorld concrete;e::GridBase&
  /// obj=concrete;obj.reset(1);auto value=obj.IsWall(0,0);
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  bool IsWall(int r, int c) const { return walls_.count({r, c}) > 0; }
  /// @brief Test whether the cell is the configured absorbing goal.
  /// @param r Board row within the owning grid.
  /// @param c Board column within the owning grid.
  /// @return Owning value or borrowed reference exactly as indicated by the return type;
  /// observation/state units follow the owning class.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::GridWorld concrete;e::GridBase&
  /// obj=concrete;obj.reset(1);auto value=obj.IsGoal(0,0);
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  bool IsGoal(int r, int c) const { return std::make_pair(r, c) == goal_; }
  /// @brief Insert a valid non-endpoint cell into the wall set.
  /// @param r Board row within the owning grid.
  /// @param c Board column within the owning grid.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @throws std::invalid_argument Invalid checked bounds, geometry or action.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::GridWorld concrete;e::GridBase&
  /// obj=concrete;obj.reset(1);obj.AddWall(1,1);
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  void AddWall(int r, int c) {
    if (r < 0 || r >= height_ || c < 0 || c >= width_ || std::make_pair(r, c) == start_ ||
        std::make_pair(r, c) == goal_)
      throw std::invalid_argument("wall outside grid or on endpoint");
    walls_.insert({r, c});
  }

  /// @brief Reset the simulation state and per-episode counter, retaining the current local RNG
  /// stream.
  /// @return Owning initial observation for environments; board Reset mutates in place.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::GridWorld concrete;e::GridBase&
  /// obj=concrete;obj.reset(1);obj.Reset();
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  Obs Reset() override {
    BeginEpisode();
    row_ = start_.first;
    col_ = start_.second;
    return StateObs();
  }

  /// @brief Advance the mutable simulation once, returning observation, reward and distinct
  /// terminal/time-limit flags.
  /// @param action Action in this instance space; board actions must appear in LegalActions().
  /// Continuous physics clips finite torque/throttle to bounds.
  /// @return Owning StepResult; vector stepping returns one owning row per environment with
  /// separate final_obs on auto-reset.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @throws std::invalid_argument Invalid checked bounds, geometry or action.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::GridWorld concrete;e::GridBase&
  /// obj=concrete;obj.reset(1);auto result=obj.step(e::Action(0));
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  StepResult step(const Action& action) override {
    ValidateAction(action_space(), action);
    int a = action.discrete;
    if (slip_prob_ > 0 && Uniform(0, 1) < slip_prob_)
      a = (a + (rng()() % 2 ? 1 : 3)) % 4;
    Outcome o = Apply(row_, col_, a);
    row_      = o.row;
    col_      = o.col;
    StepResult r;
    r.obs        = StateObs();
    r.reward     = o.reward;
    r.terminated = o.terminal;
    r.truncated  = !o.terminal && CountStep();
    return r;
  }

  // Deterministic outcome of action a from (r, c); subclasses customise.
  /// @brief Own the Outcome simulation state or interface.
  ///
  /// This value owns its described fields; it has no global state or hidden random generator.
  /// Returned owning values remain valid after the source object is destroyed. Borrowed buffers
  /// require the owner to remain alive and expire when their storage changes.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::GridBase::Outcome obj{};
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, VecEnv
  struct Outcome {
    int   row;      ///< Successor row.
    int   col;      ///< Successor column.
    float reward;   ///< Immediate transition reward.
    bool  terminal; ///< True when this deterministic outcome reaches an MDP terminal state.
  };
  /// @brief Apply the legal board move or compute the deterministic grid outcome, as specified by
  /// the owning class.
  /// @param r Board row within the owning grid.
  /// @param c Board column within the owning grid.
  /// @param a Discrete action index or owning action vector, according to this overload.
  /// @return Owning value or borrowed reference exactly as indicated by the return type;
  /// observation/state units follow the owning class.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @throws std::invalid_argument Invalid checked bounds, geometry or action.
  /// @throws std::out_of_range Checked cell, arm, action or environment index is out of range.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::GridWorld concrete;e::GridBase&
  /// obj=concrete;obj.reset(1);auto outcome=obj.Apply(0,0,1);
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  virtual Outcome Apply(int r, int c, int a) const {
    if (r < 0 || r >= height_ || c < 0 || c >= width_ || a < 0 || a >= 4)
      throw std::out_of_range("grid cell/action");
    static const int dr[4] = {-1, 0, 1, 0}, dc[4] = {0, 1, 0, -1};
    int              nr = r + dr[a], nc = c + dc[a];
    if (nr < 0 || nr >= height_ || nc < 0 || nc >= width_ || IsWall(nr, nc)) {
      nr = r;
      nc = c;
    }
    return {nr, nc, -1.f, IsGoal(nr, nc)};
  }

  /// @brief Enumerate owning probability/reward/successor records for one finite-MDP state and
  /// action.
  /// @param s Finite MDP state index in0..num_states()-1.
  /// @param a Discrete action index or owning action vector, according to this overload.
  /// @return Owning value or borrowed reference exactly as indicated by the return type;
  /// observation/state units follow the owning class.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @throws std::invalid_argument Invalid checked bounds, geometry or action.
  /// @throws std::out_of_range Checked cell, arm, action or environment index is out of range.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::GridWorld concrete;e::GridBase&
  /// obj=concrete;obj.reset(1);auto outcomes=obj.transitions(0,1);
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  std::vector<Transition> transitions(int s, int a) const override {
    if (s < 0 || s >= num_states() || a < 0 || a >= 4)
      throw std::out_of_range("grid transition index");
    // Plain ints rather than a structured binding: the lambda below captures
    // them, which C++17 does not allow for structured bindings.
    const int               r = Coords(s).first, c = Coords(s).second;
    std::vector<Transition> out;
    if (IsGoal(r, c) || IsWall(r, c))
      return out; // absorbing / unreachable
    /// @brief Append one deterministic outcome with its explicit slip probability.
    /// @param act Intended or perpendicular action index.
    /// @param p Probability of this source transition branch.
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
  template<typename Overlay>
  /// @brief Render the grid or game as owning ASCII text; an overlay callback is evaluated once per
  /// cell.
  /// @param overlay Caller callback(row,column) returning an owning replacement glyph or empty for
  /// the default.
  /// @return Owning value or borrowed reference exactly as indicated by the return type;
  /// observation/state units follow the owning class.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::GridWorld concrete;e::GridBase&
  /// obj=concrete;obj.reset(1);auto picture=obj.Render([](int,int){return std::string{};});
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  std::string Render(Overlay overlay) const {
    std::string s;
    for (int r = 0; r < height_; ++r) {
      for (int c = 0; c < width_; ++c) {
        std::string glyph = overlay(r, c);
        if (glyph.empty()) {
          if (r == row_ && c == col_)
            glyph = "A";
          else if (IsWall(r, c))
            glyph = "#";
          else if (IsGoal(r, c))
            glyph = "G";
          else if (std::make_pair(r, c) == start_)
            glyph = "S";
          else
            glyph = CellGlyph(r, c);
        }
        s += glyph + " ";
      }
      s += "\n";
    }
    return s;
  }
  /// @brief Return an owning ASCII picture of the current simulation state.
  /// @return Owning value or borrowed reference exactly as indicated by the return type;
  /// observation/state units follow the owning class.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::GridWorld concrete;e::GridBase&
  /// obj=concrete;obj.reset(1);auto value=obj.render();
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  std::string render() const override {
    return Render([](int, int) { return std::string(); });
  }
  /// @brief Return the default terrain glyph for the cell.
  /// @param r Board row within the owning grid.
  /// @param c Board column within the owning grid.
  /// @return Owning value or borrowed reference exactly as indicated by the return type;
  /// observation/state units follow the owning class.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::GridWorld concrete;e::GridBase&
  /// obj=concrete;obj.reset(1);auto value=obj.CellGlyph(0,0);
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  virtual std::string CellGlyph(int r, int c) const { return "."; }

 protected:
  int                 height_;          ///< Owned grid height in cells.
  int                 width_;           ///< Owned grid width in cells.
  std::pair<int, int> start_;           ///< Owned start row/column.
  std::pair<int, int> goal_;            ///< Owned absorbing-goal row/column.
  float               slip_prob_;       ///< Per-step perpendicular replacement probability in[0,1].
  std::set<std::pair<int, int>> walls_; ///< Owned wall-coordinate set.
  int                           row_ = 0; ///< Current agent row.
  int                           col_ = 0; ///< Current agent column.
};

// A 5x6 room with a wall in the middle. Reward -1 per step until the goal.
/// @brief Source5x6 wall grid with negative unit step rewards.
///
/// Instances own mutable state and a local RNG; concurrent access to one instance requires external
/// synchronization. Returned observations own their values. Reset before stepping and after done.
/// VecEnv owns its children; client calls remain serialized, while workers access distinct
/// children.
/// @par Example
/// @code
/// namespace e=dm::rl::environments;e::GridWorld obj;obj.reset(1);
/// @endcode
/// @par References
/// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/ ;
/// https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
/// - Sutton and Barto (2018), Reinforcement Learning, second edition,
/// http://incompleteideas.net/book/the-book-2nd.html .
/// @see Env, VecEnv
class GridWorld : public GridBase {
 public:
  /// @brief Construct GridWorld with independent owned simulation state.
  /// @param slip_prob Finite probability in[0,1] of a perpendicular replacement move.
  /// @param max_steps Nonnegative source time limit; zero means unlimited.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @throws std::invalid_argument Invalid checked bounds, geometry or action.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::GridWorld obj;obj.reset(1);
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  explicit GridWorld(float slip_prob = 0.f, int max_steps = 100)
      : GridBase(5, 6, {0, 0}, {4, 5}, slip_prob, max_steps) {
    for (int r = 0; r < 4; ++r)
      AddWall(r, 3);
  }
  /// @brief Return the stable human-readable simulation name.
  /// @return Owning value or borrowed reference exactly as indicated by the return type;
  /// observation/state units follow the owning class.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::GridWorld obj;obj.reset(1);auto value=obj.name();
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  std::string name() const override { return "GridWorld"; }
};

// Sutton & Barto's cliff: 4x12, start bottom-left, goal bottom-right, and a
// cliff along the bottom row. Stepping into the cliff costs -100 and sends
// the agent back to the start without ending the episode.
/// @brief Source4x12 cliff grid; cliff costs100 and returns to the start without terminality.
///
/// Instances own mutable state and a local RNG; concurrent access to one instance requires external
/// synchronization. Returned observations own their values. Reset before stepping and after done.
/// VecEnv owns its children; client calls remain serialized, while workers access distinct
/// children.
/// @par Example
/// @code
/// namespace e=dm::rl::environments;e::CliffWalking obj;obj.reset(1);
/// @endcode
/// @par References
/// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/ ;
/// https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
/// - Sutton and Barto (2018), Reinforcement Learning, second edition,
/// http://incompleteideas.net/book/the-book-2nd.html .
/// @see Env, VecEnv
class CliffWalking : public GridBase {
 public:
  /// @brief Construct CliffWalking with independent owned simulation state.
  /// @param max_steps Nonnegative source time limit; zero means unlimited.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @throws std::invalid_argument Invalid checked bounds, geometry or action.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::CliffWalking obj;obj.reset(1);
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  explicit CliffWalking(int max_steps = 500) : GridBase(4, 12, {3, 0}, {3, 11}, 0.f, max_steps) {}
  /// @brief Return the stable human-readable simulation name.
  /// @return Owning value or borrowed reference exactly as indicated by the return type;
  /// observation/state units follow the owning class.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::CliffWalking obj;obj.reset(1);auto value=obj.name();
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  std::string name() const override { return "CliffWalking"; }
  /// @brief Test the bottom-row non-endpoint cliff cells of the Sutton-Barto task.
  /// @param r Board row within the owning grid.
  /// @param c Board column within the owning grid.
  /// @return Owning value or borrowed reference exactly as indicated by the return type;
  /// observation/state units follow the owning class.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::CliffWalking obj;obj.reset(1);auto value=obj.IsCliff(0,0);
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  bool IsCliff(int r, int c) const { return r == 3 && c >= 1 && c <= 10; }
  /// @brief Apply the legal board move or compute the deterministic grid outcome, as specified by
  /// the owning class.
  /// @param r Board row within the owning grid.
  /// @param c Board column within the owning grid.
  /// @param a Discrete action index or owning action vector, according to this overload.
  /// @return Owning value or borrowed reference exactly as indicated by the return type;
  /// observation/state units follow the owning class.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @throws std::invalid_argument Invalid checked bounds, geometry or action.
  /// @throws std::out_of_range Checked cell, arm, action or environment index is out of range.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::CliffWalking obj;obj.reset(1);auto
  /// outcome=obj.Apply(0,0,1);
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  Outcome Apply(int r, int c, int a) const override {
    Outcome o = GridBase::Apply(r, c, a);
    if (IsCliff(o.row, o.col))
      return {start_.first, start_.second, -100.f, false};
    return o;
  }
  /// @brief Return the default terrain glyph for the cell.
  /// @param r Board row within the owning grid.
  /// @param c Board column within the owning grid.
  /// @return Owning value or borrowed reference exactly as indicated by the return type;
  /// observation/state units follow the owning class.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::CliffWalking obj;obj.reset(1);auto
  /// value=obj.CellGlyph(0,0);
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  std::string CellGlyph(int r, int c) const override { return IsCliff(r, c) ? "~" : "."; }
};

} // namespace datamunge::rl::environments
