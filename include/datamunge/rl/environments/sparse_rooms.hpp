#pragma once
/// @file sparse_rooms.hpp
/// @brief Original9x9 sparse-reward four-room MDP.
#include <datamunge/rl/environments/grid_world.hpp>
namespace datamunge::rl::environments {

/// @brief Original nine-by-nine four-room environment with a sparse goal reward.
///
/// Cross walls leave doorways at coordinates1 and7. Start(0,0), goal(8,8), no slip, limit250.
/// Only the goal gives reward1; all other steps give0. Inherits explicit seeding and owning
/// observations.
///
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// dm::rl::environments::SparseRooms env;auto observation=env.reset(1);auto step=env.step(1);
/// @endcode
/// @par References
/// - Sutton and Barto, Reinforcement Learning, second edition, Chapter2
/// (http://incompleteideas.net/book/the-book-2nd.html).
/// @see GridBase
class SparseRooms : public GridBase {
 public:
  /// @brief Construct the fixed source geometry, reward and time limit.
  ///
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// dm::rl::environments::SparseRooms env;
  /// @endcode
  /// @par References
  /// - Sutton and Barto, Reinforcement Learning, second edition, Chapter2
  /// (http://incompleteideas.net/book/the-book-2nd.html).
  /// @see Apply()
  SparseRooms() : GridBase(9, 9, {0, 0}, {8, 8}, 0.f, 250) {
    for (int i = 0; i < 9; ++i) {
      if (i != 1 && i != 7) {
        AddWall(4, i);
        AddWall(i, 4);
      }
    }
  }

  /// @brief Return the stable environment name.
  ///
  /// @return SparseFourRooms.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// dm::rl::environments::SparseRooms env;auto name=env.name();
  /// @endcode
  /// @par References
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  std::string name() const override { return "SparseFourRooms"; }

  /// @brief Compute one deterministic wall-aware transition without mutating state.
  ///
  /// Caller supplies valid coordinates/actions as for GridBase::Apply().
  ///
  /// @param r Source row in grid.
  /// @param c Source column in grid.
  /// @param a Action index0..3, up/right/down/left.
  /// @return GridBase outcome with reward1 exactly for terminal goal transitions, otherwise0.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// dm::rl::environments::SparseRooms env;auto next=env.Apply(0,0,1);
  /// @endcode
  /// @par References
  /// - Sutton and Barto, Reinforcement Learning, second edition, Chapter2
  /// (http://incompleteideas.net/book/the-book-2nd.html).
  /// @see GridBase::Apply()
  Outcome Apply(int r, int c, int a) const override {
    auto o   = GridBase::Apply(r, c, a);
    o.reward = o.terminal ? 1.f : 0.f;
    return o;
  }
};

} // namespace datamunge::rl::environments
