// datamunge::nn port: copied from tflow_examples src/common/rl/board_games.h (TF-free dynamics;
// checked native public inputs). Two-player, perfect-information board games for self-play:
// tic-tac-toe, Connect Four and Othello. Players are 1 and 2; actions are cell indices
// (tic-tac-toe, Othello) or columns (Connect Four). Othello includes a pass
// action when a player has no legal move.
#pragma once
/// @file board_games.hpp
/// @brief Native board games simulation components for explicit caller-owned rollouts.

#include <datamunge/rl/environments/interface.hpp>
#include <memory>
#include <string>
#include <vector>

namespace datamunge::rl::environments {

/// @brief Two-player perfect-information game interface with current-player observations.
///
/// Instances own mutable board state and no RNG; concurrent access requires external
/// synchronization. Returned observations own their values. Reset before stepping and after done.
/// VecEnv owns its children; client calls remain serialized, while workers access distinct
/// children.
/// @par Example
/// @code
/// namespace e=dm::rl::environments;e::TicTacToe concrete;e::Game& obj=concrete;obj.Reset();
/// @endcode
/// @par References
/// - Repository original local simulation protocol, preserved in
/// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
/// - Sutton and Barto (2018), Reinforcement Learning, second edition,
/// http://incompleteideas.net/book/the-book-2nd.html .
/// @see Env, VecEnv
class Game {
 public:
  /// @brief Release owned simulation resources; VecEnv first joins every worker.
  virtual ~Game() = default;
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
  /// namespace e=dm::rl::environments;e::TicTacToe concrete;e::Game& obj=concrete;obj.Reset();auto
  /// value=obj.name();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  virtual std::string name() const = 0;
  /// @brief Return an independently mutable deep copy of this game state.
  /// @return Unique owning game copy with independent board storage.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::TicTacToe concrete;e::Game& obj=concrete;obj.Reset();auto
  /// value=obj.Clone();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  virtual std::unique_ptr<Game> Clone() const = 0;
  /// @brief Reset the simulation state and per-episode counter, retaining the current local RNG
  /// stream.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::TicTacToe concrete;e::Game&
  /// obj=concrete;obj.Reset();obj.Reset();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  virtual void Reset() = 0;
  /// @brief Return the full action-index capacity, including Othello pass64 where applicable.
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
  /// namespace e=dm::rl::environments;e::TicTacToe concrete;e::Game& obj=concrete;obj.Reset();auto
  /// value=obj.num_actions();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  virtual int num_actions() const = 0;
  /// @brief Return an owning list of legal moves; a terminal game returns none.
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
  /// namespace e=dm::rl::environments;e::TicTacToe concrete;e::Game& obj=concrete;obj.Reset();auto
  /// value=obj.LegalActions();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  virtual std::vector<int> LegalActions() const = 0;
  /// @brief Apply the legal board move or compute the deterministic grid outcome, as specified by
  /// the owning class.
  /// @param action Action in this instance space; board actions must appear in LegalActions().
  /// Continuous physics clips finite torque/throttle to bounds.
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
  /// namespace e=dm::rl::environments;e::TicTacToe concrete;e::Game&
  /// obj=concrete;obj.Reset();obj.Apply(obj.LegalActions().front());
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  virtual void Apply(int action) = 0;
  /// @brief Test whether the board game has ended.
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
  /// namespace e=dm::rl::environments;e::TicTacToe concrete;e::Game& obj=concrete;obj.Reset();auto
  /// value=obj.Terminal();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  virtual bool Terminal() const = 0;
  // 0 for a draw or an unfinished game, else the winning player (1 or 2).
  /// @brief Return player1/2 for a win or0 for a draw or unfinished game.
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
  /// namespace e=dm::rl::environments;e::TicTacToe concrete;e::Game& obj=concrete;obj.Reset();auto
  /// value=obj.Winner();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  virtual int Winner() const = 0;
  /// @brief Return the side to move, encoded1 or2.
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
  /// namespace e=dm::rl::environments;e::TicTacToe concrete;e::Game& obj=concrete;obj.Reset();auto
  /// value=obj.CurrentPlayer();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  virtual int CurrentPlayer() const = 0;
  /// @brief Render the grid or game as owning ASCII text; an overlay callback is evaluated once per
  /// cell.
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
  /// namespace e=dm::rl::environments;e::TicTacToe concrete;e::Game& obj=concrete;obj.Reset();auto
  /// picture=obj.Render();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  virtual std::string Render() const = 0;
  // Board from the current player's point of view: +1 own stones, -1
  // opponent's, 0 empty, one float per cell.
  /// @brief Return owning row-major board values: own stones+1, opponent-1, empty0.
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
  /// namespace e=dm::rl::environments;e::TicTacToe concrete;e::Game& obj=concrete;obj.Reset();auto
  /// value=obj.Observation();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  virtual Obs Observation() const = 0;
  /// @brief Return the other player ID.
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
  /// namespace e=dm::rl::environments;e::TicTacToe concrete;e::Game& obj=concrete;obj.Reset();auto
  /// value=obj.Opponent();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  int Opponent() const { return 3 - CurrentPlayer(); }
};

// Shared grid storage and rendering for the three games.
/// @brief Own the GridGame simulation state or interface.
///
/// Instances own mutable board state and no RNG; concurrent access requires external
/// synchronization. Returned observations own their values. Reset before stepping and after done.
/// VecEnv owns its children; client calls remain serialized, while workers access distinct
/// children.
/// @par Example
/// @code
/// namespace e=dm::rl::environments;e::TicTacToe concrete;e::GridGame& obj=concrete;obj.Reset();
/// @endcode
/// @par References
/// - Repository original local simulation protocol, preserved in
/// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
/// - Sutton and Barto (2018), Reinforcement Learning, second edition,
/// http://incompleteideas.net/book/the-book-2nd.html .
/// @see Env, VecEnv
class GridGame : public Game {
 public:
  /// @brief Construct GridGame with independent owned simulation state.
  /// @param rows Positive board row count, at most1024.
  /// @param cols Positive board column count, at most1024.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @throws std::invalid_argument Invalid checked bounds, geometry or action.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::TicTacToe concrete;e::GridGame& obj=concrete;obj.Reset();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  GridGame(int rows, int cols)
      : rows_(rows), cols_(cols), cells_(CheckedCount(rows, 1024) * CheckedCount(cols, 1024), 0) {}
  /// @brief Return the board row count.
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
  /// namespace e=dm::rl::environments;e::TicTacToe concrete;e::GridGame&
  /// obj=concrete;obj.Reset();auto value=obj.rows();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  int rows() const { return rows_; }
  /// @brief Return the board column count.
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
  /// namespace e=dm::rl::environments;e::TicTacToe concrete;e::GridGame&
  /// obj=concrete;obj.Reset();auto value=obj.cols();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  int cols() const { return cols_; }
  /// @brief Read a checked row/column board cell:0 empty,1/2 player.
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
  /// @throws std::invalid_argument Invalid checked bounds, geometry or action.
  /// @throws std::out_of_range Checked cell, arm, action or environment index is out of range.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::TicTacToe concrete;e::GridGame&
  /// obj=concrete;obj.Reset();auto value=obj.At(0,0);
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  int At(int r, int c) const {
    if (r < 0 || r >= rows_ || c < 0 || c >= cols_)
      throw std::out_of_range("board cell");
    return cells_[r * cols_ + c];
  }
  /// @brief Return the side to move, encoded1 or2.
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
  /// namespace e=dm::rl::environments;e::TicTacToe concrete;e::GridGame&
  /// obj=concrete;obj.Reset();auto value=obj.CurrentPlayer();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  int CurrentPlayer() const override { return player_; }
  /// @brief Test whether the board game has ended.
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
  /// namespace e=dm::rl::environments;e::TicTacToe concrete;e::GridGame&
  /// obj=concrete;obj.Reset();auto value=obj.Terminal();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  bool Terminal() const override { return terminal_; }
  /// @brief Return player1/2 for a win or0 for a draw or unfinished game.
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
  /// namespace e=dm::rl::environments;e::TicTacToe concrete;e::GridGame&
  /// obj=concrete;obj.Reset();auto value=obj.Winner();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  int Winner() const override { return winner_; }
  /// @brief Return owning row-major board values: own stones+1, opponent-1, empty0.
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
  /// namespace e=dm::rl::environments;e::TicTacToe concrete;e::GridGame&
  /// obj=concrete;obj.Reset();auto value=obj.Observation();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  Obs Observation() const override {
    Obs o(cells_.size());
    for (size_t i = 0; i < cells_.size(); ++i)
      o[i] = cells_[i] == 0 ? 0.f : (cells_[i] == player_ ? 1.f : -1.f);
    return o;
  }
  /// @brief Render the grid or game as owning ASCII text; an overlay callback is evaluated once per
  /// cell.
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
  /// namespace e=dm::rl::environments;e::TicTacToe concrete;e::GridGame&
  /// obj=concrete;obj.Reset();auto picture=obj.Render();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  std::string Render() const override {
    std::string s;
    for (int r = 0; r < rows_; ++r) {
      for (int c = 0; c < cols_; ++c)
        s += ".XO"[At(r, c)], s += ' ';
      s += '\n';
    }
    return s;
  }

 protected:
  /// @brief Write one internal board cell.
  /// @param r Board row within the owning grid.
  /// @param c Board column within the owning grid.
  /// @param v Scalar or board-cell value in the owning operation units.
  void Set(int r, int c, int v) { cells_[r * cols_ + c] = v; }
  /// @brief Clear board storage and reset player1, terminal flag and winner.
  void ResetGrid() {
    std::fill(cells_.begin(), cells_.end(), 0);
    player_   = 1;
    terminal_ = false;
    winner_   = 0;
  }
  int              rows_;             ///< Owned board row count.
  int              cols_;             ///< Owned board column count.
  std::vector<int> cells_;            ///< Owned row-major board cells:0 empty,1/2 player.
  int              player_   = 1;     ///< Current player1 or2.
  bool             terminal_ = false; ///< Stored game-completion flag.
  int              winner_   = 0;     ///< Stored winner1/2 or0 for draw/unfinished.
};

/// @brief 3x3 alternating-placement game with legal-action checks and deep-copy support.
///
/// Instances own mutable board state and no RNG; concurrent access requires external
/// synchronization. Returned observations own their values. Reset before stepping and after done.
/// VecEnv owns its children; client calls remain serialized, while workers access distinct
/// children.
/// @par Example
/// @code
/// namespace e=dm::rl::environments;e::TicTacToe obj;obj.Reset();
/// @endcode
/// @par References
/// - Repository original local simulation protocol, preserved in
/// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
/// - Sutton and Barto (2018), Reinforcement Learning, second edition,
/// http://incompleteideas.net/book/the-book-2nd.html .
/// @see Env, VecEnv
class TicTacToe : public GridGame {
 public:
  /// @brief Construct TicTacToe with independent owned simulation state.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::TicTacToe obj;obj.Reset();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  TicTacToe() : GridGame(3, 3) {}
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
  /// namespace e=dm::rl::environments;e::TicTacToe obj;obj.Reset();auto value=obj.name();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  std::string name() const override { return "TicTacToe"; }
  /// @brief Return an independently mutable deep copy of this game state.
  /// @return Unique owning game copy with independent board storage.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::TicTacToe obj;obj.Reset();auto value=obj.Clone();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  std::unique_ptr<Game> Clone() const override { return std::make_unique<TicTacToe>(*this); }
  /// @brief Reset the simulation state and per-episode counter, retaining the current local RNG
  /// stream.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::TicTacToe obj;obj.Reset();obj.Reset();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  void Reset() override { ResetGrid(); }
  /// @brief Return the full action-index capacity, including Othello pass64 where applicable.
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
  /// namespace e=dm::rl::environments;e::TicTacToe obj;obj.Reset();auto value=obj.num_actions();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  int num_actions() const override { return 9; }
  /// @brief Return an owning list of legal moves; a terminal game returns none.
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
  /// namespace e=dm::rl::environments;e::TicTacToe obj;obj.Reset();auto value=obj.LegalActions();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  std::vector<int> LegalActions() const override {
    std::vector<int> a;
    if (!terminal_)
      for (int i = 0; i < 9; ++i)
        if (cells_[i] == 0)
          a.push_back(i);
    return a;
  }
  /// @brief Apply the legal board move or compute the deterministic grid outcome, as specified by
  /// the owning class.
  /// @param action Action in this instance space; board actions must appear in LegalActions().
  /// Continuous physics clips finite torque/throttle to bounds.
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
  /// namespace e=dm::rl::environments;e::TicTacToe
  /// obj;obj.Reset();obj.Apply(obj.LegalActions().front());
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  void Apply(int action) override {
    const auto legal = LegalActions();
    if (std::find(legal.begin(), legal.end(), action) == legal.end())
      throw std::invalid_argument("illegal board move");
    cells_[action]               = player_;
    static const int lines[8][3] = {
        {0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {0, 3, 6}, {1, 4, 7}, {2, 5, 8}, {0, 4, 8}, {2, 4, 6}};
    for (const auto& l : lines)
      if (cells_[l[0]] == player_ && cells_[l[1]] == player_ && cells_[l[2]] == player_) {
        winner_   = player_;
        terminal_ = true;
      }
    if (!terminal_ && LegalActions().empty())
      terminal_ = true;
    if (!terminal_)
      player_ = 3 - player_;
  }
};

/// @brief 6x7 alternating gravity-placement game with legal-column checks.
///
/// Instances own mutable board state and no RNG; concurrent access requires external
/// synchronization. Returned observations own their values. Reset before stepping and after done.
/// VecEnv owns its children; client calls remain serialized, while workers access distinct
/// children.
/// @par Example
/// @code
/// namespace e=dm::rl::environments;e::ConnectFour obj;obj.Reset();
/// @endcode
/// @par References
/// - Repository original local simulation protocol, preserved in
/// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
/// - Sutton and Barto (2018), Reinforcement Learning, second edition,
/// http://incompleteideas.net/book/the-book-2nd.html .
/// @see Env, VecEnv
class ConnectFour : public GridGame {
 public:
  /// @brief Construct ConnectFour with independent owned simulation state.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::ConnectFour obj;obj.Reset();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  ConnectFour() : GridGame(6, 7) {}
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
  /// namespace e=dm::rl::environments;e::ConnectFour obj;obj.Reset();auto value=obj.name();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  std::string name() const override { return "ConnectFour"; }
  /// @brief Return an independently mutable deep copy of this game state.
  /// @return Unique owning game copy with independent board storage.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::ConnectFour obj;obj.Reset();auto value=obj.Clone();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  std::unique_ptr<Game> Clone() const override { return std::make_unique<ConnectFour>(*this); }
  /// @brief Reset the simulation state and per-episode counter, retaining the current local RNG
  /// stream.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::ConnectFour obj;obj.Reset();obj.Reset();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  void Reset() override { ResetGrid(); }
  /// @brief Return the full action-index capacity, including Othello pass64 where applicable.
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
  /// namespace e=dm::rl::environments;e::ConnectFour obj;obj.Reset();auto value=obj.num_actions();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  int num_actions() const override { return 7; }
  /// @brief Return an owning list of legal moves; a terminal game returns none.
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
  /// namespace e=dm::rl::environments;e::ConnectFour obj;obj.Reset();auto value=obj.LegalActions();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  std::vector<int> LegalActions() const override {
    std::vector<int> a;
    if (!terminal_)
      for (int c = 0; c < 7; ++c)
        if (At(0, c) == 0)
          a.push_back(c);
    return a;
  }
  /// @brief Apply the legal board move or compute the deterministic grid outcome, as specified by
  /// the owning class.
  /// @param column Legal Connect Four column in0..6.
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
  /// namespace e=dm::rl::environments;e::ConnectFour
  /// obj;obj.Reset();obj.Apply(obj.LegalActions().front());
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  void Apply(int column) override {
    const auto legal = LegalActions();
    if (std::find(legal.begin(), legal.end(), column) == legal.end())
      throw std::invalid_argument("illegal board move");
    int r = rows_ - 1;
    while (At(r, column) != 0)
      --r;
    Set(r, column, player_);
    static const int dirs[4][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};
    for (const auto& d : dirs) {
      int count = 1;
      for (int sign : {-1, 1})
        for (int k = 1; k < 4; ++k) {
          int rr = r + sign * k * d[0], cc = column + sign * k * d[1];
          if (rr < 0 || rr >= rows_ || cc < 0 || cc >= cols_ || At(rr, cc) != player_)
            break;
          ++count;
        }
      if (count >= 4) {
        winner_   = player_;
        terminal_ = true;
      }
    }
    if (!terminal_ && LegalActions().empty())
      terminal_ = true;
    if (!terminal_)
      player_ = 3 - player_;
  }
};

/// @brief 8x8 alternating-placement game with bracketed flips and pass action64.
///
/// Instances own mutable board state and no RNG; concurrent access requires external
/// synchronization. Returned observations own their values. Reset before stepping and after done.
/// VecEnv owns its children; client calls remain serialized, while workers access distinct
/// children.
/// @par Example
/// @code
/// namespace e=dm::rl::environments;e::Othello obj;obj.Reset();
/// @endcode
/// @par References
/// - Repository original local simulation protocol, preserved in
/// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
/// - Sutton and Barto (2018), Reinforcement Learning, second edition,
/// http://incompleteideas.net/book/the-book-2nd.html .
/// @see Env, VecEnv
class Othello : public GridGame {
 public:
  static constexpr int kPass = 64; ///< Othello pass action64; legal only when no placement exists.

  /// @brief Construct Othello with independent owned simulation state.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::Othello obj;obj.Reset();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  Othello() : GridGame(8, 8) { Reset(); }
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
  /// namespace e=dm::rl::environments;e::Othello obj;obj.Reset();auto value=obj.name();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  std::string name() const override { return "Othello"; }
  /// @brief Return an independently mutable deep copy of this game state.
  /// @return Unique owning game copy with independent board storage.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::Othello obj;obj.Reset();auto value=obj.Clone();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  std::unique_ptr<Game> Clone() const override { return std::make_unique<Othello>(*this); }
  /// @brief Reset the simulation state and per-episode counter, retaining the current local RNG
  /// stream.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::Othello obj;obj.Reset();obj.Reset();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  void Reset() override {
    ResetGrid();
    Set(3, 3, 2);
    Set(4, 4, 2);
    Set(3, 4, 1);
    Set(4, 3, 1);
  }
  /// @brief Return the full action-index capacity, including Othello pass64 where applicable.
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
  /// namespace e=dm::rl::environments;e::Othello obj;obj.Reset();auto value=obj.num_actions();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  int num_actions() const override { return 65; }
  /// @brief Return an owning list of legal moves; a terminal game returns none.
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
  /// namespace e=dm::rl::environments;e::Othello obj;obj.Reset();auto value=obj.LegalActions();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  std::vector<int> LegalActions() const override {
    std::vector<int> a;
    if (terminal_)
      return a;
    for (int i = 0; i < 64; ++i)
      if (cells_[i] == 0 && !Flips(i / 8, i % 8, player_).empty())
        a.push_back(i);
    if (a.empty())
      a.push_back(kPass);
    return a;
  }
  /// @brief Apply the legal board move or compute the deterministic grid outcome, as specified by
  /// the owning class.
  /// @param action Action in this instance space; board actions must appear in LegalActions().
  /// Continuous physics clips finite torque/throttle to bounds.
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
  /// namespace e=dm::rl::environments;e::Othello
  /// obj;obj.Reset();obj.Apply(obj.LegalActions().front());
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  void Apply(int action) override {
    const auto legal = LegalActions();
    if (std::find(legal.begin(), legal.end(), action) == legal.end())
      throw std::invalid_argument("illegal board move");
    if (action != kPass) {
      const int r = action / 8, c = action % 8;
      Set(r, c, player_);
      for (int idx : Flips(r, c, player_))
        cells_[idx] = player_;
    }
    player_ = 3 - player_;
    // The game ends when neither player can move.
    if (!HasMove(player_)) {
      if (!HasMove(3 - player_)) {
        terminal_ = true;
        int n1 = 0, n2 = 0;
        for (int v : cells_)
          n1 += v == 1, n2 += v == 2;
        winner_ = n1 > n2 ? 1 : (n2 > n1 ? 2 : 0);
      }
      // else: the current player must pass (kPass is the only legal action)
    }
  }
  /// @brief Count player1 and player2 stones on the Othello board.
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
  /// namespace e=dm::rl::environments;e::Othello obj;obj.Reset();auto value=obj.Counts();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  std::pair<int, int> Counts() const {
    int n1 = 0, n2 = 0;
    for (int v : cells_)
      n1 += v == 1, n2 += v == 2;
    return {n1, n2};
  }

 private:
  /// @brief Test whether the player has a legal Othello placement.
  /// @param player Player ID1 or2.
  /// @return Owning value or borrowed reference exactly as indicated by the return type;
  /// observation/state units follow the owning class.
  bool HasMove(int player) const {
    for (int i = 0; i < 64; ++i)
      if (cells_[i] == 0 && !Flips(i / 8, i % 8, player).empty())
        return true;
    return false;
  }
  // Indices of opponent stones that placing `player` at (r, c) would flip.
  /// @brief Enumerate opponent stones bracketed by a hypothetical Othello placement.
  /// @param r Board row within the owning grid.
  /// @param c Board column within the owning grid.
  /// @param player Player ID1 or2.
  /// @return Owning value or borrowed reference exactly as indicated by the return type;
  /// observation/state units follow the owning class.
  std::vector<int> Flips(int r, int c, int player) const {
    std::vector<int> out;
    for (int dr = -1; dr <= 1; ++dr)
      for (int dc = -1; dc <= 1; ++dc) {
        if (dr == 0 && dc == 0)
          continue;
        std::vector<int> line;
        int              rr = r + dr, cc = c + dc;
        while (rr >= 0 && rr < 8 && cc >= 0 && cc < 8 && At(rr, cc) == 3 - player) {
          line.push_back(rr * 8 + cc);
          rr += dr;
          cc += dc;
        }
        if (!line.empty() && rr >= 0 && rr < 8 && cc >= 0 && cc < 8 && At(rr, cc) == player)
          out.insert(out.end(), line.begin(), line.end());
      }
    return out;
  }
};

} // namespace datamunge::rl::environments
