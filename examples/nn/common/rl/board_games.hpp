// datamunge::nn port: copied from tflow_examples src/common/rl/board_games.h (TF-free code, unchanged).
// Two-player, perfect-information board games for self-play: tic-tac-toe,
// Connect Four and Othello. Players are 1 and 2; actions are cell indices
// (tic-tac-toe, Othello) or columns (Connect Four). Othello includes a pass
// action when a player has no legal move.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "common/rl/env.hpp"

namespace rl {

class Game {
 public:
  virtual ~Game() = default;
  virtual std::string name() const = 0;
  virtual std::unique_ptr<Game> Clone() const = 0;
  virtual void Reset() = 0;
  virtual int num_actions() const = 0;
  virtual std::vector<int> LegalActions() const = 0;
  virtual void Apply(int action) = 0;
  virtual bool Terminal() const = 0;
  // 0 for a draw or an unfinished game, else the winning player (1 or 2).
  virtual int Winner() const = 0;
  virtual int CurrentPlayer() const = 0;
  virtual std::string Render() const = 0;
  // Board from the current player's point of view: +1 own stones, -1
  // opponent's, 0 empty, one float per cell.
  virtual Obs Observation() const = 0;
  int Opponent() const { return 3 - CurrentPlayer(); }
};

// Shared grid storage and rendering for the three games.
class GridGame : public Game {
 public:
  GridGame(int rows, int cols) : rows_(rows), cols_(cols), cells_(rows * cols, 0) {}
  int rows() const { return rows_; }
  int cols() const { return cols_; }
  int At(int r, int c) const { return cells_[r * cols_ + c]; }
  int CurrentPlayer() const override { return player_; }
  bool Terminal() const override { return terminal_; }
  int Winner() const override { return winner_; }
  Obs Observation() const override {
    Obs o(cells_.size());
    for (size_t i = 0; i < cells_.size(); ++i)
      o[i] = cells_[i] == 0 ? 0.f : (cells_[i] == player_ ? 1.f : -1.f);
    return o;
  }
  std::string Render() const override {
    std::string s;
    for (int r = 0; r < rows_; ++r) {
      for (int c = 0; c < cols_; ++c) s += ".XO"[At(r, c)], s += ' ';
      s += '\n';
    }
    return s;
  }

 protected:
  void Set(int r, int c, int v) { cells_[r * cols_ + c] = v; }
  void ResetGrid() {
    std::fill(cells_.begin(), cells_.end(), 0);
    player_ = 1;
    terminal_ = false;
    winner_ = 0;
  }
  int rows_, cols_;
  std::vector<int> cells_;
  int player_ = 1;
  bool terminal_ = false;
  int winner_ = 0;
};

class TicTacToe : public GridGame {
 public:
  TicTacToe() : GridGame(3, 3) {}
  std::string name() const override { return "TicTacToe"; }
  std::unique_ptr<Game> Clone() const override { return std::make_unique<TicTacToe>(*this); }
  void Reset() override { ResetGrid(); }
  int num_actions() const override { return 9; }
  std::vector<int> LegalActions() const override {
    std::vector<int> a;
    if (!terminal_)
      for (int i = 0; i < 9; ++i)
        if (cells_[i] == 0) a.push_back(i);
    return a;
  }
  void Apply(int action) override {
    cells_[action] = player_;
    static const int lines[8][3] = {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {0, 3, 6},
                                    {1, 4, 7}, {2, 5, 8}, {0, 4, 8}, {2, 4, 6}};
    for (const auto& l : lines)
      if (cells_[l[0]] == player_ && cells_[l[1]] == player_ && cells_[l[2]] == player_) {
        winner_ = player_;
        terminal_ = true;
      }
    if (!terminal_ && LegalActions().empty()) terminal_ = true;
    if (!terminal_) player_ = 3 - player_;
  }
};

class ConnectFour : public GridGame {
 public:
  ConnectFour() : GridGame(6, 7) {}
  std::string name() const override { return "ConnectFour"; }
  std::unique_ptr<Game> Clone() const override { return std::make_unique<ConnectFour>(*this); }
  void Reset() override { ResetGrid(); }
  int num_actions() const override { return 7; }
  std::vector<int> LegalActions() const override {
    std::vector<int> a;
    if (!terminal_)
      for (int c = 0; c < 7; ++c)
        if (At(0, c) == 0) a.push_back(c);
    return a;
  }
  void Apply(int column) override {
    int r = rows_ - 1;
    while (At(r, column) != 0) --r;
    Set(r, column, player_);
    static const int dirs[4][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};
    for (const auto& d : dirs) {
      int count = 1;
      for (int sign : {-1, 1})
        for (int k = 1; k < 4; ++k) {
          int rr = r + sign * k * d[0], cc = column + sign * k * d[1];
          if (rr < 0 || rr >= rows_ || cc < 0 || cc >= cols_ || At(rr, cc) != player_) break;
          ++count;
        }
      if (count >= 4) {
        winner_ = player_;
        terminal_ = true;
      }
    }
    if (!terminal_ && LegalActions().empty()) terminal_ = true;
    if (!terminal_) player_ = 3 - player_;
  }
};

class Othello : public GridGame {
 public:
  static constexpr int kPass = 64;
  Othello() : GridGame(8, 8) { Reset(); }
  std::string name() const override { return "Othello"; }
  std::unique_ptr<Game> Clone() const override { return std::make_unique<Othello>(*this); }
  void Reset() override {
    ResetGrid();
    Set(3, 3, 2);
    Set(4, 4, 2);
    Set(3, 4, 1);
    Set(4, 3, 1);
  }
  int num_actions() const override { return 65; }
  std::vector<int> LegalActions() const override {
    std::vector<int> a;
    if (terminal_) return a;
    for (int i = 0; i < 64; ++i)
      if (cells_[i] == 0 && !Flips(i / 8, i % 8, player_).empty()) a.push_back(i);
    if (a.empty()) a.push_back(kPass);
    return a;
  }
  void Apply(int action) override {
    if (action != kPass) {
      const int r = action / 8, c = action % 8;
      Set(r, c, player_);
      for (int idx : Flips(r, c, player_)) cells_[idx] = player_;
    }
    player_ = 3 - player_;
    // The game ends when neither player can move.
    if (!HasMove(player_)) {
      if (!HasMove(3 - player_)) {
        terminal_ = true;
        int n1 = 0, n2 = 0;
        for (int v : cells_) n1 += v == 1, n2 += v == 2;
        winner_ = n1 > n2 ? 1 : (n2 > n1 ? 2 : 0);
      }
      // else: the current player must pass (kPass is the only legal action)
    }
  }
  std::pair<int, int> Counts() const {
    int n1 = 0, n2 = 0;
    for (int v : cells_) n1 += v == 1, n2 += v == 2;
    return {n1, n2};
  }

 private:
  bool HasMove(int player) const {
    for (int i = 0; i < 64; ++i)
      if (cells_[i] == 0 && !Flips(i / 8, i % 8, player).empty()) return true;
    return false;
  }
  // Indices of opponent stones that placing `player` at (r, c) would flip.
  std::vector<int> Flips(int r, int c, int player) const {
    std::vector<int> out;
    for (int dr = -1; dr <= 1; ++dr)
      for (int dc = -1; dc <= 1; ++dc) {
        if (dr == 0 && dc == 0) continue;
        std::vector<int> line;
        int rr = r + dr, cc = c + dc;
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

}  // namespace rl

