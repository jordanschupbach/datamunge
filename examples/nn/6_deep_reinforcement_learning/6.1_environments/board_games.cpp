// CMD: just run-nn board_games
// Tic-tac-toe, Connect Four and Othello, for self-play.
//
// Port of tflow's board_games. Board games use rl::Game
// (common/rl/board_games.hpp, copied from tflow) rather than rl::Env: two
// players alternate, only legal actions may be applied, and the observation is
// always from the current player's point of view (+1 own, -1 opponent), which
// is what self-play algorithms need. tflow films one more random game per
// board on vsim's 3D board; the film is omitted but that game is still played
// (same moves, same RNG draws), so the Othello stone count printed after it
// matches tflow. The port adds checks on the random-play statistics, the
// observation perspective and Clone().

#include <cctype>
#include <cstdio>
#include <iostream>
#include <memory>
#include <vector>

#include "common/checks.hpp"
#include "common/rl/board_games.hpp"

namespace {

// One game between two uniformly random players; returns the winner.
int RandomGame(rl::Game& game, std::mt19937_64& rng, int* moves) {
  game.Reset();
  *moves = 0;
  while (!game.Terminal()) {
    std::vector<int> legal = game.LegalActions();
    game.Apply(legal[rng() % legal.size()]);
    ++*moves;
  }
  return game.Winner();
}

struct DemoStats {
  double p1 = 0, p2 = 0, draw = 0, length = 0;
};

DemoStats Demo(rl::GridGame& game, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::cout << "== " << game.name() << ": " << game.num_actions() << " actions, observation of "
            << game.Observation().size() << " cells\n";
  int moves = 0;
  int winner = RandomGame(game, rng, &moves);
  std::cout << "  random game, " << moves << " moves, "
            << (winner ? std::string("winner: player ") + std::to_string(winner) : "draw") << "\n";
  std::string board = game.Render();
  for (size_t pos = 0; pos < board.size(); pos = board.find('\n', pos) + 1) {
    size_t end = board.find('\n', pos);
    std::cout << "    " << board.substr(pos, end - pos) << "\n";
    if (end == std::string::npos) break;
  }

  int wins[3] = {0, 0, 0};
  long total_moves = 0;
  const int games = 2000;
  for (int g = 0; g < games; ++g) {
    wins[RandomGame(game, rng, &moves)]++;
    total_moves += moves;
  }
  std::printf("  %d random games: player 1 wins %.1f%%, player 2 wins %.1f%%, draws %.1f%%, "
              "mean length %.1f\n",
              games, 100.0 * wins[1] / games, 100.0 * wins[2] / games, 100.0 * wins[0] / games,
              static_cast<double>(total_moves) / games);
  // One more random game (filmed on the 3D board in tflow; here only played).
  game.Reset();
  while (!game.Terminal()) {
    const std::vector<int> legal = game.LegalActions();
    game.Apply(legal[rng() % legal.size()]);
  }
  return {100.0 * wins[1] / games, 100.0 * wins[2] / games, 100.0 * wins[0] / games,
          static_cast<double>(total_moves) / games};
}

}  // namespace

int main() {
  try {
    rl::TicTacToe ttt;
    const DemoStats t = Demo(ttt, 1);
    std::cout << "  observation from X's view after X plays centre: ";
    ttt.Reset();
    ttt.Apply(4);
    for (float v : ttt.Observation()) std::printf("%+.0f ", v);
    std::cout << " (now O to move, so the centre reads -1)\n\n";
    check::Require(ttt.Observation()[4] == -1.f, "after X plays centre, O sees the centre as -1");
    // Random tic-tac-toe: X wins about 58.5%, O 28.8%, draws 12.7%.
    check::Near(t.p1, 58.5, 4, "tic-tac-toe player 1 win rate");
    check::Near(t.draw, 12.7, 3, "tic-tac-toe draw rate");

    rl::ConnectFour c4;
    const DemoStats c = Demo(c4, 2);
    std::cout << "\n";
    check::Require(c.p1 > c.p2 && c.length > 7 && c.length < 42, "Connect Four random play: first player favoured");

    rl::Othello oth;
    const DemoStats o = Demo(oth, 3);
    auto [n1, n2] = oth.Counts();
    std::printf("  final stone count: X %d, O %d\n", n1, n2);
    std::cout << "  legal first moves for X: ";
    oth.Reset();
    const std::vector<int> first = oth.LegalActions();
    for (int a : first) std::printf("(%d,%d) ", a / 8, a % 8);
    std::cout << "\n";
    check::Require(first.size() == 4, "Othello opens with four legal moves");
    check::Require(n1 + n2 <= 64 && n1 + n2 > 4 && o.length > 50, "Othello games fill most of the board");

    // Clone() lets search algorithms (MCTS, 6.6) try moves without touching
    // the real game.
    std::unique_ptr<rl::Game> copy = c4.Clone();
    c4.Reset();
    copy->Reset();
    copy->Apply(3);
    std::printf("\n  Clone(): original has %zu legal moves and is at move 0; the clone has advanced "
                "(player %d to move)\n",
                c4.LegalActions().size(), copy->CurrentPlayer());
    check::Require(c4.LegalActions().size() == 7 && c4.CurrentPlayer() != copy->CurrentPlayer(),
                   "a move on the clone leaves the original untouched");
    std::cout << "\nchecks passed (vsim videos omitted)\n";
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
