// CMD: just run-nn mcts
// Monte Carlo tree search (MCTS) on board games.
//
// Port of tflow's mcts (TF-free: the search, the games and the RNG stream are
// the same code, so the numbers match tflow exactly).
// UCT: repeatedly (1) walk down the tree picking the child with the best
// upper-confidence score, (2) expand a leaf, (3) estimate its value with a
// random playout, (4) back the result up, negating it at each ply. The
// shared implementation is common/rl/mcts.hpp (copied from tflow); with a
// network evaluator instead of random playouts it becomes the search inside
// AlphaZero.
//
// tflow films three more MCTS games on vsim's 3D boards; the films are omitted
// but the games are still played (same moves, same RNG draws) and their
// outcomes printed. The port adds checks (exit 1 on failure).

#include <algorithm>
#include <array>
#include <cstdio>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "common/checks.hpp"
#include "common/rl/board_games.hpp"
#include "common/rl/mcts.hpp"

namespace {

// Plays `games` games; player 1 uses `sims1` simulations, player 2 `sims2`
// (0 = random moves). Returns (wins1, wins2, draws).
std::array<int, 3> Match(rl::Game& game, int sims1, int sims2, int games, std::mt19937_64& rng) {
  rl::Mcts mcts(rl::RolloutEvaluator(&rng));
  std::array<int, 3> tally{0, 0, 0};
  for (int g = 0; g < games; ++g) {
    game.Reset();
    // Alternate who starts so neither side gets the first-move advantage.
    const bool swap = g % 2 == 1;
    while (!game.Terminal()) {
      const int sims = (game.CurrentPlayer() == 1) != swap ? sims1 : sims2;
      std::vector<int> legal = game.LegalActions();
      int action = legal[rng() % legal.size()];
      if (sims > 0) {
        std::vector<int> counts = mcts.Search(game, sims, rng);
        action = static_cast<int>(std::max_element(counts.begin(), counts.end()) - counts.begin());
      }
      game.Apply(action);
    }
    const int w = game.Winner();
    if (w == 0) tally[2]++;
    else if ((w == 1) != swap) tally[0]++;
    else tally[1]++;
  }
  return tally;
}

}  // namespace

int main() {
  try {
    std::mt19937_64 rng(0);
    std::printf("== MCTS with random playouts (UCT), 100 games per pairing, first move alternates\n");
    std::printf("  %-12s %-22s %6s %6s %6s\n", "game", "matchup", "win", "loss", "draw");
    rl::TicTacToe ttt;
    rl::ConnectFour c4;
    rl::Othello oth;
    struct Pair {
      const char* label;
      int sims1, sims2;
    };
    for (rl::Game* game : std::vector<rl::Game*>{&ttt, &c4, &oth}) {
      const int big = game->name() == "Othello" ? 200 : 400;
      for (Pair p : {Pair{"MCTS(50) vs random", 50, 0}, Pair{"MCTS(big) vs MCTS(50)", big, 50}}) {
        const int games = game->name() == "Othello" ? 20 : 100;
        std::array<int, 3> t = Match(*game, p.sims1, p.sims2, games, rng);
        std::printf("  %-12s %-22s %6d %6d %6d   (%d games)\n", game->name().c_str(), p.label, t[0], t[1], t[2], games);
        // The stronger side must come out ahead (it never loses more than it wins).
        check::Require(t[0] >= t[1], game->name() + " " + p.label + ": the stronger player wins at least as often as it loses");
        if (p.sims2 == 0 && game->name() != "TicTacToe")
          check::Require(t[0] >= games * 3 / 4, game->name() + ": MCTS(50) beats a random player in >= 75% of games");
        if (p.sims2 == 0 && game->name() == "TicTacToe")
          check::Require(t[1] <= games / 10, "TicTacToe: MCTS(50) rarely loses to a random player");
      }
    }

    std::printf("\n-- Visit counts from the root: 2000 simulations on a tic-tac-toe position\n");
    ttt.Reset();
    ttt.Apply(0);  // X corner
    ttt.Apply(1);  // O next to it (a mistake)
    std::cout << ttt.Render();
    rl::Mcts mcts(rl::RolloutEvaluator(&rng));
    std::vector<int> counts = mcts.Search(ttt, 2000, rng);
    std::printf("  X to move; visits per cell:\n");
    for (int r = 0; r < 3; ++r) {
      std::printf("   ");
      for (int c = 0; c < 3; ++c) std::printf(" %5d", counts[r * 3 + c]);
      std::printf("\n");
    }
    std::printf("  (cell 4, the centre, or cell 2/6 should dominate: they start a double threat)\n");
    {
      int total = 0;
      for (int c : counts) total += c;
      check::Require(total == 2000 && counts[0] == 0 && counts[1] == 0, "visits sum to the simulations, occupied cells get none");
      // X wins by force with 3, 4 or 6 (each makes a threat whose block lets X fork).
      const int best = static_cast<int>(std::max_element(counts.begin(), counts.end()) - counts.begin());
      check::Require(best == 3 || best == 4 || best == 6, "the most visited move is a winning one (3, 4 or 6)");
      check::Require(mcts.last_root_value() > 0.5f, "the root value shows X is winning");
    }

    std::printf("\n-- MCTS games tflow films on the 3D boards (played here without the video)\n");
    auto film = [&](rl::GridGame& game, int sims1, int sims2, const std::string& name) {
      game.Reset();
      while (!game.Terminal()) {
        const int sims = game.CurrentPlayer() == 1 ? sims1 : sims2;
        std::vector<int> c = mcts.Search(game, sims, rng);
        game.Apply(static_cast<int>(std::max_element(c.begin(), c.end()) - c.begin()));
      }
      const std::string outcome = game.Winner() ? "player " + std::to_string(game.Winner()) + " wins" : "draw";
      std::printf("  %s: %s\n", name.c_str(), outcome.c_str());
      return game.Winner();
    };
    const int w_ttt = film(ttt, 400, 400, "mcts_tic_tac_toe_400_vs_400");
    film(c4, 400, 50, "mcts_connect_four_400_vs_50");
    film(oth, 200, 50, "mcts_othello_200_vs_50");
    check::Require(w_ttt == 0, "MCTS(400) vs MCTS(400) tic-tac-toe is a draw");
    std::printf("\nchecks passed (vsim videos omitted)\n");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "CHECK FAILED: %s\n", e.what());
    return 1;
  }
  return 0;
}
