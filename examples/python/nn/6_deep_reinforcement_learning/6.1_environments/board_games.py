# Native Python NN example. Tic-tac-toe, Connect Four and Othello, for self-play.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/6_deep_reinforcement_learning/6.1_environments/board_games.py
"""Tic-tac-toe, Connect Four and Othello, for self-play.

Port of tflow's board_games. Board games use rl::Game
(common/rl/board_games.hpp, copied from tflow) rather than rl::Env: two
players alternate, only legal actions may be applied, and the observation is
always from the current player's point of view (+1 own, -1 opponent), which
is what self-play algorithms need. tflow films one more random game per
board on vsim's 3D board; the film is omitted but that game is still played
(same moves, same RNG draws), so the Othello stone count printed after it
matches tflow. The port adds checks on the random-play statistics, the
observation perspective and Clone().
"""
import os

from pydatamunge import nn
from pydatamunge.nn_workflows import helper, options

checks = helper("checks")
rl_board_games = helper("rl_board_games")


def random_game(game, rng):
    """One game between two uniformly random players; returns (winner, moves)."""
    game.reset()
    moves = 0
    while not game.terminal():
        legal = game.legal_actions()
        game.apply(legal[int(rng.modulo(len(legal)))])
        moves += 1
    return game.winner(), moves


def demo(game, seed):
    """A shown random game, 2000 more for statistics, and the one tflow films; returns the rates."""
    rng = nn.Mt19937_64(seed)
    print("== %s: %d actions, observation of %d cells" % (game.name(), game.num_actions(), len(game.observation())))
    winner, moves = random_game(game, rng)
    print("  random game, %d moves, %s" % (moves, "winner: player %d" % winner if winner else "draw"))
    for row in game.render().rstrip("\n").split("\n"):
        print("    " + row)

    wins = [0, 0, 0]
    total_moves = 0
    games = 2000
    for _ in range(games):
        winner, moves = random_game(game, rng)
        wins[winner] += 1
        total_moves += moves
    print("  %d random games: player 1 wins %.1f%%, player 2 wins %.1f%%, draws %.1f%%, mean length %.1f"
          % (games, 100.0 * wins[1] / games, 100.0 * wins[2] / games, 100.0 * wins[0] / games, total_moves / games))
    # One more random game (filmed on the 3D board in tflow; here only played).
    game.reset()
    while not game.terminal():
        legal = game.legal_actions()
        game.apply(legal[int(rng.modulo(len(legal)))])
    return dict(p1=100.0 * wins[1] / games, p2=100.0 * wins[2] / games, draw=100.0 * wins[0] / games,
                length=total_moves / games)


def main(arguments=None):
    options(arguments, dict(full=os.environ.get("DATAMUNGE_EXAMPLES_FULL") == "1"))
    ttt = rl_board_games.TicTacToe()
    t = demo(ttt, 1)
    ttt.reset()
    ttt.apply(4)
    print("  observation from X's view after X plays centre: "
          + "".join("%+.0f " % v for v in ttt.observation()) + " (now O to move, so the centre reads -1)\n")
    checks.require(ttt.observation()[4] == -1.0, "after X plays centre, O sees the centre as -1")
    # Random tic-tac-toe: X wins about 58.5%, O 28.8%, draws 12.7%.
    checks.near(t["p1"], 58.5, 4, "tic-tac-toe player 1 win rate")
    checks.near(t["draw"], 12.7, 3, "tic-tac-toe draw rate")

    c4 = rl_board_games.ConnectFour()
    c = demo(c4, 2)
    print()
    checks.require(c["p1"] > c["p2"] and 7 < c["length"] < 42, "Connect Four random play: first player favoured")

    oth = rl_board_games.Othello()
    o = demo(oth, 3)
    n1, n2 = oth.counts()
    print("  final stone count: X %d, O %d" % (n1, n2))
    oth.reset()
    first = oth.legal_actions()
    print("  legal first moves for X: " + "".join("(%d,%d) " % (a // 8, a % 8) for a in first))
    checks.require(len(first) == 4, "Othello opens with four legal moves")
    checks.require(4 < n1 + n2 <= 64 and o["length"] > 50, "Othello games fill most of the board")

    # clone() lets search algorithms (MCTS, 6.6) try moves without touching the real game.
    copy = c4.clone()
    c4.reset()
    copy.reset()
    copy.apply(3)
    print("\n  Clone(): original has %d legal moves and is at move 0; the clone has advanced (player %d to move)"
          % (len(c4.legal_actions()), copy.current_player()))
    checks.require(len(c4.legal_actions()) == 7 and c4.current_player() != copy.current_player(),
                   "a move on the clone leaves the original untouched")
    print("\nchecks passed (vsim videos omitted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
