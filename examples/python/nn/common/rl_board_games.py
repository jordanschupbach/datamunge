"""Two-player board games for self-play: tic-tac-toe, Connect Four, Othello
(examples/nn/common/rl/board_games.hpp).

Players are 1 and 2; actions are cell indices (tic-tac-toe, Othello) or columns (Connect
Four). Othello has a pass action when a player has no legal move. Observations are from the
current player's point of view: +1 own stones, -1 opponent's, 0 empty.
"""
import copy


class Game:
    def name(self):
        raise NotImplementedError

    def clone(self):
        return copy.deepcopy(self)

    def opponent(self):
        return 3 - self.current_player()


class GridGame(Game):
    """Shared grid storage and rendering."""

    def __init__(self, rows, cols):
        self._rows, self._cols = rows, cols
        self.cells = [0] * (rows * cols)
        self.player = 1
        self.terminal_ = False
        self.winner_ = 0

    def rows(self):
        return self._rows

    def cols(self):
        return self._cols

    def at(self, r, c):
        return self.cells[r * self._cols + c]

    def current_player(self):
        return self.player

    def terminal(self):
        return self.terminal_

    def winner(self):
        return self.winner_

    def observation(self):
        p = self.player
        return [0.0 if v == 0 else (1.0 if v == p else -1.0) for v in self.cells]

    def render(self):
        lines = []
        for r in range(self._rows):
            lines.append("".join(".XO"[self.at(r, c)] + " " for c in range(self._cols)) + "\n")
        return "".join(lines)

    def set(self, r, c, v):
        self.cells[r * self._cols + c] = v

    def reset_grid(self):
        self.cells = [0] * (self._rows * self._cols)
        self.player = 1
        self.terminal_ = False
        self.winner_ = 0


class TicTacToe(GridGame):
    LINES = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6))

    def __init__(self):
        super().__init__(3, 3)

    def name(self):
        return "TicTacToe"

    def reset(self):
        self.reset_grid()

    def num_actions(self):
        return 9

    def legal_actions(self):
        if self.terminal_:
            return []
        return [i for i in range(9) if self.cells[i] == 0]

    def apply(self, action):
        p, cells = self.player, self.cells
        cells[action] = p
        for a, b, c in TicTacToe.LINES:
            if cells[a] == p and cells[b] == p and cells[c] == p:
                self.winner_ = p
                self.terminal_ = True
        if not self.terminal_ and not self.legal_actions():
            self.terminal_ = True
        if not self.terminal_:
            self.player = 3 - p


class ConnectFour(GridGame):
    DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))

    def __init__(self):
        super().__init__(6, 7)

    def name(self):
        return "ConnectFour"

    def reset(self):
        self.reset_grid()

    def num_actions(self):
        return 7

    def legal_actions(self):
        if self.terminal_:
            return []
        return [c for c in range(7) if self.cells[c] == 0]

    def apply(self, column):
        r = self._rows - 1
        while self.at(r, column) != 0:
            r -= 1
        p = self.player
        self.set(r, column, p)
        for dr, dc in ConnectFour.DIRS:
            count = 1
            for sign in (-1, 1):
                for k in range(1, 4):
                    rr, cc = r + sign * k * dr, column + sign * k * dc
                    if rr < 0 or rr >= self._rows or cc < 0 or cc >= self._cols or self.at(rr, cc) != p:
                        break
                    count += 1
            if count >= 4:
                self.winner_ = p
                self.terminal_ = True
        if not self.terminal_ and not self.legal_actions():
            self.terminal_ = True
        if not self.terminal_:
            self.player = 3 - p


def _othello_rays():
    """For every cell, the cells along each of the 8 directions (nearest first)."""
    rays = []
    for i in range(64):
        r, c = divmod(i, 8)
        cell = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                ray, rr, cc = [], r + dr, c + dc
                while 0 <= rr < 8 and 0 <= cc < 8:
                    ray.append(rr * 8 + cc)
                    rr += dr
                    cc += dc
                if len(ray) >= 2:
                    cell.append(ray)
        rays.append(cell)
    return rays


class Othello(GridGame):
    PASS = 64
    RAYS = _othello_rays()

    def __init__(self):
        super().__init__(8, 8)
        self.reset()

    def name(self):
        return "Othello"

    def reset(self):
        self.reset_grid()
        self.set(3, 3, 2)
        self.set(4, 4, 2)
        self.set(3, 4, 1)
        self.set(4, 3, 1)

    def num_actions(self):
        return 65

    def legal_actions(self):
        if self.terminal_:
            return []
        legal = [i for i in range(64) if self.cells[i] == 0 and self.flips(i, self.player)]
        return legal or [Othello.PASS]

    def apply(self, action):
        p = self.player
        if action != Othello.PASS:
            self.cells[action] = p
            for index in self.flips(action, p):
                self.cells[index] = p
        self.player = 3 - p
        # The game ends when neither player can move.
        if not self.has_move(self.player) and not self.has_move(3 - self.player):
            self.terminal_ = True
            n1, n2 = self.counts()
            self.winner_ = 1 if n1 > n2 else (2 if n2 > n1 else 0)

    def counts(self):
        return self.cells.count(1), self.cells.count(2)

    def has_move(self, player):
        return any(self.cells[i] == 0 and self.flips(i, player) for i in range(64))

    def flips(self, index, player):
        """Indices of opponent stones that placing `player` at index would flip (scan order as in C++)."""
        out, cells, other = [], self.cells, 3 - player
        for ray in Othello.RAYS[index]:
            k = 0
            while k < len(ray) and cells[ray[k]] == other:
                k += 1
            if 0 < k < len(ray) and cells[ray[k]] == player:
                out.extend(ray[:k])
        return out
