"""Grid-world environments: a configurable GridWorld and Sutton & Barto's CliffWalking
(examples/nn/common/rl/grid_world.hpp).

Four actions (up, right, down, left); walls block movement, the goal ends the episode and
optional slip noise replaces the chosen action with a perpendicular one. Both expose their
transition model for dynamic programming.
"""
from pydatamunge.nn_workflows import helper

rl_env = helper("rl_env")
f32 = rl_env.f32
ActionSpace, StepResult, TabularEnv, Transition = (rl_env.ActionSpace, rl_env.StepResult,
                                                  rl_env.TabularEnv, rl_env.Transition)

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ARROWS = ("^", ">", "v", "<")
_DR, _DC = (-1, 0, 1, 0), (0, 1, 0, -1)


class Outcome:
    __slots__ = ("row", "col", "reward", "terminal")

    def __init__(self, row, col, reward, terminal):
        self.row, self.col, self.reward, self.terminal = row, col, reward, terminal


class GridBase(TabularEnv):
    UP, RIGHT, DOWN, LEFT = UP, RIGHT, DOWN, LEFT
    ARROWS = ARROWS

    def __init__(self, height, width, start, goal, slip_prob, max_steps):
        super().__init__(max_steps)
        self._height, self._width = height, width
        self.start, self.goal = tuple(start), tuple(goal)
        self.slip_prob = f32(slip_prob)
        self.walls = set()
        self.row = self.col = 0

    def height(self):
        return self._height

    def width(self):
        return self._width

    def num_states(self):
        return self._height * self._width

    def state(self):
        return self.index(self.row, self.col)

    def action_space(self):
        return ActionSpace.discrete(4)

    def action_names(self):
        return ["up", "right", "down", "left"]

    def index(self, r, c):
        return r * self._width + c

    def coords(self, s):
        return divmod(s, self._width)

    def is_wall(self, r, c):
        return (r, c) in self.walls

    def is_goal(self, r, c):
        return (r, c) == self.goal

    def add_wall(self, r, c):
        self.walls.add((r, c))

    def _reset(self):
        self.begin_episode()
        self.row, self.col = self.start
        return self.state_obs()

    def step(self, action):
        a = action.discrete
        if self.slip_prob > 0 and self.uniform(0.0, 1.0) < self.slip_prob:
            a = (a + (1 if self.rng().modulo(2) else 3)) % 4
        o = self.apply(self.row, self.col, a)
        self.row, self.col = o.row, o.col
        truncated = not o.terminal and self.count_step()
        return StepResult(self.state_obs(), o.reward, o.terminal, truncated)

    def apply(self, r, c, a):
        """Deterministic outcome of action a from (r, c); subclasses customise."""
        nr, nc = r + _DR[a], c + _DC[a]
        if nr < 0 or nr >= self._height or nc < 0 or nc >= self._width or self.is_wall(nr, nc):
            nr, nc = r, c
        return Outcome(nr, nc, -1.0, self.is_goal(nr, nc))

    def transitions(self, s, a):
        r, c = self.coords(s)
        out = []
        if self.is_goal(r, c) or self.is_wall(r, c):
            return out  # absorbing / unreachable

        def add(act, p):
            o = self.apply(r, c, act)
            out.append(Transition(p, self.index(o.row, o.col), o.reward, o.terminal))

        if self.slip_prob > 0:
            add(a, f32(1 - self.slip_prob))
            add((a + 1) % 4, f32(self.slip_prob / 2))
            add((a + 3) % 4, f32(self.slip_prob / 2))
        else:
            add(a, 1.0)
        return out

    def render_with(self, overlay):
        """The grid; overlay(r, c) may return a one-character glyph or "" for the default."""
        text = ""
        for r in range(self._height):
            for c in range(self._width):
                glyph = overlay(r, c)
                if not glyph:
                    if r == self.row and c == self.col:
                        glyph = "A"
                    elif self.is_wall(r, c):
                        glyph = "#"
                    elif self.is_goal(r, c):
                        glyph = "G"
                    elif (r, c) == self.start:
                        glyph = "S"
                    else:
                        glyph = self.cell_glyph(r, c)
                text += glyph + " "
            text += "\n"
        return text

    def render(self):
        return self.render_with(lambda r, c: "")

    def cell_glyph(self, r, c):
        return "."


class GridWorld(GridBase):
    """A 5x6 room with a wall in the middle; -1 per step until the goal."""

    def __init__(self, slip_prob=0.0, max_steps=100):
        super().__init__(5, 6, (0, 0), (4, 5), slip_prob, max_steps)
        for r in range(4):
            self.add_wall(r, 3)

    def name(self):
        return "GridWorld"


class CliffWalking(GridBase):
    """4x12; stepping into the cliff costs -100 and returns to the start without ending."""

    def __init__(self, max_steps=500):
        super().__init__(4, 12, (3, 0), (3, 11), 0.0, max_steps)

    def name(self):
        return "CliffWalking"

    def is_cliff(self, r, c):
        return r == 3 and 1 <= c <= 10

    def apply(self, r, c, a):
        o = GridBase.apply(self, r, c, a)
        if self.is_cliff(o.row, o.col):
            return Outcome(self.start[0], self.start[1], -100.0, False)
        return o

    def cell_glyph(self, r, c):
        return "~" if self.is_cliff(r, c) else "."
