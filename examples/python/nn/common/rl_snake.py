"""Snake with a pixel-style observation (examples/nn/common/rl/snake.hpp).

Three planes (head, body, food) of height x width, flattened to a float vector. Eating food
gives +1, dying -1, every other step 0; the episode is truncated after too long without food.
"""
from collections import deque

from pydatamunge.nn_workflows import helper

rl_env = helper("rl_env")
ActionSpace, EnvBase, StepResult = rl_env.ActionSpace, rl_env.EnvBase, rl_env.StepResult

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
_DR, _DC = (-1, 0, 1, 0), (0, 1, 0, -1)


class Snake(EnvBase):
    UP, RIGHT, DOWN, LEFT = UP, RIGHT, DOWN, LEFT

    def __init__(self, height=8, width=8):
        super().__init__(0)
        self.h, self.w = height, width
        self._body = deque()
        self.dir = RIGHT
        self._food = (0, 0)
        self.since_food = 0

    def name(self):
        return "Snake"

    def obs_size(self):
        return 3 * self.h * self.w

    def obs_shape(self):
        """[planes, height, width], for reshaping into an image."""
        return [3, self.h, self.w]

    def action_space(self):
        return ActionSpace.discrete(4)

    def action_names(self):
        return ["up", "right", "down", "left"]

    def length(self):
        return len(self._body)

    def head(self):
        return self._body[0]

    def body(self):
        """Body cells from the head to the tail."""
        return self._body

    def food(self):
        return self._food

    def height(self):
        return self.h

    def width(self):
        return self.w

    def occupied(self, r, c):
        return (r, c) in self._body

    def _reset(self):
        self.begin_episode()
        self._body = deque([(self.h // 2, self.w // 2), (self.h // 2, self.w // 2 - 1)])
        self.dir = RIGHT
        self.since_food = 0
        self.place_food()
        return self.obs_now()

    def step(self, action):
        a = action.discrete
        if (a + 2) % 4 != self.dir:  # reversing into the body is ignored
            self.dir = a
        hr, hc = self._body[0]
        nxt = (hr + _DR[self.dir], hc + _DC[self.dir])
        ate = nxt == self._food
        if not ate:
            self._body.pop()  # the tail moves away unless we grow
        hit_wall = nxt[0] < 0 or nxt[0] >= self.h or nxt[1] < 0 or nxt[1] >= self.w
        if hit_wall or nxt in self._body:
            return StepResult(self.obs_now(), -1.0, True, False)
        self._body.appendleft(nxt)
        reward, terminated, truncated = 0.0, False, False
        if ate:
            reward = 1.0
            self.since_food = 0
            if self.length() == self.h * self.w:
                terminated = True  # the board is full: won
            else:
                self.place_food()
        else:
            self.since_food += 1
            if self.since_food > 4 * self.h * self.w:
                truncated = True
        return StepResult(self.obs_now(), reward, terminated, truncated)

    def render(self):
        grid = [["."] * self.w for _ in range(self.h)]
        for r, c in self._body:
            grid[r][c] = "o"
        hr, hc = self._body[0]
        grid[hr][hc] = "@"
        grid[self._food[0]][self._food[1]] = "*"
        return "".join("".join(row) + "\n" for row in grid)

    def obs_now(self):
        plane = self.h * self.w
        obs = [0.0] * (3 * plane)
        hr, hc = self._body[0]
        obs[hr * self.w + hc] = 1.0
        for i in range(1, len(self._body)):
            r, c = self._body[i]
            obs[plane + r * self.w + c] = 1.0
        obs[2 * plane + self._food[0] * self.w + self._food[1]] = 1.0
        return obs

    def place_food(self):
        body = set(self._body)
        free = [(r, c) for r in range(self.h) for c in range(self.w) if (r, c) not in body]
        self._food = free[int(self.rng().modulo(len(free)))]
