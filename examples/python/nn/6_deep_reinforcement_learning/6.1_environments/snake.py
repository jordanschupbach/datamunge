# Native Python NN example. Snake: a simple Atari-like pixel game.
"""Snake: a simple Atari-like pixel game.

CMD: just python-example snake

Port of tflow's snake. The observation is an image: three planes (head,
body, food) over the grid, flattened to a float vector. That makes Snake the
stand-in for Atari in the pixel-based examples later on (DQN on pixels,
6.3). The environment is common/rl/snake.hpp (copied from tflow, TF-free).
tflow films the greedy episode on a vsim 3D board; the port replays the same
episode without vsim/ffmpeg and instead writes its frames, drawn from the
board (common/rl/snake_frames.hpp), as one PNG filmstrip to
output/nn/frames/snake_greedy_policy.png. Checks are added.
"""
import os

from pydatamunge import nn
from pydatamunge.nn_workflows import helper, options

checks = helper("checks")
rl_env = helper("rl_env")
rl_utils = helper("rl_utils")
rl_playback = helper("rl_playback")
rl_snake = helper("rl_snake")
rl_snake_frames = helper("rl_snake_frames")
f32 = rl_env.f32
Snake = rl_snake.Snake


def run_episodes(env, episodes, seed, policy):
    """Episodes from env.reset(seed + e) under policy(obs); returns their EpisodeStats."""
    stats = rl_utils.EpisodeStats()
    for e in range(episodes):
        obs = env.reset(seed + e)
        ret, length = 0.0, 0
        while True:
            result = env.step(rl_env.as_action(policy(obs)))
            ret = f32(ret + result.reward)
            length += 1
            obs = result.obs
            if result.done():
                break
        stats.returns.append(ret)
        stats.lengths.append(length)
    return stats


def run_random(env, episodes, seed):
    rng = nn.Mt19937_64(seed)
    space = env.action_space()
    return run_episodes(env, episodes, seed, lambda obs: space.sample(rng))


class GreedyPolicy:
    """Head for the food along whichever axis is further off, but never into a wall or the
    body if another safe move exists."""

    DR, DC = (-1, 0, 1, 0), (0, 1, 0, -1)

    def __init__(self, env):
        self.env = env

    def __call__(self, obs):
        hr, hc = self.env.head()
        fr, fc = self.env.food()
        vertical = Snake.UP if fr < hr else Snake.DOWN
        horizontal = Snake.LEFT if fc < hc else Snake.RIGHT
        prefer = [vertical, horizontal] if abs(fr - hr) >= abs(fc - hc) else [horizontal, vertical]
        prefer += [0, 1, 2, 3]
        for direction in prefer:
            if self.safe(direction):
                return rl_env.Action(direction)
        return rl_env.Action(prefer[0])

    def safe(self, direction):
        hr, hc = self.env.head()
        r, c = hr + self.DR[direction], hc + self.DC[direction]
        if r < 0 or r >= self.env.height() or c < 0 or c >= self.env.width():
            return False
        return not self.env.occupied(r, c)


def main(arguments=None):
    options(arguments, dict(full=os.environ.get("DATAMUNGE_EXAMPLES_FULL") == "1"))
    env = Snake(8, 8)
    shape = env.obs_shape()
    print("== %s: observation [planes=%d, height=%d, width=%d] = %d floats, %s, +1 food, -1 death"
          % (env.name(), shape[0], shape[1], shape[2], env.obs_size(), env.action_space()))

    random = run_random(env, 200, 0)
    rl_utils.print_stats("random, 200 episodes", random)
    greedy = GreedyPolicy(env)
    greedy_stats = run_episodes(env, 200, 0, greedy)
    rl_utils.print_stats("greedy toward food, 200 episodes", greedy_stats)

    print("\n-- A greedy episode")
    obs = env.reset(3)
    steps, game_over, t = 0, False, 0
    while True:
        result = env.step(greedy(obs))
        obs = result.obs
        steps = t + 1
        if t % 15 == 0 or result.done():
            print("  t=%d length=%d reward=%+.0f%s" % (t, env.length(), result.reward,
                                                      "  GAME OVER" if result.terminated else ""))
        if result.done():
            game_over = result.terminated
            break
        t += 1

    print("\n-- The observation planes for that final state")
    counts = [0, 0, 0]
    for p, plane in enumerate(("head", "body", "food")):
        print("  " + plane + ":")
        for r in range(shape[1]):
            row = ""
            for c in range(shape[2]):
                on = obs[p * shape[1] * shape[2] + r * shape[2] + c] > 0
                counts[p] += on
                row += "1" if on else "."
            print("    " + row)

    print("\n-- The greedy episode replayed as frames (tflow films it on a 3D board; video omitted)")
    strip = rl_snake_frames.SnakeFilmstrip()
    pb = rl_playback.play_episode(env, greedy, "snake_greedy_policy", 3, lambda e, hud: strip.add(e))
    png = checks.output_dir("frames") / "snake_greedy_policy.png"
    strip.save(png)
    print("  wrote %s (%d frames, one per step plus the reset)" % (png, strip.frames()))

    checks.require(greedy_stats.mean_return() > 10.0 and greedy_stats.min_return() > 0.0,
                   "the greedy policy should eat food in every episode, > 10 on average")
    checks.require(random.mean_return() < 0.0, "random moves should mostly die before eating")
    checks.require(game_over, "the traced greedy episode should end in a collision")
    checks.require(counts[0] == 1 and counts[2] == 1 and counts[1] == env.length() - 1,
                   "the planes should hold one head, one food and length-1 body cells")
    checks.require(pb.steps == steps and pb.ret == float(env.length() - 2),
                   "the replay should reproduce the traced episode (food eaten - 1 = final length - 2, the fatal step drops the tail)")
    checks.require(strip.frames() == steps + 1, "the filmstrip should have one frame per step plus the reset")
    print("\nchecks passed (vsim video replaced by a PNG filmstrip)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
