# Native Python NN example. Environment interface: reset, step, render, seeding.
"""Environment interface: reset, step, render, seeding.

CMD: just python-example env_interface

Port of tflow's env_interface. All RL examples talk to environments through
rl::Env (common/rl/env.hpp, copied from tflow), a small Gymnasium-style
interface:
  reset()            -> first observation (optionally with a seed)
  step(action)       -> {obs, reward, terminated, truncated}
  render()           -> text picture of the state
  action_space()     -> Discrete(n) or Continuous(low, high)
This program writes a tiny environment against that interface and then
drives it, and a built-in one, through the same generic loop. The vsim video
of a random CartPole episode is omitted; the port adds checks that seeding
reproduces an episode and that every corridor episode ends one of the three
ways.
"""
import os

from pydatamunge import nn
from pydatamunge.nn_workflows import helper, options

checks = helper("checks")
rl_env = helper("rl_env")
rl_utils = helper("rl_utils")
rl_classic_control = helper("rl_classic_control")
f32 = rl_env.f32


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


class Corridor(rl_env.EnvBase):
    """A 1-D corridor: start in the middle of `length` cells, reach the right end (+1) and
    avoid the left end (-1). A move succeeds with probability 0.8, otherwise the agent slips
    the other way, so the environment needs its RNG and seeding matters."""

    def __init__(self, length=7, max_steps=20):
        super().__init__(max_steps)
        self.length = length
        self.pos = 0

    def name(self):
        return "Corridor"

    def obs_size(self):
        return 1

    def action_space(self):
        return rl_env.ActionSpace.discrete(2)

    def _reset(self):
        self.begin_episode()  # resets the step counter used for truncation
        self.pos = self.length // 2
        return [float(self.pos)]

    def step(self, action):
        direction = 1 if action.discrete == 1 else -1
        if self.uniform(0.0, 1.0) > f32(0.8):  # slip
            direction = -direction
        self.pos += direction
        result = rl_env.StepResult([float(self.pos)])
        if self.pos == self.length - 1:
            result.reward, result.terminated = 1.0, True
        elif self.pos == 0:
            result.reward, result.terminated = -1.0, True
        else:
            result.truncated = self.count_step()  # time limit: not a terminal state
        return result

    def render(self):
        cells = ["."] * self.length
        cells[0] = "x"
        cells[self.length - 1] = "G"
        cells[self.pos] = "A"
        return "".join(cells)


def trace(env, seed, max_print, show_render=True):
    """One episode through the generic interface, printing every step; returns the episode as text."""
    log = []
    rng = nn.Mt19937_64(seed)
    space = env.action_space()
    obs = env.reset(seed)
    print("  t=0  obs=" + rl_utils.format_obs(obs) + "  " + (env.render() if show_render else ""))
    log.append(rl_utils.format_obs(obs))
    t = 1
    while True:
        action = space.sample(rng)
        result = env.step(action)
        log.append("%d:%g:%s" % (action.discrete, result.reward, rl_utils.format_obs(result.obs)))
        if t <= max_print:
            print("  t=%-2d a=%d  r=%+.2f  obs=%s  %s%s%s" % (
                t, action.discrete, result.reward, rl_utils.format_obs(result.obs),
                env.render() if show_render else "", "  TERMINATED" if result.terminated else "",
                "  TRUNCATED" if result.truncated else ""))
        if result.done():
            if t > max_print:
                print("  ... episode ended at t=%d" % t)
            break
        t += 1
    return " ".join(log)


def main(arguments=None):
    options(arguments, dict(full=os.environ.get("DATAMUNGE_EXAMPLES_FULL") == "1"))
    corridor = Corridor()
    print("== %s: obs_size=%d action_space=%s max_steps=%d" % (
        corridor.name(), corridor.obs_size(), corridor.action_space(), corridor.max_steps()))
    print("\n-- A random episode (seed 1)")
    trace(corridor, 1, 25)

    print("\n-- Seeding: the same seed reproduces the episode exactly")
    a = trace(corridor, 7, 4)
    b = trace(corridor, 7, 4)
    print("-- A different seed does not")
    c = trace(corridor, 8, 4)
    checks.require(a == b, "the same seed should reproduce the episode")
    checks.require(a != c, "a different seed should give a different episode")

    print("\n-- terminated vs truncated over 1000 random episodes")
    rng = nn.Mt19937_64(0)
    goal = pit = timeout = 0
    for e in range(1000):
        corridor.reset(100 + e)
        while True:
            result = corridor.step(corridor.action_space().sample(rng))
            if result.terminated:
                if result.reward > 0:
                    goal += 1
                else:
                    pit += 1
            if result.truncated:
                timeout += 1
            if result.done():
                break
    print("  reached goal %d, fell in pit %d, hit the %d-step limit %d" % (goal, pit, corridor.max_steps(), timeout))
    checks.require(goal + pit + timeout == 1000, "every episode ends exactly one way")
    checks.require(goal > 300 and pit > 300 and timeout > 0,
                   "a symmetric random walk reaches both ends about equally often")

    print("\n-- The same loop works on any Env: CartPole through the base class")
    env = rl_classic_control.CartPole()
    print("  %s: obs_size=%d action_space=%s" % (env.name(), env.obs_size(), env.action_space()))
    trace(env, 3, 3, show_render=False)
    rl_utils.print_stats("random policy, 20 episodes", run_random(env, 20, 0))

    print("\n-- Continuous action spaces sample vectors")
    pendulum = rl_classic_control.Pendulum()
    rng = nn.Mt19937_64(0)
    sample = pendulum.action_space().sample(rng)
    print("  %s action_space=%s  sample=%s" % (pendulum.name(), pendulum.action_space(),
                                              rl_utils.format_obs(sample.continuous)))
    checks.require(len(sample.continuous) == 1 and -2.0 <= sample.continuous[0] <= 2.0,
                   "a Pendulum action is one torque in [-2, 2]")
    print("\nchecks passed (vsim video omitted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
