# Native Python NN example. Grid world and cliff walking.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/6_deep_reinforcement_learning/6.1_environments/grid_world.py
"""Grid world and cliff walking.

Port of tflow's grid_world. Two small finite MDPs used by the tabular
examples in 6.2 (common/rl/grid_world.hpp, copied from tflow). Both expose
their transition model (TabularEnv::transitions) so dynamic programming can
plan on them, and both render as text. The vsim videos of the policies are
omitted; the port adds checks that the transition models are distributions
and that the cliff paths have Sutton & Barto's returns (-13 and -17).
"""
import os

from pydatamunge import nn
from pydatamunge.nn_workflows import helper, options

checks = helper("checks")
rl_env = helper("rl_env")
rl_utils = helper("rl_utils")
rl_grid_world = helper("rl_grid_world")
f32 = rl_env.f32
GridBase = rl_grid_world.GridBase


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


def describe(env):
    print("== %s: %dx%d, %d states, %s, max_steps=%d" % (
        env.name(), env.height(), env.width(), env.num_states(), env.action_space(), env.max_steps()))
    env.reset(0)
    print(env.render(), end="")
    print("  (A agent, S start, G goal, # wall, ~ cliff)")


def show_model(env, s):
    r, c = env.coords(s)
    print("  model from state %d (row %d, col %d):" % (s, r, c))
    for a in range(4):
        line = "    action %s:" % GridBase.ARROWS[a]
        for t in env.transitions(s, a):
            line += "  p=%.2f -> s'=%d r=%+.0f%s" % (t.prob, t.next_state, t.reward, " (terminal)" if t.terminal else "")
        print(line)


def check_model(env):
    """Every (s, a) model sums to probability 1 (walls and terminal states have no model)."""
    for s in range(env.num_states()):
        for a in range(4):
            model = env.transitions(s, a)
            if not model:
                continue
            checks.near(sum(t.prob for t in model), 1.0, 1e-5,
                        "%s: transition probabilities from state %d" % (env.name(), s))


def main(arguments=None):
    options(arguments, dict(full=os.environ.get("DATAMUNGE_EXAMPLES_FULL") == "1"))
    grid = rl_grid_world.GridWorld(0.2)
    describe(grid)
    show_model(grid, grid.index(0, 2))
    check_model(grid)
    rl_utils.print_stats("random policy, 200 episodes", run_random(grid, 200, 0))

    def manual(obs):
        """Go down the left side, along the bottom, to G."""
        r, c = grid.coords(int(obs[0]))
        return GridBase.DOWN if r < 4 else GridBase.RIGHT

    rl_utils.print_stats("manual policy, slip 0.2", run_episodes(grid, 200, 0, manual))
    det = rl_grid_world.GridWorld(0.0)
    det_stats = run_episodes(det, 10, 0, manual)
    rl_utils.print_stats("manual policy, no slip", det_stats)
    checks.require(det_stats.min_return() == det_stats.max_return(), "without slip the manual policy is deterministic")

    print()
    cliff = rl_grid_world.CliffWalking()
    describe(cliff)
    show_model(cliff, cliff.index(2, 1))
    check_model(cliff)
    rl_utils.print_stats("random policy, 100 episodes", run_random(cliff, 100, 0))

    # The two policies from Sutton & Barto's figure: hug the cliff, or go up and around.
    # Both reach the goal; the safe one takes 4 extra steps.
    def risky(obs):
        r, c = cliff.coords(int(obs[0]))
        if r == 3 and c == 0:
            return GridBase.UP
        if r == 2 and c < 11:
            return GridBase.RIGHT
        return GridBase.DOWN

    def safe(obs):
        r, c = cliff.coords(int(obs[0]))
        if c == 0 and r > 0:
            return GridBase.UP
        if r == 0 and c < 11:
            return GridBase.RIGHT
        return GridBase.DOWN

    risky_stats = run_episodes(cliff, 5, 0, risky)
    safe_stats = run_episodes(cliff, 5, 0, safe)
    rl_utils.print_stats("risky path (along the cliff)", risky_stats)
    rl_utils.print_stats("safe path (top row)", safe_stats)
    checks.near(risky_stats.mean_return(), -13, 1e-6, "risky cliff path return")
    checks.near(safe_stats.mean_return(), -17, 1e-6, "safe cliff path return")

    print("\n-- Walking into the cliff: -100 and back to the start, episode continues")
    cliff.reset(0)
    cliff.step(rl_env.Action(GridBase.UP))
    result = cliff.step(rl_env.Action(GridBase.RIGHT))
    result = cliff.step(rl_env.Action(GridBase.DOWN))
    print("  reward %+.0f terminated=%d, now at state %d" % (result.reward, result.terminated, cliff.state()))
    checks.require(result.reward == -100.0 and not result.terminated and cliff.state() == cliff.index(3, 0),
                   "the cliff costs -100 and sends the agent back to the start")
    print("\nchecks passed (vsim videos omitted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
