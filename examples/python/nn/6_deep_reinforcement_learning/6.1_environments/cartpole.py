# Native Python NN example. CartPole: the balancing benchmark.
"""CartPole: the balancing benchmark.

CMD: just python-example cartpole

Port of tflow's cartpole. Same physics and constants as Gymnasium's
CartPole-v1 (common/rl/classic_control.hpp, copied from tflow). A random
policy lasts about 20 steps; a two-line hand-written controller holds the
pole for the full 500, which is the ceiling later learning algorithms aim
for. The vsim videos of both policies are omitted; the port adds checks on
the returns and on the always-right failure.
"""
import os

from pydatamunge import nn
from pydatamunge.nn_workflows import helper, options

checks = helper("checks")
rl_env = helper("rl_env")
rl_utils = helper("rl_utils")
rl_hud = helper("rl_hud")
rl_classic_control = helper("rl_classic_control")
f32, fma = rl_env.f32, rl_env.fma


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


def controller(obs):
    """Push toward the side the pole is falling to; the angular velocity term adds damping."""
    lean = fma(0.5, obs[3], obs[2])  # positive: the pole is falling to the right
    return rl_hud.note(1 if lean > 0 else 0, "theta + 0.5 theta_dot = " + rl_hud.hud_signed(lean))


def main(arguments=None):
    options(arguments, dict(full=os.environ.get("DATAMUNGE_EXAMPLES_FULL") == "1"))
    env = rl_classic_control.CartPole()
    print("== %s: obs [x, x_dot, theta, theta_dot], %s (0 push left, 1 push right), reward +1/step, max_steps=%d"
          % (env.name(), env.action_space(), env.max_steps()))

    print("\n-- Random policy")
    random = run_random(env, 100, 0)
    rl_utils.print_stats("random, 100 episodes", random)

    print("\n-- Hand-written controller: action = (theta + 0.5 theta_dot > 0)")
    ctrl = run_episodes(env, 100, 0, controller)
    rl_utils.print_stats("controller, 100 episodes", ctrl)

    print("\n-- Failure: always pushing right")
    env.reset(5)
    theta, fell_at, t = 0.0, 0, 1
    while True:
        result = env.step(rl_env.Action(1))
        if result.done():
            print("  episode over at t=%d (theta=%+.3f rad, limit is +-0.209)" % (t, result.obs[2]))
            theta, fell_at = result.obs[2], t
            break
        t += 1

    checks.require(10 < random.mean_length() < 40, "a random policy should last about 20 steps")
    checks.require(ctrl.mean_return() > 450, "the hand-written controller should balance (nearly) every episode to 500")
    checks.require(fell_at < 50 and abs(theta) > f32(0.2), "always pushing right should topple the pole past 0.209 rad")
    print("\nchecks passed (vsim videos omitted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
