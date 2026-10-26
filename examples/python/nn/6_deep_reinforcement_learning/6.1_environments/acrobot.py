# Native Python NN example. Acrobot: a two-link underactuated swing-up.
"""Acrobot: a two-link underactuated swing-up.

CMD: just python-example acrobot

Port of tflow's acrobot. Gymnasium's Acrobot-v1, including its RK4
integrator. Only the elbow is actuated, with torque -1, 0 or +1; the goal is
to swing the tip above a height of one link length. Reward is -1 per step,
so shorter is better. The environment is common/rl/classic_control.hpp
(copied from tflow, TF-free). The vsim video of the pump controller is
omitted; the port adds checks that the pump reaches the goal in every
episode and random actions in none.
"""
import math
import os

from pydatamunge import nn
from pydatamunge.nn_workflows import helper, options

checks = helper("checks")
rl_env = helper("rl_env")
rl_utils = helper("rl_utils")
rl_hud = helper("rl_hud")
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


def pump(obs):
    """Pump energy with the elbow torque opposing the first link's swing."""
    dtheta1 = obs[4]
    action = 0 if dtheta1 > 0 else (2 if dtheta1 < 0 else 1)
    return rl_hud.note(action, "theta1_dot " + rl_hud.hud_signed(dtheta1) + ": torque against it")


def main(arguments=None):
    options(arguments, dict(full=os.environ.get("DATAMUNGE_EXAMPLES_FULL") == "1"))
    env = rl_classic_control.Acrobot()
    print("== %s: obs [cos t1, sin t1, cos t2, sin t2, t1_dot, t2_dot], Discrete(3) torque -1/0/+1, "
          "-1 per step, goal: tip height > 1, max_steps=%d" % (env.name(), env.max_steps()))
    random = run_random(env, 30, 0)
    rl_utils.print_stats("random, 30 episodes", random)
    pumped = run_episodes(env, 30, 0, pump)
    rl_utils.print_stats("pump (torque against t1_dot)", pumped)

    print("\n-- Tip height over one episode under the pump controller")
    obs = env.reset(1)
    goal = False
    for t in range(1, env.max_steps() + 1):
        result = env.step(pump(obs).action)
        obs = result.obs
        if t % 20 == 0 or result.done():
            print("  t=%-3d height=%+.2f  t1=%+.2f t2=%+.2f%s" % (
                t, env.tip_height(), rl_env.atan2f(obs[1], obs[0]), rl_env.atan2f(obs[3], obs[2]),
                "  GOAL" if result.terminated else ""))
        if result.terminated:
            goal = env.tip_height() > 1.0
        if result.done():
            break

    checks.require(pumped.min_return() > -float(env.max_steps()) and pumped.mean_length() < 150,
                   "the pump controller should reach the goal quickly in every episode")
    checks.require(pumped.mean_length() < random.mean_length(), "the pump should beat random actions")
    checks.require(goal, "the traced pump episode should end at the goal with the tip above 1")
    print("\nchecks passed (vsim video omitted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
