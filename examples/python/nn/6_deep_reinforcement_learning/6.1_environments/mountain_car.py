# Native Python NN example. MountainCar, discrete and continuous.
"""MountainCar, discrete and continuous.

CMD: just python-example mountain_car

Port of tflow's mountain_car. The car's engine is too weak to climb
directly, so it has to build momentum. Random actions never reach the flag
within the step limit; a one-line energy-pumping controller (accelerate in
the direction you are already moving) does it in about 100 steps. The
environments are common/rl/classic_control.hpp (copied from tflow,
TF-free). The vsim videos are omitted: the filmed episodes are played
without a video (common/rl/playback.hpp) and checks are added.
"""
import os

from pydatamunge import nn
from pydatamunge.nn_workflows import helper, options

checks = helper("checks")
rl_env = helper("rl_env")
rl_utils = helper("rl_utils")
rl_hud = helper("rl_hud")
rl_playback = helper("rl_playback")
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


def position_trace(env, policy, seed):
    """A one-line "chart" of the car's position over an episode."""
    trace = ""
    obs = env.reset(seed)
    for t in range(400):
        result = env.step(rl_env.as_action(policy(obs)))
        obs = result.obs
        if t % 5 == 0:
            level = int(f32(f32(f32(obs[0] + f32(1.2)) / f32(1.8)) * 5))
            trace += "._-=^"[min(4, max(0, level))]
        if result.done():
            break
    return trace


def pump(obs):
    return rl_hud.note(2 if obs[1] >= 0 else 0, "velocity " + rl_hud.hud_signed(obs[1]) + ": push along it")


def full_throttle(obs):
    return [1.0 if obs[1] >= 0 else -1.0]


def half_throttle(obs):
    # The continuous task rewards fuel economy: half throttle still makes it and pays a
    # quarter of the action cost.
    return rl_hud.note([0.5 if obs[1] >= 0 else -0.5], "half throttle along the velocity")


def main(arguments=None):
    options(arguments, dict(full=os.environ.get("DATAMUNGE_EXAMPLES_FULL") == "1"))
    print("== MountainCar-v0: obs [position, velocity], Discrete(3) = push left/none/right, "
          "-1 per step, flag at x=0.5, 200-step limit")
    env = rl_classic_control.MountainCar()
    random = run_random(env, 50, 0)
    rl_utils.print_stats("random, 50 episodes", random)
    pumped = run_episodes(env, 50, 0, pump)
    rl_utils.print_stats("pump energy (push along velocity)", pumped)
    print("  position over time, low ('.') to high ('^'):")
    rng = nn.Mt19937_64(0)
    print("    random: " + position_trace(env, lambda obs: env.action_space().sample(rng), 1))
    pump_trace = position_trace(env, pump, 1)
    print("    pump:   " + pump_trace)
    print("  the pump controller on the hill (tflow films it in 3D; video omitted):")
    pb = rl_playback.play_episode(env, pump, "mountain_car_pump", 1)

    print("\n== MountainCarContinuous-v0: action in [-1, 1], reward -0.1 a^2 per step, "
          "+100 at x=0.45, 999-step limit")
    cenv = rl_classic_control.MountainCarContinuous()
    crandom = run_random(cenv, 20, 0)
    rl_utils.print_stats("random, 20 episodes", crandom)
    full = run_episodes(cenv, 20, 0, full_throttle)
    rl_utils.print_stats("pump energy (full throttle)", full)
    half = run_episodes(cenv, 20, 0, half_throttle)
    rl_utils.print_stats("pump energy (half throttle)", half)
    print("    pump:   " + position_trace(cenv, full_throttle, 1))
    print("  half throttle (video omitted):")
    cpb = rl_playback.play_episode(cenv, half_throttle, "mountain_car_continuous_half_throttle", 1)

    checks.require(random.max_return() == -200.0, "random actions should never reach the flag in 200 steps")
    checks.require(pumped.min_return() > -200.0 and pumped.mean_length() < 150.0,
                   "the pump controller should reach the flag in every episode")
    checks.require(pump_trace != "" and pump_trace[-1] == "^", "the pump trace should end at the top")
    checks.require(pb.steps < 200 and pb.ret == -float(pb.steps), "the played pump episode should reach the flag")
    checks.require(crandom.max_return() < 0.0 and crandom.mean_length() == 999.0,
                   "random throttle should time out in the continuous task")
    checks.require(full.min_return() > 80.0 and half.min_return() > 80.0, "both pump throttles should reach the flag")
    checks.require(half.mean_return() > full.mean_return(), "half throttle should pay less fuel than full throttle")
    checks.require(cpb.ret > 90.0, "the played half-throttle episode should reach the flag")
    print("\nchecks passed (vsim videos omitted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
