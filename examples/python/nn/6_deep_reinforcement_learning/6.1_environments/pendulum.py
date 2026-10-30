# Native Python NN example. Pendulum: continuous-control swing-up.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/6_deep_reinforcement_learning/6.1_environments/pendulum.py
"""Pendulum: continuous-control swing-up.

Port of tflow's pendulum. Gymnasium's Pendulum-v1: torque in [-2, 2] is
not enough to lift the pendulum directly, so a controller must pump energy
by pushing along the current rotation, then catch it at the top. Reward is
minus a quadratic cost, so the best possible episode return is 0 and a
random policy scores around -1200. The environment is
common/rl/classic_control.hpp (copied from tflow, TF-free). The vsim video is
omitted: the filmed episode is played without a video
(common/rl/playback.hpp) and checks are added.
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
f32, fma, clamp = rl_env.f32, rl_env.fma, rl_classic_control.clamp


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


def swing_up(obs):
    """Energy-based swing-up plus a PD catch near the top. With g=10, m=l=1 and the rod's
    inertia ml^2/3, the energy relative to the top is
      E = theta_dot^2 / 6 + 5 (cos theta - 1)        (0 at the top, -10 at rest)"""
    cos_t, sin_t, thdot = obs
    theta = rl_env.atan2f(sin_t, cos_t)
    if abs(theta) < 0.5 and abs(thdot) < 3.0:
        return rl_hud.note([clamp(fma(-8.0, theta, -2.0 * thdot), -2.0, 2.0)], "PD regulator near the top")
    energy = fma(5.0, f32(cos_t - 1.0), f32(f32(thdot * thdot) / 6.0))
    direction = 1.0 if thdot >= 0 else -1.0
    return rl_hud.note([2.0 * direction if energy < 0 else -2.0 * direction],
                       "energy " + rl_hud.hud_signed(energy) + (": pump along the swing" if energy < 0 else ": brake"))


def main(arguments=None):
    options(arguments, dict(full=os.environ.get("DATAMUNGE_EXAMPLES_FULL") == "1"))
    env = rl_classic_control.Pendulum()
    print("== %s: obs [cos theta, sin theta, theta_dot], %s, 200 steps, "
          "reward = -(theta^2 + 0.1 theta_dot^2 + 0.001 u^2)" % (env.name(), env.action_space()))
    random = run_random(env, 50, 0)
    rl_utils.print_stats("random, 50 episodes", random)
    ctrl = run_episodes(env, 50, 0, swing_up)
    rl_utils.print_stats("energy swing-up + PD", ctrl)

    print("\n-- One episode under the controller (theta=0 is upright)")
    obs = env.reset(2)
    ret = 0.0
    for t in range(200):
        result = env.step(swing_up(obs).action)
        ret = f32(ret + result.reward)
        obs = result.obs
        if t % 25 == 0 or t == 199:
            print("  t=%-3d theta=%+.2f theta_dot=%+.2f reward=%+.3f" % (
                t, rl_env.atan2f(obs[1], obs[0]), obs[2], result.reward))
    print("  return %.1f" % ret)
    final_theta = rl_env.atan2f(obs[1], obs[0])
    print("\n-- The same episode replayed (tflow films it in the pendulum's 3D world; video omitted)")
    pb = rl_playback.play_episode(env, swing_up, "pendulum_energy_swing_up", 2)

    checks.require(random.mean_return() < -900.0, "a random policy should score far below zero (about -1200)")
    checks.require(ctrl.mean_return() > -300.0 and ctrl.max_return() > -1.0,
                   "the swing-up controller should average above -300 and balance perfectly from some starts")
    checks.require(abs(final_theta) < f32(0.01), "the traced episode should end upright")
    checks.require(pb.steps == 200 and abs(f32(pb.ret - ret)) < f32(1e-3), "the replay should reproduce the traced episode")
    print("\nchecks passed (vsim video omitted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
