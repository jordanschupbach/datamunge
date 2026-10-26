# Native Python NN example. LunarLander-style 2D physics (simplified).
"""LunarLander-style 2D physics (simplified).

CMD: just python-example lunar_lander

Port of tflow's lunar_lander. A rigid body with a main engine and two side
thrusters has to touch down softly on the pad. The physics is a hand-rolled
point mass with rotation (no Box2D), but the observation, actions and reward
shaping follow Gymnasium's LunarLander so later agents transfer with little
change. The environment is common/rl/lunar_lander.hpp (copied from tflow,
TF-free). The vsim video of the heuristic landing is omitted: the port plays
the same episode without filming (common/rl/playback.hpp) and adds checks
(the heuristic lands every episode, random crashes, the replay matches).
"""
import os

from pydatamunge import nn
from pydatamunge.nn_workflows import helper, options

checks = helper("checks")
rl_env = helper("rl_env")
rl_hud = helper("rl_hud")
rl_playback = helper("rl_playback")
rl_classic_control = helper("rl_classic_control")
rl_lunar_lander = helper("rl_lunar_lander")
f32, fma, clamp = rl_env.f32, rl_env.fma, rl_classic_control.clamp


def heuristic(obs):
    """A proportional controller in the spirit of Gymnasium's heuristic: tilt toward the pad,
    cancel horizontal speed, descend slowly, stay upright."""
    x, y, vx, vy, angle, ang_vel = obs[:6]
    angle_target = clamp(fma(f32(0.6), x, vx), f32(-0.4), f32(0.4))
    angle_todo = fma(f32(angle_target - angle), 2.0, -ang_vel)
    hover_target = 0.5 * abs(x)  # come down once over the pad
    hover_todo = fma(f32(hover_target - y), 0.5, -vy)
    note = "hover " + rl_hud.hud_signed(hover_todo) + "   angle " + rl_hud.hud_signed(angle_todo)
    if hover_todo > abs(angle_todo) and hover_todo > f32(0.05):
        return rl_hud.note(2, note)  # main engine
    if angle_todo < f32(-0.05):
        return rl_hud.note(3, note)  # right thruster: rotate clockwise
    if angle_todo > f32(0.05):
        return rl_hud.note(1, note)  # left thruster: rotate counter-clockwise
    return rl_hud.note(0, note)


def outcomes(env, episodes, seed, policy, label):
    """Episodes from env.reset(seed + e): landed/crashed/timed-out counts and the mean return."""
    landed = crashed = timeout = 0
    total = 0.0
    for e in range(episodes):
        obs = env.reset(seed + e)
        while True:
            result = env.step(rl_env.as_action(policy(obs)))
            total = f32(total + result.reward)
            obs = result.obs
            if result.done():
                if env.state().landed:
                    landed += 1
                elif env.state().crashed:
                    crashed += 1
                else:
                    timeout += 1
                break
    mean = f32(total / episodes)
    print("  %-22s mean return %8.1f   landed %3d  crashed %3d  timed out %3d  (of %d)"
          % (label, mean, landed, crashed, timeout, episodes))
    return dict(landed=landed, crashed=crashed, timeout=timeout, mean=mean)


def main(arguments=None):
    options(arguments, dict(full=os.environ.get("DATAMUNGE_EXAMPLES_FULL") == "1"))
    env = rl_lunar_lander.LunarLander()
    print("== %s: obs [x, y, vx, vy, angle, ang_vel, leg_l, leg_r], Discrete(4) = noop/left/main/right\n"
          "   +100 soft landing on the pad, -100 crash, shaping toward slow+upright+close, fuel cost" % env.name())
    rng = nn.Mt19937_64(0)
    random = outcomes(env, 100, 0, lambda obs: env.action_space().sample(rng), "random")
    heuristic_outcome = outcomes(env, 100, 0, heuristic, "heuristic controller")

    print("\n-- One heuristic landing")
    obs = env.reset(4)
    ret, steps, landed, t = 0.0, 0, False, 1
    while True:
        result = env.step(heuristic(obs).action)
        ret = f32(ret + result.reward)
        obs = result.obs
        steps = t
        if t % 60 == 0 or result.done():
            status = "  LANDED" if env.state().landed else "  CRASHED" if env.state().crashed else ""
            print("  t=%-4d x=%+.2f y=%.2f vx=%+.2f vy=%+.2f angle=%+.2f  return so far %.1f%s"
                  % (t, obs[0], obs[1], obs[2], obs[3], obs[4], ret, status))
        if result.done():
            landed = env.state().landed
            break
        t += 1
    print("  the same landing replayed (tflow films it in the 3D world; video omitted):")
    pb = rl_playback.play_episode(env, heuristic, "lunar_lander_pilot", 4)

    checks.require(heuristic_outcome["landed"] == 100, "the heuristic controller should land in every episode")
    checks.require(random["landed"] == 0 and random["crashed"] > 90, "random actions should (almost) always crash")
    checks.require(heuristic_outcome["mean"] > 200.0 and heuristic_outcome["mean"] > random["mean"] + 200.0,
                   "the heuristic should score > 200 on average and far above random")
    checks.require(landed, "the traced episode should end with a landing")
    checks.require(pb.steps == steps and abs(f32(pb.ret - ret)) < f32(1e-3),
                   "replaying seed 4 should reproduce the traced episode exactly")
    print("\nchecks passed (vsim video omitted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
