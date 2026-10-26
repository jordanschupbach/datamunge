# Native Python NN example. vsim tasks through the binding: InvertedPendulum and Hopper.
"""vsim tasks through the binding: InvertedPendulum and Hopper.

CMD: just python-example vsim_envs

Port of tflow's vsim_envs -- PARTIAL. In tflow, rl::VsimEnv loads the
articulated scenes in data/vsim/ (inverted_pendulum.json, hopper.json),
steps them with vsim's Jolt-based PhysicsWorld (compiled C++20 library
tflow_rl_vsim) and films them with vsim's renderer. The vsim engine and its
renderer are out of scope for the datamunge ports (plan: "RL envs that need
ALE/vsim binaries"), so neither task can be simulated here and none of
tflow's statistics (random 10.65 / feedback 1000.0 on InvertedPendulum,
random 10.11 / zero-torque 136.02 on Hopper) are reproduced.

What remains testable without the engine is the agent-side contract the
example uses: policies emit unit-box actions in [-1, 1] that ScaleAction
maps onto the task's action box (InvertedPendulum: cart force / 100 in
[-3, 3]; Hopper: three joint controls in [-1, 1]), including the linear
feedback controller's clipping. This program checks that mapping and says
that the simulation part is skipped; it never prints simulated numbers.
"""
import os

from pydatamunge.nn_workflows import helper, options

checks = helper("checks")
rl_env = helper("rl_env")
f32, fma = rl_env.f32, rl_env.fma


def scale_action(unit, space):
    """tflow's rl::ScaleAction: clip each unit control to [-1, 1] and map it affinely onto [low, high]."""
    action = []
    for i, value in enumerate(unit):
        u = max(-1.0, min(1.0, value))
        action.append(fma(f32(f32(u + 1) * 0.5), f32(space.high[i] - space.low[i]), space.low[i]))
    return rl_env.Action(continuous=action)


def feedback(obs):
    """A linear feedback controller: push against the angle and its rate."""
    u = fma(5.0, obs[1], obs[3])
    u = fma(f32(0.2), obs[0], u)
    u = fma(f32(0.3), obs[2], u)
    return [max(-1.0, min(1.0, u))]


def main(arguments=None):
    options(arguments, dict(full=os.environ.get("DATAMUNGE_EXAMPLES_FULL") == "1"))
    print("== InvertedPendulum: obs [x, angle, x_dot, angle_dot], action = cart force / 100 in [-3, 3], +1 per step up to 1000")
    print("  SKIPPED: needs vsim's physics engine and scene data/vsim/inverted_pendulum.json (out of scope)")
    pend_space = rl_env.ActionSpace.continuous([-3.0], [3.0])
    print("  feedback controller -> cart action (unit control scaled to %s):" % pend_space)
    pend_ok = True
    for obs in ([0, 0, 0, 0], [0, 0.05, 0, 0.1], [0.1, -0.02, -0.2, 0], [0, 0.5, 0, 1.0]):
        obs = [f32(v) for v in obs]
        u = feedback(obs)[0]
        a = scale_action(feedback(obs), pend_space).continuous[0]
        print("    obs [%+.2f %+.2f %+.2f %+.2f]  u=%+.3f  action=%+.3f" % (obs[0], obs[1], obs[2], obs[3], u, a))
        pend_ok &= abs(fma(-3.0, u, a)) < f32(1e-5) and abs(a) <= 3.0

    print("\n== Hopper: obs [height, pitch, 3 joint angles, 6 rates], 3 joint torques = gear * u with u in [-1, 1], "
          "reward = forward speed + 1 - 1e-3 |u|^2")
    print("  SKIPPED: needs vsim's physics engine and scene data/vsim/hopper.json (out of scope)")
    hop_space = rl_env.ActionSpace.continuous([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
    zero = scale_action([0.0, 0.0, 0.0], hop_space)
    clipped = scale_action([2.0, -5.0, 0.25], hop_space)
    print("  zero torques -> [%+.2f %+.2f %+.2f]; unit [+2 -5 +0.25] clipped -> [%+.2f %+.2f %+.2f]"
          % tuple(zero.continuous + clipped.continuous))

    checks.require(pend_ok, "ScaleAction should map the unit control onto the [-3, 3] cart box")
    checks.require(zero.continuous == [0.0, 0.0, 0.0], "zero controls should stay zero on the symmetric Hopper box")
    checks.require(clipped.continuous == [1.0, -1.0, 0.25], "out-of-range controls should be clipped to the box")
    print("\nchecks passed (simulation and vsim videos skipped: no vsim engine)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
