"""Native CPU pendulum/hopper rollouts through datamunge's generated binding.

Uses datamunge's own articulated task definitions. Reference Jolt returns and
renderer output are separate validation targets in docs/simulation-plan.md.
"""
import math
import random
from pydatamunge import datamunge as dm


def episode(kind, feedback=False):
    count = 8
    task = dm.TaskBatch.create(kind, count, 2, 42, 200)
    actions = [0.0] * (count * task.action_size())
    returns, active = [0.0] * count, [1] * count
    rng = random.Random(7)
    for _ in range(200):
        if not any(active):
            break
        obs = task.output().observations
        if kind == dm.Task_InvertedPendulum:
            for i in range(count):
                x, angle, speed, rate = obs[i*4:i*4+4]
                actions[i] = max(-3.0, min(3.0, .08*x + .15*speed - 1.5*angle - .3*rate)) if feedback else rng.uniform(-3, 3)
        result = task.step(actions, active)
        for i in range(count):
            if active[i]:
                returns[i] += result.rewards[i]
                active[i] = int(not (result.terminated[i] or result.truncated[i]))
    assert all(math.isfinite(value) for value in returns)
    return sum(returns) / count


def main(arguments=None):
    random_return = episode(dm.Task_InvertedPendulum)
    feedback_return = episode(dm.Task_InvertedPendulum, True)
    hopper_return = episode(dm.Task_Hopper)
    print(f"Native CPU InvertedPendulum: random mean return={random_return:.3f}, feedback={feedback_return:.3f}")
    print(f"Native CPU Hopper: zero-torque mean return={hopper_return:.3f}")
    print("checks passed (native physics rollouts; explicit termination and truncation)")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
