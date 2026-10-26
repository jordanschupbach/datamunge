# Native Python NN example. Vectorized environments for parallel rollouts.
"""Vectorized environments for parallel rollouts.

CMD: just python-example vec_env

Port of tflow's vec_env. rl::VecEnv (common/rl/vec_env.hpp, copied from
tflow, TF-free) steps N environments with one call and resets any that
finish, so a learner always has a full batch of observations. It can use a
thread pool; this program measures when that is worth it. The port adds
checks: the auto-reset bookkeeping (final_obs, done flags, episode count)
and that a threaded VecEnv produces exactly the single-threaded rollout.
Throughput numbers depend on the machine and its load.
"""
import math
import os
import time

from pydatamunge import nn
from pydatamunge.nn_workflows import helper, options

checks = helper("checks")
rl_env = helper("rl_env")
rl_vec_env = helper("rl_vec_env")
rl_classic_control = helper("rl_classic_control")
f32 = rl_env.f32


class Expensive(rl_env.Env):
    """Wraps an environment and burns CPU on every step, standing in for an expensive simulator."""

    def __init__(self, inner, work):
        self.inner, self.work = inner, work

    def name(self):
        return "Expensive(" + self.inner.name() + ")"

    def obs_size(self):
        return self.inner.obs_size()

    def action_space(self):
        return self.inner.action_space()

    def _reset(self):
        return self.inner.reset()

    def seed(self, seed):
        self.inner.seed(seed)

    def step(self, action):
        sink = 0.0
        for i in range(self.work):
            sink += math.sin(float(i))
        return self.inner.step(action)


def steps_per_second(venv, iterations, seed):
    rng = nn.Mt19937_64(seed)
    space = venv.action_space()
    venv.reset()
    start = time.perf_counter()
    for _ in range(iterations):
        actions = [space.sample(rng) for _ in range(venv.size())]
        venv.step(actions)
    return venv.size() * iterations / (time.perf_counter() - start)


def same_rollout(a, b, iterations, seed):
    """Steps two VecEnvs with the same random actions; True when every batch matches."""
    rng = nn.Mt19937_64(seed)
    space = a.action_space()
    if a.reset() != b.reset():
        return False
    for _ in range(iterations):
        actions = [space.sample(rng) for _ in range(a.size())]
        sa, sb = a.step(actions), b.step(actions)
        if sa.obs != sb.obs or sa.reward != sb.reward or sa.terminated != sb.terminated or sa.truncated != sb.truncated:
            return False
    return True


def main(arguments=None):
    options(arguments, dict(full=os.environ.get("DATAMUNGE_EXAMPLES_FULL") == "1"))
    print("== Batched stepping with auto-reset")
    venv = rl_vec_env.VecEnv(rl_classic_control.CartPole, 8, 0)
    obs = venv.reset()
    print("  %d x %s, batch observation %d x %d" % (venv.size(), venv.env(0).name(), len(obs), venv.obs_size()))
    rng = nn.Mt19937_64(0)
    episodes_finished = 0
    bookkeeping_ok = True
    returns = [0.0] * venv.size()
    limit_x, limit_theta, start_limit = f32(2.4), f32(0.2094), f32(0.05)
    for t in range(1, 101):
        actions = [venv.action_space().sample(rng) for _ in range(venv.size())]
        s = venv.step(actions)
        bookkeeping_ok &= len(s.obs) == venv.size()
        for i in range(venv.size()):
            returns[i] = f32(returns[i] + s.reward[i])
            if s.done(i):
                episodes_finished += 1
                # CartPole ends when |x| > 2.4 or |theta| > 0.2095 (or at 500 steps); a fresh
                # episode starts within +-0.05 of rest.
                final = s.final_obs[i]
                bookkeeping_ok &= len(final) == 4 and (abs(final[0]) > limit_x or abs(final[2]) > limit_theta)
                bookkeeping_ok &= abs(s.obs[i][0]) <= start_limit and abs(s.obs[i][2]) <= start_limit
                if episodes_finished <= 5:
                    print("  t=%-3d env %d finished with return %.0f; final x=%+.2f, new episode starts at x=%+.2f"
                          % (t, i, returns[i], final[0], s.obs[i][0]))
                returns[i] = 0.0
    print("  after 100 batched steps: %d episodes finished across %d environments" % (episodes_finished, venv.size()))

    print("\n== Throughput: environment steps per second (64 envs, random actions)")
    print("  %-28s %10s %10s %10s %10s" % ("", "1 thread", "2 threads", "4 threads", "8 threads"))
    for work in (0, 2000, 20000):
        label = "CartPole (cheap)" if work == 0 else "CartPole + %d sins" % work
        line = "  %-28s" % label
        for threads in (1, 2, 4, 8):
            def factory(work=work):
                env = rl_classic_control.CartPole()
                return env if work == 0 else Expensive(env, work)

            v = rl_vec_env.VecEnv(factory, 64, 0, threads)
            iterations = 2000 if work == 0 else (200 if work < 10000 else 30)
            # Interpreted steps are far slower than the C++ ones: quick mode times a tenth of the
            # C++ iterations (rates are per step, so they stay comparable).
            iterations = checks.limit(iterations, max(3, iterations // 10))
            line += " %10.0f" % steps_per_second(v, iterations, 1)
            v.close()
        print(line, flush=True)
    print("  Threads only help once a step costs more than the synchronisation.")
    print("  (Python threads share the interpreter lock, so pure-Python steps do not run in parallel.)")

    serial = rl_vec_env.VecEnv(rl_classic_control.CartPole, 64, 0, 1)
    threaded = rl_vec_env.VecEnv(rl_classic_control.CartPole, 64, 0, 4)
    same = same_rollout(serial, threaded, 300, 7)
    threaded.close()
    print("\n  4-thread rollout identical to the 1-thread rollout (64 envs, 300 steps): %s" % ("yes" if same else "NO"))

    checks.require(len(obs) == 8 and venv.obs_size() == 4, "the batch observation should be 8 x 4")
    checks.require(bookkeeping_ok, "finished environments should report a terminal final_obs and restart near rest")
    checks.require(episodes_finished == 34, "random CartPole from seed 0 should finish 34 episodes in 100 steps (as in tflow)")
    checks.require(same, "threading must not change the rollout")
    print("\nchecks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
