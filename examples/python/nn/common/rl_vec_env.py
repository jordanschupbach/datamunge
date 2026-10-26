"""A batch of environments stepped together, for parallel rollouts (examples/nn/common/rl/vec_env.hpp).

step() takes one action per environment and returns batched results. Environments that finish
are reset automatically; the observation returned for them is the first observation of the
new episode, and the last observation of the finished episode is kept in `final_obs`.
With threads > 1 a persistent worker pool steps the environments (Python threads share the
interpreter lock, so the pool only overlaps work that releases it).
"""
from concurrent.futures import ThreadPoolExecutor


class VecStep:
    def __init__(self, n):
        self.obs = [None] * n
        self.reward = [0.0] * n
        self.terminated = [False] * n
        self.truncated = [False] * n
        self.final_obs = [[] for _ in range(n)]  # only set for environments that just ended

    def done(self, i):
        return self.terminated[i] or self.truncated[i]


class VecEnv:
    def __init__(self, factory, n, seed, threads=1):
        self.envs = []
        for i in range(n):
            env = factory()
            env.seed(seed + i)
            self.envs.append(env)
        self.threads = threads
        self.pool = ThreadPoolExecutor(max_workers=threads) if threads > 1 else None

    def close(self):
        if self.pool is not None:
            self.pool.shutdown()
            self.pool = None

    def __del__(self):
        self.close()

    def size(self):
        return len(self.envs)

    def env(self, i):
        return self.envs[i]

    def obs_size(self):
        return self.envs[0].obs_size()

    def action_space(self):
        return self.envs[0].action_space()

    def for_each(self, fn):
        """fn(i) for every environment, sequentially or on the worker pool."""
        if self.pool is None:
            for i in range(len(self.envs)):
                fn(i)
        else:
            list(self.pool.map(fn, range(len(self.envs))))

    def reset(self):
        obs = [None] * len(self.envs)

        def one(i):
            obs[i] = self.envs[i].reset()

        self.for_each(one)
        return obs

    def step(self, actions):
        out = VecStep(len(self.envs))

        def one(i):
            r = self.envs[i].step(actions[i])
            out.reward[i] = r.reward
            out.terminated[i] = r.terminated
            out.truncated[i] = r.truncated
            if r.done():
                out.final_obs[i] = r.obs
                out.obs[i] = self.envs[i].reset()
            else:
                out.obs[i] = r.obs

        self.for_each(one)
        return out
