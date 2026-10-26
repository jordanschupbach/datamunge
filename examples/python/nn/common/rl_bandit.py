"""Bandit environments: k-armed (stateless) and contextual (examples/nn/common/rl/bandit.hpp).

Both are one-step episodes: reset() gives the (possibly empty) context, step(arm) gives the
reward and terminates.
"""
from pydatamunge import nn
from pydatamunge.nn_workflows import helper

rl_env = helper("rl_env")
f32, fma, expf = rl_env.f32, rl_env.fma, rl_env.expf
ActionSpace, EnvBase, StepResult = rl_env.ActionSpace, rl_env.EnvBase, rl_env.StepResult


def arm_names(arms):
    return ["arm %d" % a for a in range(arms)]


class MultiArmedBandit(EnvBase):
    GAUSSIAN, BERNOULLI = 0, 1

    def __init__(self, means, stddev):
        """Gaussian arms: reward ~ N(mean[arm], stddev)."""
        super().__init__()
        self.kind = MultiArmedBandit.GAUSSIAN
        self.means = [f32(m) for m in means]
        self.stddev = f32(stddev)

    @staticmethod
    def bernoulli(probs):
        """Bernoulli arms: reward is 1 with probability p[arm], else 0."""
        bandit = MultiArmedBandit(probs, 0.0)
        bandit.kind = MultiArmedBandit.BERNOULLI
        return bandit

    @staticmethod
    def testbed(k, seed):
        """Sutton & Barto's testbed: means from one N(0, 1) distribution over mt19937_64(seed)."""
        rng = nn.Mt19937_64(int(seed))
        return MultiArmedBandit([rng.normal32() for _ in range(k)], 1.0)

    def name(self):
        kind = "Gaussian" if self.kind == MultiArmedBandit.GAUSSIAN else "Bernoulli"
        return "%sBandit(%d)" % (kind, self.num_arms())

    def num_arms(self):
        return len(self.means)

    def obs_size(self):
        return 0

    def action_space(self):
        return ActionSpace.discrete(self.num_arms())

    def action_names(self):
        return arm_names(self.num_arms())

    def mean(self, arm):
        return self.means[arm]

    def best_arm(self):
        best = 0
        for a in range(1, len(self.means)):
            if self.means[a] > self.means[best]:
                best = a
        return best

    def best_mean(self):
        return self.means[self.best_arm()]

    def _reset(self):
        self.begin_episode()
        return []

    def step(self, action):
        return StepResult([], self.pull(action.discrete), True, False)

    def pull(self, arm):
        if self.kind == MultiArmedBandit.GAUSSIAN:
            return self.normal(self.means[arm], self.stddev)
        return 1.0 if self.uniform(0.0, 1.0) < self.means[arm] else 0.0


class ContextualBandit(EnvBase):
    """x ~ N(0, I) each round; arm a pays 1 with probability sigmoid(w_a . x + b_a)."""

    def __init__(self, context_dim, num_arms, seed):
        super().__init__()
        self.dim, self.arms = context_dim, num_arms
        rng = nn.Mt19937_64(int(seed))
        self.w = [[rng.normal32() for _ in range(context_dim)] for _ in range(num_arms)]
        self.b = [f32(0.5 * rng.normal32()) for _ in range(num_arms)]
        self._context = []

    def name(self):
        return "ContextualBandit"

    def obs_size(self):
        return self.dim

    def action_space(self):
        return ActionSpace.discrete(self.arms)

    def action_names(self):
        return arm_names(self.arms)

    def num_arms(self):
        return self.arms

    def _reset(self):
        self.begin_episode()
        self._context = [self.normal(0.0, 1.0) for _ in range(self.dim)]
        return list(self._context)

    def context(self):
        return self._context

    def expected_reward(self, arm):
        """P(reward = 1 | context, arm): the quantity a learner is trying to find."""
        z = self.b[arm]
        for i in range(self.dim):
            z = fma(self.w[arm][i], self._context[i], z)
        return f32(1.0 / f32(1.0 + expf(-z)))

    def best_arm(self):
        best = 0
        for a in range(1, self.arms):
            if self.expected_reward(a) > self.expected_reward(best):
                best = a
        return best

    def step(self, action):
        reward = 1.0 if self.uniform(0.0, 1.0) < self.expected_reward(action.discrete) else 0.0
        return StepResult([], reward, True, False)
