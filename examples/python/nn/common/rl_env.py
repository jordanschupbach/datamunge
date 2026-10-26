"""Gymnasium-style environment interface shared by the RL examples (examples/nn/common/rl/env.hpp).

reset() returns the first observation, step() a StepResult with the next observation, a
reward and the two end-of-episode flags: `terminated` (the MDP reached a terminal state) and
`truncated` (a time limit cut the episode short; bootstrapping is still valid).

The C++ environments compute in float32 and draw from std::mt19937_64. The ports reproduce
both: every environment RNG is an nn.Mt19937_64 (same engine, same libc++ distributions) and
the physics rounds each operation to float32 with f32(). The canonical build (clang, arm64,
-ffp-contract=on) fuses `a + b * c` written in one expression into a single-rounding fma;
fma() reproduces that, so trajectories match the C++ bit for bit, not just at print precision.
"""
import math
import struct

from pydatamunge import nn

_FLOAT = struct.Struct("<f")
_pack, _unpack = _FLOAT.pack, _FLOAT.unpack


def f32(x):
    """x rounded to the nearest float32 (as a C++ float expression would store it)."""
    return _unpack(_pack(x))[0]


def fma(a, b, c):
    """float32 a * b + c with one rounding: clang's contraction of `c + a * b` for floats.

    For float32 operands a * b is exact in double, so only the sum is rounded (to double,
    then to float; the double rounding is harmless except in astronomically rare ties)."""
    return _unpack(_pack(a * b + c))[0]


def cpp_round(x):
    """std::round: halves away from zero (Python's round() rounds halves to even)."""
    r = math.floor(abs(x))
    if abs(x) - r >= 0.5:
        r += 1
    return int(math.copysign(r, x))


def cpp_float(x):
    """std::to_string(float): printf("%f")."""
    return "%f" % x


# The C++ calls the platform's float libm (cosf, sinf, atan2f), which is not always correctly
# rounded, so rounding the double result could differ by an ulp. nn.cos/nn.sin/nn.atan2 on a
# float32 tensor call exactly those functions, so the ports use them for scalars too.
# Where one C++ function takes both the sine and the cosine of the same float, clang on Darwin
# merges them into one __sincosf_stret call. Its results are not always the libm cosf/sinf
# values either, but they are the correctly rounded ones in all but ~0.03% of arguments (a
# 1e7-sample scan), so sincosf() uses the correctly rounded pair. No binding exposes the
# platform's float sincos, so this residual 1-ulp mismatch is the one known deviation.
def sincosf(x):
    """(cos x, sin x) as clang's merged sincos computes them (up to its rare 1-ulp errors)."""
    return f32(math.cos(x)), f32(math.sin(x))


def cosf(x):
    return nn.cos(nn.from_vector([x], [1])).item()


def sinf(x):
    return nn.sin(nn.from_vector([x], [1])).item()


def atan2f(y, x):
    return nn.atan2(nn.from_vector([y], [1]), nn.from_vector([x], [1])).item()


def sqrtf(x):
    return f32(math.sqrt(x))


def expf(x):
    """libm expf (nn.exp on fewer than 32 floats is scalar std::exp)."""
    return nn.exp(nn.from_vector([x], [1])).item()


class Action:
    """Either a discrete action (an index) or a continuous one (a vector)."""

    __slots__ = ("discrete", "continuous")

    def __init__(self, discrete=-1, continuous=None):
        self.discrete = discrete
        self.continuous = [] if continuous is None else list(continuous)

    def __repr__(self):
        return f"Action({self.discrete})" if self.discrete >= 0 else f"Action({self.continuous})"

    def __eq__(self, other):
        return isinstance(other, Action) and self.discrete == other.discrete and self.continuous == other.continuous


def as_action(value):
    """The C++ implicit conversions: int -> discrete, sequence -> continuous, Decision -> its action."""
    if isinstance(value, Action):
        return value
    action = getattr(value, "action", None)  # a hud.Decision stands in for its action
    if isinstance(action, Action):
        return action
    if isinstance(value, int):
        return Action(int(value))
    return Action(continuous=[f32(v) for v in value])


class ActionSpace:
    DISCRETE, CONTINUOUS = 0, 1

    def __init__(self, kind=0, n=0, low=None, high=None):
        self.kind = kind
        self.n = n
        self.low = [] if low is None else [f32(v) for v in low]
        self.high = [] if high is None else [f32(v) for v in high]
        self.dim = len(self.low)

    @staticmethod
    def discrete(n):
        return ActionSpace(ActionSpace.DISCRETE, n)

    @staticmethod
    def continuous(low, high):
        return ActionSpace(ActionSpace.CONTINUOUS, 0, low, high)

    def sample(self, rng):
        """A uniformly random action; rng is an nn.Mt19937_64 (rng() % n, uniform_real<float>)."""
        if self.kind == ActionSpace.DISCRETE:
            return Action(int(rng.modulo(self.n)))
        return Action(continuous=[rng.uniform32(self.low[i], self.high[i]) for i in range(self.dim)])

    def to_string(self):
        if self.kind == ActionSpace.DISCRETE:
            return f"Discrete({self.n})"
        bounds = ", ".join(cpp_float(lo) + ".." + cpp_float(hi) for lo, hi in zip(self.low, self.high))
        return f"Box({self.dim}) [{bounds}]"

    __str__ = to_string


class StepResult:
    __slots__ = ("obs", "reward", "terminated", "truncated")

    def __init__(self, obs=None, reward=0.0, terminated=False, truncated=False):
        self.obs = [] if obs is None else obs
        self.reward = reward
        self.terminated = terminated
        self.truncated = truncated

    def done(self):
        return self.terminated or self.truncated


class Env:
    """rl::Env. Subclasses implement name, obs_size, action_space, step, seed and _reset."""

    def name(self):
        raise NotImplementedError

    def obs_size(self):
        raise NotImplementedError

    def action_space(self):
        raise NotImplementedError

    def reset(self, seed=None):
        """Start a new episode; with a seed the environment's RNG is reseeded first."""
        if seed is not None:
            self.seed(seed)
        return self._reset()

    def _reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def render(self):
        return ""

    def action_names(self):
        return []

    def seed(self, seed):
        raise NotImplementedError

    def max_steps(self):
        return 0


class EnvBase(Env):
    """Common plumbing: an mt19937_64 RNG, a step counter and time-limit truncation."""

    def __init__(self, max_steps=0):
        self._max_steps = max_steps
        self._steps = 0
        self._rng = nn.Mt19937_64(0)

    def seed(self, seed):
        self._rng = nn.Mt19937_64(int(seed))

    def max_steps(self):
        return self._max_steps

    def steps(self):
        return self._steps

    def rng(self):
        return self._rng

    def begin_episode(self):
        self._steps = 0

    def count_step(self):
        """True if the time limit is reached after this step."""
        if self._max_steps <= 0:
            return False
        self._steps += 1
        return self._steps >= self._max_steps

    def uniform(self, lo, hi):
        return self._rng.uniform32(lo, hi)

    def normal(self, mean, stddev):
        # EnvBase builds a fresh std::normal_distribution per draw: no cached second sample.
        self._rng.reset_normal()
        return self._rng.normal32(mean, stddev)


class Transition:
    """Outcome of taking action a in state s, for planning methods."""

    __slots__ = ("prob", "next_state", "reward", "terminal")

    def __init__(self, prob, next_state, reward, terminal):
        self.prob = prob
        self.next_state = next_state
        self.reward = reward
        self.terminal = terminal


class TabularEnv(EnvBase):
    """A finite MDP: states are indices, observations are the index as a float."""

    def num_states(self):
        raise NotImplementedError

    def state(self):
        raise NotImplementedError

    def transitions(self, s, a):
        raise NotImplementedError

    def obs_size(self):
        return 1

    def state_obs(self):
        return [float(self.state())]
