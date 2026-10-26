"""Episode statistics, argmax helpers and observation printing (examples/nn/common/rl/utils.hpp).

The episode runners (RunEpisodes / RunRandom) are experiment loops and live in each example.
"""
from pydatamunge.nn_workflows import helper

rl_env = helper("rl_env")
f32 = rl_env.f32


class EpisodeStats:
    def __init__(self):
        self.returns = []  # float32 episode returns
        self.lengths = []

    def mean_return(self):
        # std::accumulate over floats with a float accumulator, then / size.
        total = 0.0
        for value in self.returns:
            total = f32(total + value)
        return f32(total / len(self.returns)) if self.returns else 0.0

    def mean_length(self):
        total = 0.0
        for value in self.lengths:
            total = f32(total + value)
        return f32(total / len(self.lengths)) if self.lengths else 0.0

    def min_return(self):
        return min(self.returns)

    def max_return(self):
        return max(self.returns)


def print_stats(label, stats):
    print("  %-32s return mean %8.2f  min %8.2f  max %8.2f   length mean %6.1f"
          % (label, stats.mean_return(), stats.min_return(), stats.max_return(), stats.mean_length()))


def argmax(values):
    """Index of the first maximum."""
    best = 0
    for i in range(1, len(values)):
        if values[i] > values[best]:
            best = i
    return best


def argmax_random_tie(q, n, rng):
    """Argmax with ties broken uniformly at random (rng: nn.Mt19937_64, rng() % ties)."""
    best = max(q[:n])
    ties = [i for i in range(n) if q[i] == best]
    return ties[int(rng.modulo(len(ties)))]


def format_obs(obs, max_elems=8):
    text = " ".join("%.3f" % v for v in list(obs)[:max_elems])
    if len(obs) > max_elems:
        text += " ..."
    return "[" + text + "]"
