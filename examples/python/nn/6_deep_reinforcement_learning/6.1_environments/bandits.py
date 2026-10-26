# Native Python NN example. Multi-armed bandits.
"""Multi-armed bandits.

CMD: just python-example bandits

Port of tflow's bandits. Bandits are one-step environments: no state (or a
context that does not depend on the action), a reward per pull, and the
episode ends. This program shows the three variants in
common/rl/bandit.hpp (copied from tflow) and the notion of regret that
bandit algorithms (6.2) try to minimise. The vsim video of 15 random and 15
best-arm pulls is omitted (its pulls are still drawn, so the RNG stream and
every later number match tflow); the port adds checks on the sample means,
the regret and the contextual oracle.
"""
import math
import os

from pydatamunge import nn
from pydatamunge.nn_workflows import helper, options

checks = helper("checks")
rl_env = helper("rl_env")
rl_utils = helper("rl_utils")
rl_bandit = helper("rl_bandit")
f32 = rl_env.f32


def main(arguments=None):
    options(arguments, dict(full=os.environ.get("DATAMUNGE_EXAMPLES_FULL") == "1"))
    print("== 10-armed Gaussian testbed (Sutton & Barto 2.3)")
    testbed = rl_bandit.MultiArmedBandit.testbed(10, 42)
    testbed.seed(0)
    print("  %s  action_space=%s  obs_size=%d" % (testbed.name(), testbed.action_space(), testbed.obs_size()))
    print("  %-4s %10s %12s" % ("arm", "true mean", "sample mean"))
    for a in range(testbed.num_arms()):
        total = 0.0
        for _ in range(1000):
            testbed.reset()
            total += testbed.step(rl_env.Action(a)).reward
        print("  %-4d %10.3f %12.3f%s" % (a, testbed.mean(a), total / 1000,
                                         "  <- best" if a == testbed.best_arm() else ""))
        # Unit-variance rewards: the mean of 1000 pulls is within 4 standard errors.
        checks.near(total / 1000, testbed.mean(a), 4 / math.sqrt(1000.0), "sample mean of arm %d" % a)

    print("\n== Regret: how much a policy loses against always pulling the best arm")
    T = 2000
    rng = nn.Mt19937_64(0)
    random_reward = best_reward = 0.0
    for _ in range(T):
        testbed.reset()
        random_reward += testbed.step(testbed.action_space().sample(rng)).reward
        testbed.reset()
        best_reward += testbed.step(rl_env.Action(testbed.best_arm())).reward
    mean_of_means = 0.0
    for a in range(testbed.num_arms()):
        mean_of_means = f32(mean_of_means + testbed.mean(a))
    mean_of_means = f32(mean_of_means / testbed.num_arms())
    expected_regret = f32(T * f32(testbed.best_mean() - mean_of_means))
    print("  over %d pulls: random policy %.1f, best arm %.1f, regret %.1f (expected %.1f)"
          % (T, random_reward, best_reward, best_reward - random_reward, expected_regret))
    # Standard deviation of the regret: about sqrt(2T) from the reward noise plus the spread of the means.
    checks.near(best_reward - random_reward, expected_regret, 0.1 * expected_regret + 4 * math.sqrt(4.0 * T),
                "regret of the random policy")

    # tflow films 15 random pulls and 15 pulls of the best arm here; the film is omitted but
    # the pulls are made so the generators stay in step.
    for t in range(30):
        arm = int(rng.modulo(testbed.num_arms())) if t < 15 else testbed.best_arm()
        testbed.reset()
        testbed.step(rl_env.Action(arm))

    print("\n== Bernoulli arms")
    coins = rl_bandit.MultiArmedBandit.bernoulli([0.1, 0.5, 0.7, 0.65])
    coins.seed(1)
    for a in range(coins.num_arms()):
        wins = 0
        for _ in range(1000):
            coins.reset()
            wins += coins.step(rl_env.Action(a)).reward > 0
        print("  arm %d: p=%.2f  observed %d/1000" % (a, coins.mean(a), wins))
        checks.near(wins / 1000.0, coins.mean(a), 4 * math.sqrt(0.25 / 1000), "win rate of coin %d" % a)

    print("\n== Contextual bandit: the best arm depends on the context")
    ctx = rl_bandit.ContextualBandit(4, 3, 3)
    ctx.seed(0)
    for _ in range(4):
        x = ctx.reset()
        line = "  context %s  P(reward|arm) =" % rl_utils.format_obs(x)
        line += "".join(" %.2f" % ctx.expected_reward(a) for a in range(ctx.num_arms()))
        result = ctx.step(rl_env.Action(ctx.best_arm()))
        print(line + "  best arm %d -> reward %.0f" % (ctx.best_arm(), result.reward))
    # Average reward of three fixed policies over many rounds.
    R = 5000
    fixed = [0.0] * ctx.num_arms()
    oracle = random = 0.0
    for _ in range(R):
        ctx.reset()
        for a in range(ctx.num_arms()):
            fixed[a] += ctx.expected_reward(a)
        oracle += ctx.expected_reward(ctx.best_arm())
        random += ctx.expected_reward(int(rng.modulo(ctx.num_arms())))
    print("  mean reward over %d rounds: random %.3f, always-arm" % (R, random / R)
          + "".join(" %d=%.3f" % (a, fixed[a] / R) for a in range(ctx.num_arms()))
          + ", oracle %.3f" % (oracle / R))
    for a in range(ctx.num_arms()):
        checks.require(oracle > fixed[a], "the context-aware oracle should beat always pulling arm %d" % a)
    checks.require(oracle > random, "the oracle should beat random pulls")
    print("\nchecks passed (vsim video omitted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
