// CMD: just run-nn bandits
// Multi-armed bandits.
//
// Port of tflow's bandits. Bandits are one-step environments: no state (or a
// context that does not depend on the action), a reward per pull, and the
// episode ends. This program shows the three variants in
// common/rl/bandit.hpp (copied from tflow) and the notion of regret that
// bandit algorithms (6.2) try to minimise. The vsim video of 15 random and 15
// best-arm pulls is omitted (its pulls are still drawn, so the RNG stream and
// every later number match tflow); the port adds checks on the sample means,
// the regret and the contextual oracle.

#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

#include "common/checks.hpp"
#include "common/rl/bandit.hpp"
#include "common/rl/utils.hpp"

int main() {
  try {
    std::cout << "== 10-armed Gaussian testbed (Sutton & Barto 2.3)\n";
    rl::MultiArmedBandit testbed = rl::MultiArmedBandit::Testbed(10, /*seed=*/42);
    testbed.Seed(0);
    std::cout << "  " << testbed.name() << "  action_space=" << testbed.action_space().ToString()
              << "  obs_size=" << testbed.obs_size() << "\n";
    std::printf("  %-4s %10s %12s\n", "arm", "true mean", "sample mean");
    for (int a = 0; a < testbed.num_arms(); ++a) {
      double sum = 0;
      for (int i = 0; i < 1000; ++i) {
        testbed.reset();
        sum += testbed.step(a).reward;
      }
      std::printf("  %-4d %10.3f %12.3f%s\n", a, testbed.mean(a), sum / 1000,
                  a == testbed.best_arm() ? "  <- best" : "");
      // Unit-variance rewards: the mean of 1000 pulls is within 4 standard errors.
      check::Near(sum / 1000, testbed.mean(a), 4 / std::sqrt(1000.0), "sample mean of arm " + std::to_string(a));
    }

    std::cout << "\n== Regret: how much a policy loses against always pulling the best arm\n";
    const int T = 2000;
    std::mt19937_64 rng(0);
    double random_reward = 0, best_reward = 0;
    for (int t = 0; t < T; ++t) {
      testbed.reset();
      random_reward += testbed.step(testbed.action_space().Sample(rng)).reward;
      testbed.reset();
      best_reward += testbed.step(testbed.best_arm()).reward;
    }
    float mean_of_means = 0;
    for (int a = 0; a < testbed.num_arms(); ++a) mean_of_means += testbed.mean(a);
    mean_of_means /= testbed.num_arms();
    const double expected_regret = T * (testbed.best_mean() - mean_of_means);
    std::printf("  over %d pulls: random policy %.1f, best arm %.1f, regret %.1f (expected %.1f)\n",
                T, random_reward, best_reward, best_reward - random_reward, expected_regret);
    // Standard deviation of the regret: about sqrt(2T) from the reward noise plus the spread of the means.
    check::Near(best_reward - random_reward, expected_regret, 0.1 * expected_regret + 4 * std::sqrt(4.0 * T),
                "regret of the random policy");

    // tflow films 15 random pulls and 15 pulls of the best arm here; the film
    // is omitted but the pulls are made so the generators stay in step.
    for (int t = 0; t < 30; ++t) {
      const int arm = t < 15 ? static_cast<int>(rng() % testbed.num_arms()) : testbed.best_arm();
      testbed.reset();
      testbed.step(arm);
    }

    std::cout << "\n== Bernoulli arms\n";
    rl::MultiArmedBandit coins = rl::MultiArmedBandit::Bernoulli({0.1f, 0.5f, 0.7f, 0.65f});
    coins.Seed(1);
    for (int a = 0; a < coins.num_arms(); ++a) {
      int wins = 0;
      for (int i = 0; i < 1000; ++i) {
        coins.reset();
        wins += coins.step(a).reward > 0;
      }
      std::printf("  arm %d: p=%.2f  observed %d/1000\n", a, coins.mean(a), wins);
      check::Near(wins / 1000.0, coins.mean(a), 4 * std::sqrt(0.25 / 1000), "win rate of coin " + std::to_string(a));
    }

    std::cout << "\n== Contextual bandit: the best arm depends on the context\n";
    rl::ContextualBandit ctx(/*context_dim=*/4, /*num_arms=*/3, /*seed=*/3);
    ctx.Seed(0);
    for (int round = 0; round < 4; ++round) {
      rl::Obs x = ctx.reset();
      std::printf("  context %s  P(reward|arm) =", rl::FormatObs(x).c_str());
      for (int a = 0; a < ctx.num_arms(); ++a) std::printf(" %.2f", ctx.ExpectedReward(a));
      rl::StepResult r = ctx.step(ctx.BestArm());
      std::printf("  best arm %d -> reward %.0f\n", ctx.BestArm(), r.reward);
    }
    // Average reward of three fixed policies over many rounds.
    const int R = 5000;
    std::vector<double> fixed(ctx.num_arms(), 0.0);
    double oracle = 0, random = 0;
    for (int t = 0; t < R; ++t) {
      ctx.reset();
      for (int a = 0; a < ctx.num_arms(); ++a) fixed[a] += ctx.ExpectedReward(a);
      oracle += ctx.ExpectedReward(ctx.BestArm());
      random += ctx.ExpectedReward(static_cast<int>(rng() % ctx.num_arms()));
    }
    std::printf("  mean reward over %d rounds: random %.3f, always-arm", R, random / R);
    for (int a = 0; a < ctx.num_arms(); ++a) std::printf(" %d=%.3f", a, fixed[a] / R);
    std::printf(", oracle %.3f\n", oracle / R);
    for (int a = 0; a < ctx.num_arms(); ++a)
      check::Require(oracle > fixed[a], "the context-aware oracle should beat always pulling arm " + std::to_string(a));
    check::Require(oracle > random, "the oracle should beat random pulls");
    std::cout << "\nchecks passed (vsim video omitted)\n";
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
