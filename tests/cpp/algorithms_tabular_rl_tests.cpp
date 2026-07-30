#include <gtest/gtest.h>

#include <datamunge/algorithms/tabular_rl.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

using datamunge::algorithms::q_learning;
using datamunge::algorithms::RLParameters;
using datamunge::algorithms::sarsa;
using datamunge::algorithms::StepResult;
using datamunge::algorithms::td_prediction;
using datamunge::algorithms::TabularEnvironment;

namespace {

// A deterministic corridor of `n` states: state 0 is the start, state n-1 is the
// absorbing goal. Action 1 ("right") increments the state; action 0 ("left")
// decrements it (clamped at 0). EVERY step costs -1, so the agent is pushed to
// reach the goal as fast as possible: the optimal policy is "always go right".
// Under the always-right policy the goal is d = n-1-s steps away, so the return is
//   V(s) = -(1 + gamma + ... + gamma^{d-1}) = -(1 - gamma^d) / (1 - gamma).
// (Starting Q/V at 0 is "optimistic" against negative rewards and drives exploration.)
TabularEnvironment corridor(std::size_t n) {
    TabularEnvironment env;
    env.num_states  = n;
    env.num_actions = 2;
    env.reset       = [] { return std::size_t{0}; };
    env.step        = [n](std::size_t s, std::size_t a) -> StepResult {
        std::size_t next = s;
        if (a == 1)
            next = s + 1;
        else if (s > 0)
            next = s - 1;
        const bool terminal = (next == n - 1);
        return {next, -1.0, terminal};
    };
    return env;
}

// Analytic state-value of the always-right policy: -(1 - gamma^d)/(1 - gamma), d = n-1-s.
double corridor_value(std::size_t n, std::size_t s, double gamma) {
    const double d = static_cast<double>(n - 1 - s);
    return -(1.0 - std::pow(gamma, d)) / (1.0 - gamma);
}

}  // namespace

TEST(TabularRL, QLearningFindsOptimalCorridorPolicy) {
    const auto   env = corridor(6);
    RLParameters p;
    p.learning_rate = 0.2;
    p.discount      = 0.9;
    p.epsilon       = 0.2;
    p.episodes      = 3000;
    p.seed          = 1;

    const auto r = q_learning(env, p);
    // Every non-terminal state's greedy action must be "right" (=1).
    for (std::size_t s = 0; s + 1 < env.num_states; ++s) EXPECT_EQ(r.greedy_policy[s], 1u) << "state " << s;
    // The optimal action-value at the start equals the always-right return from state 0.
    EXPECT_NEAR(r.q[0][1], corridor_value(6, 0, 0.9), 0.1);
}

TEST(TabularRL, SarsaFindsOptimalCorridorPolicy) {
    const auto   env = corridor(6);
    RLParameters p;
    p.learning_rate = 0.2;
    p.discount      = 0.9;
    p.epsilon       = 0.1;
    p.episodes      = 5000;
    p.seed          = 2;

    const auto r = sarsa(env, p);
    for (std::size_t s = 0; s + 1 < env.num_states; ++s) EXPECT_EQ(r.greedy_policy[s], 1u) << "state " << s;
}

TEST(TabularRL, TDPredictionMatchesAnalyticValues) {
    const std::size_t n   = 6;
    const auto        env = corridor(n);
    RLParameters      p;
    p.learning_rate = 0.05;
    p.discount      = 0.9;
    p.episodes      = 20000;
    p.seed          = 3;

    const std::vector<std::size_t> always_right(n, 1);
    const auto                     r = td_prediction(env, always_right, p);

    // V(s) = -(1 - gamma^d)/(1 - gamma), d = n-1-s. Terminal-state value stays 0.
    for (std::size_t s = 0; s + 1 < n; ++s)
        EXPECT_NEAR(r.value[s], corridor_value(n, s, 0.9), 0.05) << "state " << s;
}

TEST(TabularRL, QLearningReturnsImproveOverTraining) {
    const auto   env = corridor(6);
    RLParameters p;
    p.learning_rate = 0.2;
    p.discount      = 0.9;
    p.epsilon       = 0.1;
    p.episodes      = 2000;
    p.seed          = 5;

    const auto r = q_learning(env, p);
    ASSERT_EQ(r.episode_returns.size(), p.episodes);
    // The optimal path from state 0 takes 5 steps at -1 each, an undiscounted return of -5.
    // Averaged over the last 100 episodes (with only 10% exploration) it should be near that.
    double late = 0.0;
    for (std::size_t i = r.episode_returns.size() - 100; i < r.episode_returns.size(); ++i) late += r.episode_returns[i];
    EXPECT_GT(late / 100.0, -8.0);
}
