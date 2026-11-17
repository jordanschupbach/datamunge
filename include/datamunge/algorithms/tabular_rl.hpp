#pragma once
#include <datamunge/algorithms/td_prediction.hpp>

/// \file tabular_rl.hpp
/// \brief Tabular, model-free reinforcement learning: temporal-difference (TD(0))
///        prediction and the Q-learning and SARSA control algorithms.
///
/// An agent interacts with a finite Markov decision process (MDP): at each step it
/// observes a state \f$s\in\{0,\dots,|S|-1\}\f$, chooses an action
/// \f$a\in\{0,\dots,|A|-1\}\f$, and the environment returns a reward \f$r\f$ and a next
/// state \f$s'\f$, terminating episodes when a goal or absorbing state is reached. The
/// goal is a policy maximizing the expected discounted return
/// \f$G_t=\sum_{k\ge 0}\gamma^k r_{t+k+1}\f$ with discount \f$\gamma\in[0,1]\f$.
///
/// All three methods are *model-free* (they never see the transition/reward tables,
/// only sampled transitions) and *tabular* (one entry per state or state-action pair),
/// and all rest on the temporal-difference idea: bootstrap a value estimate toward a
/// one-step sample \f$r+\gamma(\cdot)\f$ rather than waiting for a full Monte-Carlo return.
///
///   - TD(0) *prediction* evaluates a fixed policy \f$\pi\f$:
///     \f$V(s)\leftarrow V(s)+\alpha\,[\,r+\gamma V(s')-V(s)\,]\f$.
///   - SARSA is *on-policy* control -- it bootstraps from the action the (exploring)
///     policy actually takes next:
///     \f$Q(s,a)\leftarrow Q(s,a)+\alpha\,[\,r+\gamma Q(s',a')-Q(s,a)\,]\f$.
///   - Q-learning is *off-policy* control -- it bootstraps from the greedy action
///     regardless of what is taken next:
///     \f$Q(s,a)\leftarrow Q(s,a)+\alpha\,[\,r+\gamma\max_{a'}Q(s',a')-Q(s,a)\,]\f$.
///
/// Behavior during learning is \f$\varepsilon\f$-greedy: with probability
/// \f$\varepsilon\f$ a uniformly random action is taken, otherwise the current greedy
/// action. The environment is supplied as callbacks so any simulator can be plugged in.

#include <datamunge/algorithms/q_learning.hpp>
#include <datamunge/algorithms/sarsa.hpp>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <vector>

namespace datamunge::algorithms {

namespace detail {

/// Greedy action for state \p s, breaking ties toward the lowest index.
inline std::size_t argmax_action(const std::vector<double>& row) {
    std::size_t best     = 0;
    double      best_val = -std::numeric_limits<double>::infinity();
    for (std::size_t a = 0; a < row.size(); ++a)
        if (row[a] > best_val) {
            best_val = row[a];
            best     = a;
        }
    return best;
}

/// \f$\varepsilon\f$-greedy action selection.
template <class Rng>
std::size_t epsilon_greedy(const std::vector<double>& row, double epsilon, Rng& rng) {
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    if (unit(rng) < epsilon) {
        std::uniform_int_distribution<std::size_t> pick(0, row.size() - 1);
        return pick(rng);
    }
    return argmax_action(row);
}

}  // namespace detail

}  // namespace datamunge::algorithms
