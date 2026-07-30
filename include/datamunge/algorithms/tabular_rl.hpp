#pragma once

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

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <vector>

namespace datamunge::algorithms {

/// One environment transition returned by an MDP step.
struct StepResult {
    std::size_t next_state;      ///< Resulting state \f$s'\f$.
    double      reward;          ///< Immediate reward \f$r\f$.
    bool        terminal;        ///< True if \f$s'\f$ ends the episode.
};

/// A tabular episodic environment described entirely by callbacks.
struct TabularEnvironment {
    std::size_t                                            num_states  = 0;
    std::size_t                                            num_actions = 0;
    std::function<std::size_t()>                           reset;  ///< Sample and return a start state.
    std::function<StepResult(std::size_t, std::size_t)>    step;   ///< (state, action) -> transition.
};

/// Hyper-parameters shared by the control algorithms.
struct RLParameters {
    double        learning_rate    = 0.1;   ///< Step size \f$\alpha\f$.
    double        discount         = 0.99;  ///< Discount factor \f$\gamma\f$.
    double        epsilon          = 0.1;   ///< Exploration rate \f$\varepsilon\f$ (constant).
    std::size_t   episodes         = 500;   ///< Number of training episodes.
    std::size_t   max_steps        = 1000;  ///< Per-episode step cap (guards non-terminating runs).
    std::uint64_t seed             = 0;     ///< RNG seed for reproducibility.
};

/// Learned action-value function and derived greedy policy.
struct QLearningResult {
    std::vector<std::vector<double>> q;                 ///< \f$Q(s,a)\f$, \f$|S|\times|A|\f$.
    std::vector<std::size_t>         greedy_policy;     ///< \f$\arg\max_a Q(s,a)\f$ per state.
    std::vector<double>              episode_returns;   ///< Undiscounted return collected each episode.
};

/// Learned state-value function from TD(0) prediction.
struct TDPredictionResult {
    std::vector<double> value;              ///< \f$V(s)\f$, length \f$|S|\f$.
    std::vector<double> episode_returns;    ///< Undiscounted return collected each episode.
};

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

/// \brief Q-learning: off-policy temporal-difference control (Watkins 1989).
///
/// Uses \f$\varepsilon\f$-greedy behavior but updates toward the *greedy* bootstrap
/// \f$r+\gamma\max_{a'}Q(s',a')\f$, so it learns the optimal action-value function
/// \f$Q^\star\f$ independently of how much it explores. For a terminal transition the
/// bootstrap term is dropped (the future value of an absorbing state is zero).
///
/// \return the learned \f$Q\f$ table, the greedy policy it induces, and per-episode returns.
inline QLearningResult q_learning(const TabularEnvironment& env, const RLParameters& params) {
    QLearningResult result;
    result.q.assign(env.num_states, std::vector<double>(env.num_actions, 0.0));
    result.episode_returns.reserve(params.episodes);
    std::mt19937_64 rng(params.seed);

    for (std::size_t ep = 0; ep < params.episodes; ++ep) {
        std::size_t state          = env.reset();
        double      episode_return = 0.0;
        for (std::size_t t = 0; t < params.max_steps; ++t) {
            const std::size_t action = detail::epsilon_greedy(result.q[state], params.epsilon, rng);
            const StepResult  sr     = env.step(state, action);
            episode_return += sr.reward;

            const double best_next =
                sr.terminal ? 0.0 : result.q[sr.next_state][detail::argmax_action(result.q[sr.next_state])];
            const double target = sr.reward + params.discount * best_next;
            result.q[state][action] += params.learning_rate * (target - result.q[state][action]);

            state = sr.next_state;
            if (sr.terminal) break;
        }
        result.episode_returns.push_back(episode_return);
    }

    result.greedy_policy.resize(env.num_states);
    for (std::size_t s = 0; s < env.num_states; ++s) result.greedy_policy[s] = detail::argmax_action(result.q[s]);
    return result;
}

/// \brief SARSA: on-policy temporal-difference control (Rummery & Niranjan 1994).
///
/// The name is the transition tuple \f$(s,a,r,s',a')\f$ used in its update. It picks the
/// next action \f$a'\f$ with the same \f$\varepsilon\f$-greedy policy it is evaluating and
/// bootstraps from \f$Q(s',a')\f$, so the values it learns account for the cost of
/// exploration -- on stochastic or cliff-like problems SARSA tends to learn a safer policy
/// than Q-learning. For terminal transitions the bootstrap term is dropped.
inline QLearningResult sarsa(const TabularEnvironment& env, const RLParameters& params) {
    QLearningResult result;
    result.q.assign(env.num_states, std::vector<double>(env.num_actions, 0.0));
    result.episode_returns.reserve(params.episodes);
    std::mt19937_64 rng(params.seed);

    for (std::size_t ep = 0; ep < params.episodes; ++ep) {
        std::size_t state          = env.reset();
        std::size_t action         = detail::epsilon_greedy(result.q[state], params.epsilon, rng);
        double      episode_return = 0.0;
        for (std::size_t t = 0; t < params.max_steps; ++t) {
            const StepResult sr = env.step(state, action);
            episode_return += sr.reward;

            std::size_t  next_action = 0;
            double       next_value  = 0.0;
            if (!sr.terminal) {
                next_action = detail::epsilon_greedy(result.q[sr.next_state], params.epsilon, rng);
                next_value  = result.q[sr.next_state][next_action];
            }
            const double target = sr.reward + params.discount * next_value;
            result.q[state][action] += params.learning_rate * (target - result.q[state][action]);

            state  = sr.next_state;
            action = next_action;
            if (sr.terminal) break;
        }
        result.episode_returns.push_back(episode_return);
    }

    result.greedy_policy.resize(env.num_states);
    for (std::size_t s = 0; s < env.num_states; ++s) result.greedy_policy[s] = detail::argmax_action(result.q[s]);
    return result;
}

/// \brief TD(0) prediction: estimate \f$V^\pi\f$ for a fixed policy (Sutton 1988).
///
/// Unlike the control methods this does not improve the policy; it *evaluates* the given
/// \p policy by the update \f$V(s)\leftarrow V(s)+\alpha[r+\gamma V(s')-V(s)]\f$ along
/// sampled trajectories, bootstrapping each estimate from the successor's current estimate.
/// It is the prediction primitive underneath SARSA and Q-learning.
///
/// \param env     Environment to sample transitions from.
/// \param policy  Deterministic action per state to be evaluated.
/// \param params  Hyper-parameters (\p epsilon is ignored; the policy is followed exactly).
inline TDPredictionResult td_prediction(const TabularEnvironment&       env,
                                         const std::vector<std::size_t>& policy,
                                         const RLParameters&             params) {
    TDPredictionResult result;
    result.value.assign(env.num_states, 0.0);
    result.episode_returns.reserve(params.episodes);
    std::mt19937_64 rng(params.seed);  // reserved for stochastic start states via env.reset()
    (void)rng;

    for (std::size_t ep = 0; ep < params.episodes; ++ep) {
        std::size_t state          = env.reset();
        double      episode_return = 0.0;
        for (std::size_t t = 0; t < params.max_steps; ++t) {
            const std::size_t action = policy[state];
            const StepResult  sr     = env.step(state, action);
            episode_return += sr.reward;

            const double next_value = sr.terminal ? 0.0 : result.value[sr.next_state];
            const double target     = sr.reward + params.discount * next_value;
            result.value[state] += params.learning_rate * (target - result.value[state]);

            state = sr.next_state;
            if (sr.terminal) break;
        }
        result.episode_returns.push_back(episode_return);
    }
    return result;
}

}  // namespace datamunge::algorithms
