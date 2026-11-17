#pragma once
/// @file q_learning.hpp
/// @brief Checked native Q-learning with actual behavior draws, temporal-difference updates and episode outcomes.
#include <datamunge/algorithms/tabular_rl_types.hpp>
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {
/// @brief Table, episode, actual-transition and retained-entry policies.
/// Zero independently disables each cap; validation and callback work are not timed by these limits.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::QLearningOptions{}.max_transitions); // 1000000
/// @endcode
struct QLearningOptions {
  std::size_t max_table_entries{
      1000000};                      ///< Admitted num_states*num_actions entries before allocation.
  std::size_t max_episodes{1000000}; ///< Admitted requested episodes before callback execution.
  std::size_t max_transitions{
      1000000}; ///< Actual transitions admitted before each behavior draw/step callback.
  std::size_t max_trace_entries{
      1000000}; ///< Actual initial Q scalars and producing transition records; excludes ordinary outcome storage.
};
/// @brief One actual behavior selection, callback transition and Q update.
/// Terminal transitions use bootstrap=0 and bootstrap_action=0 sentinel. Exploration draw and flag come from the actual native RNG calls; no reseeding/replay creates them.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::q_learning_trace(dm::algorithms::tabular_corridor(2),dm::algorithms::RLParameters{.episodes=1,.max_steps=1}).steps.size()); // 1
/// @endcode
struct QLearningStep {
  std::size_t episode;          ///< Actual zero-based episode.
  std::size_t step;             ///< Actual zero-based transition within that episode.
  std::size_t state;            ///< Actual current state passed to the callback.
  std::size_t action;           ///< Actual chosen action.
  std::size_t next_state;       ///< Actual validated callback state.
  double      reward;           ///< Actual finite immediate reward.
  bool        terminal;         ///< Actual callback termination flag.
  double      exploration_draw; ///< Actual uniform[0,1) branch-selection draw.
  bool        explored;         ///< True when this draw selected uniform random action sampling.
  std::size_t
         bootstrap_action; ///< Actual greedy next action, zero sentinel for terminal transitions.
  double bootstrap;        ///< Actual greedy next-state Q value, zero for terminal transitions.
  double before;           ///< Actual Q(state,action) immediately before update.
  double target;           ///< Actual reward+discount*bootstrap.
  double after;            ///< Actual Q(state,action) immediately after the additive update.
  double episode_return;   ///< Actual accumulated undiscounted return after this transition.
};
/// @brief Ordinary output and actual episode/callback work.
/// A truncated episode stops because max_steps was reached without a terminal transition; zero-step episodes are truncated. No automatic convergence or optimality certification is implied.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::q_learning_run(dm::algorithms::tabular_corridor(2),dm::algorithms::RLParameters{.episodes=0}).result.q.size()); // 2
/// @endcode
struct QLearningRun {
  QLearningResult          result;         ///< Actual final table/policy and training returns.
  std::vector<std::size_t> episode_steps;  ///< Actual transition count per episode.
  std::vector<bool>        terminated;     ///< Actual terminal transition occurred in each episode.
  std::size_t              reset_calls{0}; ///< Actual reset callback invocations.
  std::size_t              transitions{0}; ///< Actual validated step callback transitions.
  std::size_t exploratory_actions{0};      ///< Actual behavior draws selecting uniform exploration.
  std::size_t greedy_actions{0};           ///< Actual behavior draws selecting greedy action.
};
/// @brief Owned zero initialization and actual producing scalar Q updates.
/// Initial Q is captured from the allocated native table and updates are recorded during that same training execution.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::q_learning_trace(dm::algorithms::tabular_corridor(2),dm::algorithms::RLParameters{.episodes=0}).initial_q[0][0]); // 0
/// @endcode
struct QLearningTrace {
  QLearningRun                     run; ///< Actual final output and selected callback/action work.
  std::vector<std::vector<double>> initial_q; ///< Actual initialized all-zero table.
  std::vector<QLearningStep>
      steps; ///< Actual behavior/callback/update records, no training replay.
};
/// @brief Train by the single checked native off-policy temporal-difference producer.
/// Preserves the local mt19937_64 behavior stream, epsilon-greedy draw order, lowest-index ties and additive Q += alpha*(reward+gamma*max_next-Q) expression. Terminal bootstrap is zero. Validates complete geometry/parameters/callback presence before allocation/callbacks and validates each actual callback result before indexing. Zero episodes produce a zero table; zero max_steps still resets each episode and records return0. Constant learning/exploration rates and finite training do not establish asymptotic convergence or policy optimality. Work is O(S*A+T*A+episodes) plus arbitrary callback work for T actual transitions; ordinary storage O(S*A+episodes). Callback side effects/exceptions propagate and are not rolled back; outputs retain no callback references. Calls are independent only when callbacks/captured state are independent.
/// @param env Borrowed positive state/action geometry and synchronous reset/step callbacks.
/// @param params Finite valid learning/discount/exploration parameters and requested counts.
/// @return Owned actual final table/policy/training returns.
/// @throws std::invalid_argument If geometry/callbacks/parameters are invalid or a callback returns a nonfinite reward or out-of-range state, including terminal states.
/// @throws std::length_error If table/episode/transition/history policy or representable storage is exceeded.
/// @throws std::overflow_error If a producing return/target/update is nonfinite or a selected count cannot fit.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::q_learning(dm::algorithms::tabular_corridor(2),dm::algorithms::RLParameters{.episodes=0}).q.size()); // 2
/// @endcode
/// @par References
/// - C. J. C. H. Watkins and P. Dayan (1992), Q-learning, Machine Learning8,279–292, doi:10.1007/BF00992698.
QLearningResult q_learning(const TabularEnvironment& env, const RLParameters& params);
/// @brief Train by the single checked native off-policy temporal-difference producer.
/// Preserves the local mt19937_64 behavior stream, epsilon-greedy draw order, lowest-index ties and additive Q += alpha*(reward+gamma*max_next-Q) expression. Terminal bootstrap is zero. Validates complete geometry/parameters/callback presence before allocation/callbacks and validates each actual callback result before indexing. Zero episodes produce a zero table; zero max_steps still resets each episode and records return0. Constant learning/exploration rates and finite training do not establish asymptotic convergence or policy optimality. Work is O(S*A+T*A+episodes) plus arbitrary callback work for T actual transitions; ordinary storage O(S*A+episodes). Callback side effects/exceptions propagate and are not rolled back; outputs retain no callback references. Calls are independent only when callbacks/captured state are independent.
/// @param env Borrowed positive state/action geometry and synchronous reset/step callbacks.
/// @param params Finite valid learning/discount/exploration parameters and requested counts.
/// @param options Table/episode/actual-transition/actual-history policies.
/// @return Owned actual final table/policy/training returns and episode/work outcomes.
/// @throws std::invalid_argument If geometry/callbacks/parameters are invalid or a callback returns a nonfinite reward or out-of-range state, including terminal states.
/// @throws std::length_error If table/episode/transition/history policy or representable storage is exceeded.
/// @throws std::overflow_error If a producing return/target/update is nonfinite or a selected count cannot fit.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::q_learning_run(dm::algorithms::tabular_corridor(2),dm::algorithms::RLParameters{.episodes=0}).result.q.size()); // 2
/// @endcode
/// @par References
/// - C. J. C. H. Watkins and P. Dayan (1992), Q-learning, Machine Learning8,279–292, doi:10.1007/BF00992698.
QLearningRun q_learning_run(
    const TabularEnvironment& env,
    const RLParameters&       params,
    QLearningOptions          options = {});
/// @brief Train by the single checked native off-policy temporal-difference producer.
/// Preserves the local mt19937_64 behavior stream, epsilon-greedy draw order, lowest-index ties and additive Q += alpha*(reward+gamma*max_next-Q) expression. Terminal bootstrap is zero. Validates complete geometry/parameters/callback presence before allocation/callbacks and validates each actual callback result before indexing. Zero episodes produce a zero table; zero max_steps still resets each episode and records return0. Constant learning/exploration rates and finite training do not establish asymptotic convergence or policy optimality. Work is O(S*A+T*A+episodes) plus arbitrary callback work for T actual transitions; ordinary storage O(S*A+episodes). Callback side effects/exceptions propagate and are not rolled back; outputs retain no callback references. Calls are independent only when callbacks/captured state are independent.
/// @param env Borrowed positive state/action geometry and synchronous reset/step callbacks.
/// @param params Finite valid learning/discount/exploration parameters and requested counts.
/// @param options Table/episode/actual-transition/actual-history policies.
/// @return Owned actual final table/policy/training returns, episode/work outcomes and producing update history.
/// @throws std::invalid_argument If geometry/callbacks/parameters are invalid or a callback returns a nonfinite reward or out-of-range state, including terminal states.
/// @throws std::length_error If table/episode/transition/history policy or representable storage is exceeded.
/// @throws std::overflow_error If a producing return/target/update is nonfinite or a selected count cannot fit.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::q_learning_trace(dm::algorithms::tabular_corridor(2),dm::algorithms::RLParameters{.episodes=0}).run.result.q.size()); // 2
/// @endcode
/// @par References
/// - C. J. C. H. Watkins and P. Dayan (1992), Q-learning, Machine Learning8,279–292, doi:10.1007/BF00992698.
QLearningTrace q_learning_trace(
    const TabularEnvironment& env,
    const RLParameters&       params,
    QLearningOptions          options = {});
/// @brief Construct the reusable deterministic two-action corridor environment.
/// States0..states-1, reset0. Action0 moves left/clamps0; action1 moves right/clamps the goal. Every step has the supplied finite reward and the last state is terminal. Lambdas capture scalar values only; no mutable environment state is shared.
/// @param states At least2 corridor positions, default6.
/// @param reward Finite per-transition reward, default-1.
/// @return Owned callbacks and explicit tabular geometry.
/// @throws std::invalid_argument If states<2 or reward is nonfinite.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::tabular_corridor(2).step(0,1).terminal); // true
/// @endcode
/// @see q_learning_trace()
TabularEnvironment tabular_corridor(std::size_t states = 6, double reward = -1);
/// @brief Construct the reusable original square four-action grid environment.
/// Reset is the top-left state0; bottom-right is terminal. Actions0,1,2,3 mean up,right,down,left; blocked moves stay in place. Row-major state ids are preserved. Each transition receives the finite supplied reward. Scalar-capture callbacks have no mutable shared state.
/// @param side At least2 grid side length, default4; side*side is checked.
/// @param reward Finite immediate reward, default-1.
/// @return Owned callbacks and complete state/action geometry.
/// @throws std::invalid_argument If side<2 or reward is nonfinite.
/// @throws std::length_error If side*side overflows size_t.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::tabular_grid(4).step(0,1).next_state); // 1
/// @endcode
/// @see q_learning_trace(), tabular_corridor()
TabularEnvironment tabular_grid(std::size_t side = 4, double reward = -1);
/// @brief Owned actual fixed-policy callback trajectory.
/// States contains the actual reset state followed by each actual next state; the other vectors have one entry per transition.
/// @par Example
/// @code
/// const auto r=dm::algorithms::tabular_policy_rollout(dm::algorithms::tabular_corridor(2),{1,1});std::print("{}\n",r.total_reward); // -1
/// @endcode
struct TabularPolicyRollout {
  std::vector<std::size_t> states;          ///< Actual reset and visited states.
  std::vector<std::size_t> actions;         ///< Actual caller-policy actions.
  std::vector<double>      rewards;         ///< Actual callback rewards.
  double                   total_reward{0}; ///< Actual finite undiscounted sum.
  bool terminal{false}; ///< Whether an actual transition ended at a terminal state.
};
/// @brief Execute an already chosen deterministic tabular policy through actual callbacks.
/// Validates complete geometry, callback presence and every policy action before reset. Validates every actual returned state/reward before retaining it, including terminal states. Zero max_steps still resets once. External callback side effects/exceptions propagate without rollback. O(S+T) native work and O(T) owned trajectory storage plus arbitrary callback work; this finite rollout is not a proof of general policy optimality.
/// @param env Borrowed synchronous finite environment.
/// @param policy Exactly one in-range action per state.
/// @param max_steps Actual transition cap, default1000.
/// @param max_entries Retained actual states/actions/rewards cap, defaultone million; zero disables.
/// @return Owned actual trajectory and terminal/return result.
/// @throws std::invalid_argument If geometry/callbacks/policy or returned state/reward are invalid.
/// @throws std::length_error If actual retained entries or representable storage are exceeded.
/// @throws std::overflow_error If actual total reward becomes nonfinite.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::tabular_policy_rollout(dm::algorithms::tabular_corridor(3),{1,1,1}).actions.size()); // 2
/// @endcode
/// @see q_learning_run()
TabularPolicyRollout tabular_policy_rollout(
    const TabularEnvironment&       env,
    const std::vector<std::size_t>& policy,
    std::size_t                     max_steps   = 1000,
    std::size_t                     max_entries = 1000000);
/// @brief Compute finite trailing episode-return means with partial initial windows.
/// Element i averages returns[max(0,i+1-window)..i]. Uses a long-double sliding sum, so rounding can differ from repeated double summation; this is descriptive smoothing, not an uncertainty interval or convergence test. O(n) work and output storage, no shared state.
/// @param returns Borrowed complete finite observed returns, possibly empty.
/// @param window Positive trailing window length, default20; values exceeding the sample length are allowed.
/// @return Owned one mean per actual episode return.
/// @throws std::invalid_argument If window is zero or any return is nonfinite.
/// @throws std::overflow_error If sliding sum or a returned mean becomes nonfinite.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::trailing_return_mean({1,3,8},2)); // [1, 2, 5.5]
/// @endcode
std::vector<double> trailing_return_mean(
    const std::vector<double>& returns,
    std::size_t                window = 20);
} // namespace datamunge::algorithms
