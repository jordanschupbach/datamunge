#pragma once
/// @file sarsa.hpp
/// @brief Checked on-policy SARSA with actual behavior choices and sampled transition updates.
#include <cstddef>
#include <datamunge/algorithms/q_learning.hpp>
#include <datamunge/algorithms/tabular_rl_types.hpp>
#include <vector>
/// @brief Canonical tabular on-policy control and owned actual histories.
namespace datamunge::algorithms {
/// @brief Table/episode, actual-transition and optional full-history policies; zero disables a cap.
/// @par Example
/// @code
/// const dm::algorithms::SarsaOptions value{};
/// std::print("{}\n",value.max_transitions); // 1000000
/// @endcode
struct SarsaOptions {
  std::size_t max_table_entries{1000000}; ///< Complete states*actions allocation.
  std::size_t max_episodes{1000000};      ///< Complete reset/return count.
  std::size_t max_transitions{1000000};   ///< Actual environment step callbacks.
  std::size_t max_trace_entries{
      1000000}; ///< Initial Q scalars plus actual selection/update records.
};
/// @brief One actual epsilon-greedy behavior selection, including an unused final next action.
/// @par Example
/// @code
/// const dm::algorithms::SarsaAction value{};
/// std::print("{}\n",value.explored); // false
/// @endcode
struct SarsaAction {
  std::size_t episode{};  ///< Actual zero-based episode.
  std::size_t state{};    ///< Actual validated state where this action was selected.
  std::size_t action{};   ///< Actual selected in-range action.
  double      draw{};     ///< Actual uniform[0,1) branch-selection value from local mt19937_64.
  bool        explored{}; ///< Whether the real draw selected uniform random action sampling.
};
/// @brief One actual sampled transition and on-policy Q update.
/// @par Example
/// @code
/// const dm::algorithms::SarsaStep value{};
/// std::print("{}\n",value.terminal); // false
/// @endcode
struct SarsaStep {
  std::size_t episode{};    ///< Actual episode.
  std::size_t time{};       ///< Actual zero-based transition within the episode.
  std::size_t state{};      ///< Actual source state.
  std::size_t action{};     ///< Actual carried behavior action.
  std::size_t next_state{}; ///< Actual validated callback state, including terminal states.
  double      reward{};     ///< Actual finite callback reward.
  bool        terminal{};   ///< Actual terminal callback flag.
  std::size_t
         next_action{}; ///< Actual next behavior action; zero sentinel for terminal transitions.
  double bootstrap{}; ///< Actual Q(next_state,next_action) before this update; zero when terminal.
  double before{};    ///< Actual Q(state,action) before the update.
  double target{};    ///< Actual reward+discount*bootstrap.
  double after{};     ///< Actual additive update result.
};
/// @brief Owned actual tabular training output and selected reset/behavior/update work.
/// @par Example
/// @code
/// const dm::algorithms::SarsaRun value{};
/// std::print("{}\n",value.transitions); // 0
/// @endcode
struct SarsaRun {
  QLearningResult
      result; ///< Actual final Q table, lowest-index greedy policy and undiscounted training returns.
  std::vector<std::size_t> episode_steps; ///< Actual completed transition count per episode.
  std::vector<char>
      terminated; ///< Actual per-episode terminal outcome, distinct from reaching its step cap.
  std::size_t reset_calls{}; ///< Actual environment reset callbacks.
  std::size_t transitions{}; ///< Actual validated environment step callbacks.
  std::size_t
      action_selections{}; ///< Actual initial and nonterminal-next epsilon-greedy selections.
  std::size_t exploratory_actions{}; ///< Actual selections whose branch drew a uniform action.
  std::size_t greedy_actions{}; ///< Actual selections choosing the current lowest-index argmax.
  std::vector<std::vector<double>> initial_q; ///< Actual zero table, only captured by trace.
  std::vector<SarsaAction>
      actions; ///< Actual optional behavior choices, including selected-but-unused next actions.
  std::vector<SarsaStep> steps; ///< Actual optional sampled transition/Q updates, never replayed.
};
/// @brief Execute checked native on-policy SARSA.
/// Validates positive state/action geometry, callback presence, finite learning_rate in(0,1], discount/epsilon in[0,1], complete table and episode bounds before callbacks. Preserves the original local mt19937_64 epsilon-greedy draw order and additive update Q+=alpha*(reward+discount*Qnext-Q). At each reset selects an initial behavior action even when max_steps=0. Every nonterminal transition selects its next behavior action before updating, including the final step-capped transition; that actual selected action may remain unused. Terminal bootstrap is exactly zero and selects no next action. Lowest-index greedy ties match the canonical Q-learning helper. Actual returned state ids and finite rewards are validated before indexing, even for terminal transitions. Zero episodes return zero Q; zero max_steps still resets and selects per episode, retaining return0. Work O(S*A+(T+episodes)*A) plus arbitrary callback work; O(S*A+episodes) intrinsic output and optional actual history. Captured units are initial table scalars plus one per actual behavior selection/update record, admitted before their work. Callback side effects/exceptions propagate without rollback; no callback references survive. Finite constant-rate training is not an optimality or convergence certificate.
/// @param env Borrowed positive finite tabular geometry and synchronous reset/step callbacks.
/// @param params Complete tabular control parameters and behavior seed.
/// @param options Input, actual-transition and capture bounds.
/// @return Owned actual final Q/returns/policy, selected work and optional producing action/update records.
/// @throws std::invalid_argument Invalid geometry/parameters/callbacks or returned state/reward.
/// @throws std::length_error Complete table/episodes, actual transitions/history or counter capacity exceeded.
/// @throws std::overflow_error Actual return, bootstrap target or additive update becomes nonfinite.
/// @par Example
/// @code
/// auto r=dm::algorithms::sarsa_run(dm::algorithms::tabular_corridor(2),dm::algorithms::RLParameters{.episodes=1,.max_steps=1});
/// std::print("{}\n",r.transitions); // 1
/// @endcode
/// @par References
/// - Sutton and Barto, Reinforcement Learning: An Introduction, second edition, chapter 6. https://incompleteideas.net/book/the-book-2nd.html
/// @see sarsa(), q_learning_run(), tabular_policy_rollout()
SarsaRun sarsa_run(
    const TabularEnvironment& env,
    const RLParameters&       params,
    const SarsaOptions&       options = {});
/// @brief Execute checked native on-policy SARSA with actual action/update histories.
/// Validates positive state/action geometry, callback presence, finite learning_rate in(0,1], discount/epsilon in[0,1], complete table and episode bounds before callbacks. Preserves the original local mt19937_64 epsilon-greedy draw order and additive update Q+=alpha*(reward+discount*Qnext-Q). At each reset selects an initial behavior action even when max_steps=0. Every nonterminal transition selects its next behavior action before updating, including the final step-capped transition; that actual selected action may remain unused. Terminal bootstrap is exactly zero and selects no next action. Lowest-index greedy ties match the canonical Q-learning helper. Actual returned state ids and finite rewards are validated before indexing, even for terminal transitions. Zero episodes return zero Q; zero max_steps still resets and selects per episode, retaining return0. Work O(S*A+(T+episodes)*A) plus arbitrary callback work; O(S*A+episodes) intrinsic output and optional actual history. Captured units are initial table scalars plus one per actual behavior selection/update record, admitted before their work. Callback side effects/exceptions propagate without rollback; no callback references survive. Finite constant-rate training is not an optimality or convergence certificate.
/// @param env Borrowed positive finite tabular geometry and synchronous reset/step callbacks.
/// @param params Complete tabular control parameters and behavior seed.
/// @param options Input, actual-transition and capture bounds.
/// @return Owned actual final Q/returns/policy, selected work and optional producing action/update records.
/// @throws std::invalid_argument Invalid geometry/parameters/callbacks or returned state/reward.
/// @throws std::length_error Complete table/episodes, actual transitions/history or counter capacity exceeded.
/// @throws std::overflow_error Actual return, bootstrap target or additive update becomes nonfinite.
/// @par Example
/// @code
/// auto r=dm::algorithms::sarsa_trace(dm::algorithms::tabular_corridor(2),dm::algorithms::RLParameters{.episodes=1,.max_steps=1});
/// std::print("{}\n",r.transitions); // 1
/// @endcode
/// @par References
/// - Sutton and Barto, Reinforcement Learning: An Introduction, second edition, chapter 6. https://incompleteideas.net/book/the-book-2nd.html
/// @see sarsa(), q_learning_run(), tabular_policy_rollout()
SarsaRun sarsa_trace(
    const TabularEnvironment& env,
    const RLParameters&       params,
    const SarsaOptions&       options = {});
/// @brief Return canonical checked SARSA learning values through the preserved signature.
/// Uses the same native on-policy producer/default policies and preserves ordinary valid behavior streams. Invalid environment results now reject before indexing. Returned values and constant-rate training have the same finite-run limits as sarsa_run().
/// @param env Borrowed tabular callbacks and geometry.
/// @param params Complete validated training controls.
/// @return Owned actual Q, greedy policy and undiscounted episode returns.
/// @throws std::invalid_argument Invalid input or callback result.
/// @throws std::length_error Default input/work policies exceeded.
/// @throws std::overflow_error Nonfinite actual update/return.
/// @par Example
/// @code
/// auto r=dm::algorithms::sarsa(dm::algorithms::tabular_corridor(2),dm::algorithms::RLParameters{.episodes=0});
/// std::print("{}\n",r.q.size()); // 2
/// @endcode
/// @par References
/// - Sutton and Barto, Reinforcement Learning: An Introduction, second edition, chapter 6. https://incompleteideas.net/book/the-book-2nd.html
/// @see sarsa_run()
QLearningResult sarsa(const TabularEnvironment& env, const RLParameters& params);
/// @brief Construct the original deterministic four-action cliff environment in the native library.
/// Row-major states, reset at bottom-left, goal at bottom-right. Actions0,1,2,3 mean up,right,down,left; boundary moves clamp. Entering an interior bottom-row cliff cell returns the start state, the cliff reward and terminal=false. Other moves receive step_reward and terminate only on reaching the goal. Owns scalar-capture callbacks with no shared mutable state, preserving the original4x12 fixture; validates every directly supplied state/action too.
/// @param rows At least2 rows, default4.
/// @param columns At least3 columns, default12.
/// @param step_reward Finite ordinary reward, default-1.
/// @param cliff_reward Finite cliff-entry reward, default-100.
/// @return Owned callbacks and complete cliff geometry.
/// @throws std::invalid_argument Invalid dimensions/rewards or direct callback state/action.
/// @throws std::length_error Complete rows*columns overflows size_t.
/// @par Example
/// @code
/// auto e=dm::algorithms::tabular_cliff();auto s=e.step(36,1);
/// std::print("{} {}\n",s.next_state,s.reward); // 36 -100
/// @endcode
/// @par References
/// - Sutton and Barto, Reinforcement Learning: An Introduction, second edition, chapter 6. https://incompleteideas.net/book/the-book-2nd.html
/// @see sarsa_trace(), q_learning_trace()
TabularEnvironment tabular_cliff(
    std::size_t rows         = 4,
    std::size_t columns      = 12,
    double      step_reward  = -1,
    double      cliff_reward = -100);
} // namespace datamunge::algorithms
