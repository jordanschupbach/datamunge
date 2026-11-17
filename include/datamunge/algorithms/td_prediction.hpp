#pragma once
/// @file td_prediction.hpp
/// @brief Checked actual fixed-policy TD(0) updates, episode outcomes and reproducible random walk.
#include <datamunge/algorithms/tabular_rl_types.hpp>
namespace datamunge::algorithms {
/// @brief Own the TDPredictionOptions contract for one independent native execution.
/// Field comments specify actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record does not run the algorithm, retain caller storage or create shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::TDPredictionOptions r{};
/// (void)r;
/// @endcode
/// @see td_prediction_run()
struct TDPredictionOptions {
  std::size_t max_states{1000000};   ///< Complete state geometry; zero disables.
  std::size_t max_episodes{1000000}; ///< Complete requested resets/episode records; zero disables.
  std::size_t max_transitions{20000000}; ///< Actual callback transitions/TD updates; zero disables.
  std::size_t max_history_values{
      1000000}; ///< Actual captured step/epoch records and epoch-value scalar copies; zero disables.
};
/// @brief Own the TDPredictionStep contract for one independent native execution.
/// Field comments specify actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record does not run the algorithm, retain caller storage or create shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::TDPredictionStep r{};
/// (void)r;
/// @endcode
/// @see td_prediction_run()
struct TDPredictionStep {
  std::size_t episode{};    ///< Actual zero-based episode.
  std::size_t step{};       ///< Actual zero-based transition within episode.
  std::size_t state{};      ///< Actual current state.
  std::size_t action{};     ///< Actual supplied deterministic policy action.
  std::size_t next_state{}; ///< Actual environment successor, validated even at termination.
  double      reward{};     ///< Actual finite immediate reward.
  bool        terminal{};   ///< Actual callback terminal flag.
  double      before{};     ///< Actual current state value before this update.
  double      successor{};  ///< Actual pre-update successor value, forced0 at termination.
  double      target{};     ///< Actual reward+discount*successor.
  double      error{};      ///< Actual target-before.
  double      after{};      ///< Actual before+learning_rate*error.
};
/// @brief Own the TDPredictionEpisode contract for one independent native execution.
/// Field comments specify actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record does not run the algorithm, retain caller storage or create shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::TDPredictionEpisode r{};
/// (void)r;
/// @endcode
/// @see td_prediction_run()
struct TDPredictionEpisode {
  std::size_t episode{};      ///< Actual completed episode count; initial values use0.
  std::size_t transitions{};  ///< Actual transitions in this episode; zero for initial capture.
  double      return_value{}; ///< Actual undiscounted sum of this episode rewards.
  bool        terminated{};   ///< Actual terminal flag ended the episode.
  bool truncated{}; ///< Actual step cap ended it without a terminal flag, including zero steps.
  std::vector<double> values; ///< Actual complete owned values after this episode or initial zeros.
};
/// @brief Own the TDPredictionRun contract for one independent native execution.
/// Field comments specify actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record does not run the algorithm, retain caller storage or create shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::TDPredictionRun r{};
/// (void)r;
/// @endcode
/// @see td_prediction_run()
struct TDPredictionRun {
  TDPredictionResult result;        ///< Actual final value table and undiscounted episode returns.
  std::size_t        resets{};      ///< Actual reset callback calls.
  std::size_t        transitions{}; ///< Actual environment step calls and TD updates.
  std::size_t        terminated{};  ///< Actual terminal-ended episodes.
  std::size_t        truncated{};   ///< Actual step-cap-ended episodes without terminal flag.
  std::vector<TDPredictionStep>    steps;    ///< Optional actual step updates; no callback replay.
  std::vector<TDPredictionEpisode> episodes; ///< Optional actual initial/post-episode values.
};
/// @brief Evaluate a fixed policy through actual online TD(0) transitions.
/// Finite tabular geometry must have positive states/actions, nonempty synchronous callbacks and one valid action for every state. learning_rate finite in(0,1], discount finite in[0,1]. Epsilon and params.seed are unused: the supplied deterministic policy is followed and callback RNG remains callback-owned. Initialize every value to0, including terminals. After reset, validate state; after each actual step validate even terminal successor and reward. Update V(s)+=alpha*(reward+gamma*(terminal?0:V(next))-V(s)) using pre-update successor, including self transitions. Episode returns are undiscounted. Zero episodes invokes no callback; zero max_steps still resets and records a truncated episode. Constant alpha does not generally converge exactly and errors need not decrease monotonically. Classical asymptotic prediction results require appropriate visitation, step sizes and task assumptions; they are not certified by a finite run. Callback exceptions/side effects propagate without rollback. O(states+actual transitions), optional real state/episode histories and no internal shared state; caller owns callback concurrency safety.
/// @param env Borrowed finite synchronous environment; callbacks may mutate caller state.
/// @param policy Borrowed complete deterministic action per state.
/// @param params Shared RL controls: rate,discount,episode/step caps; epsilon/seed unused.
/// @param options Complete state/episode and actual transition/capture policies; zero disables.
/// @return Owned actual final values/returns, terminal/truncation counts and optional producing histories.
/// @throws std::invalid_argument Malformed geometry, callbacks, policy or numeric controls.
/// @throws std::out_of_range Actual reset or transition state outside environment.
/// @throws std::domain_error Actual reward is nonfinite.
/// @throws std::overflow_error Actual return/target/error/update arithmetic is nonfinite.
/// @throws std::length_error Complete state/episode or actual transition/history/counter capacity exceeded.
/// @par Example
/// @code
/// dm::algorithms::TabularEnvironment e{1,1,[]{return std::size_t{0};},[](std::size_t,std::size_t){return dm::algorithms::StepResult{0,2,true};}};
/// dm::algorithms::RLParameters p;p.episodes=1;p.learning_rate=.5;
/// auto r=dm::algorithms::td_prediction_run(e,{0},p);
/// std::print("{}\n",r.result.value[0]); // 1
/// @endcode
/// @par References
/// - Sutton, Learning to Predict by the Methods of Temporal Differences, Machine Learning3 (1988),9-44.
/// @see td_prediction()
TDPredictionRun td_prediction_run(
    const TabularEnvironment&       env,
    const std::vector<std::size_t>& policy,
    const RLParameters&             params,
    const TDPredictionOptions&      options = {});
/// @brief Evaluate a fixed policy through actual online TD(0) transitions.
/// Finite tabular geometry must have positive states/actions, nonempty synchronous callbacks and one valid action for every state. learning_rate finite in(0,1], discount finite in[0,1]. Epsilon and params.seed are unused: the supplied deterministic policy is followed and callback RNG remains callback-owned. Initialize every value to0, including terminals. After reset, validate state; after each actual step validate even terminal successor and reward. Update V(s)+=alpha*(reward+gamma*(terminal?0:V(next))-V(s)) using pre-update successor, including self transitions. Episode returns are undiscounted. Zero episodes invokes no callback; zero max_steps still resets and records a truncated episode. Constant alpha does not generally converge exactly and errors need not decrease monotonically. Classical asymptotic prediction results require appropriate visitation, step sizes and task assumptions; they are not certified by a finite run. Callback exceptions/side effects propagate without rollback. O(states+actual transitions), optional real state/episode histories and no internal shared state; caller owns callback concurrency safety.
/// @param env Borrowed finite synchronous environment; callbacks may mutate caller state.
/// @param policy Borrowed complete deterministic action per state.
/// @param params Shared RL controls: rate,discount,episode/step caps; epsilon/seed unused.
/// @param options Complete state/episode and actual transition/capture policies; zero disables.
/// @return Owned actual final values/returns, terminal/truncation counts and optional producing histories.
/// @throws std::invalid_argument Malformed geometry, callbacks, policy or numeric controls.
/// @throws std::out_of_range Actual reset or transition state outside environment.
/// @throws std::domain_error Actual reward is nonfinite.
/// @throws std::overflow_error Actual return/target/error/update arithmetic is nonfinite.
/// @throws std::length_error Complete state/episode or actual transition/history/counter capacity exceeded.
/// @par Example
/// @code
/// dm::algorithms::TabularEnvironment e{1,1,[]{return std::size_t{0};},[](std::size_t,std::size_t){return dm::algorithms::StepResult{0,2,true};}};
/// dm::algorithms::RLParameters p;p.episodes=1;p.learning_rate=.5;
/// auto r=dm::algorithms::td_prediction_trace(e,{0},p);
/// std::print("{}\n",r.result.value[0]); // 1
/// @endcode
/// @par References
/// - Sutton, Learning to Predict by the Methods of Temporal Differences, Machine Learning3 (1988),9-44.
/// @see td_prediction()
TDPredictionRun td_prediction_trace(
    const TabularEnvironment&       env,
    const std::vector<std::size_t>& policy,
    const RLParameters&             params,
    const TDPredictionOptions&      options = {});
/// @brief Preserve checked fixed-policy TD(0) final values and episode returns.
/// Finite tabular geometry must have positive states/actions, nonempty synchronous callbacks and one valid action for every state. learning_rate finite in(0,1], discount finite in[0,1]. Epsilon and params.seed are unused: the supplied deterministic policy is followed and callback RNG remains callback-owned. Initialize every value to0, including terminals. After reset, validate state; after each actual step validate even terminal successor and reward. Update V(s)+=alpha*(reward+gamma*(terminal?0:V(next))-V(s)) using pre-update successor, including self transitions. Episode returns are undiscounted. Zero episodes invokes no callback; zero max_steps still resets and records a truncated episode. Constant alpha does not generally converge exactly and errors need not decrease monotonically. Classical asymptotic prediction results require appropriate visitation, step sizes and task assumptions; they are not certified by a finite run. Callback exceptions/side effects propagate without rollback. O(states+actual transitions), optional real state/episode histories and no internal shared state; caller owns callback concurrency safety.
/// @param env Borrowed finite synchronous environment; callbacks may mutate caller state.
/// @param policy Borrowed complete deterministic action per state.
/// @param params Shared RL controls: rate,discount,episode/step caps; epsilon/seed unused.
/// @return Owned actual values/returns, without optional history.
/// @throws std::invalid_argument Malformed geometry/callback/policy/controls.
/// @throws std::out_of_range Actual callback state invalid.
/// @throws std::domain_error Nonfinite reward.
/// @throws std::overflow_error Actual nonfinite arithmetic.
/// @throws std::length_error Default geometry/transition capacity exceeded.
/// @par Example
/// @code
/// auto e=dm::algorithms::td_random_walk_environment();
/// dm::algorithms::RLParameters p;p.episodes=0;
/// std::print("{}\n",dm::algorithms::td_prediction(e,std::vector<std::size_t>(7),p).value.size()); // 7
/// @endcode
/// @see td_prediction_run()
TDPredictionResult td_prediction(
    const TabularEnvironment&       env,
    const std::vector<std::size_t>& policy,
    const RLParameters&             params);
/// @brief Create the original seven-state symmetric random-walk callbacks.
/// States0 and6 are terminal, reset always3, one action. A local shared mt19937_64 seed2024 and uniform integer coin select left/right; reward1 only when entering6. Each factory call owns a fresh independent engine; copying the returned environment shares its engine with that copy. Stepping terminal or out-of-range states/actions throws. No implicit use of RLParameters.seed.
/// @param seed Callback-owned RNG seed, default2024.
/// @return Owned callbacks and7-state1-action geometry; engine lifetime follows callback copies.
/// @par Example
/// @code
/// auto e=dm::algorithms::td_random_walk_environment();
/// std::print("{}\n",e.reset()); // 3
/// @endcode
/// @see td_random_walk_truth()
TabularEnvironment td_random_walk_environment(std::uint64_t seed = 2024);
/// @brief Return exact-form symmetric random-walk reference values.
/// States1..5 have mathematical undiscounted values k/6 for the fair walk with right-terminal reward1; represented in double. Terminal values0 and6 are0 because terminal continuation is excluded.
/// @return Owned seven-state reference vector.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::td_random_walk_truth()[3]); // 0.5
/// @endcode
/// @see td_random_walk_environment()
std::vector<double> td_random_walk_truth();
/// @brief Measure actual five-interior-state random-walk prediction RMSE.
/// Requires exactly seven finite state values. Uses the fixed fair-walk references k/6 for states1..5, ignores both terminal coordinates after validating them, and accumulates errors with hypot to avoid avoidable square overflow. Constant work, no mutation/shared state.
/// @param values Borrowed complete seven-state finite estimates.
/// @return Actual finite root-mean-square interior prediction error.
/// @throws std::invalid_argument Wrong geometry or nonfinite supplied estimate.
/// @throws std::overflow_error Actual error norm not representable.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::td_random_walk_rmse(dm::algorithms::td_random_walk_truth())); // 0
/// @endcode
/// @see td_random_walk_truth()
double td_random_walk_rmse(const std::vector<double>& values);
} // namespace datamunge::algorithms
