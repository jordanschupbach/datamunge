/// @file q_learning.cpp
/// @brief Single checked behavior/transition/update producer and deterministic corridor fixture.
#include <datamunge/algorithms/q_learning.hpp>
#include <datamunge/algorithms/tabular_rl.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Actual local behavior RNG, callback validation and optional update capture.
namespace q_learning_detail {
/// @brief Require a producing floating-point result to remain finite.
/// @param v Actual computed scalar.
/// @return Unchanged finite scalar.
/// @throws std::overflow_error If a producing return/target/update is nonfinite or a selected count cannot fit.
double finite(double v) {
  if (!std::isfinite(v))
    throw std::overflow_error("Q-learning nonfinite arithmetic");
  return v;
}
/// @brief Validate geometry/parameters then execute one complete actual training run.
/// @param env Borrowed synchronous environment.
/// @param params Complete scalar training parameters.
/// @param options Policies.
/// @param capture Whether actual initial table/updates are retained.
/// @return Owned actual output/work and optional history.
/// @throws std::invalid_argument If geometry/callbacks/parameters are invalid or a callback returns a nonfinite reward or out-of-range state, including terminal states.
/// @throws std::length_error If table/episode/transition/history policy or representable storage is exceeded.
/// @throws std::overflow_error If a producing return/target/update is nonfinite or a selected count cannot fit.
QLearningTrace produce(
    const TabularEnvironment& env,
    const RLParameters&       params,
    QLearningOptions          options,
    bool                      capture) {
  if (!env.num_states || !env.num_actions || !env.reset || !env.step ||
      !std::isfinite(params.learning_rate) || params.learning_rate <= 0 ||
      params.learning_rate > 1 || !std::isfinite(params.discount) || params.discount < 0 ||
      params.discount > 1 || !std::isfinite(params.epsilon) || params.epsilon < 0 ||
      params.epsilon > 1)
    throw std::invalid_argument("Q-learning geometry or parameters");
  if (env.num_states > std::numeric_limits<std::size_t>::max() / env.num_actions)
    throw std::length_error("Q-learning table geometry");
  auto entries = env.num_states * env.num_actions;
  if ((options.max_table_entries && entries > options.max_table_entries) ||
      (options.max_episodes && params.episodes > options.max_episodes) ||
      (capture && options.max_trace_entries && entries > options.max_trace_entries))
    throw std::length_error("Q-learning admission cap");
  QLearningTrace out;
  auto&          r      = out.run;
  auto&          result = r.result;
  result.q.assign(env.num_states, std::vector<double>(env.num_actions, 0));
  result.episode_returns.reserve(params.episodes);
  r.episode_steps.reserve(params.episodes);
  r.terminated.reserve(params.episodes);
  if (capture)
    out.initial_q = result.q;
  std::mt19937_64 rng(params.seed);
  for (std::size_t ep = 0; ep < params.episodes; ++ep) {
    auto state = env.reset();
    ++r.reset_calls;
    if (state >= env.num_states)
      throw std::invalid_argument("Q-learning reset state");
    double      episode_return = 0;
    std::size_t steps          = 0;
    bool        terminal       = false;
    for (std::size_t t = 0; t < params.max_steps; ++t) {
      if (r.transitions == std::numeric_limits<std::size_t>::max() ||
          (options.max_transitions && r.transitions >= options.max_transitions))
        throw std::length_error("Q-learning transition cap");
      if (capture && (entries == std::numeric_limits<std::size_t>::max() ||
                      (options.max_trace_entries && entries >= options.max_trace_entries)))
        throw std::length_error("Q-learning history cap");
      std::uniform_real_distribution<double> unit(0, 1);
      double                                 draw     = unit(rng);
      bool                                   explored = draw < params.epsilon;
      std::size_t                            action;
      if (explored) {
        std::uniform_int_distribution<std::size_t> pick(0, env.num_actions - 1);
        action = pick(rng);
        ++r.exploratory_actions;
      } else {
        action = detail::argmax_action(result.q[state]);
        ++r.greedy_actions;
      }
      auto sr = env.step(state, action);
      if (sr.next_state >= env.num_states || !std::isfinite(sr.reward))
        throw std::invalid_argument("Q-learning transition output");
      ++r.transitions;
      ++steps;
      episode_return = finite(episode_return + sr.reward);
      auto next_action =
          sr.terminal ? std::size_t{0} : detail::argmax_action(result.q[sr.next_state]);
      double bootstrap        = sr.terminal ? 0 : result.q[sr.next_state][next_action];
      double target           = finite(sr.reward + params.discount * bootstrap),
             before           = result.q[state][action];
      double after            = finite(before + params.learning_rate * finite(target - before));
      result.q[state][action] = after;
      if (capture) {
        ++entries;
        out.steps.push_back(
            {ep,
             t,
             state,
             action,
             sr.next_state,
             sr.reward,
             sr.terminal,
             draw,
             explored,
             next_action,
             bootstrap,
             before,
             target,
             after,
             episode_return});
      }
      state = sr.next_state;
      if (sr.terminal) {
        terminal = true;
        break;
      }
    }
    result.episode_returns.push_back(episode_return);
    r.episode_steps.push_back(steps);
    r.terminated.push_back(terminal);
  }
  result.greedy_policy.resize(env.num_states);
  for (std::size_t s = 0; s < env.num_states; ++s)
    result.greedy_policy[s] = detail::argmax_action(result.q[s]);
  return out;
}
} // namespace q_learning_detail
QLearningResult q_learning(const TabularEnvironment& env, const RLParameters& params) {
  return q_learning_detail::produce(env, params, {}, false).run.result;
}
QLearningRun q_learning_run(
    const TabularEnvironment& env,
    const RLParameters&       params,
    QLearningOptions          options) {
  return q_learning_detail::produce(env, params, options, false).run;
}
QLearningTrace q_learning_trace(
    const TabularEnvironment& env,
    const RLParameters&       params,
    QLearningOptions          options) {
  return q_learning_detail::produce(env, params, options, true);
}
TabularEnvironment tabular_corridor(std::size_t states, double reward) {
  if (states < 2 || !std::isfinite(reward))
    throw std::invalid_argument("corridor geometry or reward");
  TabularEnvironment env;
  env.num_states  = states;
  env.num_actions = 2;
  /// @brief Reset this deterministic scalar-capture corridor to state zero.
  /// @return Initial state0.
  env.reset = [] { return std::size_t{0}; };
  /// @brief Execute one actual deterministic corridor step.
  /// @param state Valid current state.
  /// @param action Zero for left,one for right.
  /// @return Actual clamped next state,reward and goal flag.
  /// @throws std::invalid_argument On an invalid state/action.
  env.step = [states, reward](std::size_t state, std::size_t action) {
    if (state >= states || action >= 2)
      throw std::invalid_argument("corridor state/action");
    auto next = action == 1 ? (state < states - 1 ? state + 1 : state) : (state ? state - 1 : 0);
    return StepResult{next, reward, next == states - 1};
  };
  return env;
}
TabularEnvironment tabular_grid(std::size_t side, double reward) {
  if (side < 2 || !std::isfinite(reward))
    throw std::invalid_argument("grid side/reward");
  if (side > std::numeric_limits<std::size_t>::max() / side)
    throw std::length_error("grid state geometry");
  auto               states = side * side;
  TabularEnvironment env;
  env.num_states  = states;
  env.num_actions = 4;
  /// @brief Reset this deterministic grid to its top-left corner.
  /// @return Actual initial state0.
  env.reset = [] { return std::size_t{0}; };
  /// @brief Execute one original row-major grid move with blocked boundary steps.
  /// @param state Current valid row-major state.
  /// @param action Up,right,down,left encoded0,1,2,3.
  /// @return Actual next state,reward and bottom-right termination flag.
  /// @throws std::invalid_argument On an invalid state or action.
  env.step = [side, states, reward](std::size_t state, std::size_t action) {
    if (state >= states || action >= 4)
      throw std::invalid_argument("grid state/action");
    auto row = state / side, col = state % side;
    if (action == 0 && row > 0)
      --row;
    else if (action == 1 && col < side - 1)
      ++col;
    else if (action == 2 && row < side - 1)
      ++row;
    else if (action == 3 && col > 0)
      --col;
    auto next = row * side + col;
    return StepResult{next, reward, next == states - 1};
  };
  return env;
}
TabularPolicyRollout tabular_policy_rollout(
    const TabularEnvironment&       env,
    const std::vector<std::size_t>& policy,
    std::size_t                     max_steps,
    std::size_t                     max_entries) {
  if (!env.num_states || !env.num_actions || !env.reset || !env.step ||
      policy.size() != env.num_states)
    throw std::invalid_argument("policy rollout geometry");
  for (auto action : policy)
    if (action >= env.num_actions)
      throw std::invalid_argument("policy rollout action");
  TabularPolicyRollout out;
  auto                 state = env.reset();
  if (state >= env.num_states)
    throw std::invalid_argument("policy reset state");
  out.states.push_back(state);
  std::size_t entries = 1;
  for (std::size_t step = 0; step < max_steps; ++step) {
    if (entries > std::numeric_limits<std::size_t>::max() - 3 ||
        (max_entries && (entries > max_entries || 3 > max_entries - entries)))
      throw std::length_error("policy rollout history cap");
    auto action = policy[state];
    auto sr     = env.step(state, action);
    if (sr.next_state >= env.num_states || !std::isfinite(sr.reward))
      throw std::invalid_argument("policy transition output");
    out.total_reward = q_learning_detail::finite(out.total_reward + sr.reward);
    out.actions.push_back(action);
    out.rewards.push_back(sr.reward);
    out.states.push_back(sr.next_state);
    entries += 3;
    state = sr.next_state;
    if (sr.terminal) {
      out.terminal = true;
      break;
    }
  }
  return out;
}
std::vector<double> trailing_return_mean(const std::vector<double>& returns, std::size_t window) {
  if (!window)
    throw std::invalid_argument("return smoothing window");
  for (auto value : returns)
    if (!std::isfinite(value))
      throw std::invalid_argument("return smoothing finite input");
  std::vector<double> means;
  means.reserve(returns.size());
  long double sum = 0;
  for (std::size_t i = 0; i < returns.size(); ++i) {
    if (i >= window)
      sum -= returns[i - window];
    sum += returns[i];
    if (!std::isfinite(sum))
      throw std::overflow_error("return smoothing sum");
    auto count = std::min(i + 1, window);
    means.push_back(q_learning_detail::finite(double(sum / count)));
  }
  return means;
}
} // namespace datamunge::algorithms
