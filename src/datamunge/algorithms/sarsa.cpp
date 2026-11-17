/// @file sarsa.cpp
/// @brief One checked actual on-policy behavior stream, callback updates and original cliff fixture.
#include <cmath>
#include <datamunge/algorithms/sarsa.hpp>
#include <datamunge/algorithms/tabular_rl.hpp>
#include <limits>
#include <random>
#include <stdexcept>
/// @brief Canonical tabular on-policy implementation.
namespace datamunge::algorithms {
/// @brief Private actual behavior/update execution.
/// @brief Private invocation-local checked algorithm helpers.
namespace {
/// @brief Require one actual producing scalar to remain finite.
/// @param value Computed return, target or update.
/// @return Unchanged finite scalar.
/// @throws std::overflow_error For nonfinite arithmetic.
double sarsa_finite(double value) {
  if (!std::isfinite(value))
    throw std::overflow_error("SARSA: nonfinite arithmetic");
  return value;
}
/// @brief Validate complete geometry then execute actual on-policy decisions once.
/// @param env Borrowed callbacks and positive table dimensions.
/// @param params Complete controls.
/// @param o Policies.
/// @param capture Retain actual initial table/actions/updates.
/// @return Owned actual output and selected work.
SarsaRun solve(
    const TabularEnvironment& env,
    const RLParameters&       params,
    const SarsaOptions&       o,
    bool                      capture) {
  const auto max = std::numeric_limits<std::size_t>::max();
  if (!env.num_states || !env.num_actions || !env.reset || !env.step ||
      !std::isfinite(params.learning_rate) || params.learning_rate <= 0 ||
      params.learning_rate > 1 || !std::isfinite(params.discount) || params.discount < 0 ||
      params.discount > 1 || !std::isfinite(params.epsilon) || params.epsilon < 0 ||
      params.epsilon > 1)
    throw std::invalid_argument("SARSA: geometry/callbacks/parameters");
  if (env.num_states > max / env.num_actions)
    throw std::length_error("SARSA: table capacity");
  const auto entries = env.num_states * env.num_actions;
  if ((o.max_table_entries && entries > o.max_table_entries) ||
      (o.max_episodes && params.episodes > o.max_episodes))
    throw std::length_error("SARSA: complete input policy");
  std::size_t retained = 0;
  /// @brief Admit actual optional capture geometry before its work.
  /// @param count Initial table entries or one real event record.
  const auto admit = [&](std::size_t count) {
    if (!capture)
      return;
    if (count > max - retained || (o.max_trace_entries && (retained > o.max_trace_entries ||
                                                           count > o.max_trace_entries - retained)))
      throw std::length_error("SARSA: capture policy");
    retained += count;
  };
  admit(entries);
  SarsaRun out;
  out.result.q.assign(env.num_states, std::vector<double>(env.num_actions, 0));
  out.result.episode_returns.reserve(params.episodes);
  out.episode_steps.reserve(params.episodes);
  out.terminated.reserve(params.episodes);
  if (capture)
    out.initial_q = out.result.q;
  std::mt19937_64 rng(params.seed);
  /// @brief Select one actual original-stream epsilon-greedy action.
  /// @param episode Current episode.
  /// @param state Current validated state.
  /// @return Actual branch draw/action, also retained when capturing.
  const auto select = [&](std::size_t episode, std::size_t state) {
    if (out.action_selections == max)
      throw std::length_error("SARSA: selection capacity");
    admit(1);
    std::uniform_real_distribution<double> unit(0, 1);
    SarsaAction                            choice;
    choice.episode  = episode;
    choice.state    = state;
    choice.draw     = unit(rng);
    choice.explored = choice.draw < params.epsilon;
    if (choice.explored) {
      std::uniform_int_distribution<std::size_t> pick(0, env.num_actions - 1);
      choice.action = pick(rng);
      ++out.exploratory_actions;
    } else {
      choice.action = detail::argmax_action(out.result.q[state]);
      ++out.greedy_actions;
    }
    ++out.action_selections;
    if (capture)
      out.actions.push_back(choice);
    return choice;
  };
  for (std::size_t episode = 0; episode < params.episodes; ++episode) {
    if (out.reset_calls == max)
      throw std::length_error("SARSA: reset capacity");
    ++out.reset_calls;
    auto state = env.reset();
    if (state >= env.num_states)
      throw std::invalid_argument("SARSA: reset state");
    auto        choice     = select(episode, state);
    double      reward_sum = 0;
    std::size_t completed  = 0;
    bool        terminal   = false;
    for (std::size_t time = 0; time < params.max_steps; ++time) {
      if (out.transitions == max || (o.max_transitions && out.transitions >= o.max_transitions))
        throw std::length_error("SARSA: transition policy");
      admit(1);
      const auto transition = env.step(state, choice.action);
      ++out.transitions;
      if (transition.next_state >= env.num_states || !std::isfinite(transition.reward))
        throw std::invalid_argument("SARSA: returned state/reward");
      reward_sum = sarsa_finite(reward_sum + transition.reward);
      SarsaAction next;
      double      bootstrap = 0;
      if (!transition.terminal) {
        next      = select(episode, transition.next_state);
        bootstrap = out.result.q[transition.next_state][next.action];
      }
      const double before = out.result.q[state][choice.action],
                   target = sarsa_finite(transition.reward + params.discount * bootstrap),
                   after  = sarsa_finite(before + params.learning_rate * (target - before));
      out.result.q[state][choice.action] = after;
      if (capture)
        out.steps.push_back(
            {episode,
             time,
             state,
             choice.action,
             transition.next_state,
             transition.reward,
             transition.terminal,
             next.action,
             bootstrap,
             before,
             target,
             after});
      ++completed;
      state  = transition.next_state;
      choice = next;
      if (transition.terminal) {
        terminal = true;
        break;
      }
    }
    out.result.episode_returns.push_back(reward_sum);
    out.episode_steps.push_back(completed);
    out.terminated.push_back(terminal);
  }
  out.result.greedy_policy.resize(env.num_states);
  for (std::size_t state = 0; state < env.num_states; ++state)
    out.result.greedy_policy[state] = detail::argmax_action(out.result.q[state]);
  return out;
}
} // namespace
SarsaRun sarsa_run(const TabularEnvironment& e, const RLParameters& p, const SarsaOptions& o) {
  return solve(e, p, o, false);
}
SarsaRun sarsa_trace(const TabularEnvironment& e, const RLParameters& p, const SarsaOptions& o) {
  return solve(e, p, o, true);
}
QLearningResult sarsa(const TabularEnvironment& e, const RLParameters& p) {
  return solve(e, p, {}, false).result;
}
TabularEnvironment
tabular_cliff(std::size_t rows, std::size_t columns, double step_reward, double cliff_reward) {
  if (rows < 2 || columns < 3 || !std::isfinite(step_reward) || !std::isfinite(cliff_reward))
    throw std::invalid_argument("cliff: geometry/rewards");
  if (rows > std::numeric_limits<std::size_t>::max() / columns)
    throw std::length_error("cliff: state capacity");
  const auto         states = rows * columns, start = (rows - 1) * columns, goal = states - 1;
  TabularEnvironment env;
  env.num_states  = states;
  env.num_actions = 4;
  /// @brief Reset to the original bottom-left cliff start.
  /// @return Actual start state.
  env.reset = [start]() { return start; };
  /// @brief Execute one actual clamped grid/cliff transition.
  /// @param state Valid row-major source state.
  /// @param action Four-way move id.
  /// @return Actual reset-on-cliff or ordinary/goal transition.
  env.step = [=](std::size_t state, std::size_t action) {
    if (state >= states || action >= 4)
      throw std::invalid_argument("cliff: state/action");
    auto row = state / columns, column = state % columns;
    if (action == 0 && row > 0)
      --row;
    else if (action == 1 && column + 1 < columns)
      ++column;
    else if (action == 2 && row + 1 < rows)
      ++row;
    else if (action == 3 && column > 0)
      --column;
    if (row == rows - 1 && column >= 1 && column < columns - 1)
      return StepResult{start, cliff_reward, false};
    const auto next = row * columns + column;
    return StepResult{next, step_reward, next == goal};
  };
  return env;
}
} // namespace datamunge::algorithms
