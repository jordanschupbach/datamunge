/// @file td_prediction.cpp
/// @brief Actual checked fixed-policy TD update kernel and original random-walk environment.
#include <cmath>
#include <datamunge/algorithms/td_prediction.hpp>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private finite checks and per-invocation producer.
namespace {
/// @brief Reject actual nonfinite learning arithmetic.
double finite(double x) {
  if (!std::isfinite(x))
    throw std::overflow_error("TD arithmetic");
  return x;
}
/// @brief Execute validated fixed-policy callbacks and actual bootstrap updates.
TDPredictionRun execute(
    const TabularEnvironment&       env,
    const std::vector<std::size_t>& policy,
    const RLParameters&             p,
    const TDPredictionOptions&      o,
    bool                            capture) {
  if (!env.num_states || !env.num_actions || !env.reset || !env.step ||
      policy.size() != env.num_states || !std::isfinite(p.learning_rate) || p.learning_rate <= 0 ||
      p.learning_rate > 1 || !std::isfinite(p.discount) || p.discount < 0 || p.discount > 1)
    throw std::invalid_argument("TD geometry/controls");
  for (auto a : policy)
    if (a >= env.num_actions)
      throw std::invalid_argument("TD policy action");
  if ((o.max_states && env.num_states > o.max_states) ||
      (o.max_episodes && p.episodes > o.max_episodes) ||
      p.episodes == std::numeric_limits<std::size_t>::max())
    throw std::length_error("TD complete geometry");
  TDPredictionRun r;
  r.result.value.assign(env.num_states, 0);
  r.result.episode_returns.reserve(p.episodes);
  std::size_t retained = 0;
  /// @brief Bound actual captured value units before copying one history object.
  auto retain = [&](std::size_t n) {
    auto cap = std::numeric_limits<std::size_t>::max();
    if (retained > cap - n || (o.max_history_values && (retained > o.max_history_values ||
                                                        n > o.max_history_values - retained)))
      throw std::length_error("TD history");
    retained += n;
  };
  if (capture) {
    if (env.num_states == std::numeric_limits<std::size_t>::max())
      throw std::length_error("TD initial history");
    retain(env.num_states + 1);
    r.episodes.push_back({0, 0, 0, false, false, r.result.value});
  }
  for (std::size_t ep = 0; ep < p.episodes; ++ep) {
    auto state = env.reset();
    ++r.resets;
    if (state >= env.num_states)
      throw std::out_of_range("TD reset state");
    double      total    = 0;
    bool        terminal = false;
    std::size_t taken    = 0;
    for (std::size_t t = 0; t < p.max_steps; ++t) {
      if (r.transitions == std::numeric_limits<std::size_t>::max() ||
          (o.max_transitions && r.transitions >= o.max_transitions))
        throw std::length_error("TD transitions");
      auto action = policy[state];
      auto sr     = env.step(state, action);
      ++r.transitions;
      if (sr.next_state >= env.num_states)
        throw std::out_of_range("TD successor state");
      if (!std::isfinite(sr.reward))
        throw std::domain_error("TD finite reward");
      total                 = finite(total + sr.reward);
      double before         = r.result.value[state],
             successor      = sr.terminal ? 0 : r.result.value[sr.next_state],
             target         = finite(sr.reward + finite(p.discount * successor)),
             error          = finite(target - before),
             after          = finite(before + finite(p.learning_rate * error));
      r.result.value[state] = after;
      ++taken;
      if (capture) {
        retain(1);
        r.steps.push_back(
            {ep,
             t,
             state,
             action,
             sr.next_state,
             sr.reward,
             sr.terminal,
             before,
             successor,
             target,
             error,
             after});
      }
      state = sr.next_state;
      if (sr.terminal) {
        terminal = true;
        break;
      }
    }
    r.result.episode_returns.push_back(total);
    if (terminal)
      ++r.terminated;
    else
      ++r.truncated;
    if (capture) {
      retain(env.num_states + 1);
      r.episodes.push_back({ep + 1, taken, total, terminal, !terminal, r.result.value});
    }
  }
  return r;
}
} // namespace
TDPredictionRun td_prediction_run(
    const TabularEnvironment&       env,
    const std::vector<std::size_t>& policy,
    const RLParameters&             params,
    const TDPredictionOptions&      options) {
  return execute(env, policy, params, options, false);
}
TDPredictionRun td_prediction_trace(
    const TabularEnvironment&       env,
    const std::vector<std::size_t>& policy,
    const RLParameters&             params,
    const TDPredictionOptions&      options) {
  return execute(env, policy, params, options, true);
}
TDPredictionResult td_prediction(
    const TabularEnvironment&       env,
    const std::vector<std::size_t>& policy,
    const RLParameters&             params) {
  return td_prediction_run(env, policy, params).result;
}
TabularEnvironment td_random_walk_environment(std::uint64_t seed) {
  auto rng = std::make_shared<std::mt19937_64>(seed);
  return {
      7,
      1,
      [] { return std::size_t{3}; },
      [rng](std::size_t s, std::size_t a) {
        if (s < 1 || s > 5 || a != 0)
          throw std::out_of_range("random walk active state/action");
        std::uniform_int_distribution<int> coin(0, 1);
        auto                               next = coin(*rng) ? s + 1 : s - 1;
        return StepResult{next, next == 6 ? 1. : 0., next == 0 || next == 6};
      }};
}
std::vector<double> td_random_walk_truth() {
  return {0, 1. / 6, 2. / 6, 3. / 6, 4. / 6, 5. / 6, 0};
}
double td_random_walk_rmse(const std::vector<double>& values) {
  if (values.size() != 7)
    throw std::invalid_argument("walk RMSE geometry");
  for (double x : values)
    if (!std::isfinite(x))
      throw std::invalid_argument("walk RMSE finite values");
  double norm = 0;
  for (std::size_t i = 1; i <= 5; ++i)
    norm = finite(std::hypot(norm, values[i] - double(i) / 6));
  return norm / std::sqrt(5.);
}
} // namespace datamunge::algorithms
