#pragma once
/// @file bootstrapped_returns.hpp
/// @brief Source Float32 GAE with separate next-state bootstrap and reset trace boundaries.
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>
namespace datamunge::rl::returns {
/// @brief Compute source Float32 GAE with explicit next-state bootstraps and trace boundaries.
///
/// delta=r+gamma*bootstrap-value, then A=delta+gamma*lambda*(1-done)*next_A. A time limit may keep
/// a nonzero bootstrap while still cutting the trace. No graph, objective, optimizer or collection
/// is retained; Float32 operation order matches source.
///
/// @param rewards Finite per-transition rewards, copied only into output arithmetic.
/// @param values Finite pre-action value predictions, same length.
/// @param bootstraps Finite actual next-state values, same length; caller supplies0 for physical
/// terminals.
/// @param done Binary terminal-or-time-limit boundaries, same length; cuts recurrence across reset.
/// @param gamma Finite discount in[0,1].
/// @param lambda Finite trace decay in[0,1].
/// @return Owned advantages and value targets in chronological order; empty input yields two empty
/// vectors.
/// @throws std::invalid_argument Shape, coefficients, value finiteness or boundary masks are
/// invalid.
/// @par Example
/// @code
/// auto result=dm::rl::returns::NativeGae({1,2},{.5f,.5f},{.5f,4},{0,1},.9f,.95f);
/// @endcode
/// @par References
/// - Schulman et al. (2016), Generalized Advantage Estimation, https://arxiv.org/abs/1506.02438 .
/// @see Gae()
inline std::pair<std::vector<float>, std::vector<float>> NativeGae(
    const std::vector<float>& rewards,
    const std::vector<float>& values,
    const std::vector<float>& bootstraps,
    const std::vector<float>& done,
    float                     gamma,
    float                     lambda) {
  const auto n = rewards.size();
  if (values.size() != n || bootstraps.size() != n || done.size() != n || !std::isfinite(gamma) ||
      gamma < 0 || gamma > 1 || !std::isfinite(lambda) || lambda < 0 || lambda > 1)
    throw std::invalid_argument("native GAE shape or coefficient");
  std::vector<float> advantages(n), targets(n);
  float              advantage = 0;
  for (auto t = n; t-- > 0;) {
    if (!std::isfinite(rewards[t]) || !std::isfinite(values[t]) || !std::isfinite(bootstraps[t]) ||
        (done[t] != 0 && done[t] != 1))
      throw std::invalid_argument("native GAE nonfinite value or nonbinary boundary");
    const float delta = rewards[t] + gamma * bootstraps[t] - values[t];
    advantage         = delta + gamma * lambda * (1 - done[t]) * advantage;
    advantages[t]     = advantage;
    targets[t]        = advantage + values[t];
  }
  return {std::move(advantages), std::move(targets)};
}

} // namespace datamunge::rl::returns
