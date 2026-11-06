#pragma once
/// @file detail.hpp
/// @brief Testable private masked Bellman, GAE and PPO objective primitives.
#include <algorithm>
#include <cmath>
#include <datamunge/nn/autograd.hpp>
#include <datamunge/nn/ops.hpp>
#include <stdexcept>
#include <vector>
/// @brief Private analytic targets and objectives shared by native learners and tests.
namespace datamunge::rl::detail {
/// @brief First maximum among permitted entries, including negative-valued actions.
/// @param scores Finite scores aligned to mask.
/// @param mask Binary flags with at least one true.
/// @return First maximizing allowed index.
/// @throws std::invalid_argument Shape mismatch or empty support.
inline std::size_t argmax(
    const std::vector<float>&         scores,
    const std::vector<unsigned char>& mask) {
  if (scores.size() != mask.size())
    throw std::invalid_argument("RL score/mask shape mismatch");
  std::size_t best = scores.size();
  for (std::size_t a = 0; a < scores.size(); ++a)
    if (mask[a] && (best == scores.size() || scores[a] > scores[best]))
      best = a;
  if (best == scores.size())
    throw std::invalid_argument("RL mask has no allowed action");
  return best;
}
/// @brief Double-Q bootstrap: online selection, independent target evaluation.
/// @param reward Already composed transition reward.
/// @param discount Elapsed-time discount.
/// @param terminal True only for a genuine terminal, not a time limit.
/// @param online Next online scores.
/// @param target Next target-network scores.
/// @param mask Actual successor feasibility.
/// @return Reward plus discounted target value, or just reward at termination.
/// @throws std::invalid_argument Score/mask shape mismatch or nonterminal empty support.
inline double double_target(
    double                            reward,
    double                            discount,
    bool                              terminal,
    const std::vector<float>&         online,
    const std::vector<float>&         target,
    const std::vector<unsigned char>& mask) {
  if (terminal)
    return reward;
  if (target.size() != online.size())
    throw std::invalid_argument("RL target shape mismatch");
  return reward + discount * target[argmax(online, mask)];
}
/// @brief One pre-update PPO value sample with explicit final-observation bootstrap.
struct ValueSample {
  double reward;   ///< Composed reward.
  double discount; ///< Elapsed-time discount.
  double value;    ///< Old value at the observation.
  double next;     ///< Old value at actual successor, zero only at true termination.
  bool   boundary; ///< Cut traces across terminal/truncation/window resets.
};
/// @brief Variable-duration GAE, retaining truncation bootstrap and cutting reset traces.
/// @param samples Contiguous rollout with actual successor values.
/// @param lambda Per-decision trace decay in [0,1].
/// @return Owned advantages; return targets are advantage plus old value.
inline std::vector<float> gae(const std::vector<ValueSample>& samples, double lambda) {
  std::vector<float> result(samples.size());
  double             following = 0;
  for (std::size_t i = samples.size(); i-- > 0;) {
    const auto& s = samples[i];
    following     = s.reward + s.discount * s.next - s.value +
                (s.boundary ? 0 : s.discount * lambda * following);
    result[i] = float(following);
  }
  return result;
}
/// @brief Clipped PPO actor objective, retaining the sign-dependent minimum.
/// @param logp Current selected-action log-probabilities.
/// @param old Frozen behavior log-probabilities.
/// @param advantage Frozen normalized advantages.
/// @param radius Positive ratio clip radius below one.
/// @return Mean negative clipped surrogate with autograd history.
inline nn::Tensor ppo_loss(
    const nn::Tensor& logp,
    const nn::Tensor& old,
    const nn::Tensor& advantage,
    double            radius) {
  auto ratio = nn::exp(logp - old);
  return -nn::mean(
      nn::minimum(ratio * advantage, nn::clip(ratio, 1 - radius, 1 + radius) * advantage));
}
} // namespace datamunge::rl::detail
