#pragma once
/// @file off_policy.hpp
/// @brief Explicit span-based Retrace and V-trace target calculations, without replay or learner
/// state.
#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>
/// @brief Explicit off-policy target algebra; callers own trajectories, replay and every learner
/// loop.
namespace datamunge::rl::off_policy {
/// @brief Internal finite sequence/span validation.
namespace detail {

/// @brief Validate a bounded finite reward sequence and binary terminality.
///
/// @param reward At most16million finite Float32 entries.
/// @param done Aligned binary flags; time-limit truncation uses0.
/// @param gamma Finite discount in[0,1].
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par References
/// - Kapturowski et al. (2019), Recurrent Experience Replay,
/// https://openreview.net/forum?id=r1lyTjAqYX; Sutton and Barto (2018),
/// http://incompleteideas.net/book/the-book-2nd.html .
inline void Sequence(std::span<const float> reward, std::span<const float> done, float gamma) {
  if (reward.size() != done.size() || reward.size() > 16000000 || !std::isfinite(gamma) ||
      gamma < 0 || gamma > 1)
    throw std::invalid_argument(
        "aligned bounded rewards/terminality and discount in[0,1] required");
  for (size_t t = 0; t < reward.size(); ++t)
    if (!std::isfinite(reward[t]) || (done[t] != 0 && done[t] != 1))
      throw std::invalid_argument("finite rewards and binary terminality required");
}

/// @brief Validate finite caller-borrowed Float32 entries.
///
/// @param x Span valid for this call; empty allowed.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par References
/// - Kapturowski et al. (2019), Recurrent Experience Replay,
/// https://openreview.net/forum?id=r1lyTjAqYX; Sutton and Barto (2018),
/// http://incompleteideas.net/book/the-book-2nd.html .
inline void Finite(std::span<const float> x) {
  for (float v : x)
    if (!std::isfinite(v))
      throw std::invalid_argument("finite off-policy values required");
}
} // namespace detail

/// @brief Compute the source clipped-importance Retrace targets for one trajectory.
///
/// rho_bar=min(1,pi_taken/max(mu_taken,1e-8)). Terminality zeros the backward bootstrap.
/// Rows are range-checked but normalization is caller-owned. Spans are borrowed only during the
/// call. No graph to host inputs, RNG, replay mutation or learner state is retained.
///
/// @param rewards Finite Float32[T], T at most16million, possibly empty.
/// @param done Binary actual-terminal flags[T].
/// @param actions Discrete indices[T], each in[0,A).
/// @param behavior Behavior action probabilities[T*A], finite in[0,1]; chosen denominators floored
/// at1e-8.
/// @param probability Target action probabilities[(T+1)*A], finite in[0,1].
/// @param q Finite target/current Q values[(T+1)*A], same row-major layout.
/// @param value Finite state expectations[T+1], final entry is bootstrap.
/// @param A Action count in[1,4096]; (T+1)*A at most64million.
/// @param gamma Finite discount in[0,1].
/// @return Owning Float32 target Q returns[T], source backward accumulation order.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// auto
/// target=dm::rl::off_policy::Retrace(std::vector<float>{1},std::vector<float>{1},std::vector<int>{0},std::vector<float>{1},std::vector<float>{1,1},std::vector<float>{0,0},std::vector<float>{0,0},1,.9f);
/// @endcode
/// @par References
/// - Munos et al. (2016), https://arxiv.org/abs/1606.02647; Wang et al. (2017),
/// https://arxiv.org/abs/1611.01224 .
/// @see VTrace()
inline std::vector<float> Retrace(
    std::span<const float> rewards,
    std::span<const float> done,
    std::span<const int>   actions,
    std::span<const float> behavior,
    std::span<const float> probability,
    std::span<const float> q,
    std::span<const float> value,
    int                    A,
    float                  gamma) {
  detail::Sequence(rewards, done, gamma);
  size_t T = rewards.size();
  if (A < 1 || A > 4096 || (T + 1) * size_t(A) > 64000000 || actions.size() != T ||
      behavior.size() != T * size_t(A) || probability.size() != (T + 1) * size_t(A) ||
      q.size() != probability.size() || value.size() != T + 1)
    throw std::invalid_argument("aligned Retrace action/probability/value geometry required");
  detail::Finite(behavior);
  detail::Finite(probability);
  detail::Finite(q);
  detail::Finite(value);
  for (size_t t = 0; t < T; ++t)
    if (actions[t] < 0 || actions[t] >= A)
      throw std::invalid_argument("Retrace action index");
  for (float p : behavior)
    if (p < 0 || p > 1)
      throw std::invalid_argument("behavior probability range");
  for (float p : probability)
    if (p < 0 || p > 1)
      throw std::invalid_argument("target probability range");
  float              ret = value[T];
  std::vector<float> qret(T);
  for (int t = int(T) - 1; t >= 0; --t) {
    if (done[t] > 0)
      ret = 0;
    ret           = rewards[t] + gamma * ret;
    qret[t]       = ret;
    int   a       = actions[t];
    float rho_bar = std::min(1.f, probability[t * A + a] / std::max(behavior[t * A + a], 1e-8f));
    ret           = rho_bar * (ret - q[t * A + a]) + value[t];
  }
  return qret;
}
/// @brief Owning source-rounded V-trace value and policy-gradient targets.
struct VTracePath {
  std::vector<float> values;     ///< Corrected state values[T+1], final entry is caller bootstrap.
  std::vector<float> advantages; ///< Truncated-importance policy-gradient advantages[T].
};

/// @brief Compute source V-trace corrected values and policy-gradient advantages.
///
/// Uses rho_bar=c_bar=1, rho=min(1,exp(logp_pi-logp_behavior)) with Float32 source ordering.
/// Corrected values recurse backward; advantages use the next corrected value. Extreme finite log
/// ratios may overflow exp to infinity before min clips it to1, as in the source. No RNG, autograd
/// host path, actor scheduling, FIFO queue, parameter synchronization or learner loop is hidden
/// here.
///
/// @param logp_pi Finite learner chosen-action log-probabilities[T].
/// @param value Finite pre-step/bootstrap values[T+1].
/// @param rewards Finite rewards[T], T at most16million, possibly empty.
/// @param done Binary actual-terminal flags[T].
/// @param logp_behavior Finite recorded behavior chosen-action log-probabilities[T].
/// @param gamma Finite discount in[0,1].
/// @return Owning corrected values[T+1] and advantages[T]; inputs unchanged.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// auto
/// target=dm::rl::off_policy::VTrace(std::vector<float>{0},std::vector<float>{0,0},std::vector<float>{1},std::vector<float>{1},std::vector<float>{0},.9f);
/// @endcode
/// @par References
/// - Espeholt et al. (2018), IMPALA, https://arxiv.org/abs/1802.01561 .
/// @see Retrace()
inline VTracePath VTrace(
    std::span<const float> logp_pi,
    std::span<const float> value,
    std::span<const float> rewards,
    std::span<const float> done,
    std::span<const float> logp_behavior,
    float                  gamma) {
  detail::Sequence(rewards, done, gamma);
  int T = int(rewards.size());
  if (logp_pi.size() != size_t(T) || logp_behavior.size() != size_t(T) ||
      value.size() != size_t(T) + 1)
    throw std::invalid_argument("aligned V-trace log-probability/value geometry required");
  detail::Finite(logp_pi);
  detail::Finite(logp_behavior);
  detail::Finite(value);
  VTracePath path{std::vector<float>(T + 1), std::vector<float>(T)};
  path.values[T] = value[T];
  for (int t = T - 1; t >= 0; --t) {
    float rho = std::min(1.f, std::exp(logp_pi[t] - logp_behavior[t])), c = rho, cont = 1 - done[t];
    float delta    = rho * (rewards[t] + gamma * cont * value[t + 1] - value[t]);
    path.values[t] = value[t] + delta + gamma * cont * c * (path.values[t + 1] - value[t + 1]);
  }
  for (int t = 0; t < T; ++t) {
    float rho = std::min(1.f, std::exp(logp_pi[t] - logp_behavior[t])), cont = 1 - done[t];
    path.advantages[t] = rho * (rewards[t] + gamma * cont * path.values[t + 1] - value[t]);
  }
  return path;
}
} // namespace datamunge::rl::off_policy
