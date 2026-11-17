#pragma once
/// @file returns.hpp
/// @brief Source-order host return/GAE calculations and explicit env-major rollout data.
#include <cmath>
#include <datamunge/rl/environments/interface.hpp>
/// @brief Explicit source-order target calculations and owning env-major rollout data.
namespace datamunge::rl::returns {
/// @brief Internal return-sequence validation.
namespace return_detail {

/// @brief Validate finite aligned rewards, binary terminality and discount.
///
/// @param rewards At most16million finite Float32 entries, possibly empty.
/// @param done Equal-length binary terminal flags.
/// @param gamma Finite discount in[0,1].
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par References
/// - Sutton and Barto (2018), Chapter13, http://incompleteideas.net/book/the-book-2nd.html;
/// Schulman et al. (2016), Generalized Advantage Estimation, https://arxiv.org/abs/1506.02438 .
inline void
Trajectory(const std::vector<float>& rewards, const std::vector<float>& done, float gamma) {
  if (rewards.size() != done.size() || rewards.size() > 16000000 || !std::isfinite(gamma) ||
      gamma < 0 || gamma > 1)
    throw std::invalid_argument(
        "aligned bounded rewards/terminality and discount in[0,1] required");
  for (size_t i = 0; i < rewards.size(); ++i)
    if (!std::isfinite(rewards[i]) || (done[i] != 0 && done[i] != 1))
      throw std::invalid_argument("finite rewards and binary terminal flags required");
}
} // namespace return_detail

/// @brief Compute source-rounded backward discounted returns with an explicit bootstrap.
///
/// No autograd path to host vectors. Inputs unchanged. Terminality resets the backward recurrence;
/// callers must split episodes/provide correct final-state bootstraps at time limits.
///
/// @param r Finite Float32 rewards[T], T at most16million, possibly empty.
/// @param done Aligned binary terminal flags; time-limit truncation uses0.
/// @param gamma Finite discount in[0,1].
/// @param bootstrap Finite state value after the final transition, default0.
/// @return Independent Float32 returns[T]; G_t=r_t+gamma*(1-done_t)*G_(t+1).
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// auto values=dm::rl::returns::DiscountedReturns({1,1,1},{0,0,1},.5f);
/// @endcode
/// @par References
/// - Sutton and Barto (2018), Chapter13, http://incompleteideas.net/book/the-book-2nd.html;
/// Schulman et al. (2016), Generalized Advantage Estimation, https://arxiv.org/abs/1506.02438 .
/// @see Gae()
inline std::vector<float> DiscountedReturns(
    const std::vector<float>& r,
    const std::vector<float>& done,
    float                     gamma,
    float                     bootstrap = 0.f) {
  return_detail::Trajectory(r, done, gamma);
  if (!std::isfinite(bootstrap))
    throw std::invalid_argument("finite bootstrap required");
  std::vector<float> g(r.size());
  float              next = bootstrap;
  for (int t = static_cast<int>(r.size()) - 1; t >= 0; --t) {
    next = r[t] + gamma * (1 - done[t]) * next;
    g[t] = next;
  }
  return g;
}

// Generalized advantage estimation (Schulman et al. 2016) for one
// trajectory: delta_t = r_t + gamma (1 - done_t) V(s_{t+1}) - V(s_t),
// A_t = delta_t + gamma lambda (1 - done_t) A_{t+1}; returns = A + V.

/// @brief Compute source Float32 generalized advantages and bootstrapped return targets.
///
/// delta_t=r_t+gamma*(1-done_t)*v_next-v_t; A_t=delta_t+gamma*lam*(1-done_t)*A_(t+1).
/// Returns are A+v. Validation occurs before output assignment; outputs are resized to T.
/// Host inputs have no autograd path; callers control trajectory splitting and stopped target
/// conversion. A truncation uses done0 but still requires its actual final-state bootstrap, not an
/// auto-reset state.
///
/// @param r Finite Float32 rewards[T], possibly empty.
/// @param v Aligned finite pre-step values[T].
/// @param done Aligned binary actual-terminal flags[T].
/// @param next_v Finite bootstrap value after the final transition.
/// @param gamma Finite discount in[0,1].
/// @param lam Finite trace parameter in[0,1].
/// @param adv Non-null output vector, distinct from ret and all input vectors.
/// @param ret Non-null output vector, distinct from adv and all inputs.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// std::vector<float>adv,ret;dm::rl::returns::Gae({1,1},{.2f,.3f},{0,1},0,.9f,.95f,&adv,&ret);
/// @endcode
/// @par References
/// - Sutton and Barto (2018), Chapter13, http://incompleteideas.net/book/the-book-2nd.html;
/// Schulman et al. (2016), Generalized Advantage Estimation, https://arxiv.org/abs/1506.02438 .
/// @see DiscountedReturns()
inline void Gae(
    const std::vector<float>& r,
    const std::vector<float>& v,
    const std::vector<float>& done,
    float                     next_v,
    float                     gamma,
    float                     lam,
    std::vector<float>*       adv,
    std::vector<float>*       ret) {
  return_detail::Trajectory(r, done, gamma);
  if (v.size() != r.size() || !std::isfinite(next_v) || !std::isfinite(lam) || lam < 0 || lam > 1 ||
      !adv || !ret || adv == ret || adv == &r || adv == &v || adv == &done || ret == &r ||
      ret == &v || ret == &done)
    throw std::invalid_argument("aligned finite GAE values and distinct output vectors required");
  for (float x : v)
    if (!std::isfinite(x))
      throw std::invalid_argument("finite GAE values required");
  const int T = static_cast<int>(r.size());
  adv->assign(T, 0.f);
  ret->assign(T, 0.f);
  float next_adv = 0;
  for (int t = T - 1; t >= 0; --t) {
    const float v_next = t + 1 < T ? v[t + 1] : next_v;
    const float delta  = r[t] + gamma * (1 - done[t]) * v_next - v[t];
    next_adv           = delta + gamma * lam * (1 - done[t]) * next_adv;
    (*adv)[t]          = next_adv;
    (*ret)[t]          = next_adv + v[t];
  }
}

/// @brief Owning env-major trajectory data; the caller supplies and collects every field.
/// @details No sampler or collection loop. Flat methods concatenate stored rows as-is, without
/// validating descriptors or interpreting terminal/time-limit boundaries. Host copies are
/// independent.
struct Rollout {
  int n_envs  = 0; ///< Caller-provided environment count descriptor.
  int n_steps = 0; ///< Caller-provided per-environment horizon descriptor.
  std::vector<std::vector<environments::Obs>> obs; ///< Owning observations[env][t][feature].
  std::vector<std::vector<int>>               act; ///< Owning actions[env][t].
  std::vector<std::vector<float>>             rew; ///< Owning rewards[env][t].
  std::vector<std::vector<float>>
      done; ///< Owning terminal flags[env][t]; truncation convention is caller-controlled.
  std::vector<std::vector<float>> val;      ///< Owning pre-step value estimates[env][t].
  std::vector<std::vector<float>> logp;     ///< Owning recorded action log-probabilities[env][t].
  std::vector<environments::Obs>  last_obs; ///< Caller-supplied post-rollout observations[env].
  std::vector<float>              last_val; ///< Caller-supplied post-rollout bootstrap values[env].
  // Flatten in env-major order.

  /// @brief Copy stored observation rows in env-major order.
  ///
  /// @return Owning concatenation; raggedness is preserved for downstream validation.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// dm::rl::returns::Rollout ro;ro.obs={{{1},{2}},{{3}}};auto rows=ro.FlatObs();
  /// @endcode
  /// @par References
  /// - Sutton and Barto (2018), Chapter13, http://incompleteideas.net/book/the-book-2nd.html;
  /// Schulman et al. (2016), Generalized Advantage Estimation, https://arxiv.org/abs/1506.02438 .
  /// @see FlatAct()
  std::vector<environments::Obs> FlatObs() const {
    std::vector<environments::Obs> o;
    for (const auto& e : obs)
      o.insert(o.end(), e.begin(), e.end());
    return o;
  }

  /// @brief Copy stored action rows in env-major order.
  ///
  /// @return Owning concatenation; no action range validation.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// dm::rl::returns::Rollout ro;ro.act={{0,1},{2}};auto actions=ro.FlatAct();
  /// @endcode
  /// @par References
  /// - Sutton and Barto (2018), Chapter13, http://incompleteideas.net/book/the-book-2nd.html;
  /// Schulman et al. (2016), Generalized Advantage Estimation, https://arxiv.org/abs/1506.02438 .
  /// @see FlatObs()
  std::vector<int> FlatAct() const {
    std::vector<int> a;
    for (const auto& e : act)
      a.insert(a.end(), e.begin(), e.end());
    return a;
  }
};

} // namespace datamunge::rl::returns
