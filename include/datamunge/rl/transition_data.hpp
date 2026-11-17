#pragma once
/// @file transition_data.hpp
/// @brief Append-only owning continuous-transition data and source modulo native minibatches.
#include <cmath>
#include <datamunge/nn/tensor.hpp>
#include <datamunge/rl/environments/interface.hpp>
#include <random>
namespace datamunge::rl::planning {
/// @brief Owning Float32 CPU minibatch endpoints sampled with replacement.
///
/// @par Example
/// @code
/// dm::rl::planning::TransitionBatch batch;
/// @endcode
/// @see TransitionSet
struct TransitionBatch {
  nn::Tensor state;      ///< Sampled observations[B,S].
  nn::Tensor action;     ///< Sampled continuous actions[B,A].
  nn::Tensor next_state; ///< Sampled successor observations[B,S].
  nn::Tensor reward;     ///< Sampled rewards[B].
};
/// @brief Append owning continuous transitions and sample source-compatible minibatches.
///
/// Copies own their host rows. Add and Sample require external synchronization on a shared
/// instance; Sample mutates the caller random engine. No optimizer or experiment is owned.
///
/// @par Example
/// @code
/// dm::rl::planning::TransitionSet data;data.Add({1},{.5},{2},3);
/// std::mt19937_64 rng(0);auto batch=data.Sample(2,rng);
/// @endcode
/// @see TransitionBatch
class TransitionSet {
 private:
  std::vector<environments::Obs>  s;  ///< Owned initial observations.
  std::vector<environments::Obs>  s2; ///< Owned successor observations.
  std::vector<std::vector<float>> a;  ///< Owned continuous actions.
  std::vector<float>              r;  ///< Owned scalar rewards.
 public:
  /// @brief Copy one finite transition after validating consistent bounded geometry.
  ///
  /// At most one million rows and64 million row*(observation+action) coordinates.
  /// Inconsistent/nonfinite input throws std::invalid_argument before rows are appended. Allocation
  /// failures propagate.
  ///
  /// @param st Nonempty observation, width at most one million; copied.
  /// @param at Nonempty action, width at most one million; copied.
  /// @param st2 Successor with the same width as st; copied.
  /// @param rt Finite scalar reward.
  /// @par Example
  /// @code
  /// dm::rl::planning::TransitionSet data;data.Add({0},{.5},{1},2);
  /// @endcode
  /// @see Sample()
  void Add(
      const environments::Obs&  st,
      const std::vector<float>& at,
      const environments::Obs&  st2,
      float                     rt) {
    if (st.empty() || at.empty() || st.size() != st2.size() || st.size() > 1000000 ||
        at.size() > 1000000 || r.size() >= 1000000 ||
        int64_t(r.size() + 1) * (st.size() + at.size()) > 64000000 ||
        (!s.empty() && (st.size() != s[0].size() || at.size() != a[0].size())))
      throw std::invalid_argument("bounded consistent transition geometry required");
    for (const auto& row : {st, at, st2})
      for (float v : row)
        if (!std::isfinite(v))
          throw std::invalid_argument("finite transition observations/actions required");
    if (!std::isfinite(rt))
      throw std::invalid_argument("finite reward required");
    s.push_back(st), a.push_back(at), s2.push_back(st2), r.push_back(rt);
  }
  /// @brief Count stored transitions.
  ///
  /// @return Number of rows, between0 and one million.
  /// @see Add()
  int size() const { return static_cast<int>(r.size()); }
  /// @brief Draw an owning native minibatch with replacement using raw engine modulo.
  ///
  /// Throws std::invalid_argument for an empty dataset or invalid batch geometry. Modulo sampling
  /// deliberately preserves the original example stream rather than changing to a distribution
  /// adapter.
  ///
  /// @param n Batch size1..1000000; n*(observation+action) at most64 million.
  /// @param rng Borrowed engine; exactly n draws consumed.
  /// @return New independent Float32 CPU tensors in state/action/next_state/reward order.
  /// @par Example
  /// @code
  /// dm::rl::planning::TransitionSet data;data.Add({0},{1},{2},3);
  /// std::mt19937_64 rng(0);auto batch=data.Sample(1,rng);
  /// @endcode
  /// @see Add()
  TransitionBatch Sample(int n, std::mt19937_64& rng) const {
    if (n < 1 || n > 1000000 || r.empty() || int64_t(n) * (s[0].size() + a[0].size()) > 64000000)
      throw std::invalid_argument("positive bounded minibatch and nonempty transitions required");
    namespace nn      = datamunge::nn;
    const int obs_dim = static_cast<int>(s[0].size()), act_dim = static_cast<int>(a[0].size());
    std::vector<float> vs(static_cast<size_t>(n) * obs_dim), vs2(vs.size()),
        va(static_cast<size_t>(n) * act_dim), vr(n);
    for (int i = 0; i < n; ++i) {
      const int k = static_cast<int>(rng() % r.size());
      for (int j = 0; j < obs_dim; ++j)
        vs[i * obs_dim + j] = s[k][j], vs2[i * obs_dim + j] = s2[k][j];
      for (int j = 0; j < act_dim; ++j)
        va[i * act_dim + j] = a[k][j];
      vr[i] = r[k];
    }
    auto S = nn::from_vector(vs, {n, obs_dim}), S2 = nn::from_vector(vs2, {n, obs_dim}),
         A = nn::from_vector(va, {n, act_dim}), R = nn::from_vector(vr, {n});
    return {S, A, S2, R};
  }
};

} // namespace datamunge::rl::planning
