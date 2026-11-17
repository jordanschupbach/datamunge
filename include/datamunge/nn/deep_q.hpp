#pragma once
/// @file deep_q.hpp
/// @brief Explicit native DQN-family forward graphs and categorical Bellman projection.
#include <cmath>
#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/ops.hpp>
#include <limits>
namespace datamunge::nn::models {
/// @brief Internal shape validation for DQN-family native forward graphs.
namespace deep_q_detail {

/// @brief Validate the source Float32 CPU batch and bounded layer widths.
///
/// @param x Defined nonempty Float32 CPU [B,D].
/// @param hidden Hidden width in[1,4096].
/// @param actions Action count in[1,4096].
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par References
/// - Mnih et al. (2015), Human-level control through deep reinforcement learning,
/// https://doi.org/10.1038/nature14236; Hessel et al. (2018), Rainbow,
/// https://arxiv.org/abs/1710.02298.
inline void Input(const Tensor& x, int hidden, int actions) {
  if (!x.defined() || x.device().backend != compute::Backend::CPU || x.dtype() != DType::Float32 ||
      x.dim() != 2 || x.size(0) < 1 || x.size(1) < 1 || hidden < 1 || hidden > 4096 ||
      actions < 1 || actions > 4096)
    throw std::invalid_argument(
        "Float32 CPU [B,D] and positive bounded hidden/action widths required");
}
} // namespace deep_q_detail

/// @brief Build the source two-ReLU-hidden-layer action-value graph.
///
/// This call may create named parameters and advance the store initializer; subsequent calls reuse
/// them. Inputs and existing leaves are not updated. Native first/higher primitive derivative paths
/// are retained. There is no optimizer, target synchronization, objective or training/episode loop;
/// callers own each. Geometry must remain consistent when reusing a ParameterStore. Mnih et
/// al.2015.
///
/// @param p Caller-owned ParameterStore: missing named leaves are initialized on first use in
/// source order; existing leaves are reused.
/// @param x Nonempty Float32 CPU batch [B,D]; input values are not scanned for finiteness.
/// @param hidden Hidden width in[1,4096], source default128.
/// @param actions Action count in[1,4096], CartPole preset2.
/// @return Live Q[B,A] with mlp then q names; source4/128/128/2 has17410 parameters.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// dm::nn::ParameterStore p(0);auto q=dm::nn::models::DeepQ(p,dm::nn::zeros({2,4}),128,2);
/// @endcode
/// @par References
/// - Mnih et al. (2015), Human-level control through deep reinforcement learning,
/// https://doi.org/10.1038/nature14236; Hessel et al. (2018), Rainbow,
/// https://arxiv.org/abs/1710.02298.
/// @see DeepQ(), DuelingQ(), CategoricalQ(), QuantileQ(), ImplicitQuantileQ(), NoisyQ(), RainbowQ()
inline Tensor DeepQ(ParameterStore& p, const Tensor& x, int hidden, int actions) {
  deep_q_detail::Input(x, hidden, actions);
  auto h = mlp(p, "mlp", x, {hidden, hidden});
  return dense(p, "q", h, actions);
}
/// @brief Live endpoints of mean-centered dueling action values.
struct DuelingQPath {
  Tensor q;         ///< Q[B,A]=value+centered advantage.
  Tensor value;     ///< State values[B,1].
  Tensor advantage; ///< Centered advantages[B,A], mean zero over actions.
};

/// @brief Build the source shared-feature dueling action-value graph.
///
/// This call may create named parameters and advance the store initializer; subsequent calls reuse
/// them. Inputs and existing leaves are not updated. Native first/higher primitive derivative paths
/// are retained. There is no optimizer, target synchronization, objective or training/episode loop;
/// callers own each. Geometry must remain consistent when reusing a ParameterStore. Wang et
/// al.2016, https://arxiv.org/abs/1511.06581 .
///
/// @param p Caller-owned ParameterStore: missing named leaves are initialized on first use in
/// source order; existing leaves are reused.
/// @param x Nonempty Float32 CPU batch [B,D]; input values are not scanned for finiteness.
/// @param hidden Hidden width in[1,4096], source default128.
/// @param actions Action count in[1,4096], CartPole preset2.
/// @return Live Q[B,A], value[B,1] and centered advantage[B,A].
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// dm::nn::ParameterStore p(0);auto path=dm::nn::models::DuelingQ(p,dm::nn::zeros({2,4}),128,2);
/// @endcode
/// @par References
/// - Mnih et al. (2015), Human-level control through deep reinforcement learning,
/// https://doi.org/10.1038/nature14236; Hessel et al. (2018), Rainbow,
/// https://arxiv.org/abs/1710.02298.
/// @see DeepQ(), DuelingQ(), CategoricalQ(), QuantileQ(), ImplicitQuantileQ(), NoisyQ(), RainbowQ()
inline DuelingQPath DuelingQ(ParameterStore& p, const Tensor& x, int hidden, int actions) {
  deep_q_detail::Input(x, hidden, actions);
  auto h     = mlp(p, "mlp", x, {hidden});
  auto value = dense(p, "value", dense(p, "value_hidden", h, hidden, Activation::Relu), 1);
  auto adv =
      dense(p, "advantage", dense(p, "advantage_hidden", h, hidden, Activation::Relu), actions);
  auto centered = adv - mean(adv, {1}, true);
  return {value + centered, value, centered};
}
/// @brief Live categorical return-distribution endpoints.
struct CategoricalQPath {
  Tensor logits; ///< Unnormalized scores[B,A,Z].
  Tensor probs;  ///< Normalized probabilities[B,A,Z].
  Tensor q;      ///< Expected return[B,A] using supplied support.
};

/// @brief Build a source categorical return-distribution graph.
///
/// This call may create named parameters and advance the store initializer; subsequent calls reuse
/// them. Inputs and existing leaves are not updated. Native first/higher primitive derivative paths
/// are retained. There is no optimizer, target synchronization, objective or training/episode loop;
/// callers own each. Geometry must remain consistent when reusing a ParameterStore. Bellemare et
/// al.2017, https://arxiv.org/abs/1707.06887 .
///
/// @param p Caller-owned ParameterStore: missing named leaves are initialized on first use in
/// source order; existing leaves are reused.
/// @param x Nonempty Float32 CPU batch [B,D]; input values are not scanned for finiteness.
/// @param hidden Hidden width in[1,4096], source default128.
/// @param actions Action count in[1,4096], CartPole preset2.
/// @param support Float32 CPU atom values [Z], Z in[2,4096]; support values are caller-defined.
/// @return Live logits/probs[B,A,Z] and expectationQ[B,A].
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// dm::nn::ParameterStore p(0);auto
/// path=dm::nn::models::CategoricalQ(p,dm::nn::zeros({2,4}),128,2,dm::nn::from_vector(std::vector<float>{0,1,2}));
/// @endcode
/// @par References
/// - Mnih et al. (2015), Human-level control through deep reinforcement learning,
/// https://doi.org/10.1038/nature14236; Hessel et al. (2018), Rainbow,
/// https://arxiv.org/abs/1710.02298.
/// @see DeepQ(), DuelingQ(), CategoricalQ(), QuantileQ(), ImplicitQuantileQ(), NoisyQ(), RainbowQ()
inline CategoricalQPath
CategoricalQ(ParameterStore& p, const Tensor& x, int hidden, int actions, const Tensor& support) {
  deep_q_detail::Input(x, hidden, actions);
  if (!support.defined() || support.dim() != 1 || support.size(0) < 2 || support.size(0) > 4096 ||
      support.dtype() != DType::Float32 || support.device().backend != compute::Backend::CPU)
    throw std::invalid_argument("Float32 CPU atom support [Z] required");
  auto h = mlp(p, "mlp", x, {hidden, hidden});
  auto logits =
      reshape(dense(p, "logits", h, actions * support.size(0)), {-1, actions, support.size(0)});
  auto probs = softmax(logits, -1);
  return {logits, probs, sum(probs * support, {2})};
}
/// @brief Live fixed/implicit-quantile endpoints; theta axis order follows the builder.
struct QuantileQPath {
  Tensor theta; ///< Quantiles[B,A,N] for QuantileQ(), or[B,N,A] for ImplicitQuantileQ().
  Tensor q;     ///< Mean return per action[B,A].
};

/// @brief Build a source fixed-quantile action-value graph.
///
/// This call may create named parameters and advance the store initializer; subsequent calls reuse
/// them. Inputs and existing leaves are not updated. Native first/higher primitive derivative paths
/// are retained. There is no optimizer, target synchronization, objective or training/episode loop;
/// callers own each. Geometry must remain consistent when reusing a ParameterStore. Dabney et
/// al.2018, https://arxiv.org/abs/1710.10044 .
///
/// @param p Caller-owned ParameterStore: missing named leaves are initialized on first use in
/// source order; existing leaves are reused.
/// @param x Nonempty Float32 CPU batch [B,D]; input values are not scanned for finiteness.
/// @param hidden Hidden width in[1,4096], source default128.
/// @param actions Action count in[1,4096], CartPole preset2.
/// @param quantiles Number N in[1,4096].
/// @return Live theta[B,A,N] and meanQ[B,A]; quantiles are not sorted.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// dm::nn::ParameterStore p(0);auto path=dm::nn::models::QuantileQ(p,dm::nn::zeros({2,4}),128,2,8);
/// @endcode
/// @par References
/// - Mnih et al. (2015), Human-level control through deep reinforcement learning,
/// https://doi.org/10.1038/nature14236; Hessel et al. (2018), Rainbow,
/// https://arxiv.org/abs/1710.02298.
/// @see DeepQ(), DuelingQ(), CategoricalQ(), QuantileQ(), ImplicitQuantileQ(), NoisyQ(), RainbowQ()
inline QuantileQPath
QuantileQ(ParameterStore& p, const Tensor& x, int hidden, int actions, int quantiles) {
  deep_q_detail::Input(x, hidden, actions);
  if (quantiles < 1 || quantiles > 4096)
    throw std::invalid_argument("positive bounded quantile count required");
  auto h     = mlp(p, "mlp", x, {hidden, hidden});
  auto theta = reshape(dense(p, "theta", h, actions * quantiles), {-1, actions, quantiles});
  return {theta, mean(theta, {2})};
}

/// @brief Build the source cosine-conditioned implicit quantile graph.
///
/// This call may create named parameters and advance the store initializer; subsequent calls reuse
/// them. Inputs and existing leaves are not updated. Native first/higher primitive derivative paths
/// are retained. There is no optimizer, target synchronization, objective or training/episode loop;
/// callers own each. Geometry must remain consistent when reusing a ParameterStore. Dabney et
/// al.2018, https://arxiv.org/abs/1806.06923 .
///
/// @param p Caller-owned ParameterStore: missing named leaves are initialized on first use in
/// source order; existing leaves are reused.
/// @param x Nonempty Float32 CPU batch [B,D]; input values are not scanned for finiteness.
/// @param hidden Hidden width in[1,4096], source default128.
/// @param actions Action count in[1,4096], CartPole preset2.
/// @param taus Float32 CPU fractions [B,N], N positive; caller samples them and supplies
/// intended[0,1] values.
/// @param frequencies Float32 CPU [1,1,E], E positive; caller supplies source Float32 pi*i values.
/// @return Live theta[B,N,A] and meanQ[B,A]; unlike QuantileQ(), action is the final theta axis.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// dm::nn::ParameterStore p(0);auto
/// path=dm::nn::models::ImplicitQuantileQ(p,dm::nn::zeros({2,4}),dm::nn::zeros({2,8}),128,2,dm::nn::ones({1,1,64}));
/// @endcode
/// @par References
/// - Mnih et al. (2015), Human-level control through deep reinforcement learning,
/// https://doi.org/10.1038/nature14236; Hessel et al. (2018), Rainbow,
/// https://arxiv.org/abs/1710.02298.
/// @see DeepQ(), DuelingQ(), CategoricalQ(), QuantileQ(), ImplicitQuantileQ(), NoisyQ(), RainbowQ()
inline QuantileQPath ImplicitQuantileQ(
    ParameterStore& p,
    const Tensor&   x,
    const Tensor&   taus,
    int             hidden,
    int             actions,
    const Tensor&   frequencies) {
  deep_q_detail::Input(x, hidden, actions);
  if (!taus.defined() || taus.dim() != 2 || taus.size(0) != x.size(0) || taus.size(1) < 1 ||
      taus.dtype() != x.dtype() || taus.device().backend != compute::Backend::CPU || !frequencies.defined() ||
      frequencies.dim() != 3 || frequencies.size(0) != 1 || frequencies.size(1) != 1 ||
      frequencies.size(2) < 1 || frequencies.dtype() != x.dtype() ||
      frequencies.device().backend != compute::Backend::CPU)
    throw std::invalid_argument("aligned taus [B,N] and frequencies [1,1,E] required");
  auto psi     = mlp(p, "psi", x, {hidden});
  auto cosines = cos(expand_dims(taus, 2) * frequencies);
  auto phi     = dense(p, "phi", cosines, hidden, Activation::Relu);
  auto zf      = expand_dims(psi, 1) * phi;
  auto h       = dense(p, "head_hidden", zf, hidden, Activation::Relu);
  auto theta   = dense(p, "theta", h, actions);
  return {theta, mean(theta, {1})};
}

/// @brief Build the source three-factorized-noisy-layer Q graph.
///
/// This call may create named parameters and advance the store initializer; subsequent calls reuse
/// them. Inputs and existing leaves are not updated. Native first/higher primitive derivative paths
/// are retained. There is no optimizer, target synchronization, objective or training/episode loop;
/// callers own each. Geometry must remain consistent when reusing a ParameterStore. Fortunato et
/// al.2018, https://arxiv.org/abs/1706.10295 .
///
/// @param p Caller-owned ParameterStore: missing named leaves are initialized on first use in
/// source order; existing leaves are reused.
/// @param x Nonempty Float32 CPU batch [B,D]; input values are not scanned for finiteness.
/// @param hidden Hidden width in[1,4096], source default128.
/// @param actions Action count in[1,4096], CartPole preset2.
/// @param noise Caller generator, advanced by native noisy_dense calls, including their scale0
/// behavior.
/// @param scale Finite nonnegative noise multiplier;0 requests deterministic evaluation.
/// @return Live Q[B,A] through caller mu/sigma leaves; all noise draws retain source layer
/// ordering.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// dm::nn::ParameterStore p(0);dm::nn::Generator noise(12345);auto
/// q=dm::nn::models::NoisyQ(p,dm::nn::zeros({2,4}),128,2,noise,0);
/// @endcode
/// @par References
/// - Mnih et al. (2015), Human-level control through deep reinforcement learning,
/// https://doi.org/10.1038/nature14236; Hessel et al. (2018), Rainbow,
/// https://arxiv.org/abs/1710.02298.
/// @see DeepQ(), DuelingQ(), CategoricalQ(), QuantileQ(), ImplicitQuantileQ(), NoisyQ(), RainbowQ()
inline Tensor NoisyQ(
    ParameterStore& p,
    const Tensor&   x,
    int             hidden,
    int             actions,
    Generator&      noise,
    double          scale) {
  deep_q_detail::Input(x, hidden, actions);
  if (!std::isfinite(scale) || scale < 0)
    throw std::invalid_argument("nonnegative finite noise scale required");
  auto h1 = noisy_dense(p, "noisy1", x, hidden, noise, scale, Activation::Relu);
  auto h2 = noisy_dense(p, "noisy2", h1, hidden, noise, scale, Activation::Relu);
  return noisy_dense(p, "noisy_q", h2, actions, noise, scale);
}

/// @brief Build the source noisy dueling categorical return graph.
///
/// This call may create named parameters and advance the store initializer; subsequent calls reuse
/// them. Inputs and existing leaves are not updated. Native first/higher primitive derivative paths
/// are retained. There is no optimizer, target synchronization, objective or training/episode loop;
/// callers own each. Geometry must remain consistent when reusing a ParameterStore. Hessel et
/// al.2018, https://arxiv.org/abs/1710.02298 .
///
/// @param p Caller-owned ParameterStore: missing named leaves are initialized on first use in
/// source order; existing leaves are reused.
/// @param x Nonempty Float32 CPU batch [B,D]; input values are not scanned for finiteness.
/// @param hidden Hidden width in[1,4096], source default128.
/// @param actions Action count in[1,4096], CartPole preset2.
/// @param support Float32 CPU [Z], Z in[2,4096]; caller-defined atom values.
/// @param noise Caller generator; value-hidden,adv-hidden,value,advantage draws in original order.
/// @param scale Finite nonnegative noise multiplier.
/// @return Live logits/probs[B,A,Z] and meanQ[B,A].
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// dm::nn::ParameterStore p(0);dm::nn::Generator noise(12345);auto
/// path=dm::nn::models::RainbowQ(p,dm::nn::zeros({2,4}),128,2,dm::nn::from_vector(std::vector<float>{0,1,2}),noise,0);
/// @endcode
/// @par References
/// - Mnih et al. (2015), Human-level control through deep reinforcement learning,
/// https://doi.org/10.1038/nature14236; Hessel et al. (2018), Rainbow,
/// https://arxiv.org/abs/1710.02298.
/// @see DeepQ(), DuelingQ(), CategoricalQ(), QuantileQ(), ImplicitQuantileQ(), NoisyQ(), RainbowQ()
inline CategoricalQPath RainbowQ(
    ParameterStore& p,
    const Tensor&   x,
    int             hidden,
    int             actions,
    const Tensor&   support,
    Generator&      noise,
    double          scale) {
  deep_q_detail::Input(x, hidden, actions);
  if (!support.defined() || support.dim() != 1 || support.size(0) < 2 || support.size(0) > 4096 ||
      support.dtype() != DType::Float32 || support.device().backend != compute::Backend::CPU ||
      !std::isfinite(scale) || scale < 0)
    throw std::invalid_argument("Float32 CPU atom support and nonnegative noise scale required");
  int64_t atoms = support.size(0);
  auto    h     = mlp(p, "mlp", x, {hidden});
  auto    vh    = noisy_dense(p, "value_hidden", h, hidden, noise, scale, Activation::Relu);
  auto    ah    = noisy_dense(p, "adv_hidden", h, hidden, noise, scale, Activation::Relu);
  auto    value = reshape(noisy_dense(p, "value", vh, atoms, noise, scale), {-1, 1, atoms});
  auto    adv =
      reshape(noisy_dense(p, "advantage", ah, actions * atoms, noise, scale), {-1, actions, atoms});
  auto logits = value + (adv - mean(adv, {1}, true));
  auto probs  = softmax(logits, -1);
  return {logits, probs, sum(probs * support, {2})};
}

/// @brief Project a categorical Bellman target onto equally spaced Float32 support atoms.
///
/// Host-only diagnostic/target transformation: no autograd path or retained buffers.
/// Clipped r+gamma*(1-done)*z mass is split by linear interpolation; exact atoms receive all mass.
/// Validation precedes writes. Integer atom indices are clamped to prevent roundoff beyond support
/// endpoints.
///
/// @param probs_next Borrowed normalized Float32[B,Z] next-action distributions, entries in[0,1],
/// mass tolerance1e-5.
/// @param r Borrowed finite Float32[B] rewards.
/// @param done Borrowed binary Float32[B] terminal flags; truncation must use0.
/// @param B Batch count in[1,1000000].
/// @param atoms Atom count Z in[2,4096]; B*Z at most64million.
/// @param vmin Finite lower support bound.
/// @param vmax Finite upper support bound, strictly greater; finite positive representable atom
/// spacing required.
/// @param gamma Finite discount in[0,1], or caller-computed gamma^n for n-step targets.
/// @param out Writable Float32[B,Z], nonoverlapping input storage; cleared before source-ordered
/// accumulation.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// float
/// p[]={.5f,.5f},r[]={.5f},done[]={1},out[2];dm::nn::models::ProjectDistribution(p,r,done,1,2,0,1,.9f,out);
/// @endcode
/// @par References
/// - Bellemare et al.2017, https://arxiv.org/abs/1707.06887 .
/// @see CategoricalQ(), RainbowQ()
inline void ProjectDistribution(
    const float* probs_next,
    const float* r,
    const float* done,
    int          B,
    int          atoms,
    float        vmin,
    float        vmax,
    float        gamma,
    float*       out) {
  if (!probs_next || !r || !done || !out || B < 1 || atoms < 2 || B > 1000000 || atoms > 4096 ||
      int64_t(B) * atoms > 64000000 || !std::isfinite(vmin) || !std::isfinite(vmax) ||
      vmax <= vmin || !std::isfinite(vmax - vmin) || !std::isfinite(gamma) || gamma < 0 ||
      gamma > 1)
    throw std::invalid_argument("valid categorical projection geometry/support/discount required");
  for (int i = 0; i < B; ++i) {
    if (!std::isfinite(r[i]) || (done[i] != 0 && done[i] != 1))
      throw std::invalid_argument("finite rewards and binary terminal flags required");
    double mass = 0;
    for (int j = 0; j < atoms; ++j) {
      float value = probs_next[i * atoms + j];
      if (!std::isfinite(value) || value < 0 || value > 1)
        throw std::invalid_argument("valid return probabilities required");
      mass += value;
    }
    if (std::abs(mass - 1) > 1e-5)
      throw std::invalid_argument("unit probability row required");
  }

  const float dz = (vmax - vmin) / (atoms - 1);
  if (!std::isfinite(dz) || dz <= 0)
    throw std::invalid_argument("representable positive atom spacing required");
  std::fill(out, out + B * atoms, 0.f);
  for (int i = 0; i < B; ++i)
    for (int j = 0; j < atoms; ++j) {
      const float z  = vmin + j * dz;
      const float tz = std::min(vmax, std::max(vmin, r[i] + gamma * (1 - done[i]) * z));
      const float b  = (tz - vmin) / dz;
      const int   l  = std::clamp(static_cast<int>(std::floor(b)), 0, atoms - 1),
                u    = std::clamp(static_cast<int>(std::ceil(b)), 0, atoms - 1);
      const float p  = probs_next[i * atoms + j];
      if (l == u) {
        out[i * atoms + l] += p;
      } else {
        out[i * atoms + l] += p * (u - b);
        out[i * atoms + u] += p * (b - l);
      }
    }
}

} // namespace datamunge::nn::models
