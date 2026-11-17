#pragma once
/// @file energy_components.hpp
/// @brief Tiny exact binary energy graphs and continuous energy/retrieval components.
#include <cmath>
#include <datamunge/nn/attention.hpp>
#include <datamunge/nn/memory_networks.hpp>
#include <datamunge/nn/midpoint_uniform.hpp>
namespace datamunge::nn::models {
/// @brief Enumerate all binary rows with bit j of row b equal to (b >> j) & 1.
///
/// Inputs and returned handles own their tensor storage; live paths retain first and higher
/// derivatives where the primitive supports them. No optimizer, sampling cache or fitting loop.
/// Mathematical presets use Float32 CPU tensors. Caller validates observations and random banks.
///
/// @param bits Width in[1,12]; exponential enumeration is intentionally bounded.
/// @return Owning Float32 CPU[2^bits,bits] constant.
/// @throws std::invalid_argument Invalid explicit bounds, or incompatible tensor geometry/dtype.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;auto y=m::BinaryStates(3);
/// @endcode
/// @par References
/// - Smolensky (1986), Information Processing in Dynamical Systems: Foundations of Harmony Theory,
/// https://stanford.edu/~jlmcc/papers/PDP/Volume%201/Chap6_PDP86.pdf .
/// @see BinaryStates(), RbmFreeEnergy()
inline Tensor BinaryStates(int bits) {
  if (bits < 1 || bits > 12)
    throw std::invalid_argument("binary enumeration bits must be 1..12");
  std::vector<float> s(static_cast<std::size_t>((1 << bits) * bits));
  for (int b = 0; b < (1 << bits); ++b)
    for (int j = 0; j < bits; ++j)
      s[b * bits + j] = float((b >> j) & 1);
  return from_vector(s, {std::int64_t(1) << bits, bits});
}

/// @brief Create a tiny Bernoulli RBM with normal(.1) weights and zero biases.
///
/// Inputs and returned handles own their tensor storage; live paths retain first and higher
/// derivatives where the primitive supports them. No optimizer, sampling cache or fitting loop.
/// Mathematical presets use Float32 CPU tensors. Caller validates observations and random banks.
///
/// @param p Caller-owned Float32 CPU parameter store; RNG advances on first creation only.
/// @param visible Visible width in[1,12].
/// @param hidden Hidden width in[1,12].
/// @throws std::invalid_argument Invalid explicit bounds, or incompatible tensor geometry/dtype.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(1);m::RbmParameters(p,8,4);
/// @endcode
/// @par References
/// - Smolensky (1986), Information Processing in Dynamical Systems: Foundations of Harmony Theory,
/// https://stanford.edu/~jlmcc/papers/PDP/Volume%201/Chap6_PDP86.pdf .
/// @see BinaryStates(), RbmFreeEnergy()
inline void RbmParameters(ParameterStore& p, int visible, int hidden) {
  if (visible < 1 || visible > 12 || hidden < 1 || hidden > 12)
    throw std::invalid_argument("tiny RBM widths must be 1..12");
  p.get("W", {visible, hidden}, init::normal(.1));
  p.get("visible_bias", {visible}, init::zeros());
  p.get("hidden_bias", {hidden}, init::zeros());
}
/// @brief Compute sigmoid(v W + hidden_bias).
///
/// Inputs and returned handles own their tensor storage; live paths retain first and higher
/// derivatives where the primitive supports them. No optimizer, sampling cache or fitting loop.
/// Mathematical presets use Float32 CPU tensors. Caller validates observations and random banks.
///
/// @param p Initialized RBM store.
/// @param v Float32 CPU[N,visible] observations.
/// @return Live CPU[N,hidden] conditional probabilities.
/// @throws std::invalid_argument Invalid explicit bounds, or incompatible tensor geometry/dtype.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(1);m::RbmParameters(p,8,4);auto
/// y=m::RbmHidden(p,n::zeros({2,8}));
/// @endcode
/// @par References
/// - Smolensky (1986), Information Processing in Dynamical Systems: Foundations of Harmony Theory,
/// https://stanford.edu/~jlmcc/papers/PDP/Volume%201/Chap6_PDP86.pdf .
/// @see BinaryStates(), RbmFreeEnergy()
inline Tensor RbmHidden(const ParameterStore& p, const Tensor& v) {
  return sigmoid(bias_add(matmul(v, p.at("W")), p.at("hidden_bias")));
}
/// @brief Compute sigmoid(h W^T + visible_bias).
///
/// Inputs and returned handles own their tensor storage; live paths retain first and higher
/// derivatives where the primitive supports them. No optimizer, sampling cache or fitting loop.
/// Mathematical presets use Float32 CPU tensors. Caller validates observations and random banks.
///
/// @param p Initialized RBM store.
/// @param h Float32 CPU[N,hidden] states.
/// @return Live CPU[N,visible] conditional probabilities.
/// @throws std::invalid_argument Invalid explicit bounds, or incompatible tensor geometry/dtype.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(1);m::RbmParameters(p,8,4);auto
/// y=m::RbmVisible(p,n::zeros({2,4}));
/// @endcode
/// @par References
/// - Smolensky (1986), Information Processing in Dynamical Systems: Foundations of Harmony Theory,
/// https://stanford.edu/~jlmcc/papers/PDP/Volume%201/Chap6_PDP86.pdf .
/// @see BinaryStates(), RbmFreeEnergy()
inline Tensor RbmVisible(const ParameterStore& p, const Tensor& h) {
  return sigmoid(bias_add(matmul(h, p.at("W"), false, true), p.at("visible_bias")));
}
/// @brief Sum out hidden Bernoulli states to obtain RBM free energy.
///
/// Inputs and returned handles own their tensor storage; live paths retain first and higher
/// derivatives where the primitive supports them. No optimizer, sampling cache or fitting loop.
/// Mathematical presets use Float32 CPU tensors. Caller validates observations and random banks.
///
/// @param p Initialized RBM store.
/// @param v Float32 CPU[N,visible] observations.
/// @return Live CPU[N] negative visible bias potential minus summed softplus.
/// @throws std::invalid_argument Invalid explicit bounds, or incompatible tensor geometry/dtype.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(1);m::RbmParameters(p,8,4);auto
/// y=m::RbmFreeEnergy(p,n::zeros({2,8}));
/// @endcode
/// @par References
/// - Smolensky (1986), Information Processing in Dynamical Systems: Foundations of Harmony Theory,
/// https://stanford.edu/~jlmcc/papers/PDP/Volume%201/Chap6_PDP86.pdf .
/// @see BinaryStates(), RbmFreeEnergy()
inline Tensor RbmFreeEnergy(const ParameterStore& p, const Tensor& v) {
  const Tensor hidden = bias_add(matmul(v, p.at("W")), p.at("hidden_bias"));
  return neg(add(sum(mul(v, p.at("visible_bias")), {1}), sum(softplus(hidden), {1})));
}
/// @brief Evaluate stable log-sum-exp with a stopped maximum.
///
/// Inputs and returned handles own their tensor storage; live paths retain first and higher
/// derivatives where the primitive supports them. No optimizer, sampling cache or fitting loop.
/// Mathematical presets use Float32 CPU tensors. Caller validates observations and random banks.
///
/// @param x Floating tensor with nonempty reduction dimension.
/// @param axis Valid positive or negative reduction axis.
/// @return Live floating tensor with the axis removed; first/higher derivatives retained.
/// @throws std::invalid_argument Invalid explicit bounds, or incompatible tensor geometry/dtype.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;auto y=m::LogSumExp(n::zeros({2,3}),1);
/// @endcode
/// @par References
/// - Smolensky (1986), Information Processing in Dynamical Systems: Foundations of Harmony Theory,
/// https://stanford.edu/~jlmcc/papers/PDP/Volume%201/Chap6_PDP86.pdf .
/// @see BinaryStates(), RbmFreeEnergy()
inline Tensor LogSumExp(const Tensor& x, std::int64_t axis) {
  const Tensor m = stop_gradient(max(x, {axis}, true));
  return squeeze(add(m, log(sum(exp(sub(x, m)), {axis}, true))), {axis});
}
/// @brief Compute an exact small-RBM partition by visible-state enumeration.
///
/// Inputs and returned handles own their tensor storage; live paths retain first and higher
/// derivatives where the primitive supports them. No optimizer, sampling cache or fitting loop.
/// Mathematical presets use Float32 CPU tensors. Caller validates observations and random banks.
///
/// @param p Initialized RBM store.
/// @param visible Visible width in[1,12], matching W.
/// @return Live scalar log Z; exponential work.
/// @throws std::invalid_argument Invalid explicit bounds, or incompatible tensor geometry/dtype.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(1);m::RbmParameters(p,8,4);auto
/// y=m::RbmLogPartition(p,8);
/// @endcode
/// @par References
/// - Smolensky (1986), Information Processing in Dynamical Systems: Foundations of Harmony Theory,
/// https://stanford.edu/~jlmcc/papers/PDP/Volume%201/Chap6_PDP86.pdf .
/// @see BinaryStates(), RbmFreeEnergy()
inline Tensor RbmLogPartition(const ParameterStore& p, int visible) {
  return LogSumExp(neg(RbmFreeEnergy(p, BinaryStates(visible))), 0);
}

/// @brief Own the live RbmGibbsPath mathematical endpoints.
///
/// Handles retain their storage and graph; no optimizer or mutable cache.
/// @par Example
/// @code
/// dm::nn::models::RbmGibbsPath path;
/// @endcode
/// @par References
/// - Hinton et al. (2006), https://doi.org/10.1162/neco.2006.18.7.1527 ;
/// Salakhutdinov and Hinton (2009), https://proceedings.mlr.press/v5/salakhutdinov09a.html ;
/// Ramsauer et al. (2021), https://arxiv.org/abs/2008.02217 .
/// @see RbmFreeEnergy(), ConfiningEnergy(), ModernHopfield()
struct RbmGibbsPath {
  Tensor hidden_probability;          ///< Conditional hidden probabilities.
  Tensor hidden;                      ///< Hidden samples or activations.
  Tensor visible_probability;         ///< Conditional visible probabilities.
  Tensor visible;                     ///< Sampled visible states.
  Tensor negative_hidden_probability; ///< Hidden conditional after the visible draw.
};
/// @brief Perform one sampled blocked RBM Gibbs sweep with explicit random banks.
///
/// Inputs and returned handles own their tensor storage; live paths retain first and higher
/// derivatives where the primitive supports them. No optimizer, sampling cache or fitting loop.
/// Mathematical presets use Float32 CPU tensors. Caller validates observations and random banks.
///
/// @param p Initialized RBM store.
/// @param v Float32 CPU[N,visible] starting states.
/// @param hidden_uniform Float32 CPU[N,hidden] caller uniforms in[0,1].
/// @param visible_uniform Float32 CPU[N,visible] caller uniforms in[0,1].
/// @return Live conditional probabilities and Float32 sampled bits; comparisons stop sampled paths.
/// @throws std::invalid_argument Invalid explicit bounds, or incompatible tensor geometry/dtype.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(1);m::RbmParameters(p,8,4);auto
/// y=m::RbmGibbs(p,n::zeros({2,8}),n::full({2,4},.5),n::full({2,8},.5));
/// @endcode
/// @par References
/// - Smolensky (1986), Information Processing in Dynamical Systems: Foundations of Harmony Theory,
/// https://stanford.edu/~jlmcc/papers/PDP/Volume%201/Chap6_PDP86.pdf .
/// @see BinaryStates(), RbmFreeEnergy()
inline RbmGibbsPath RbmGibbs(
    const ParameterStore& p,
    const Tensor&         v,
    const Tensor&         hidden_uniform,
    const Tensor&         visible_uniform) {
  const Tensor hp   = RbmHidden(p, v);
  const Tensor h    = cast(less(hidden_uniform, hp), DType::Float32);
  const Tensor vp   = RbmVisible(p, h);
  const Tensor next = cast(less(visible_uniform, vp), DType::Float32);
  return {hp, h, vp, next, RbmHidden(p, next)};
}

/// @brief Own the live DeepBeliefPath mathematical endpoints.
///
/// Handles retain their storage and graph; no optimizer or mutable cache.
/// @par Example
/// @code
/// dm::nn::models::DeepBeliefPath path;
/// @endcode
/// @par References
/// - Hinton et al. (2006), https://doi.org/10.1162/neco.2006.18.7.1527 ;
/// Salakhutdinov and Hinton (2009), https://proceedings.mlr.press/v5/salakhutdinov09a.html ;
/// Ramsauer et al. (2021), https://arxiv.org/abs/2008.02217 .
/// @see RbmFreeEnergy(), ConfiningEnergy(), ModernHopfield()
struct DeepBeliefPath {
  Tensor log_probability;         ///< Normalized per-observation log probability.
  Tensor nll;                     ///< Mean exact negative log likelihood.
  Tensor top_log_probability;     ///< Top-RBM16-state normalized log probabilities.
  Tensor decoder_logits;          ///< Directed decoder logits for16 hidden states.
  Tensor decoder_probability;     ///< Directed decoder probabilities.
  Tensor log_joint;               ///< Visible/hidden joint log weights.
  Tensor posterior;               ///< Posterior over16 hidden states.
  Tensor posterior_mean;          ///< Posterior mean lower hidden states.
  Tensor recognition_probability; ///< Bottom recognition conditional; not the generative posterior.
};

/// @brief Marginalize the 8-4-3 hybrid directed/undirected deep belief preset.
///
/// Inputs and returned handles own their tensor storage; live paths retain first and higher
/// derivatives where the primitive supports them. No optimizer, sampling cache or fitting loop.
/// Mathematical presets use Float32 CPU tensors. Caller validates observations and random banks.
///
/// @param bottom Initialized 8-visible4-hidden RBM; hidden_bias is recognition-only.
/// @param top Initialized4-visible3-hidden top RBM.
/// @param visible Float32 CPU[N,8] observations.
/// @return Live log probabilities, posterior over16 hidden states and recognition diagnostics.
/// @throws std::invalid_argument Invalid explicit bounds, or incompatible tensor geometry/dtype.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore
/// p(1),q(2);m::RbmParameters(p,8,4);m::RbmParameters(q,4,3);auto
/// y=m::DeepBelief(p,q,n::zeros({2,8}));
/// @endcode
/// @par References
/// - Hinton, Osindero and Teh (2006), A Fast Learning Algorithm for Deep Belief Nets,
/// https://doi.org/10.1162/neco.2006.18.7.1527 .
/// @see BinaryStates(), RbmFreeEnergy()
inline DeepBeliefPath
DeepBelief(const ParameterStore& bottom, const ParameterStore& top, const Tensor& visible) {
  const Tensor states   = BinaryStates(4);
  const Tensor top_logp = neg(add(RbmFreeEnergy(top, states), RbmLogPartition(top, 4)));
  const Tensor logits =
      bias_add(matmul(states, bottom.at("W"), false, true), bottom.at("visible_bias"));
  const Tensor conditional = sub(matmul(visible, logits, false, true), sum(softplus(logits), {1}));
  const Tensor joint       = add(conditional, top_logp);
  const Tensor logp        = LogSumExp(joint, 1);
  const Tensor posterior   = softmax(joint, 1);
  return {
      logp,
      neg(mean(logp, {0})),
      top_logp,
      logits,
      sigmoid(logits),
      joint,
      posterior,
      matmul(posterior, states),
      RbmHidden(bottom, visible)};
}

/// @brief Create the exact 8-4-3 deep Boltzmann preset parameters.
///
/// Inputs and returned handles own their tensor storage; live paths retain first and higher
/// derivatives where the primitive supports them. No optimizer, sampling cache or fitting loop.
/// Mathematical presets use Float32 CPU tensors. Caller validates observations and random banks.
///
/// @param p Caller-owned Float32 CPU store; normal(.2) W1/W2 and zero biases.
/// @throws std::invalid_argument Invalid explicit bounds, or incompatible tensor geometry/dtype.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(1);m::DeepBoltzmannParameters(p);
/// @endcode
/// @par References
/// - Salakhutdinov and Hinton (2009), Deep Boltzmann Machines,
/// https://proceedings.mlr.press/v5/salakhutdinov09a.html .
/// @see BinaryStates(), RbmFreeEnergy()
inline void DeepBoltzmannParameters(ParameterStore& p) {
  p.get("W1", {8, 4}, init::normal(.2));
  p.get("W2", {4, 3}, init::normal(.2));
  p.get("visible_bias", {8}, init::zeros());
  p.get("hidden1_bias", {4}, init::zeros());
  p.get("hidden2_bias", {3}, init::zeros());
}
/// @brief Compute the upper hidden conditional of the deep Boltzmann preset.
///
/// Inputs and returned handles own their tensor storage; live paths retain first and higher
/// derivatives where the primitive supports them. No optimizer, sampling cache or fitting loop.
/// Mathematical presets use Float32 CPU tensors. Caller validates observations and random banks.
///
/// @param p Initialized 8-4-3 store.
/// @param h1 Float32 CPU[N,4] lower hidden states.
/// @return Live CPU[N,3] probabilities.
/// @throws std::invalid_argument Invalid explicit bounds, or incompatible tensor geometry/dtype.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore
/// p(1);m::DeepBoltzmannParameters(p);auto y=m::DbmHidden2(p,n::zeros({2,4}));
/// @endcode
/// @par References
/// - Salakhutdinov and Hinton (2009), Deep Boltzmann Machines,
/// https://proceedings.mlr.press/v5/salakhutdinov09a.html .
/// @see BinaryStates(), RbmFreeEnergy()
inline Tensor DbmHidden2(const ParameterStore& p, const Tensor& h1) {
  return sigmoid(bias_add(matmul(h1, p.at("W2")), p.at("hidden2_bias")));
}
/// @brief Compute the visible conditional of the deep Boltzmann preset.
///
/// Inputs and returned handles own their tensor storage; live paths retain first and higher
/// derivatives where the primitive supports them. No optimizer, sampling cache or fitting loop.
/// Mathematical presets use Float32 CPU tensors. Caller validates observations and random banks.
///
/// @param p Initialized 8-4-3 store.
/// @param h1 Float32 CPU[N,4] lower hidden states.
/// @return Live CPU[N,8] probabilities.
/// @throws std::invalid_argument Invalid explicit bounds, or incompatible tensor geometry/dtype.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore
/// p(1);m::DeepBoltzmannParameters(p);auto y=m::DbmVisible(p,n::zeros({2,4}));
/// @endcode
/// @par References
/// - Salakhutdinov and Hinton (2009), Deep Boltzmann Machines,
/// https://proceedings.mlr.press/v5/salakhutdinov09a.html .
/// @see BinaryStates(), RbmFreeEnergy()
inline Tensor DbmVisible(const ParameterStore& p, const Tensor& h1) {
  return sigmoid(bias_add(matmul(h1, p.at("W1"), false, true), p.at("visible_bias")));
}
/// @brief Compute the lower hidden conditional given both neighbouring layers.
///
/// Inputs and returned handles own their tensor storage; live paths retain first and higher
/// derivatives where the primitive supports them. No optimizer, sampling cache or fitting loop.
/// Mathematical presets use Float32 CPU tensors. Caller validates observations and random banks.
///
/// @param p Initialized 8-4-3 store.
/// @param v Float32 CPU[N,8] visible states.
/// @param h2 Float32 CPU[N,3] upper hidden states.
/// @return Live CPU[N,4] probabilities.
/// @throws std::invalid_argument Invalid explicit bounds, or incompatible tensor geometry/dtype.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore
/// p(1);m::DeepBoltzmannParameters(p);auto y=m::DbmHidden1(p,n::zeros({2,8}),n::zeros({2,3}));
/// @endcode
/// @par References
/// - Salakhutdinov and Hinton (2009), Deep Boltzmann Machines,
/// https://proceedings.mlr.press/v5/salakhutdinov09a.html .
/// @see BinaryStates(), RbmFreeEnergy()
inline Tensor DbmHidden1(const ParameterStore& p, const Tensor& v, const Tensor& h2) {
  return sigmoid(add(
      bias_add(matmul(v, p.at("W1")), p.at("hidden1_bias")), matmul(h2, p.at("W2"), false, true)));
}

/// @brief Own the live DbmUnnormalizedPath mathematical endpoints.
///
/// Handles retain their storage and graph; no optimizer or mutable cache.
/// @par Example
/// @code
/// dm::nn::models::DbmUnnormalizedPath path;
/// @endcode
/// @par References
/// - Hinton et al. (2006), https://doi.org/10.1162/neco.2006.18.7.1527 ;
/// Salakhutdinov and Hinton (2009), https://proceedings.mlr.press/v5/salakhutdinov09a.html ;
/// Ramsauer et al. (2021), https://arxiv.org/abs/2008.02217 .
/// @see RbmFreeEnergy(), ConfiningEnergy(), ModernHopfield()
struct DbmUnnormalizedPath {
  Tensor log_weight;           ///< Unnormalized visible log mass.
  Tensor hidden1_posterior;    ///< Posterior over16 lower hidden states.
  Tensor hidden1_mean;         ///< Exact lower hidden posterior mean.
  Tensor hidden2_mean;         ///< Exact upper hidden posterior mean.
  Tensor hidden2_conditionals; ///< Upper conditional probabilities for16 lower states.
};
/// @brief Enumerate16 lower hidden states and analytically sum the upper layer.
///
/// Inputs and returned handles own their tensor storage; live paths retain first and higher
/// derivatives where the primitive supports them. No optimizer, sampling cache or fitting loop.
/// Mathematical presets use Float32 CPU tensors. Caller validates observations and random banks.
///
/// @param p Initialized 8-4-3 store.
/// @param v Float32 CPU[N,8] visible states.
/// @return Live unnormalized log weights and exact hidden posterior statistics.
/// @throws std::invalid_argument Invalid explicit bounds, or incompatible tensor geometry/dtype.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore
/// p(1);m::DeepBoltzmannParameters(p);auto y=m::DbmUnnormalized(p,n::zeros({2,8}));
/// @endcode
/// @par References
/// - Salakhutdinov and Hinton (2009), Deep Boltzmann Machines,
/// https://proceedings.mlr.press/v5/salakhutdinov09a.html .
/// @see BinaryStates(), RbmFreeEnergy()
inline DbmUnnormalizedPath DbmUnnormalized(const ParameterStore& p, const Tensor& v) {
  const Tensor h            = BinaryStates(4);
  const Tensor upper_logits = bias_add(matmul(h, p.at("W2")), p.at("hidden2_bias"));
  const Tensor hidden_potential =
      add(sum(mul(h, p.at("hidden1_bias")), {1}), sum(softplus(upper_logits), {1}));
  const Tensor joint = add(matmul(matmul(v, p.at("W1")), h, false, true), hidden_potential);
  const Tensor visible_potential = sum(mul(v, p.at("visible_bias")), {1});
  const Tensor posterior         = softmax(joint, 1);
  const Tensor upper             = sigmoid(upper_logits);
  return {
      add(visible_potential, LogSumExp(joint, 1)),
      posterior,
      matmul(posterior, h),
      matmul(posterior, upper),
      upper};
}

/// @brief Own the live DbmMeanFieldPath mathematical endpoints.
///
/// Handles retain their storage and graph; no optimizer or mutable cache.
/// @par Example
/// @code
/// dm::nn::models::DbmMeanFieldPath path;
/// @endcode
/// @par References
/// - Hinton et al. (2006), https://doi.org/10.1162/neco.2006.18.7.1527 ;
/// Salakhutdinov and Hinton (2009), https://proceedings.mlr.press/v5/salakhutdinov09a.html ;
/// Ramsauer et al. (2021), https://arxiv.org/abs/2008.02217 .
/// @see RbmFreeEnergy(), ConfiningEnergy(), ModernHopfield()
struct DbmMeanFieldPath {
  Tensor hidden1;     ///< Lower hidden states/probabilities.
  Tensor hidden2;     ///< Upper hidden states/probabilities.
  Tensor free_energy; ///< Per-observation variational free energy.
};
/// @brief Iterate clamped mean-field inference for the deep Boltzmann preset.
///
/// Inputs and returned handles own their tensor storage; live paths retain first and higher
/// derivatives where the primitive supports them. No optimizer, sampling cache or fitting loop.
/// Mathematical presets use Float32 CPU tensors. Caller validates observations and random banks.
///
/// @param p Initialized 8-4-3 store.
/// @param v Float32 CPU[N,8] clamped observations.
/// @param iterations Alternating update count in[1,10000], default20.
/// @return Live hidden probabilities and variational free energy; not exact likelihood.
/// @throws std::invalid_argument Invalid explicit bounds, or incompatible tensor geometry/dtype.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore
/// p(1);m::DeepBoltzmannParameters(p);auto y=m::DbmMeanField(p,n::zeros({2,8}));
/// @endcode
/// @par References
/// - Salakhutdinov and Hinton (2009), Deep Boltzmann Machines,
/// https://proceedings.mlr.press/v5/salakhutdinov09a.html .
/// @see BinaryStates(), RbmFreeEnergy()
inline DbmMeanFieldPath
DbmMeanField(const ParameterStore& p, const Tensor& v, int iterations = 20) {
  if (iterations < 1 || iterations > 10000)
    throw std::invalid_argument("mean-field iterations must be 1..10000");
  const Tensor bottom = bias_add(matmul(v, p.at("W1")), p.at("hidden1_bias"));
  Tensor       h1 = sigmoid(bottom), h2 = DbmHidden2(p, h1);
  for (int k = 0; k < iterations; ++k) {
    h1 = DbmHidden1(p, v, h2);
    h2 = DbmHidden2(p, h1);
  }
  auto entropy_term = [](const Tensor& q) {
    const Tensor complement = sub(1.0, q);
    return sum(add(mul(q, log(q)), mul(complement, log(complement))), {1});
  };
  const Tensor expected_energy = neg(
      add(add(sum(mul(v, p.at("visible_bias")), {1}), sum(mul(h1, bottom), {1})),
          add(sum(mul(h2, p.at("hidden2_bias")), {1}), sum(mul(matmul(h1, p.at("W2")), h2), {1}))));
  return {h1, h2, add(expected_energy, add(entropy_term(h1), entropy_term(h2)))};
}

/// @brief Own the live DbmGibbsPath mathematical endpoints.
///
/// Handles retain their storage and graph; no optimizer or mutable cache.
/// @par Example
/// @code
/// dm::nn::models::DbmGibbsPath path;
/// @endcode
/// @par References
/// - Hinton et al. (2006), https://doi.org/10.1162/neco.2006.18.7.1527 ;
/// Salakhutdinov and Hinton (2009), https://proceedings.mlr.press/v5/salakhutdinov09a.html ;
/// Ramsauer et al. (2021), https://arxiv.org/abs/2008.02217 .
/// @see RbmFreeEnergy(), ConfiningEnergy(), ModernHopfield()
struct DbmGibbsPath {
  Tensor visible_probability; ///< Conditional visible probabilities.
  Tensor visible;             ///< Sampled visible states.
  Tensor hidden2_probability; ///< Upper conditional probabilities.
  Tensor hidden2;             ///< Upper hidden states/probabilities.
  Tensor hidden1_probability; ///< Lower conditional probabilities.
  Tensor hidden1;             ///< Lower hidden states/probabilities.
};
/// @brief Sample (visible,upper) given lower states, then resample lower states.
///
/// Inputs and returned handles own their tensor storage; live paths retain first and higher
/// derivatives where the primitive supports them. No optimizer, sampling cache or fitting loop.
/// Mathematical presets use Float32 CPU tensors. Caller validates observations and random banks.
///
/// @param p Initialized 8-4-3 store.
/// @param old_h1 Float32 CPU[N,4] previous lower states.
/// @param uv Float32 CPU[N,8] explicit uniforms.
/// @param u2 Float32 CPU[N,3] explicit uniforms.
/// @param u1 Float32 CPU[N,4] explicit uniforms.
/// @return Live conditional probabilities and sampled Float32 states.
/// @throws std::invalid_argument Invalid explicit bounds, or incompatible tensor geometry/dtype.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore
/// p(1);m::DeepBoltzmannParameters(p);auto
/// y=m::DbmGibbs(p,n::zeros({2,4}),n::full({2,8},.5),n::full({2,3},.5),n::full({2,4},.5));
/// @endcode
/// @par References
/// - Salakhutdinov and Hinton (2009), Deep Boltzmann Machines,
/// https://proceedings.mlr.press/v5/salakhutdinov09a.html .
/// @see BinaryStates(), RbmFreeEnergy()
inline DbmGibbsPath DbmGibbs(
    const ParameterStore& p,
    const Tensor&         old_h1,
    const Tensor&         uv,
    const Tensor&         u2,
    const Tensor&         u1) {
  const Tensor vp = DbmVisible(p, old_h1), p2 = DbmHidden2(p, old_h1);
  const Tensor v = cast(less(uv, vp), DType::Float32), h2 = cast(less(u2, p2), DType::Float32);
  const Tensor p1 = DbmHidden1(p, v, h2);
  return {vp, v, p2, h2, p1, cast(less(u1, p1), DType::Float32)};
}

/// @brief Own the live ConfiningEnergyPath mathematical endpoints.
///
/// Handles retain their storage and graph; no optimizer or mutable cache.
/// @par Example
/// @code
/// dm::nn::models::ConfiningEnergyPath path;
/// @endcode
/// @par References
/// - Hinton et al. (2006), https://doi.org/10.1162/neco.2006.18.7.1527 ;
/// Salakhutdinov and Hinton (2009), https://proceedings.mlr.press/v5/salakhutdinov09a.html ;
/// Ramsauer et al. (2021), https://arxiv.org/abs/2008.02217 .
/// @see RbmFreeEnergy(), ConfiningEnergy(), ModernHopfield()
struct ConfiningEnergyPath {
  Tensor              energy;       ///< Per-observation energy.
  Tensor              hidden;       ///< Hidden samples or activations.
  std::vector<Tensor> hessian_rows; ///< Two live CPU[N,2] analytic Hessian rows.
};

/// @brief Create a quadratic plus bounded-tanh-residual energy preset.
///
/// Inputs and returned handles own their tensor storage; live paths retain first and higher
/// derivatives where the primitive supports them. No optimizer, sampling cache or fitting loop.
/// Mathematical presets use Float32 CPU tensors. Caller validates observations and random banks.
///
/// @param p Caller-owned Float32 CPU store;64 parameters in three tensors.
/// @throws std::invalid_argument Invalid explicit bounds, or incompatible tensor geometry/dtype.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(1);m::ConfiningEnergyParameters(p);
/// @endcode
/// @par References
/// - Welling and Teh (2011), Bayesian Learning via Stochastic Gradient Langevin Dynamics,
/// https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf . This API is the energy
/// graph; callers choose the sampler and training rule.
/// @see BinaryStates(), RbmFreeEnergy()
inline void ConfiningEnergyParameters(ParameterStore& p) {
  p.get("hidden/W", {2, 16}, init::normal(.3));
  p.get("hidden/b", {16}, init::zeros());
  p.get("output/W", {16, 1}, init::normal(.1));
}

/// @brief Evaluate the confining two-dimensional energy and analytic Hessian rows.
///
/// Inputs and returned handles own their tensor storage; live paths retain first and higher
/// derivatives where the primitive supports them. No optimizer, sampling cache or fitting loop.
/// Mathematical presets use Float32 CPU tensors. Caller validates observations and random banks.
///
/// @param p Initialized64-parameter store; may be null only when oracle=true.
/// @param x Float32 CPU[N,2] states; finite values required.
/// @param oracle Defaultfalse; true uses Gaussian mean(.3,-.2), variance(.49,.25).
/// @return Live per-row energy, hidden activations and two Hessian rows. No gradients are consumed.
/// @throws std::invalid_argument Invalid explicit bounds, or incompatible tensor geometry/dtype.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore
/// p(1);m::ConfiningEnergyParameters(p);auto y=m::ConfiningEnergy(&p,n::zeros({2,2}));
/// @endcode
/// @par References
/// - Welling and Teh (2011), Bayesian Learning via Stochastic Gradient Langevin Dynamics,
/// https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf . This API is the energy
/// graph; callers choose the sampler and training rule.
/// @see BinaryStates(), RbmFreeEnergy()
inline ConfiningEnergyPath
ConfiningEnergy(const ParameterStore* p, const Tensor& x, bool oracle = false) {
  if (x.dim() != 2 || x.size(1) != 2 || (!oracle && p == nullptr))
    throw std::invalid_argument("energy requires [N,2] states and a store");
  if (oracle) {
    const Tensor delta = sub(x, from_vector(std::vector<float>{.3f, -.2f}, {1, 2}));
    const Tensor energy =
        mul(sum(div(square(delta), from_vector(std::vector<float>{.49f, .25f}, {1, 2})), {1}), .5);
    const Tensor zeros = mul(x, 0.0);
    return {
        energy,
        x,
        {add(zeros, from_vector(std::vector<float>{1 / .49f, 0.f}, {1, 2})),
         add(zeros, from_vector(std::vector<float>{0.f, 4.f}, {1, 2}))}};
  }
  const Tensor        hidden    = tanh(bias_add(matmul(x, p->at("hidden/W")), p->at("hidden/b")));
  const Tensor        residual  = reshape(matmul(hidden, p->at("output/W")), {-1});
  const Tensor        energy    = mul(add(sum(square(x), {1}), residual), .5);
  const Tensor        v         = reshape(p->at("output/W"), {1, 16});
  const Tensor        curvature = neg(mul(mul(v, hidden), sub(1.0, square(hidden))));
  const Tensor        w         = p->at("hidden/W");
  std::vector<Tensor> rows;
  for (int a = 0; a < 2; ++a) {
    std::vector<Tensor> columns;
    for (int b = 0; b < 2; ++b) {
      const Tensor wa = slice_axis(w, 0, a, a + 1), wb = slice_axis(w, 0, b, b + 1);
      columns.push_back(add(sum(mul(curvature, mul(wa, wb)), {1}, true), a == b ? 1.0 : 0.0));
    }
    rows.push_back(concat(columns, 1));
  }
  return {energy, hidden, rows};
}

/// @brief Own the live ModernHopfieldPath mathematical endpoints.
///
/// Handles retain their storage and graph; no optimizer or mutable cache.
/// @par Example
/// @code
/// dm::nn::models::ModernHopfieldPath path;
/// @endcode
/// @par References
/// - Hinton et al. (2006), https://doi.org/10.1162/neco.2006.18.7.1527 ;
/// Salakhutdinov and Hinton (2009), https://proceedings.mlr.press/v5/salakhutdinov09a.html ;
/// Ramsauer et al. (2021), https://arxiv.org/abs/2008.02217 .
/// @see RbmFreeEnergy(), ConfiningEnergy(), ModernHopfield()
struct ModernHopfieldPath {
  Tensor logits;    ///< Inverse-temperature query-memory dot products.
  Tensor attention; ///< Normalized memory attention weights.
  Tensor next;      ///< Retrieved next query states.
  Tensor energy;    ///< Per-observation energy.
};

/// @brief Retrieve tied key/value memories and evaluate continuous Hopfield energy.
///
/// Inputs and returned handles own their tensor storage; live paths retain first and higher
/// derivatives where the primitive supports them. No optimizer, sampling cache or fitting loop.
/// Mathematical presets use Float32 CPU tensors. Caller validates observations and random banks.
///
/// @param state Float32 CPU[N,D] query states; need not have unit norm.
/// @param memory Float32 CPU[M,D] fixed or differentiable memory matrix.
/// @param beta Finite positive inverse temperature.
/// @return Live logits, attention, retrieved state and per-query energy; no model mutation.
/// @throws std::invalid_argument Invalid explicit bounds, or incompatible tensor geometry/dtype.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;auto
/// y=m::ModernHopfield(n::zeros({2,3}),n::ones({4,3}),8);
/// @endcode
/// @par References
/// - Ramsauer et al. (2021), Hopfield Networks is All You Need, https://arxiv.org/abs/2008.02217 .
/// @see BinaryStates(), RbmFreeEnergy()
inline ModernHopfieldPath ModernHopfield(const Tensor& state, const Tensor& memory, float beta) {
  if (!std::isfinite(beta) || beta <= 0 || state.dim() != 2 || memory.dim() != 2 ||
      state.size(1) != memory.size(1) || memory.size(0) < 1)
    throw std::invalid_argument("Hopfield positive beta and aligned matrices required");
  const Tensor     logits = mul(matmul(state, memory, false, true), double(beta));
  AttentionOptions o;
  o.scale               = beta;
  o.need_weights        = true;
  o.impl                = AttentionImpl::Plain;
  const auto   attended = scaled_dot_product_attention(state, memory, memory, o);
  const Tensor energy =
      sub(mul(sum(square(state), {1}), .5), div(LogSumExp(logits, 1), double(beta)));
  return {logits, attended.weights, attended.output, energy};
}

/// @brief Own the reproducible noisy single-bar fixture on a2x4 grid.
/// @par Example
/// @code
/// dm::nn::models::BinaryBarData data;
/// @endcode
/// @see BinaryBars()
struct BinaryBarData {
  Tensor points;     ///< Float32 CPU[N,8] noisy binary observations.
  Tensor uniforms;   ///< Float64 CPU[N,9] midpoint-MT draw bank.
  Tensor components; ///< Int32 CPU[N] selected vertical0..3 or horizontal4..5 bar.
  Tensor source_ids; ///< Int32 CPU[N] caller-offset provenance IDs.
};
/// @brief Draw uniform single bars with independent five-percent bit flips.
///
/// The first midpoint MT19937 draw selects one of six bars; eight subsequent draws flip bits.
/// No global RNG or model state is touched. This is a controlled teaching fixture, not public data.
/// @param n Row count in[1,1000000].
/// @param seed Local UInt32 MT19937 seed.
/// @param offset Nonnegative source-ID offset; offset+n must fit Int32.
/// @return Owning observations, exact Double draw bank, components and IDs.
/// @throws std::invalid_argument Invalid count or source-ID bounds.
/// @par Example
/// @code
/// auto data=dm::nn::models::BinaryBars(512,21001,0);
/// @endcode
/// @par References
/// - Repository noisy single-bar teaching protocol, original restricted_boltzmann.cpp;
/// docs/mepml/nn/3_generative_models/3.5_energy_based_models/restricted_boltzmann.mepml .
/// @see BinaryStates(), MidpointUniform()
inline BinaryBarData BinaryBars(std::int64_t n, std::uint32_t seed, int offset = 0) {
  if (n < 1 || n > 1000000 || offset < 0 || n + offset > 2147483647)
    throw std::invalid_argument("bar fixture bounds");
  auto               bank = MidpointUniform(n, 9, seed, DType::Float64);
  auto               u    = bank.to_vector<double>();
  std::vector<float> values(n * 8);
  std::vector<int>   components(n), ids(n);
  for (int b = 0; b < n; ++b) {
    int c         = int(6 * u[b * 9]);
    components[b] = c;
    ids[b]        = offset + b;
    for (int j = 0; j < 8; ++j) {
      bool bit          = c < 4 ? j % 4 == c : j / 4 == c - 4;
      values[b * 8 + j] = float(bit != (u[b * 9 + j + 1] < .05));
    }
  }
  return {from_vector(values, {n, 8}), bank, from_vector(components), from_vector(ids)};
}
} // namespace datamunge::nn::models
