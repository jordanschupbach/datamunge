#pragma once
/// @file attention_diagnostics.hpp
/// @brief Attention concentration and source-rounded key-removal counterfactual diagnostics.
#include <cmath>
#include <datamunge/nn/attention.hpp>
#include <datamunge/nn/autograd.hpp>
#include <datamunge/nn/ops.hpp>
namespace datamunge::nn::models {
/// @brief Per-query attention concentration tensors, with key dimension reduced.
struct AttentionRowStatistics {
  /// Negative sum p*log(max(p,1e-30)); empty rows give0.
  Tensor entropy;
  /// exp(entropy) on nonempty rows, otherwise0.
  Tensor effective_support;
  /// Largest key weight, zero on empty rows.
  Tensor max_weight;
};

/// @brief Reduce normalized attention rows to entropy, effective support and maximum weight.
///
/// Uses a1e-30 logarithm floor. First/higher derivatives follow native primitives away from max/tie
/// kinks. Concentration describes weights, not a causal explanation of the output.
///
/// @param weights Finite Float32/Float64 CPU [...,K], K positive; entries in[0,1], row mass0 or1
/// within3e-6.
/// @return Same-dtype [...]-shaped live tensors; empty support0, singleton support1. Inputs
/// unchanged.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// auto
/// rows=dm::nn::models::AttentionConcentration(dm::nn::from_vector(std::vector<float>{.5f,.5f,0,0},{2,2}));
/// @endcode
/// @par References
/// - Vaswani et al., Attention Is All You Need (2017), https://arxiv.org/abs/1706.03762; Jain and
/// Wallace, Attention is not Explanation (2019), https://aclanthology.org/N19-1357/.
/// @see AttentionKeyRemoval()
inline AttentionRowStatistics AttentionConcentration(const Tensor& weights) {
  if (!weights.defined() || weights.device().backend != compute::Backend::CPU ||
      weights.dim() < 1 || weights.size(-1) < 1 ||
      (weights.dtype() != DType::Float32 && weights.dtype() != DType::Float64))
    throw std::invalid_argument("nonempty floating CPU attention weights required");
  for (double v : weights.to_vector<double>())
    if (!std::isfinite(v) || v < 0 || v > 1)
      throw std::invalid_argument("attention weight outside[0,1]");
  auto mass = sum(weights, {-1});
  for (double v : mass.to_vector<double>())
    if (v != 0 && std::abs(v - 1) > 3e-6)
      throw std::invalid_argument("attention rows must sum to zero or one");
  auto entropy  = neg(sum(weights * log(maximum(weights, scalar_like(1e-30, weights))), {-1}));
  auto nonempty = greater(mass, 0.);
  auto support  = exp(entropy) * cast(nonempty, weights.dtype());
  return {entropy, support, max(weights, {-1})};
}
/// @brief Owning detached native results of one original and K removal interventions.
struct AttentionRemovalDiagnostics {
  /// Original keep-mask attention weights [...,T,K].
  Tensor weights;
  /// Original output [...,T,Dv].
  Tensor output;
  /// Euclidean output changes [...,T,K].
  Tensor effects;
  /// Outputs with each key removed [K,...,T,Dv].
  Tensor removed_outputs;
  /// Concentration of the original weights.
  AttentionRowStatistics concentration;
};

/// @brief Measure output changes by masking each key and rerunning plain native attention.
///
/// NoGradGuard intentionally prevents retained training graphs. Inputs unchanged.
/// Subtraction is rounded in input dtype before a Double sum-of-squares norm, then cast back.
/// This is a specific mask intervention; equal values can make concentrated weights have zero
/// removal effect. No optimizer, rollout or training schedule is owned here.
///
/// @param q Nonempty finite Float32/Float64 CPU queries [...,T,Dk].
/// @param k Matching finite CPU keys [...,K,Dk], same rank/dtype.
/// @param v Matching finite CPU values [...,K,Dv], same rank/dtype.
/// @param allowed Boolean CPU keep mask broadcastable to [...,T,K]; false entries are invisible.
/// @param key_width Positive normalization width;0 selects q last axis; scale1/sqrt(width).
/// @return Detached diagnostics with each key removed once; empty attention rows/outputs are zero.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// auto q=dm::nn::ones({1,2,3});auto mask=dm::nn::ones({1,2,2},dm::nn::DType::Bool);auto
/// result=dm::nn::models::AttentionKeyRemoval(q,q,q,mask);
/// @endcode
/// @par References
/// - Vaswani et al., Attention Is All You Need (2017), https://arxiv.org/abs/1706.03762; Jain and
/// Wallace, Attention is not Explanation (2019), https://aclanthology.org/N19-1357/.
/// @see AttentionConcentration(), scaled_dot_product_attention()
inline AttentionRemovalDiagnostics AttentionKeyRemoval(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& allowed,
    int64_t       key_width = 0) {
  for (const auto& x : {q, k, v}) {
    if (!x.defined() || x.device().backend != compute::Backend::CPU ||
        (x.dtype() != DType::Float32 && x.dtype() != DType::Float64) || x.numel() < 1)
      throw std::invalid_argument("nonempty floating CPU Q/K/V required");
    for (double z : x.to_vector<double>())
      if (!std::isfinite(z))
        throw std::invalid_argument("finite attention Q/K/V required");
  }
  if (!allowed.defined() || allowed.device().backend != compute::Backend::CPU)
    throw std::invalid_argument("Boolean CPU visibility required");
  if (q.dim() < 2 || k.dim() != q.dim() || v.dim() != q.dim() || allowed.dtype() != DType::Bool ||
      k.size(-2) < 1)
    throw std::invalid_argument("aligned attention Q/K/V and Boolean visibility required");
  if (key_width == 0)
    key_width = q.size(-1);
  if (key_width < 1)
    throw std::invalid_argument("positive attention key normalization width required");
  NoGradGuard      guard;
  AttentionOptions options;
  options.mask                 = allowed;
  options.need_weights         = true;
  options.impl                 = AttentionImpl::Plain;
  options.scale                = 1. / std::sqrt(double(key_width));
  auto                original = scaled_dot_product_attention(q, k, v, options);
  std::vector<Tensor> removed, effects;
  auto                columns = arange(0, k.size(-2), 1., DType::Int64);
  for (int64_t r = 0; r < k.size(-2); ++r) {
    options.mask = logical_and(allowed, not_equal(columns, double(r)));
    auto without = scaled_dot_product_attention(q, k, v, options).output;
    removed.push_back(without);
    auto difference = cast(original.output - without, DType::Float64);
    effects.push_back(cast(sqrt(sum(square(difference), {-1})), q.dtype()));
  }
  return {
      original.weights,
      original.output,
      stack(effects, -1),
      stack(removed, 0),
      AttentionConcentration(original.weights)};
}
} // namespace datamunge::nn::models
