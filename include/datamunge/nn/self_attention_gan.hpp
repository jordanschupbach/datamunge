#pragma once
/// @file self_attention_gan.hpp
/// @brief Cached spectral spatial attention and compact patch GAN maps with explicit hinge losses.
#include "datamunge/nn/attention.hpp"
#include "datamunge/nn/cached_spectral_dense.hpp"
#include "datamunge/nn/dense_gan.hpp"
#include "datamunge/nn/masked_autoencoder.hpp"

namespace datamunge::nn::models {
/// @brief Internal positive spatial attention and activation-product validation.
namespace sag_detail {
/// @brief Validate a finite floating token cube and every dense/attention product before leaves.
/// @param input Finite Float32/64 CPU[B,T,C], positive axes and strides allowed.
/// @param query Positive shared query/key width.
/// @param value Positive value projection width.
/// @param type Required input computation dtype.
/// @throws std::invalid_argument For malformed/nonfinite inputs or invalid/overflowing
/// widths/products.
/// @throws std::logic_error For non-CPU storage.
inline void Tokens(const Tensor& input, std::int64_t query, std::int64_t value, DType type) {
  gan_detail::Finite(input);
  if (input.dim() != 3 || input.dtype() != type || query <= 0 || value <= 0)
    throw std::invalid_argument(
        "Spectral spatial attention needs matching token cubes and positive widths");
  const auto b = input.size(0), t = input.size(1), c = input.size(2),
             limit = std::numeric_limits<std::int64_t>::max();
  if (c > limit - query || c > limit - value || c > limit / query || c > limit / value ||
      b > limit / t || b * t > limit / std::max({t, c, query, value}))
    throw std::invalid_argument("Spectral spatial attention dimensions/products overflow");
}
} // namespace sag_detail

/// @brief Own spatial attention input, row probabilities and the gated residual output.
///
/// Tensor handles own storage and applicable first/higher graph paths after parent
/// or stores disappear. input/output[B,T,C], probability[B,T,T] with each query
/// row normalized over keys. No mode, optimizer, RNG or cache reference is retained.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(1),s(0);
/// auto a=n::models::SpectralSpatialAttention(p,s,n::ones({2,3,8}));
/// // a.output equals a.input exactly at the initial zero residual gate.
/// @endcode
/// @par References
/// - Zhang et al. (2018/2019), Self-Attention Generative Adversarial Networks,
///   https://arxiv.org/abs/1805.08318 .
/// @see SpectralSpatialAttention(), SelfAttentionGanGenerate()
struct SpectralSpatialOutput {
  Tensor input;       ///< Owning live supplied tokens[B,T,C], sharing their storage.
  Tensor probability; ///< Owning live row-softmax unscaled Q*K-transpose[B,T,T].
  Tensor output;      ///< Owning live input+gamma*project(probability*V)[B,T,C].
};

/// @brief Mix all spatial tokens with unscaled attention and cached spectral dense projections.
///
/// input[B,T,C] flattens to[B*T,C]. Creates name/query, key, value and project
/// biased spectral maps C->Q,C->Q,C->V,V->C, then scalar name/gamma initialized0,
/// in this order. probability=softmax(Q*K^T), without sqrt(Q) scaling; output is
/// input+gamma*project(probability*V). Reuses scaled_dot_product_attention() with
/// explicit scale1, Plain and need_weights=true; no mask, dropout or fused path.
/// The initial zero gate makes output exactly input, while probability remains
/// a live computed field. Q/K/V/project partials of an output-only loss initially
/// vanish; the gate can still receive a gradient. Every projection uses
/// cached_spectral_dense() source-rounded initialization and stopped column paths,
/// while its signed sigma remains differentiable in W. No cache advance is hidden.
///
/// Finite matching Float32/64 CPU tokens with positive B,T,C, positive Q/V,
/// representable products, strides and applicable first/higher continuous paths.
/// Distinct matching stores and nonempty name required. Input checks precede
/// leaves; existing-leaf/nonfinite failures can occur later after earlier leaves.
/// Inputs unchanged; owning outputs retain lifetime. Work O(B*T*C*(Q+V)+B*T*T*(Q+V)),
/// activation storage O(B*T*(C+Q+V)+B*T*T), plus validation. Synchronize stores.
/// Cache flags follow ParameterStore; exclude state from optimizers and checkpoint it.
/// @param parameters Mutable floating projection/gate owner.
/// @param state Distinct matching cached-column owner; unchanged after initialization.
/// @param input Finite signed matching floating CPU[B,T,C] tokens with positive axes.
/// @param query_width Positive shared query/key width Q, default2.
/// @param value_width Positive value width V, default4.
/// @param name Nonempty parameter/cache prefix, default "attention".
/// @return Owning SpectralSpatialOutput with input, probability and residual output.
/// @throws std::invalid_argument For invalid inputs/options/stores/products, incompatible leaves or
/// nonfinite output.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(1),s(0);
/// auto x=n::ones({2,16,8});
/// n::models::SpectralSpatialAttention(p,s,x);n::advance_cached_spectral_state(p,s,16);
/// auto a=n::models::SpectralSpatialAttention(p,s,x);
/// // a.probability is[2,16,16]; scoring does not advance existing cache state.
/// @endcode
/// @par References
/// - Zhang et al. (2018/2019), https://arxiv.org/abs/1805.08318 , self-attention
///   residual with a zero-initialized scalar gate. This variant uses biased dense
///   projections and explicit cached spectral normalization on every kernel.
/// @see cached_spectral_dense(), SelfAttentionGanGenerate(), SelfAttentionGanDiscriminate()
inline SpectralSpatialOutput SpectralSpatialAttention(
    ParameterStore&    parameters,
    ParameterStore&    state,
    const Tensor&      input,
    std::int64_t       query_width = 2,
    std::int64_t       value_width = 4,
    const std::string& name        = "attention") {
  if (&parameters == &state || parameters.dtype() != state.dtype() || name.empty())
    throw std::invalid_argument(
        "Spectral spatial attention needs distinct matching stores and a nonempty name");
  sag_detail::Tokens(input, query_width, value_width, parameters.dtype());
  const auto b = input.size(0), t = input.size(1), c = input.size(2);
  auto       flat = reshape(input, {b * t, c});
  auto       q    = reshape(
      cached_spectral_dense(parameters, state, name + "/query", flat, query_width),
      {b, t, query_width});
  auto k = reshape(
      cached_spectral_dense(parameters, state, name + "/key", flat, query_width),
      {b, t, query_width});
  auto v = reshape(
      cached_spectral_dense(parameters, state, name + "/value", flat, value_width),
      {b, t, value_width});
  AttentionOptions options;
  options.scale        = 1.;
  options.need_weights = true;
  options.impl         = AttentionImpl::Plain;
  auto a               = scaled_dot_product_attention(q, k, v, options);
  auto projected       = reshape(
      cached_spectral_dense(
          parameters, state, name + "/project", reshape(a.output, {b * t, value_width}), c),
      {b, t, c});
  auto gamma = parameters.get(name + "/gamma", {}, init::zeros());
  gan_detail::Finite(gamma);
  auto output = input + gamma * projected;
  gan_detail::Finite(a.weights);
  gan_detail::Finite(output);
  return {input, a.weights, output};
}

/// @brief Own generated patch pixels and three live spatial-attention endpoints.
///
/// Flat pixels[B,784], input/output[B,16,8], probability[B,16,16] retain
/// storage and applicable first/higher paths after parent or stores disappear.
/// No fitting, mode, RNG or checkpoint state is retained.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore g(5903),s(0);
/// auto out=n::models::SelfAttentionGanGenerate(g,s,n::zeros({2,16}));
/// // out.pixels is[2,784]; out.attention_probability is[2,16,16].
/// @endcode
/// @par References
/// - Zhang et al. (2018/2019), https://arxiv.org/abs/1805.08318 .
/// @see SelfAttentionGanGenerate(), SelfAttentionGanOutput
struct SelfAttentionGenerated {
  Tensor pixels;                ///< Owning live sigmoid pixels[B,784], raster ordered.
  Tensor attention_input;       ///< Owning live tanh tokens[B,16,8], before the residual block.
  Tensor attention_probability; ///< Owning live unscaled row-softmax probabilities[B,16,16].
  Tensor attention_output;      ///< Owning live gated residual tokens[B,16,8].
};

/// @brief Generate grayscale pixels from explicit noise through spectral spatial-patch attention.
///
/// noise[B,16] -> spectral input dense128/tanh ->[B,16,8] tokens ->
/// SpectralSpatialAttention(Q2,V4) -> shared spectral patch dense49/sigmoid ->
/// MaeImage() raster permutation of sixteen row-major7x7 patches. Creates input
/// W,b; four attention maps W,b; scalar attention/gamma0; patch W,b, in this order.
/// Defaults are fixed architecture,2730 parameters/13 leaves and245 cache values/12
/// leaves; no labels or image enter G. Cached spectral guards and initialization
/// use source Float32 rounding. No power iteration, warm-up, noise draw beyond
/// missing leaves, optimizer or fitting is hidden. Existing cache is unchanged.
/// Finite matching Float32/64 CPU noise with positive B, strides, representable
/// activation products and applicable first/higher noise/W paths; cache paths stop.
/// Distinct matching stores required. Inputs validate before leaves; incompatible
/// leaves/nonfinite computation may fail later. Inputs unchanged, outputs own
/// lifetime. Work O(B*(16*128+16*8*49+16*16*6)); activations O(B*(784+16*16)),
/// plus validation. Synchronize shared stores; exclude cache from optimizers.
/// @param generator Mutable floating patch-generator weight/gate owner.
/// @param state Distinct matching G cache owner, unchanged after initialization.
/// @param noise Finite signed matching floating CPU[B,16], B>0.
/// @return Owning SelfAttentionGenerated, with pixels and all G attention endpoints.
/// @throws std::invalid_argument For invalid inputs/stores/products, incompatible leaves or
/// nonfinite computation.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore g(5903),s(0);
/// n::models::SelfAttentionGanGenerate(g,s,n::zeros({2,16}));
/// n::advance_cached_spectral_state(g,s,16);
/// auto out=n::models::SelfAttentionGanGenerate(g,s,n::zeros({2,16}));
/// // Every patch shares the same learned8->49 map; state is not advanced here.
/// @endcode
/// @par References
/// - Zhang et al. (2018/2019), https://arxiv.org/abs/1805.08318 . This compact
///   dense spatial-patch teaching model is not the paper's convolutional benchmark.
/// @see SelfAttentionGanDiscriminate(), SpectralSpatialAttention(), MaeImage(), SelfAttentionGan()
inline SelfAttentionGenerated
SelfAttentionGanGenerate(ParameterStore& generator, ParameterStore& state, const Tensor& noise) {
  if (&generator == &state || generator.dtype() != state.dtype())
    throw std::invalid_argument("Self-attention G needs distinct matching stores");
  gan_detail::Matrix(noise, 16, generator.dtype());
  if (noise.size(0) > std::numeric_limits<std::int64_t>::max() / 784)
    throw std::invalid_argument("Self-attention G activation products overflow");
  auto tokens = reshape(
      cached_spectral_dense(generator, state, "input", noise, 128, Activation::Tanh),
      {noise.size(0), 16, 8});
  auto a     = SpectralSpatialAttention(generator, state, tokens);
  auto patch = cached_spectral_dense(
      generator, state, "patch", reshape(a.output, {-1, 8}), 49, Activation::Sigmoid);
  return {MaeImage(reshape(patch, {noise.size(0), 16, 49})), a.input, a.probability, a.output};
}

/// @brief Own raw patch-discriminator scores and three live attention endpoints.
///
/// score[B,1], input/output[B,16,8], probability[B,16,16] own storage and
/// applicable first/higher paths after parent/stores disappear; no sigmoid or mode.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore d(5923),s(0);
/// auto out=n::models::SelfAttentionGanDiscriminate(d,s,n::zeros({2,784}));
/// // out.score is[2,1]; out.attention_input includes learned spatial positions.
/// @endcode
/// @par References
/// - Zhang et al. (2018/2019), https://arxiv.org/abs/1805.08318 .
/// @see SelfAttentionGanDiscriminate(), SelfAttentionGanOutput
struct SelfAttentionDiscriminated {
  Tensor score;                 ///< Owning live raw source scores[B,1], without a sigmoid.
  Tensor attention_input;       ///< Owning live patch encodings plus learned positions[B,16,8].
  Tensor attention_probability; ///< Owning live unscaled row-softmax probabilities[B,16,16].
  Tensor attention_output;      ///< Owning live gated residual tokens[B,16,8].
};

/// @brief Score raster pixels through shared spectral patch embeddings and position-aware
/// attention.
///
/// image[B,784] -> MaePatches() row-major7x7 patches -> shared spectral49->8/tanh
/// -> learned normal(.01) position[16,8] -> SpectralSpatialAttention(Q2,V4) ->
/// mean over16 positions -> spectral8->1 raw score. Creates patch W,b, position,
/// four attention maps W,b, scalar attention/gamma0, output W,b, in that order:
/// 650 values/14 leaves and110 cached values/12 leaves. Both real and fake branches
/// share this store and cache. No label, noise or class head enters D.
/// No cache advance, warm-up, optimizer, fitting or RNG beyond missing leaves.
/// Finite signed matching Float32/64 CPU image with positive B, strides and
/// applicable first/higher image/weight paths. Cache columns stop, sigma stays
/// live. Distinct matching stores and representable products required. Input
/// checks precede leaves; incompatible/nonfinite leaves can fail later. Inputs
/// unchanged, outputs own lifetime. Work O(B*(16*49*8+16*16*6)); activations
/// O(B*(784+16*16)), plus validation. Synchronize stores; exclude cache from Adam.
/// @param discriminator Mutable floating patch-discriminator owner.
/// @param state Distinct matching D cache owner, unchanged after initialization.
/// @param image Finite signed matching floating CPU[B,784], B>0.
/// @return Owning SelfAttentionDiscriminated with raw score and all D attention fields.
/// @throws std::invalid_argument For invalid inputs/stores/products, incompatible leaves or
/// nonfinite computation.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore d(5923),s(0);
/// n::models::SelfAttentionGanDiscriminate(d,s,n::zeros({2,784}));
/// n::advance_cached_spectral_state(d,s,16);
/// auto out=n::models::SelfAttentionGanDiscriminate(d,s,n::zeros({2,784}));
/// // out.score is raw; no sigmoid or class prediction is hidden.
/// @endcode
/// @par References
/// - Zhang et al. (2018/2019), https://arxiv.org/abs/1805.08318 . This compact
///   dense patch variant adds learned spatial positions before attention.
/// @see SelfAttentionGanGenerate(), SpectralSpatialAttention(), MaePatches(), SelfAttentionGan()
inline SelfAttentionDiscriminated SelfAttentionGanDiscriminate(
    ParameterStore& discriminator,
    ParameterStore& state,
    const Tensor&   image) {
  if (&discriminator == &state || discriminator.dtype() != state.dtype())
    throw std::invalid_argument("Self-attention D needs distinct matching stores");
  gan_detail::Matrix(image, 784, discriminator.dtype());
  const auto b       = image.size(0);
  auto       patches = reshape(MaePatches(reshape(image, {b, 28, 28, 1})), {-1, 49});
  auto       encoded = reshape(
      cached_spectral_dense(discriminator, state, "patch", patches, 8, Activation::Tanh),
      {b, 16, 8});
  auto position = discriminator.get("position", {16, 8}, init::normal(.01));
  gan_detail::Finite(position);
  auto a     = SpectralSpatialAttention(discriminator, state, encoded + position);
  auto score = cached_spectral_dense(discriminator, state, "output", mean(a.output, {1}), 1);
  return {score, a.input, a.probability, a.output};
}

/// @brief Own twelve live patch SAGAN endpoints with separate scalar hinge objectives.
///
/// Tensor handles retain storage and applicable first/higher graphs after parent
/// and stores disappear. No optimizer, fitting counter, checkpoint, cache reference
/// or quadratic pixel MMD is retained. Real and generated pools may differ.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore g(5903),d(5923),gs(0),ds(0);
/// auto o=n::models::SelfAttentionGan(g,d,gs,ds,n::zeros({2,784}),n::zeros({3,16}));
/// // fake is[3,784], real_score[2,1]; both scalar game losses remain live.
/// @endcode
/// @par References
/// - Zhang et al. (2018/2019), https://arxiv.org/abs/1805.08318 .
/// @see SelfAttentionGan(), SelfAttentionGenerated, SelfAttentionDiscriminated
struct SelfAttentionGanOutput {
  Tensor fake;       ///< Owning live sigmoid generated pixels[M,784].
  Tensor real_score; ///< Owning live raw real source scores[N,1].
  Tensor fake_score; ///< Owning live raw generated source scores[M,1].
  Tensor
      discriminator_loss; ///< Scalar mean(relu(1-real))+mean(relu(1+fake)); zero boundary partials.
  Tensor generator_loss;  ///< Scalar -mean(fake), unhinged and without a sigmoid.
  Tensor g_attention;     ///< Owning live G row-softmax probabilities[M,16,16].
  Tensor g_attention_input;     ///< Owning live G tanh tokens[M,16,8].
  Tensor g_attention_output;    ///< Owning live G gated residual tokens[M,16,8].
  Tensor real_attention;        ///< Owning live real-D row-softmax probabilities[N,16,16].
  Tensor real_attention_input;  ///< Owning live real patch tokens plus learned positions[N,16,8].
  Tensor real_attention_output; ///< Owning live real-D gated residual tokens[N,16,8].
  Tensor fake_attention;        ///< Owning live generated-D row-softmax probabilities[M,16,16].
};

/// @brief Evaluate a compact spectral patch GAN with unscaled attention in both networks.
///
/// Composes SelfAttentionGanGenerate(), SelfAttentionGanDiscriminate() on real
/// and fake, and HingeGanObjectives(). Creates G2730/13 leaves, D650/14 leaves,
/// G cache245/12 leaves and D cache110/12 leaves:51 checkpoint tensors/3735 values,
/// of which3380 values are optimized. Stores initialize in G,D order with layer
/// orders documented by the two maps. Residual gates start0; all twelve kernels
/// use guarded cached spectral normalization with live sigma and stopped columns.
///
/// Finite signed Float32/64 CPU image[N,784],noise[M,16], positive possibly unequal
/// N/M, strides, representable products and applicable first/higher continuous
/// paths. All four stores are distinct and dtype matched. No hidden advance,
/// warm-up, optimizer, fit, freeze, checkpoint or MMD. Caller initializes, warms
/// both cache stores16 iterations, then per batch advances D cache, writes D,
/// advances G cache and recomputes a fresh G objective through updated D with
/// unchanged D cache, writing only G. Exclude both cache stores from Adam even
/// though ParameterStore flags follow normal leaves. Input checks precede leaves;
/// incompatible/nonfinite leaves may fail later. Inputs unchanged, outputs own
/// lifetime. Work/storage are the sum of the three maps, with O((N+M)*16*16)
/// attention activations plus validation. Synchronize shared stores. Finite power
/// iteration is not an exact unit norm or network Lipschitz certificate.
/// @param generator Mutable floating G parameter owner.
/// @param discriminator Distinct matching floating D parameter owner.
/// @param generator_state Third distinct matching cached G column owner.
/// @param discriminator_state Fourth distinct matching cached D column owner.
/// @param image Finite signed matching floating CPU[N,784], N>0.
/// @param noise Finite signed matching floating CPU[M,16], M>0.
/// @return Owning twelve-field SelfAttentionGanOutput with explicit scalar D/G losses.
/// @throws std::invalid_argument For invalid inputs/stores/products, incompatible leaves or
/// nonfinite computation.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore g(5903),d(5923),gs(0),ds(0);
/// auto x=n::zeros({2,784}),z=n::zeros({3,16});
/// n::models::SelfAttentionGan(g,d,gs,ds,x,z);
/// n::advance_cached_spectral_state(g,gs,16);n::advance_cached_spectral_state(d,ds,16);
/// n::advance_cached_spectral_state(d,ds);auto o=n::models::SelfAttentionGan(g,d,gs,ds,x,z);
/// auto derivative=n::grad(o.discriminator_loss,d.parameters());
/// // Write D, advance only G's cache, then recompute before selecting G partials.
/// @endcode
/// @par References
/// - Zhang, Goodfellow, Metaxas and Odena (2018/2019), Self-Attention Generative
///   Adversarial Networks, https://arxiv.org/abs/1805.08318 . This compact MNIST
///   patch architecture and explicit state/update schedule are repository teaching choices.
/// @see SelfAttentionGanGenerate(), SelfAttentionGanDiscriminate(), SpectralSpatialAttention(),
/// HingeGanObjectives()
inline SelfAttentionGanOutput SelfAttentionGan(
    ParameterStore& generator,
    ParameterStore& discriminator,
    ParameterStore& generator_state,
    ParameterStore& discriminator_state,
    const Tensor&   image,
    const Tensor&   noise) {
  ParameterStore* stores[] = {&generator, &discriminator, &generator_state, &discriminator_state};
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < i; ++j)
      if (stores[i] == stores[j] || stores[i]->dtype() != stores[j]->dtype())
        throw std::invalid_argument("Self-attention GAN needs four distinct matching stores");
  gan_detail::Matrix(image, 784, generator.dtype());
  gan_detail::Matrix(noise, 16, generator.dtype());
  if (noise.size(0) > std::numeric_limits<std::int64_t>::max() / 784)
    throw std::invalid_argument("Self-attention GAN activation products overflow");
  auto g    = SelfAttentionGanGenerate(generator, generator_state, noise);
  auto real = SelfAttentionGanDiscriminate(discriminator, discriminator_state, image);
  auto fake = SelfAttentionGanDiscriminate(discriminator, discriminator_state, g.pixels);
  auto loss = HingeGanObjectives(real.score, fake.score);
  return {
      g.pixels,
      real.score,
      fake.score,
      loss.discriminator,
      loss.generator,
      g.attention_probability,
      g.attention_input,
      g.attention_output,
      real.attention_probability,
      real.attention_input,
      real.attention_output,
      fake.attention_probability};
}
} // namespace datamunge::nn::models
