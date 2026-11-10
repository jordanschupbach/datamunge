#pragma once
/// @file adversarial_autoencoder.hpp
/// @brief Deterministic adversarial-autoencoder forward components and explicit scalar objectives.
#include <cmath>
#include <stdexcept>

#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn::models {
/// @brief Input validation shared by the independent AAE parameter groups.
namespace aae_detail {
/// @brief Validate positive architecture dimensions.
/// @param features Image width.
/// @param latent Code width.
/// @param hidden Tanh hidden width in each network.
/// @throws std::invalid_argument If any width is nonpositive.
inline void Widths(std::int64_t features, std::int64_t latent, std::int64_t hidden) {
  if (features <= 0 || latent <= 0 || hidden <= 0)
    throw std::invalid_argument("AAE widths must be positive");
}
/// @brief Validate finite matching floating rows, optionally unit-range pixel targets.
/// @param value Finite nonempty Float32/64 matrix.
/// @param width Required last dimension.
/// @param type Required computation dtype.
/// @param pixels Require values in [0,1] when true.
/// @throws std::invalid_argument For invalid shape, dtype or values.
inline void Matrix(const Tensor& value, std::int64_t width, DType type, bool pixels) {
  if (width <= 0 || !value.defined() || value.dim() != 2 || value.size(0) <= 0 || value.size(1) != width ||
      (type != DType::Float32 && type != DType::Float64) || value.dtype() != type)
    throw std::invalid_argument("AAE requires matching nonempty Float32/64 matrix rows");
  for (double x : value.contiguous().to_vector<double>())
    if (!std::isfinite(x) || (pixels && (x < 0 || x > 1)))
      throw std::invalid_argument("AAE requires finite values and unit-range pixels");
}
} // namespace aae_detail

/// @brief Sum stable Bernoulli cross-entropy over each row's pixels.
///
/// Computes sum_j(softplus(logits_ij)-target_ij*logits_ij). Targets are finite
/// [0,1] values; binary values give Bernoulli negative log likelihood and soft
/// targets give cross-entropy. No pixel averaging, sigmoid clipping or Gaussian
/// KL is added. Finite Float32/64 matrices must share nonempty [B,F] shape/dtype.
/// Strides and gradients through both arguments are preserved; inputs are not
/// mutated. Output owns lifetime; work/storage O(B*F), no shared mutable state.
/// Finite large logits use softplus, avoiding log(sigmoid) underflow; numerical
/// overflow/cancellation in the computation dtype is not repaired.
/// @param logits Finite floating log-odds [B,F].
/// @param target Finite same-shape/dtype target values in [0,1].
/// @return Owning per-row summed cross-entropy [B], in the computation dtype.
/// @throws std::invalid_argument For invalid shapes, dtype or values.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// const auto values=n::models::BernoulliPixelNll(n::zeros({2,3}),n::zeros({2,3}));
/// // Each row is 3*log(2), approximately 2.07944, not a per-pixel mean.
/// @endcode
/// @par References
/// - Makhzani, Shlens, Jaitly, Goodfellow and Frey (2015/2016), Adversarial
///   Autoencoders, https://arxiv.org/abs/1511.05644 . This component supplies the
///   migrated Bernoulli-decoder experiment's summed reconstruction term only.
/// @see AdversarialAutoencoder(), softplus()
inline Tensor BernoulliPixelNll(const Tensor& logits, const Tensor& target) {
  if (!logits.defined() || logits.dim() != 2)
    throw std::invalid_argument("Bernoulli NLL requires matrices");
  aae_detail::Matrix(logits, logits.size(1), logits.dtype(), false);
  aae_detail::Matrix(target, logits.size(1), logits.dtype(), true);
  if (logits.shape() != target.shape())
    throw std::invalid_argument("Bernoulli NLL shapes must match");
  return sum(softplus(logits) - target * logits, {1});
}

/// @brief Encode unit-range pixel rows with an independent tanh/linear deterministic network.
///
/// Uses hidden/W[features,hidden], hidden/b[hidden], code/W[hidden,latent] and
/// code/b[latent], with ordinary Glorot-uniform weights and zero biases. There
/// is no posterior noise, KL term, fitting or normalization. Input checks precede
/// leaf creation; incompatible stored leaves can fail later. Strided matching
/// Float32/64 pixels preserve input and parameter gradients. Returned codes own
/// lifetime and can be signed/unbounded. Stored/computed nonfinite values are
/// not sanitized; shared stores need external synchronization. Work
/// O(B*hidden*(features+latent)), activation storage O(B*(hidden+latent)).
/// @param encoder Mutable matching floating store, separate from other AAE groups.
/// @param image Finite nonempty [B,features] values in [0,1].
/// @param features Image width, default784.
/// @param latent Code width, default4.
/// @param hidden Positive tanh width, default16.
/// @return Owning differentiable linear code [B,latent].
/// @throws std::invalid_argument For invalid widths/input/dtype or incompatible leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// dm::nn::ParameterStore e(3501);
/// const auto code=dm::nn::models::AaeEncode(e,dm::nn::zeros({2,784}));
/// // code is [2,4]; encoder owns four leaves/12628 scalars.
/// @endcode
/// @par References
/// - Makhzani et al. (2015/2016), Adversarial Autoencoders,
///   https://arxiv.org/abs/1511.05644 . This small deterministic encoder is an
///   educational architecture, not the published benchmark configuration.
/// @see AaeDecode(), AdversarialAutoencoder()
inline Tensor AaeEncode(
    ParameterStore& encoder,
    const Tensor&   image,
    std::int64_t    features = 784,
    std::int64_t    latent   = 4,
    std::int64_t    hidden   = 16) {
  aae_detail::Widths(features, latent, hidden);
  aae_detail::Matrix(image, features, encoder.dtype(), true);
  return dense(encoder, "code", dense(encoder, "hidden", image, hidden, Activation::Tanh), latent);
}

/// @brief Decode explicit codes into Bernoulli pixel logits through a tanh/linear network.
///
/// Creates/reuses hidden/W[latent,hidden], hidden/b[hidden], output/W[hidden,features]
/// and output/b[features] with Glorot-uniform weights and zero biases. Only code
/// enters: no image, target, discriminator, copying, fitting or sigmoid is hidden.
/// Signed/zero finite codes are accepted. Strided matching Float32/64 inputs retain
/// code/parameter gradients and output ownership. Validation precedes creation;
/// incompatible leaves may fail later and nonfinite parameters/computed values
/// propagate. Shared stores need synchronization. Work
/// O(B*hidden*(latent+features)), activation storage O(B*(hidden+features)).
/// @param decoder Mutable floating store, separate from the encoder.
/// @param code Finite nonempty [B,latent] code.
/// @param features Output width, default784.
/// @param latent Code width, default4.
/// @param hidden Positive tanh width, default16.
/// @return Owning differentiable logits [B,features]; call sigmoid() for probabilities.
/// @throws std::invalid_argument For invalid widths/code/dtype or incompatible leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore g(3511);
/// const auto pixels=n::sigmoid(n::models::AaeDecode(g,n::zeros({2,4})));
/// // pixels are [2,784]; decoder owns four leaves/13408 scalars.
/// @endcode
/// @par References
/// - Makhzani et al. (2015/2016), https://arxiv.org/abs/1511.05644 . Uses the
///   migrated source's small Bernoulli decoder; outputs are probabilities, not sampled pixels.
/// @see AaeEncode(), BernoulliPixelNll(), AdversarialAutoencoder()
inline Tensor AaeDecode(
    ParameterStore& decoder,
    const Tensor&   code,
    std::int64_t    features = 784,
    std::int64_t    latent   = 4,
    std::int64_t    hidden   = 16) {
  aae_detail::Widths(features, latent, hidden);
  aae_detail::Matrix(code, latent, decoder.dtype(), false);
  return dense(
      decoder, "output", dense(decoder, "hidden", code, hidden, Activation::Tanh), features);
}

/// @brief Evaluate latent prior-versus-encoded logits through an independent tanh discriminator.
///
/// Uses hidden/W[latent,hidden], hidden/b[hidden], logit/W[hidden,1] and logit/b[1]
/// with ordinary Glorot-uniform weights and zero biases (97 default scalars).
/// No labels, sampling, detachment, loss, clipping or optimizer is hidden. Matching
/// finite Float32/64 rows, signed/zero codes and strides are accepted. Returns an
/// owning differentiable value retaining code and discriminator gradients. Input
/// validation precedes creation; incompatible leaves may fail later. Nonfinite
/// stored/computed values propagate; shared stores need synchronization. Work
/// O(B*hidden*latent), activation storage O(B*hidden).
/// @param discriminator Mutable floating store, separate from encoder and decoder.
/// @param code Finite nonempty [B,latent] prior or encoded rows.
/// @param latent Positive code width, default4.
/// @param hidden Positive tanh width, default16.
/// @return Owning differentiable log-odds [B,1], with positive meaning prior/real.
/// @throws std::invalid_argument For invalid widths/code/dtype or incompatible leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// dm::nn::ParameterStore d(3527);
/// const auto logits=dm::nn::models::AaeDiscriminate(d,dm::nn::zeros({2,4}));
/// // logits are [2,1]; no probabilities or labels are supplied to the network.
/// @endcode
/// @par References
/// - Makhzani et al. (2015/2016), https://arxiv.org/abs/1511.05644 . Labels in
///   the caller's objective assign prior=real and encoded=fake.
/// @see AdversarialAutoencoder(), softplus()
inline Tensor AaeDiscriminate(
    ParameterStore& discriminator,
    const Tensor&   code,
    std::int64_t    latent = 4,
    std::int64_t    hidden = 16) {
  aae_detail::Widths(1, latent, hidden);
  aae_detail::Matrix(code, latent, discriminator.dtype(), false);
  return dense(
      discriminator, "logit", dense(discriminator, "hidden", code, hidden, Activation::Tanh), 1);
}

/// @brief Own deterministic codes, Bernoulli outputs and the three AAE scalar objectives.
///
/// Every tensor retains autograd history and owns its lifetime. No optimizer,
/// fitted state, detach barrier or posterior sampling is stored. Full losses
/// retain all mathematical paths; the caller selects which parameter group to update.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore e(3501),g(3511),d(3527);
/// const auto out=n::models::AdversarialAutoencoder(e,g,d,n::zeros({2,784}),n::zeros({3,4}));
/// // Separate prior batch size3 is allowed; code/probability still have2 rows.
/// @endcode
/// @see AdversarialAutoencoder()
struct AaeOutput {
  Tensor code;                ///< Deterministic linear encoded rows [B,latent].
  Tensor logits;              ///< Decoder log-odds [B,features].
  Tensor probability;         ///< Decoder sigmoid probabilities [B,features], not sampled pixels.
  Tensor prior_logit;         ///< Prior/real discriminator logits [P,1].
  Tensor encoded_logit;       ///< Encoded/fake discriminator logits [B,1].
  Tensor nll;                 ///< Per-image summed Bernoulli NLL [B].
  Tensor reconstruction_loss; ///< Mean NLL over B images, scalar.
  Tensor discriminator_loss;  ///< Mean softplus(-prior_logit)+mean softplus(encoded_logit), scalar.
  Tensor adversarial_loss;    ///< Mean softplus(-encoded_logit), non-saturating encoder objective,
                              ///< scalar.
};

/// @brief Evaluate a deterministic AAE and all three explicit reconstruction/adversarial
/// objectives.
///
/// Uses independent AaeEncode(), AaeDecode() and AaeDiscriminate() groups. Real
/// discriminator examples come from explicit prior rows; encoded examples are
/// fake. Reconstruction sums Bernoulli NLL over features then averages images.
/// Discriminator loss averages prior and encoded softplus terms separately,
/// allowing P!=B; encoder adversarial loss uses non-saturating real labels.
/// The 784–16–4 /4–16–784 /4–16–1 default has12 leaves/26133 scalars.
/// There is no Gaussian posterior, per-example KL, ELBO, RNG or hidden training.
/// Prior changes affect discriminator outputs/objectives only, not code or pixels.
///
/// All fields own lifetime and retain all continuous paths, including target
/// derivatives when image requires gradients and encoder gradients of discriminator
/// loss. No group is detached implicitly: callers choose requested gradients and
/// updates. Compute encoder/decoder reconstruction gradients from one snapshot
/// before either update, then recompute for discriminator and encoder-adversarial
/// phases, with four independent optimizer states if following the source protocol.
/// Validation of widths, image/prior, dtype and distinct store identities precedes
/// leaf creation. Incompatible leaves or nonfinite computed logits may fail later.
/// Inputs are not mutated; shared stores require synchronization. Work/storage are
/// those of the three small networks; separate prior batch work is O(P*hidden*latent).
/// @param encoder Mutable floating store, distinct from decoder/discriminator.
/// @param decoder Mutable same-dtype floating store, distinct from other groups.
/// @param discriminator Mutable same-dtype floating store, distinct from other groups.
/// @param image Finite nonempty unit-range [B,features] Float32/64 pixel rows.
/// @param prior Finite same-dtype [P,latent] prior rows, P>0, supplied by caller.
/// @param features Positive image width, default784.
/// @param latent Positive code width, default4.
/// @param hidden Positive tanh width in each network, default16.
/// @return Owning nine-field forward value, with three distinct scalar losses.
/// @throws std::invalid_argument For invalid input/options, aliased/mismatched stores,
///         incompatible leaves or nonfinite logits/encoded values encountered later.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore e(3501),g(3511),d(3527);
/// const auto x=n::full({2,784},.2),prior=n::zeros({2,4});
/// const auto out=n::models::AdversarialAutoencoder(e,g,d,x,prior);
/// auto variables=e.parameters();const auto dec=g.parameters();
/// variables.insert(variables.end(),dec.begin(),dec.end());
/// const auto reconstruction_gradients=n::grad(out.reconstruction_loss,variables);
/// // Compute BOTH groups before writing either. Optimizers/phase loops belong to the caller.
/// @endcode
/// @par References
/// - Makhzani, Shlens, Jaitly, Goodfellow and Frey (2015/2016), Adversarial
///   Autoencoders, https://arxiv.org/abs/1511.05644 . This deterministic tanh,
///   Bernoulli-decoder teaching model matches an aggregate prior adversarially;
///   finite training does not certify distributional matching or useful generation.
/// @see AaeOutput, AaeEncode(), AaeDecode(), AaeDiscriminate(), BernoulliPixelNll()
inline AaeOutput AdversarialAutoencoder(
    ParameterStore& encoder,
    ParameterStore& decoder,
    ParameterStore& discriminator,
    const Tensor&   image,
    const Tensor&   prior,
    std::int64_t    features = 784,
    std::int64_t    latent   = 4,
    std::int64_t    hidden   = 16) {
  aae_detail::Widths(features, latent, hidden);
  if (&encoder == &decoder || &encoder == &discriminator || &decoder == &discriminator ||
      encoder.dtype() != decoder.dtype() || encoder.dtype() != discriminator.dtype())
    throw std::invalid_argument("AAE requires distinct matching-dtype parameter groups");
  aae_detail::Matrix(image, features, encoder.dtype(), true);
  aae_detail::Matrix(prior, latent, encoder.dtype(), false);
  AaeOutput out;
  out.code                = AaeEncode(encoder, image, features, latent, hidden);
  out.logits              = AaeDecode(decoder, out.code, features, latent, hidden);
  out.probability         = sigmoid(out.logits);
  out.nll                 = BernoulliPixelNll(out.logits, image);
  out.reconstruction_loss = mean(out.nll, {0});
  out.prior_logit         = AaeDiscriminate(discriminator, prior, latent, hidden);
  out.encoded_logit       = AaeDiscriminate(discriminator, out.code, latent, hidden);
  out.discriminator_loss =
      mean(softplus(-out.prior_logit), {0, 1}) + mean(softplus(out.encoded_logit), {0, 1});
  out.adversarial_loss = mean(softplus(-out.encoded_logit), {0, 1});
  return out;
}
} // namespace datamunge::nn::models
