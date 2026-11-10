#pragma once
/// @file ladder_variational_autoencoder.hpp
/// @brief Precision-fused Gaussian Ladder VAE forwards and explicit ancestral prior sampling.
#include "datamunge/nn/hierarchical_variational_autoencoder.hpp"

namespace datamunge::nn::models {
/// @brief Own the mean and log variance of a precision-fused diagonal Gaussian.
///
/// Fields own lifetime and retain continuous evidence/prior gradients. No state,
/// variance clipping, fitting or random draws are stored.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// const auto q=n::models::LadderPrecisionFusion(n::full({1,1},2.),n::full({1,1},std::log(4.)),
///     n::full({1,1},-1.),n::full({1,1},std::log(9.)));
/// // Fused mean14/13 and variance36/13, up to floating rounding.
/// @endcode
/// @see LadderPrecisionFusion(), ConditionalGaussianKl()
struct LadderGaussian {
  Tensor mu;           ///< Owning fused posterior mean[B,L].
  Tensor log_variance; ///< Owning fused posterior log variance[B,L].
};

/// @brief Fuse bottom-up Gaussian evidence and a top-down Gaussian prior by adding precisions.
///
/// For evidence N(em,exp(el)) and prior N(pm,exp(pl)), computes log precision
/// log(exp(-el)+exp(-pl)) using a detached maximum shift. Posterior log variance
/// is its negative; posterior mean is exp(post_lv-el)*em+exp(post_lv-pl)*pm.
/// The source parenthesization/rounding is retained. All four finite matching
/// nonempty Float32/64[B,L] inputs remain live, including second derivatives;
/// only the stabilizing maximum is detached. This is precision fusion, not
/// additive residual mean/log-variance heads. Identical input distributions halve
/// variance and retain mean up to rounding, rather than leaving variance unchanged.
/// Strides accepted, no broadcast between different shapes. No raw precision
/// exponential, RNG or input mutation. Outputs own lifetime, no shared state.
/// Work/storage O(B*L). Extreme-log cancellation and arithmetic overflow/underflow
/// are not repaired or clipped; a finite input does not guarantee accurate moments
/// at unrepresentably large log scales.
/// @param evidence_mu Finite evidence means[B,L].
/// @param evidence_lv Finite matching evidence log variances[B,L].
/// @param prior_mu Finite matching conditional prior means[B,L].
/// @param prior_lv Finite matching conditional prior log variances[B,L].
/// @return Owning differentiable fused mean and log variance.
/// @throws std::invalid_argument For invalid shapes/dtypes or nonfinite inputs.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// const auto q=n::models::LadderPrecisionFusion(n::zeros({2,3}),n::zeros({2,3}),
///     n::zeros({2,3}),n::zeros({2,3}));
/// // Fused mean0, log variance-log(2); two equal sources double precision.
/// @endcode
/// @par References
/// - Sonderby, Raiko, Maaloe, Sonderby and Winther (2016), Ladder Variational
///   Autoencoders, https://arxiv.org/abs/1602.02282 . Uses Gaussian precision fusion.
/// @see LadderGaussian, ConditionalGaussianKl(), HierarchyPosterior()
inline LadderGaussian LadderPrecisionFusion(
    const Tensor& evidence_mu,
    const Tensor& evidence_lv,
    const Tensor& prior_mu,
    const Tensor& prior_lv) {
  gaussian_detail::Match(evidence_mu, evidence_lv);
  gaussian_detail::Match(evidence_mu, prior_mu);
  gaussian_detail::Match(evidence_mu, prior_lv);
  const auto     a = -evidence_lv, b = -prior_lv;
  const auto     shift        = maximum(a, b).detach();
  const auto     logprecision = shift + log(exp(a - shift) + exp(b - shift));
  LadderGaussian out;
  out.log_variance = -logprecision;
  const auto wa = exp(out.log_variance - evidence_lv), wb = exp(out.log_variance - prior_lv);
  out.mu = wa * evidence_mu + wb * prior_mu;
  return out;
}

/// @brief Shared dimension checks and conditional prior heads for Ladder forwards.
namespace ladder_detail {
/// @brief Check positive dimensions and representable noise/activation/weight products.
/// @param batch Positive image or ancestral batch.
/// @param features Positive pixel width.
/// @param lower_latent Positive lower Gaussian width.
/// @param top_latent Positive top Gaussian width.
/// @param hidden Positive first-encoder/decoder ReLU width.
/// @param top_hidden Positive second-encoder/conditional-prior ReLU width.
/// @throws std::invalid_argument For invalid widths or int64 product/addition overflow.
inline void Geometry(
    std::int64_t batch,
    std::int64_t features,
    std::int64_t lower_latent,
    std::int64_t top_latent,
    std::int64_t hidden,
    std::int64_t top_hidden) {
  aae_detail::Widths(features, lower_latent, hidden);
  const auto limit = std::numeric_limits<std::int64_t>::max();
  if (batch <= 0 || top_latent <= 0 || top_hidden <= 0 || lower_latent > limit - top_latent ||
      features > limit / hidden || hidden > limit / top_hidden || hidden > limit / lower_latent ||
      top_hidden > limit / top_latent || top_hidden > limit / lower_latent ||
      batch > limit / std::max(features, std::max(hidden, top_hidden)) ||
      batch > limit / (lower_latent + top_latent))
    throw std::invalid_argument(
        "Ladder dimensions need positive representable products and noise width");
}
/// @brief Create/reuse the conditional Gaussian prior from explicit top codes.
/// @param parameters Validated floating trainable store.
/// @param top Finite matching top-code rows[B,T].
/// @param lower_latent Positive lower Gaussian width.
/// @param top_hidden Positive conditional-prior hidden width.
/// @return Owning prior mean and log variance, with live top/parameter paths.
inline LadderGaussian PriorHeads(
    ParameterStore& parameters,
    const Tensor&   top,
    std::int64_t    lower_latent,
    std::int64_t    top_hidden) {
  const auto h  = dense(parameters, "prior_hidden", top, top_hidden, Activation::Relu);
  const auto mu = dense(parameters, "prior_mu", h, lower_latent);
  const auto lv = dense(parameters, "prior_lv", h, lower_latent);
  return {mu, lv};
}
} // namespace ladder_detail

/// @brief Own all six explicit ancestral Ladder endpoints without an image or inference state.
///
/// Every field owns lifetime and retains applicable continuous gradients. No
/// random draws, optimizer or encoder are hidden. Only lower_z enters decoding;
/// top_z affects decoding through the conditional prior and lower sample.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(2503);
/// const auto prior=n::models::LadderPrior(p,n::zeros({2,2}),n::zeros({2,4}));
/// // Zero lower noise makes lower_z exactly prior_mu; probabilities are[2,784].
/// @endcode
/// @see LadderPrior(), GaussianVaeDecode(), LadderVariationalAutoencoder()
struct LadderPriorOutput {
  Tensor top_z;       ///< Owning supplied top samples[B,T], normally from N(0,I).
  Tensor lower_noise; ///< Owning supplied independent lower standard-normal draws[B,L].
  Tensor prior_mu;    ///< Owning learned conditional lower prior means[B,L].
  Tensor prior_lv;    ///< Owning learned conditional lower prior log variances[B,L].
  Tensor lower_z;     ///< Owning lower samples prior_mu+exp(prior_lv/2)*lower_noise[B,L].
  Tensor probability; ///< Owning sigmoid decoder probabilities[B,F], not sampled pixels.
};

/// @brief Sample a Ladder lower prior from explicit top codes/noise and decode without an image.
///
/// Applies prior_hidden ReLU, prior_mu/prior_lv linear heads, GaussianReparameterize
/// and sigmoid(GaussianVaeDecode(lower_z)). Uses the full model's same ten leaves;
/// defaults create26,216 scalars. There is no direct top-to-decoder bypass, encoder,
/// target, inference fusion, RNG draw beyond lazy initialization or hidden no-grad.
/// Finite matching nonempty Float32/64 top[B,T] and lower_noise[B,L] accept strides
/// and preserve both input/parameter paths. Zero lower noise gives the conditional
/// prior mean, not a marginal predictive mean. Caller supplies independent prior
/// draws and owns any no-grad guard, optimizer or checkpoints. Checks precede leaves;
/// incompatible existing leaves/nonfinite computed endpoints may fail later.
/// Inputs unchanged, all results own lifetime; synchronize shared stores. Work
/// O(B*(T*U+U*L+L*H+H*F)); storage includes those dense activations.
/// @param parameters Mutable matching floating store.
/// @param top Finite top prior samples[B,top_latent], B>0.
/// @param lower_noise Finite explicit independent lower draws[B,lower_latent].
/// @param features Positive pixel width, default784.
/// @param lower_latent Positive lower Gaussian width, default4.
/// @param top_latent Positive top Gaussian width, default2.
/// @param hidden Positive decoder ReLU width, default32.
/// @param top_hidden Positive conditional-prior ReLU width, default16.
/// @return Owning six-field ancestral prior output with differentiable probabilities.
/// @throws std::invalid_argument For invalid inputs/dimensions or incompatible leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(2503);
/// const auto prior=n::models::LadderPrior(p,n::ones({20,2}),n::ones({20,4}));
/// const auto decoded=n::sigmoid(n::models::GaussianVaeDecode(p,prior.lower_z));
/// // decoded is exactly prior.probability; neither call uses an image.
/// @endcode
/// @par References
/// - Sonderby et al. (2016), Ladder Variational Autoencoders,
///   https://arxiv.org/abs/1602.02282 . Uses two small Gaussian latent groups.
/// @see LadderPriorOutput, LadderVariationalAutoencoder(), GaussianVaeDecode()
inline LadderPriorOutput LadderPrior(
    ParameterStore& parameters,
    const Tensor&   top,
    const Tensor&   lower_noise,
    std::int64_t    features     = 784,
    std::int64_t    lower_latent = 4,
    std::int64_t    top_latent   = 2,
    std::int64_t    hidden       = 32,
    std::int64_t    top_hidden   = 16) {
  if (!top.defined() || top.dim() != 2)
    throw std::invalid_argument("Ladder prior needs matrix top codes");
  ladder_detail::Geometry(top.size(0), features, lower_latent, top_latent, hidden, top_hidden);
  aae_detail::Matrix(top, top_latent, parameters.dtype(), false);
  aae_detail::Matrix(lower_noise, lower_latent, parameters.dtype(), false);
  if (top.size(0) != lower_noise.size(0))
    throw std::invalid_argument("Ladder prior batches must match");
  const auto prior = ladder_detail::PriorHeads(parameters, top, lower_latent, top_hidden);
  const auto z     = GaussianReparameterize(prior.mu, prior.log_variance, lower_noise);
  return {
      top,
      lower_noise,
      prior.mu,
      prior.log_variance,
      z,
      sigmoid(GaussianVaeDecode(parameters, z, features, lower_latent, hidden))};
}

/// @brief Own sixteen Gaussian evidence, hierarchy, decoder and loss endpoints of a Ladder VAE.
///
/// Fields own lifetime and preserve autograd history. No fitting or RNG state is
/// stored. Recursive zero-noise means are not marginal posterior predictive means.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(2503);
/// const auto out=n::models::LadderVariationalAutoencoder(p,n::full({2,784},.2),n::zeros({2,6}));
/// // lower_z=lower_mu, top_z=top_mu; only the lower four coordinates enter decoding.
/// @endcode
/// @see LadderVariationalAutoencoder(), LadderGaussian, LadderPriorOutput
struct LadderVaeOutput {
  Tensor evidence_mu; ///< Owning bottom-up lower Gaussian evidence means[B,L].
  Tensor evidence_lv; ///< Owning bottom-up evidence log variances[B,L].
  Tensor top_mu;      ///< Owning top posterior means[B,T].
  Tensor top_lv;      ///< Owning top posterior log variances[B,T].
  Tensor top_z;       ///< Owning reparameterized top posterior samples[B,T].
  Tensor prior_mu;    ///< Owning lower conditional prior means[B,L], dependent on top_z.
  Tensor prior_lv;    ///< Owning lower conditional prior log variances[B,L].
  Tensor lower_mu;    ///< Owning precision-fused lower posterior means[B,L].
  Tensor lower_lv;    ///< Owning precision-fused lower posterior log variances[B,L].
  Tensor lower_z;     ///< Owning reparameterized lower posterior samples[B,L].
  Tensor logits;      ///< Owning lower-code-only decoder logits[B,F].
  Tensor probability; ///< Owning decoder sigmoid probabilities[B,F], not sampled pixels.
  Tensor nll;         ///< Owning summed Bernoulli pixel NLL[B], nats.
  Tensor kl_lower;    ///< Owning conditional KL(q1||p1)[B], nats, dependent on sampled top_z.
  Tensor kl_top;      ///< Owning standard-normal top KL[B], nats.
  Tensor loss;        ///< Owning mean(nll+(kl_lower+kl_top)) scalar negative-ELBO estimate.
};

/// @brief Evaluate a two-level precision-fusion Ladder VAE with explicit top and lower noise.
///
/// Image->hidden ReLU->top_hidden ReLU supplies lower evidence from the first
/// feature and top posterior heads from the second. Top noise samples z2; the
/// learned conditional p(z1|z2) combines with evidence via LadderPrecisionFusion.
/// Lower noise samples fused q(z1|x,z2); only z1 enters GaussianVaeDecode. The loss
/// is mean(pixel-summed NLL+(conditional lower KL+standard-normal top KL)), with
/// source reduction grouping. Conditional lower KL is sampled over top latents,
/// so inference should average all three terms over supplied top draws.
///
/// Defaults F784,L4,T2,H32,U16 have22 leaves/52,196 scalars. Creation order is
/// encoder1,encoder2,evidence_mu,evidence_lv,top_mu,top_lv,prior_hidden,prior_mu,
/// prior_lv,decoder,output, with ordinary Glorot weights and zero biases.
/// This small teaching model has no BN, KL warmup, free bits, variance clipping,
/// hidden optimizer or checkpoint selection. The paper's benchmark recipe is not
/// reproduced. No new draws beyond lazy initialization; the caller owns noise.
///
/// Finite nonempty Float32/64 image[B,F] in[0,1] and epsilon[B,L+T] must match the
/// store dtype/batch. LOWER noise comes first, TOP noise last. Strides and all
/// image/noise/parameter paths stay live, including conditional-prior gradients.
/// Validates inputs and representable products before creating leaves. Existing
/// incompatible leaves or nonfinite computed endpoints may fail later. Inputs
/// unchanged; all outputs own lifetime. Synchronize shared stores. Work includes
/// O(B*(F*H+H*U+H*L+U*T+T*U+U*L+L*H)); activation storage is linear in B.
/// @param parameters Mutable floating store; original seed2503.
/// @param image Finite unit-range pixel matrix[B,features], B>0.
/// @param epsilon Finite matching explicit Gaussian draws[B,lower_latent+top_latent].
/// @param features Positive pixel width, default784.
/// @param lower_latent Positive lower Gaussian width, default4.
/// @param top_latent Positive top Gaussian width, default2.
/// @param hidden Positive first-encoder/decoder ReLU width, default32.
/// @param top_hidden Positive second-encoder/prior ReLU width, default16.
/// @return Owning sixteen-field output with differentiable scalar negative ELBO.
/// @throws std::invalid_argument For invalid input/dimensions or incompatible leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(2503);
/// const auto out=n::models::LadderVariationalAutoencoder(p,n::full({2,784},.2),n::ones({2,6}));
/// const auto gradients=n::grad(out.loss,p.parameters());
/// // Caller owns Adam and validation selection; all22 leaves retain their paths.
/// @endcode
/// @par References
/// - Sonderby, Raiko, Maaloe, Sonderby and Winther (2016), Ladder Variational
///   Autoencoders, https://arxiv.org/abs/1602.02282 . Educational precision-fusion mechanism.
/// @see LadderVaeOutput, LadderPrecisionFusion(), ConditionalGaussianKl(), LadderPrior(),
/// GaussianVaeDecode()
inline LadderVaeOutput LadderVariationalAutoencoder(
    ParameterStore& parameters,
    const Tensor&   image,
    const Tensor&   epsilon,
    std::int64_t    features     = 784,
    std::int64_t    lower_latent = 4,
    std::int64_t    top_latent   = 2,
    std::int64_t    hidden       = 32,
    std::int64_t    top_hidden   = 16) {
  if (!image.defined() || image.dim() != 2)
    throw std::invalid_argument("Ladder image must be a pixel matrix");
  ladder_detail::Geometry(image.size(0), features, lower_latent, top_latent, hidden, top_hidden);
  aae_detail::Matrix(image, features, parameters.dtype(), true);
  aae_detail::Matrix(epsilon, lower_latent + top_latent, parameters.dtype(), false);
  if (image.size(0) != epsilon.size(0))
    throw std::invalid_argument("Ladder image and noise batches must match");
  LadderVaeOutput out;
  const auto      h1 = dense(parameters, "encoder1", image, hidden, Activation::Relu);
  const auto      h2 = dense(parameters, "encoder2", h1, top_hidden, Activation::Relu);
  out.evidence_mu    = dense(parameters, "evidence_mu", h1, lower_latent);
  out.evidence_lv    = dense(parameters, "evidence_lv", h1, lower_latent);
  out.top_mu         = dense(parameters, "top_mu", h2, top_latent);
  out.top_lv         = dense(parameters, "top_lv", h2, top_latent);
  const auto e1      = slice_axis(epsilon, 1, 0, lower_latent),
             e2      = slice_axis(epsilon, 1, lower_latent, lower_latent + top_latent);
  out.top_z          = GaussianReparameterize(out.top_mu, out.top_lv, e2);
  const auto prior   = ladder_detail::PriorHeads(parameters, out.top_z, lower_latent, top_hidden);
  out.prior_mu       = prior.mu;
  out.prior_lv       = prior.log_variance;
  const auto q =
      LadderPrecisionFusion(out.evidence_mu, out.evidence_lv, out.prior_mu, out.prior_lv);
  out.lower_mu    = q.mu;
  out.lower_lv    = q.log_variance;
  out.lower_z     = GaussianReparameterize(out.lower_mu, out.lower_lv, e1);
  out.logits      = GaussianVaeDecode(parameters, out.lower_z, features, lower_latent, hidden);
  out.probability = sigmoid(out.logits);
  out.nll         = BernoulliPixelNll(out.logits, image);
  out.kl_lower    = ConditionalGaussianKl(out.lower_mu, out.lower_lv, out.prior_mu, out.prior_lv);
  out.kl_top      = DiagonalGaussianKl(out.top_mu, out.top_lv);
  out.loss        = mean(out.nll + (out.kl_lower + out.kl_top), {0});
  return out;
}
} // namespace datamunge::nn::models
