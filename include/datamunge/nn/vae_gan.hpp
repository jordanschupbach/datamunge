#pragma once
/// @file vae_gan.hpp
/// @brief Explicit VAE-GAN components, learned-feature reconstruction and three caller-selected
/// objectives.
#include <algorithm>
#include <limits>

#include "datamunge/nn/gaussian_autoencoder.hpp"

namespace datamunge::nn::models {
/// @brief Internal architecture-product validation before VAE-GAN leaf creation.
namespace vae_gan_detail {
/// @brief Check positive widths and representable weight/activation element counts.
/// @param batch Positive image, code or prior row count.
/// @param features Positive pixel width.
/// @param latent Positive Gaussian code width.
/// @param hidden Positive encoder/generator tanh width.
/// @param feature_width Positive discriminator tanh width.
/// @throws std::invalid_argument For nonpositive dimensions or int64 product overflow.
inline void Geometry(
    std::int64_t batch,
    std::int64_t features,
    std::int64_t latent,
    std::int64_t hidden,
    std::int64_t feature_width) {
  aae_detail::Widths(features, latent, hidden);
  const auto limit = std::numeric_limits<std::int64_t>::max();
  if (feature_width <= 0 || batch <= 0 || features > limit / hidden || latent > limit / hidden ||
      features > limit / feature_width ||
      batch > limit / std::max({features, latent, hidden, feature_width}))
    throw std::invalid_argument("VAE-GAN dimensions or element products are invalid");
}
} // namespace vae_gan_detail

/// @brief Sum squared discriminator-feature residuals against a detached real target.
///
/// Returns .5*sum((real.detach()-reconstruction)^2,axis=1). Both arguments are
/// finite matching nonempty Float32/64 [B,D] matrices; values need not be pixels
/// or bounded. Feature coordinates are summed, not averaged. The unit-variance
/// Gaussian normalizing constant D*log(2*pi)/2 is omitted. This is a learned
/// feature-space residual, not an image-space likelihood or pixel ELBO.
/// Real targets are detached at every derivative order; reconstructed features
/// retain first/higher gradients and strides. Inputs are unchanged; output owns
/// lifetime. O(B*D) work/storage, without shared mutable state. Computation-dtype
/// overflow is not repaired.
/// @param real Finite floating real-image feature targets [B,D].
/// @param reconstruction Finite matching reconstructed-image features [B,D].
/// @return Owning differentiable [B] half-summed squared feature distances.
/// @throws std::invalid_argument For invalid shape, dtype or nonfinite values.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// const auto distance=n::models::VaeGanFeatureNll(n::ones({2,3}),n::zeros({2,3}));
/// // Each row is 1.5, without the Gaussian normalizing constant.
/// @endcode
/// @par References
/// - Larsen, Sønderby, Larochelle and Winther (2016), Autoencoding beyond pixels
///   using a learned similarity metric, https://proceedings.mlr.press/v48/larsen16.html ,
///   equations 6-7. Target detachment follows the migrated teaching implementation.
/// @see VaeGan(), VaeGanDiscriminator()
inline Tensor VaeGanFeatureNll(const Tensor& real, const Tensor& reconstruction) {
  gaussian_detail::Match(real, reconstruction);
  return 0.5 * sum(square(real.detach() - reconstruction), {1});
}

/// @brief Decode explicit Gaussian codes with the source tanh/sigmoid generator.
///
/// Reuses AaeDecode()'s hidden/W[latent,hidden], hidden/b[hidden],
/// output/W[hidden,features], output/b[features], then applies sigmoid. Ordinary
/// Glorot-uniform weights and zero biases give four leaves/13408 default scalars.
/// Only codes enter; there is no image, posterior, RNG, detach, fit or optimizer.
/// Finite signed/zero nonempty matching Float32/64 rows and strides retain code
/// and generator gradients. Checks precede creation; incompatible leaves or
/// nonfinite computed states can fail later. Outputs own lifetime, with no
/// input mutation. Work O(P*hidden*(latent+features)); activation storage
/// O(P*(hidden+features)). Synchronize callers sharing a mutable store.
/// @param generator Mutable floating generator store.
/// @param code Finite nonempty [P,latent] codes matching the store dtype.
/// @param features Positive output pixel width, default784.
/// @param latent Positive code width, default4.
/// @param hidden Positive tanh width, default16.
/// @return Owning differentiable sigmoid probabilities [P,features], not pixel draws.
/// @throws std::invalid_argument For invalid input/widths/products or incompatible leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// dm::nn::ParameterStore g(3307);
/// const auto pixels=dm::nn::models::VaeGanDecode(g,dm::nn::zeros({2,4}));
/// // pixels is [2,784]; decoding requires no encoder or image.
/// @endcode
/// @par References
/// - Larsen et al. (2016), https://proceedings.mlr.press/v48/larsen16.html .
///   This compact dense teaching generator differs from the paper's convolutional model.
/// @see VaeGan(), AaeDecode()
inline Tensor VaeGanDecode(
    ParameterStore& generator,
    const Tensor&   code,
    std::int64_t    features = 784,
    std::int64_t    latent   = 4,
    std::int64_t    hidden   = 16) {
  if (!code.defined() || code.dim() != 2)
    throw std::invalid_argument("VAE-GAN generator requires code matrices");
  vae_gan_detail::Geometry(code.size(0), features, latent, hidden, 1);
  return sigmoid(AaeDecode(generator, code, features, latent, hidden));
}

/// @brief Own a discriminator's learned feature rows and real-versus-fake logits.
///
/// Both fields retain autograd history and independent lifetime after this record
/// or the parameter store is destroyed. The record contains no fitted state,
/// optimizer, labels or detached fields.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// dm::nn::ParameterStore d(3313);
/// const auto out=dm::nn::models::VaeGanDiscriminator(d,dm::nn::zeros({2,784}));
/// // feature is [2,8]; logit is [2,1], with positive meaning real.
/// @endcode
/// @see VaeGanDiscriminator(), VaeGanOutput
struct VaeGanDiscriminatorOutput {
  Tensor feature; ///< Tanh discriminator representation [P,feature_width].
  Tensor logit;   ///< Linear real-image log-odds [P,1].
};

/// @brief Compute source tanh discriminator features and linear real/fake logits.
///
/// Creates/reuses feature/W[features,feature_width], feature/b[feature_width],
/// logit/W[feature_width,1] and logit/b[1] in that order, using Glorot-uniform
/// weights and zero biases (four leaves/6289 default scalars). Finite nonempty
/// matching Float32/64 matrices are accepted, including signed inputs and
/// strides; no pixel clipping, labels, sigmoid, detach or fit is hidden. Input
/// checks precede creation; incompatible leaves can fail later. Nonfinite computed
/// states propagate. Both owned outputs retain input/parameter gradients;
/// inputs are unchanged. Work O(P*features*feature_width), activation storage
/// O(P*feature_width). Synchronize shared stores.
/// @param discriminator Mutable floating discriminator store.
/// @param image Finite nonempty matching-dtype rows [P,features].
/// @param features Positive input width, default784.
/// @param feature_width Positive tanh feature width, default8.
/// @return Owning feature and logit fields, without a loss or probabilities.
/// @throws std::invalid_argument For invalid inputs/widths/products or incompatible leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore d(3313,n::DType::Float64);
/// const auto out=n::models::VaeGanDiscriminator(d,n::full({3,2},-.2,n::DType::Float64),2,4);
/// // out.feature is [3,4]; signed inputs are allowed in this standalone component.
/// @endcode
/// @par References
/// - Larsen et al. (2016), https://proceedings.mlr.press/v48/larsen16.html .
///   Features implement the compact teaching experiment's learned reconstruction metric.
/// @see VaeGanFeatureNll(), VaeGan()
inline VaeGanDiscriminatorOutput VaeGanDiscriminator(
    ParameterStore& discriminator,
    const Tensor&   image,
    std::int64_t    features      = 784,
    std::int64_t    feature_width = 8) {
  if (!image.defined() || image.dim() != 2)
    throw std::invalid_argument("VAE-GAN discriminator requires image matrices");
  vae_gan_detail::Geometry(image.size(0), features, 1, 1, feature_width);
  aae_detail::Matrix(image, features, discriminator.dtype(), false);
  auto feature = dense(discriminator, "feature", image, feature_width, Activation::Tanh);
  return {feature, dense(discriminator, "logit", feature, 1)};
}

/// @brief Own all nineteen source VAE-GAN endpoints and its three explicit objectives.
///
/// Every tensor owns lifetime and retains available autograd paths. The sole
/// deliberate feature-target barrier is inside feature_nll; real_feature itself
/// stays live. No RNG, optimizer, fitting counter or mode switch is stored.
/// The caller chooses which parameter group receives each objective's update.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore e(3301),g(3307),d(3313);
/// const auto out=n::models::VaeGan(e,g,d,n::zeros({2,784}),n::zeros({2,4}),n::zeros({3,4}));
/// // Distinct prior batch size3 is allowed; reconstruction still has2 rows.
/// @endcode
/// @see VaeGan(), VaeGanDiscriminatorOutput
struct VaeGanOutput {
  Tensor mu;                     ///< Posterior means [B,latent].
  Tensor log_variance;           ///< Posterior log variances [B,latent].
  Tensor z;                      ///< Explicit reparameterized samples [B,latent].
  Tensor probability;            ///< Reconstructed sigmoid probabilities [B,features].
  Tensor prior_probability;      ///< Independent prior-code probabilities [P,features].
  Tensor real_feature;           ///< Live real-image discriminator features [B,feature_width].
  Tensor reconstruction_feature; ///< Reconstructed-image features [B,feature_width].
  Tensor prior_feature;          ///< Prior-image discriminator features [P,feature_width].
  Tensor real_logit;             ///< Real-image discriminator logits [B,1].
  Tensor reconstruction_logit;   ///< Reconstructed-image logits [B,1].
  Tensor prior_logit;            ///< Prior-image logits [P,1].
  Tensor feature_nll;           ///< Half-summed squared features against detached real targets [B].
  Tensor kl;                    ///< Coordinate-summed analytic standard-normal Gaussian KL [B].
  Tensor pixel_nll;             ///< Clipped Bernoulli pixel-NLL diagnostic [B], not fitted.
  Tensor mse;                   ///< Pixel-mean squared reconstruction error [B], not fitted.
  Tensor encoder_loss;          ///< Scalar mean(feature_nll+kl), for encoder-only updates.
  Tensor generator_loss;        ///< Scalar mean(feature_nll)+generator_adversarial, for G only.
  Tensor discriminator_loss;    ///< Scalar real logistic loss plus half of each fake-branch loss.
  Tensor generator_adversarial; ///< Scalar mean of two non-saturating fake-as-real losses.
};

/// @brief Evaluate a Gaussian VAE-GAN with explicit posterior noise and independent prior codes.
///
/// Encoder: features->hidden tanh->mu/log_variance linear heads. Generator:
/// VaeGanDecode() is shared by posterior and prior codes. Discriminator:
/// VaeGanDiscriminator() is shared by real, reconstructed and prior pixels.
/// Creation order/names match the source: E hidden,mu,log_variance; G hidden,output;
/// D feature,logit (each W then b). Defaults give E12696+G13408+D6289=32393
/// scalars in 6+4+4 leaves. No images bypass the sampled latent into the generator.
///
/// Encoder objective is mean(feature_nll+KL). Generator objective is
/// mean(feature_nll)+.5*(mean(softplus(-reconstruction_logit))+
/// mean(softplus(-prior_logit))). Discriminator objective is
/// mean(softplus(-real_logit))+.5*(mean(softplus(reconstruction_logit))+
/// mean(softplus(prior_logit))). Each branch averages its own rows, so P may
/// differ from B. These are separate objectives, not a single pixel ELBO.
/// The source uses unit feature weight and balanced, non-saturating fake branches.
/// Pixel NLL is only diagnostic: sigmoid probabilities are clipped to the
/// source's Float32-represented 1e-6 and 1-1e-6 endpoints, even in Float64.
///
/// No fit, RNG, update, NoGradGuard or implicit parameter freezing is hidden.
/// Select only D leaves for discriminator_loss, E leaves for encoder_loss and
/// G leaves for generator_loss; recompute the forward after each written group.
/// Other mathematical paths remain live, except the real-feature target detach.
/// Finite unit-range images, finite noise/prior and all three distinct stores
/// must share Float32/64 dtype. Strides and continuous higher derivatives are
/// retained. Input/geometry checks precede leaf creation; incompatible leaves
/// or nonfinite intermediate states may fail later. Numerical overflow is not
/// repaired. Inputs are unchanged; returned fields own lifetime. Work
/// O((B+P)*features*(hidden+feature_width)+(B+P)*latent*hidden), activation
/// storage O((B+P)*(features+hidden+latent+feature_width)). Shared stores need
/// external synchronization.
/// @param encoder Mutable independent floating encoder store.
/// @param generator Mutable independent generator store, matching encoder dtype.
/// @param discriminator Mutable independent discriminator store, matching dtype.
/// @param image Finite nonempty unit-range [B,features] pixel rows.
/// @param epsilon Explicit finite standard-normal noise [B,latent], matching dtype.
/// @param prior Explicit finite independent prior codes [P,latent], matching dtype.
/// @param features Positive pixel width, default784.
/// @param latent Positive Gaussian width, default4.
/// @param hidden Positive encoder/generator tanh width, default16.
/// @param feature_width Positive discriminator tanh width, default8.
/// @return Owning nineteen-field VaeGanOutput, with three separate scalar objectives.
/// @throws std::invalid_argument For invalid inputs/options/aliased stores or incompatible leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore e(3301),g(3307),d(3313);
/// const auto out=n::models::VaeGan(e,g,d,n::zeros({2,784}),n::zeros({2,4}),n::zeros({2,4}));
/// const auto de=n::grad(out.encoder_loss,e.parameters());
/// // Apply de only to E with a caller-owned optimizer; recompute before the next group.
/// @endcode
/// @par References
/// - Larsen, Sønderby, Larochelle and Winther (2016), Autoencoding beyond pixels
///   using a learned similarity metric, https://proceedings.mlr.press/v48/larsen16.html .
///   This dense, balanced-fake, non-saturating teaching variant does not reproduce
///   the paper's convolutional architecture or RMSProp benchmark training.
/// @see VaeGanDecode(), VaeGanDiscriminator(), VaeGanFeatureNll(), GaussianReparameterize()
inline VaeGanOutput VaeGan(
    ParameterStore& encoder,
    ParameterStore& generator,
    ParameterStore& discriminator,
    const Tensor&   image,
    const Tensor&   epsilon,
    const Tensor&   prior,
    std::int64_t    features      = 784,
    std::int64_t    latent        = 4,
    std::int64_t    hidden        = 16,
    std::int64_t    feature_width = 8) {
  if (!image.defined() || image.dim() != 2 || !prior.defined() || prior.dim() != 2)
    throw std::invalid_argument("VAE-GAN requires image and prior matrices");
  vae_gan_detail::Geometry(image.size(0), features, latent, hidden, feature_width);
  vae_gan_detail::Geometry(prior.size(0), features, latent, hidden, feature_width);
  if (&encoder == &generator || &encoder == &discriminator || &generator == &discriminator ||
      encoder.dtype() != generator.dtype() || encoder.dtype() != discriminator.dtype())
    throw std::invalid_argument("VAE-GAN requires distinct matching-dtype parameter groups");
  aae_detail::Matrix(image, features, encoder.dtype(), true);
  aae_detail::Matrix(epsilon, latent, encoder.dtype(), false);
  aae_detail::Matrix(prior, latent, encoder.dtype(), false);
  if (epsilon.size(0) != image.size(0))
    throw std::invalid_argument("VAE-GAN posterior noise rows must match images");
  VaeGanOutput out;
  auto         h        = dense(encoder, "hidden", image, hidden, Activation::Tanh);
  out.mu                = dense(encoder, "mu", h, latent);
  out.log_variance      = dense(encoder, "log_variance", h, latent);
  out.z                 = GaussianReparameterize(out.mu, out.log_variance, epsilon);
  out.probability       = VaeGanDecode(generator, out.z, features, latent, hidden);
  out.prior_probability = VaeGanDecode(generator, prior, features, latent, hidden);
  auto real             = VaeGanDiscriminator(discriminator, image, features, feature_width);
  auto rec = VaeGanDiscriminator(discriminator, out.probability, features, feature_width);
  auto generated =
      VaeGanDiscriminator(discriminator, out.prior_probability, features, feature_width);
  out.real_feature           = real.feature;
  out.reconstruction_feature = rec.feature;
  out.prior_feature          = generated.feature;
  out.real_logit             = real.logit;
  out.reconstruction_logit   = rec.logit;
  out.prior_logit            = generated.logit;
  out.feature_nll            = VaeGanFeatureNll(real.feature, rec.feature);
  out.kl                     = DiagonalGaussianKl(out.mu, out.log_variance);
  out.encoder_loss           = mean(out.feature_nll + out.kl, {0});
  out.generator_adversarial =
      0.5 * (mean(softplus(neg(rec.logit)), {0, 1}) + mean(softplus(neg(generated.logit)), {0, 1}));
  out.generator_loss = mean(out.feature_nll, {0}) + out.generator_adversarial;
  out.discriminator_loss =
      mean(softplus(neg(real.logit)), {0, 1}) +
      0.5 * (mean(softplus(rec.logit), {0, 1}) + mean(softplus(generated.logit), {0, 1}));
  auto clipped  = clip(out.probability, double(1e-6f), double(1.f - 1e-6f));
  out.pixel_nll = neg(sum(image * log(clipped) + (1.0 - image) * log(1.0 - clipped), {1}));
  out.mse       = mean(square(image - out.probability), {1});
  return out;
}
} // namespace datamunge::nn::models
