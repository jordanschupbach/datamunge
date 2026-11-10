#pragma once
/// @file info_variational_autoencoder.hpp
/// @brief Explicit InfoVAE/MMD-VAE objectives and Gaussian/tanh Bernoulli forward components.
#include "datamunge/nn/kernel_discrepancy.hpp"

namespace datamunge::nn::models {
/// @brief Option and architecture validation before creating any InfoVAE leaves.
namespace info_detail {
/// @brief Validate controls and compute local KL/aggregate MMD weights in the computation dtype.
/// @param alpha Finite value in[0,1].
/// @param lambda_weight Finite value>=1.
/// @param type Float32 or Float64 computation dtype.
/// @return Local KL weight1-alpha and aggregate weight(alpha+lambda_weight)-1, promoted to double.
/// @throws std::invalid_argument For invalid/unrepresentable controls or computation dtype.
inline std::pair<double, double> Weights(double alpha, double lambda_weight, DType type) {
  if (!std::isfinite(alpha) || alpha < 0 || alpha > 1 || !std::isfinite(lambda_weight) ||
      lambda_weight < 1 || (type != DType::Float32 && type != DType::Float64))
    throw std::invalid_argument("InfoVAE needs alpha in[0,1],lambda>=1 and floating dtype");
  if (type == DType::Float32) {
    const float a = float(alpha), l = float(lambda_weight);
    if (!std::isfinite(l) || (alpha > 0 && a == 0))
      throw std::invalid_argument("InfoVAE controls must be representable");
    return {double(1.f - a), double((a + l) - 1.f)};
  }
  return {1.0 - alpha, (alpha + lambda_weight) - 1.0};
}
/// @brief Check positive architecture widths and representable weight/activation products.
/// @param batch Positive image or code batch.
/// @param features Positive pixel width.
/// @param latent Positive latent width.
/// @param hidden Positive tanh width.
/// @throws std::invalid_argument For invalid widths/count or product overflow.
inline void
Geometry(std::int64_t batch, std::int64_t features, std::int64_t latent, std::int64_t hidden) {
  aae_detail::Widths(features, latent, hidden);
  const auto limit = std::numeric_limits<std::int64_t>::max();
  if (batch <= 0 || features > limit / hidden || latent > limit / hidden ||
      batch > limit / std::max(features, std::max(latent, hidden)))
    throw std::invalid_argument("InfoVAE architecture products overflow");
}
} // namespace info_detail

/// @brief Combine reconstruction, local Gaussian KL and signed aggregate MMD with explicit
/// controls.
///
/// Returns mean(nll+(1-alpha)*kl)+(alpha+lambda_weight-1)*mmd. NLL/KL are finite
/// matching nonempty Float32/64[B] vectors; mmd has one finite element of the same
/// dtype. Finite signed values, including negative unbiased MMD, are retained.
/// alpha in[0,1],lambda_weight>=1; Float32 narrows controls and computes both
/// coefficients in float before promotion, preserving source rounding. Float64
/// uses full precision. Positive alpha underflow and lambda overflow are rejected.
/// alpha1 removes local KL; alpha0/lambda1 gives mean(NLL+KL), though a full-model
/// call still computes its MMD diagnostics and requires at least two samples.
/// This generally is not an ordinary negative ELBO or a mutual-information estimate.
/// Inputs unchanged; owning scalar, all input paths stay live (zero weights give
/// zero derivatives). No fitting, RNG or shared state. Work O(B), no clipping.
/// @param nll Finite reconstruction negative log densities[B], B>0.
/// @param kl Finite matching local Gaussian KL[B].
/// @param mmd Finite signed scalar/one-element aggregate MMD, same dtype.
/// @param alpha Finite coefficient in[0,1], default0.5.
/// @param lambda_weight Finite aggregate strength>=1, default10.
/// @return Owning scalar InfoVAE objective with live reconstruction/KL/MMD paths.
/// @throws std::invalid_argument For invalid tensors, controls or unrepresentable options.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// const auto
/// value=n::models::InfoVaeObjective(n::tensor({4.,6.}),n::tensor({2.,4.}),n::scalar(.25));
/// // mean(NLL)+.5*mean(KL)+9.5*MMD = 8.875.
/// @endcode
/// @par References
/// - Zhao, Song and Ermon (2017/2018), InfoVAE: Information Maximizing Variational
///   Autoencoders, https://arxiv.org/abs/1706.02262 . Uses IMQ aggregate discrepancy.
/// @see InfoVariationalAutoencoder(), ImqMmd()
inline Tensor InfoVaeObjective(
    const Tensor& nll,
    const Tensor& kl,
    const Tensor& mmd,
    double        alpha         = .5,
    double        lambda_weight = 10) {
  if (!nll.defined() || nll.dim() != 1 || nll.size(0) <= 0 || !kl.defined() ||
      kl.shape() != nll.shape() || kl.dtype() != nll.dtype() || !mmd.defined() ||
      mmd.numel() != 1 || mmd.dtype() != nll.dtype())
    throw std::invalid_argument(
        "InfoVAE objective needs matching floating NLL/KL vectors and one MMD value");
  const auto weights = info_detail::Weights(alpha, lambda_weight, nll.dtype());
  for (const auto& t : {nll, kl, mmd})
    for (double v : t.contiguous().to_vector<double>())
      if (!std::isfinite(v))
        throw std::invalid_argument("InfoVAE objective inputs must be finite");
  return mean(nll + weights.first * kl, {0}) + weights.second * reshape(mmd, {});
}

/// @brief Decode supplied codes through the InfoVAE tanh/linear decoder, without an image.
///
/// Uses decoder/W[L,H],decoder/b[H],output/W[H,F],output/b[F] with ordinary Glorot
/// weights and zero biases; the full model shares these four leaves. This is the
/// tanh decoder, whereas GaussianVaeDecode uses ReLU. Finite signed/zero nonempty
/// matching Float32/64 codes retain code/parameter derivatives and accept strides.
/// Returns logits; apply sigmoid for probabilities, not Bernoulli pixel draws.
/// No encoder, targets, MMD, prior draws, fitting or optimizer is invoked. Prior
/// codes are caller-supplied standard-normal samples. Width/input checks precede
/// leaves; incompatible stored leaves may fail later. Computed overflow is not
/// repaired. Inputs unchanged; result owns lifetime. Synchronize shared stores.
/// Work O(B*H*(L+F)); activation storage O(B*(H+F)). Default decoder has13,408 scalars.
/// @param parameters Mutable matching floating store containing/creating decoder leaves.
/// @param code Finite code matrix[B,latent], B>0.
/// @param features Positive pixel width, default784.
/// @param latent Positive latent width, default4.
/// @param hidden Positive tanh hidden width, default16.
/// @return Owning differentiable logits[B,features].
/// @throws std::invalid_argument For invalid input/widths/overflow or incompatible leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(3901);
/// const auto probability=n::sigmoid(n::models::InfoVaeDecode(p,n::zeros({20,4})));
/// // Twenty image-free prior-code probability images[20,784].
/// @endcode
/// @par References
/// - Zhao, Song and Ermon (2017/2018), https://arxiv.org/abs/1706.02262 . Uses the
///   migrated small tanh teaching architecture, not the paper's benchmark networks.
/// @see InfoVariationalAutoencoder(), GaussianVaeDecode()
inline Tensor InfoVaeDecode(
    ParameterStore& parameters,
    const Tensor&   code,
    std::int64_t    features = 784,
    std::int64_t    latent   = 4,
    std::int64_t    hidden   = 16) {
  if (!code.defined() || code.dim() != 2)
    throw std::invalid_argument("InfoVAE decoder needs matrix codes");
  info_detail::Geometry(code.size(0), features, latent, hidden);
  aae_detail::Matrix(code, latent, parameters.dtype(), false);
  return dense(
      parameters, "output", dense(parameters, "decoder", code, hidden, Activation::Tanh), features);
}

/// @brief Own all sixteen Gaussian, decoder, MMD and objective endpoints of a compact InfoVAE.
///
/// Every tensor owns lifetime and keeps autograd history. No optimizer, RNG or
/// fitting state is stored. MMD matrices include within diagonals for inspection;
/// only scalar within estimates exclude them. Negative unbiased MMD is retained.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(3901);
/// const auto out=n::models::InfoVariationalAutoencoder(p,n::full({2,784},.2),
///     n::ones({2,4}),n::zeros({3,4}),1);
/// // MMD-VAE has zero local KL weight, but out.kl remains a diagnostic; xy is[2,3].
/// @endcode
/// @see InfoVariationalAutoencoder(), MmdOutput, InfoVaeObjective()
struct InfoVaeOutput {
  Tensor mu;            ///< Owning Gaussian posterior means[B,L].
  Tensor log_variance;  ///< Owning posterior log variances[B,L].
  Tensor z;             ///< Owning explicit reparameterized posterior samples[B,L].
  Tensor logits;        ///< Owning tanh/linear decoder logits[B,F], using only z.
  Tensor probability;   ///< Owning sigmoid probabilities[B,F], not sampled pixels.
  Tensor nll;           ///< Owning summed Bernoulli pixel NLL[B], nats.
  Tensor kl;            ///< Owning analytic standard-normal KL[B], nats.
  Tensor negative_elbo; ///< Owning mean(nll+kl), a diagnostic distinct from the training objective.
  Tensor mmd_xx;        ///< Owning posterior-sample kernel matrix[B,B], unit diagonal included.
  Tensor mmd_yy;        ///< Owning supplied prior kernel matrix[P,P], unit diagonal included.
  Tensor mmd_xy;        ///< Owning all posterior-prior cross pairs[B,P].
  Tensor mmd_within_x;  ///< Owning posterior within mean excluding self-pairs.
  Tensor mmd_within_y;  ///< Owning prior within mean excluding self-pairs.
  Tensor mmd_cross;     ///< Owning full cross-pair mean.
  Tensor mmd;           ///< Owning signed unbiased aggregate MMD squared.
  Tensor loss;          ///< Owning scalar mean(NLL+(1-alpha)*KL)+(alpha+lambda-1)*MMD.
};

/// @brief Evaluate an explicit-noise Gaussian InfoVAE/MMD-VAE with an independent supplied prior
/// sample.
///
/// Encoder F->H tanh feeds linear mean/log-variance heads[L]. Explicit epsilon
/// gives z=mu+exp(lv/2)*epsilon; only z enters InfoVaeDecode, no image bypass.
/// The aggregate discrepancy is ImqMmd(z,prior), not a discrepancy of means.
/// Loss is InfoVaeObjective; mean(NLL+KL) remains a separate diagnostic. Ten
/// ordinary Glorot/zero-bias leaves use encoder,mu,log_variance,decoder,output
/// prefixes in that creation order. Defaults F784,H16,L4 contain26,104 scalars.
/// alpha1/lambda10 gives MMD-VAE with zero local KL; alpha.5 retains.5KL+9.5MMD.
/// Objectives with different alpha are not directly comparable optimized ELBOs.
///
/// Requires finite nonempty same-dtype Float32/64 image[B,F] in[0,1], explicit
/// epsilon[B,L] and independent caller-supplied prior[P,L], B,P>=2 (may differ).
/// All image/noise/prior/parameter paths, including second derivatives, stay live.
/// Prior changes affect MMD/loss but not posterior/decoder fields. Independence
/// is a statistical caller responsibility and cannot be inferred from arrays.
/// Strides accepted. Controls, inputs and pairwise/architecture products are
/// checked before leaves. Existing incompatible leaves or nonfinite computed
/// endpoints may fail later. No clipping of signed MMD, optimizer, fit, RNG draws
/// beyond lazy initialization or checkpoint selection is hidden. Inputs unchanged;
/// all fields own lifetime. Synchronize shared stores. Work/storage includes
/// O((B^2+P^2+B*P)*L) for kernels and O(B*H*(F+L)) for the Gaussian network.
/// @param parameters Mutable matching floating store; original seed3901.
/// @param image Finite unit-range pixel rows[B,features], B>=2.
/// @param epsilon Explicit finite matching Gaussian noise[B,latent].
/// @param prior Explicit finite independent standard-normal samples[P,latent], P>=2.
/// @param alpha Finite local/aggregate control in[0,1], default0.5.
/// @param lambda_weight Finite aggregate control>=1, default10.
/// @param features Positive pixel width, default784.
/// @param latent Positive code width, default4.
/// @param hidden Positive tanh hidden width, default16.
/// @return Owning sixteen-field output with signed MMD and separate diagnostic negative ELBO.
/// @throws std::invalid_argument For invalid inputs/options/products or incompatible leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(3901);
/// const auto out=n::models::InfoVariationalAutoencoder(p,n::full({2,784},.2),
///     n::ones({2,4}),n::zeros({3,4}));
/// const auto gradient=n::grad(out.loss,p.parameters());
/// // Caller owns Adam and checkpoint selection; all ten leaves retain gradients.
/// @endcode
/// @par References
/// - Zhao, Song and Ermon (2017/2018), InfoVAE: Information Maximizing Variational
///   Autoencoders, https://arxiv.org/abs/1706.02262 . Small three-scale IMQ teaching variant.
/// @see InfoVaeOutput, InfoVaeObjective(), ImqMmd(), InfoVaeDecode()
inline InfoVaeOutput InfoVariationalAutoencoder(
    ParameterStore& parameters,
    const Tensor&   image,
    const Tensor&   epsilon,
    const Tensor&   prior,
    double          alpha         = .5,
    double          lambda_weight = 10,
    std::int64_t    features      = 784,
    std::int64_t    latent        = 4,
    std::int64_t    hidden        = 16) {
  info_detail::Weights(alpha, lambda_weight, parameters.dtype());
  aae_detail::Matrix(image, features, parameters.dtype(), true);
  aae_detail::Matrix(epsilon, latent, parameters.dtype(), false);
  aae_detail::Matrix(prior, latent, parameters.dtype(), false);
  info_detail::Geometry(image.size(0), features, latent, hidden);
  if (image.size(0) < 2 || prior.size(0) < 2 || epsilon.size(0) != image.size(0))
    throw std::invalid_argument(
        "InfoVAE needs matching image/noise rows and at least two posterior/prior samples");
  mmd_detail::Products(image.size(0), image.size(0), latent);
  mmd_detail::Products(prior.size(0), prior.size(0), latent);
  mmd_detail::Products(image.size(0), prior.size(0), latent);
  InfoVaeOutput out;
  const auto    h        = dense(parameters, "encoder", image, hidden, Activation::Tanh);
  out.mu                 = dense(parameters, "mu", h, latent);
  out.log_variance       = dense(parameters, "log_variance", h, latent);
  out.z                  = GaussianReparameterize(out.mu, out.log_variance, epsilon);
  out.logits             = InfoVaeDecode(parameters, out.z, features, latent, hidden);
  out.probability        = sigmoid(out.logits);
  out.nll                = BernoulliPixelNll(out.logits, image);
  out.kl                 = DiagonalGaussianKl(out.mu, out.log_variance);
  out.negative_elbo      = mean(out.nll + out.kl, {0});
  const auto discrepancy = ImqMmd(out.z, prior);
  out.mmd_xx             = discrepancy.xx;
  out.mmd_yy             = discrepancy.yy;
  out.mmd_xy             = discrepancy.xy;
  out.mmd_within_x       = discrepancy.within_x;
  out.mmd_within_y       = discrepancy.within_y;
  out.mmd_cross          = discrepancy.cross;
  out.mmd                = discrepancy.value;
  out.loss               = InfoVaeObjective(out.nll, out.kl, out.mmd, alpha, lambda_weight);
  return out;
}
} // namespace datamunge::nn::models
