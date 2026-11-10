#pragma once
/// @file gaussian_autoencoder.hpp
/// @brief Explicit diagonal-Gaussian posterior operations and beta-weighted Bernoulli VAEs.
#include "datamunge/nn/adversarial_autoencoder.hpp"

namespace datamunge::nn::models {
/// @brief Internal matching-matrix validation for Gaussian posterior operations.
namespace gaussian_detail {
/// @brief Require two finite nonempty floating matrices with identical shape and dtype.
/// @param a First matrix [B,L].
/// @param b Second matrix [B,L].
/// @throws std::invalid_argument For invalid shape, values or dtype.
inline void Match(const Tensor& a, const Tensor& b) {
  if (!a.defined() || a.dim() != 2)
    throw std::invalid_argument("Gaussian posterior requires matrices");
  aae_detail::Matrix(a, a.size(1), a.dtype(), false);
  aae_detail::Matrix(b, a.size(1), a.dtype(), false);
  if (a.shape() != b.shape())
    throw std::invalid_argument("Gaussian posterior shapes must match");
}
} // namespace gaussian_detail

/// @brief Transform explicit standard-normal noise into a diagonal-Gaussian posterior sample.
///
/// Returns mu+exp(log_variance/2)*epsilon, without RNG, clipping or detachment.
/// All inputs must be finite nonempty matching Float32/64 [B,L] matrices; strides
/// and derivatives through all three inputs are retained. Zero epsilon gives mu
/// exactly for finite computed scale. Exp overflow/underflow is not repaired.
/// Inputs are unchanged; output owns lifetime. Work/storage O(B*L); no shared state.
/// @param mu Finite posterior means [B,L].
/// @param log_variance Finite log variances, same shape/dtype as mu.
/// @param epsilon Explicit finite standard-normal draws, same shape/dtype as mu.
/// @return Owning differentiable posterior sample [B,L].
/// @throws std::invalid_argument For invalid shape, dtype or values.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// const auto z=n::models::GaussianReparameterize(n::ones({1,2}),n::zeros({1,2}),n::zeros({1,2}));
/// // z is exactly [1,1]. The caller supplies the noise.
/// @endcode
/// @par References
/// - Kingma and Welling (2013/2014), Auto-Encoding Variational Bayes,
///   https://arxiv.org/abs/1312.6114 . Uses the diagonal-Gaussian pathwise estimator.
/// @see DiagonalGaussianKl(), BetaVariationalAutoencoder(), GaussianNoise()
inline Tensor
GaussianReparameterize(const Tensor& mu, const Tensor& log_variance, const Tensor& epsilon) {
  gaussian_detail::Match(mu, log_variance);
  gaussian_detail::Match(mu, epsilon);
  return mu + exp(0.5 * log_variance) * epsilon;
}

/// @brief Sum analytic KL from a diagonal Gaussian to a standard normal over latent coordinates.
///
/// Computes 0.5*sum(mu^2+exp(log_variance)-1-log_variance,axis=1), without
/// sampling, averaging coordinates or clipping small negative roundoff. Finite
/// nonempty matching Float32/64 [B,L] inputs preserve strides and both gradients.
/// Zero means/log variances yield exact zero. Computed overflow is not repaired.
/// Inputs are unchanged; owning output. O(B*L) work/storage, no shared state.
/// @param mu Finite means [B,L].
/// @param log_variance Finite log variances with identical shape/dtype.
/// @return Owning differentiable per-row KL [B], in nats.
/// @throws std::invalid_argument For invalid input shape, dtype or values.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const auto kl=dm::nn::models::DiagonalGaussianKl(dm::nn::ones({2,3}),dm::nn::zeros({2,3}));
/// // Each row has KL 1.5 nats.
/// @endcode
/// @par References
/// - Kingma and Welling (2013/2014), Auto-Encoding Variational Bayes,
///   https://arxiv.org/abs/1312.6114 , Appendix B.
/// @see GaussianReparameterize(), BetaVariationalAutoencoder()
inline Tensor DiagonalGaussianKl(const Tensor& mu, const Tensor& log_variance) {
  gaussian_detail::Match(mu, log_variance);
  return 0.5 * sum(square(mu) + exp(log_variance) - 1.0 - log_variance, {1});
}

/// @brief Decode explicit Gaussian-latent codes through a ReLU/linear Bernoulli decoder.
///
/// Uses decoder/W[latent,hidden], decoder/b[hidden], output/W[hidden,features],
/// output/b[features], ordinary Glorot-uniform weights and zero biases. The same
/// names are used by BetaVariationalAutoencoder(); reuse its store for inference.
/// Only code enters, with no image, posterior, RNG or hidden sigmoid. Finite signed
/// or zero matching Float32/64 codes retain code/parameter gradients and strides.
/// Input checks precede leaf creation; incompatible stored leaves can fail later.
/// Nonfinite computed values propagate. Inputs unchanged; output owns lifetime.
/// Work O(B*hidden*(latent+features)); shared stores need external synchronization.
/// @param parameters Mutable floating store containing or creating decoder leaves.
/// @param code Finite nonempty [B,latent] code with store dtype.
/// @param features Positive output width, default784.
/// @param latent Positive code width, default4.
/// @param hidden Positive ReLU width, default32.
/// @return Owning differentiable logits [B,features]; sigmoid gives probabilities.
/// @throws std::invalid_argument For invalid widths/code/dtype or incompatible leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(2237);
/// const auto probabilities=n::sigmoid(n::models::GaussianVaeDecode(p,n::zeros({2,4})));
/// // [2,784] probabilities, not sampled Bernoulli pixels.
/// @endcode
/// @par References
/// - Kingma and Welling (2013/2014), https://arxiv.org/abs/1312.6114 . This small
///   migrated ReLU decoder is an educational architecture, not a benchmark reproduction.
/// @see BetaVariationalAutoencoder(), GaussianReparameterize()
inline Tensor GaussianVaeDecode(
    ParameterStore& parameters,
    const Tensor&   code,
    std::int64_t    features = 784,
    std::int64_t    latent   = 4,
    std::int64_t    hidden   = 32) {
  aae_detail::Widths(features, latent, hidden);
  aae_detail::Matrix(code, latent, parameters.dtype(), false);
  return dense(
      parameters, "output", dense(parameters, "decoder", code, hidden, Activation::Relu), features);
}

/// @brief Own Gaussian posterior parameters, explicit samples and beta-weighted VAE objectives.
///
/// Every tensor owns lifetime and retains autograd history. No RNG, optimizer or
/// fitting state is stored. Decode(mu) is not the posterior predictive mean.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(2237);
/// const auto out=n::models::BetaVariationalAutoencoder(p,n::zeros({2,784}),n::zeros({2,4}),4);
/// // out.loss is mean(NLL+4*KL); out.unweighted_loss is mean(NLL+KL).
/// @endcode
/// @see BetaVariationalAutoencoder()
struct GaussianVaeOutput {
  Tensor mu;                 ///< Linear posterior means [B,latent].
  Tensor log_variance;       ///< Linear posterior log variances [B,latent].
  Tensor z;                  ///< Explicit reparameterized samples [B,latent].
  Tensor logits;             ///< Decoder log-odds [B,features].
  Tensor probability;        ///< Sigmoid decoder probabilities [B,features], not pixel draws.
  Tensor reconstruction_nll; ///< Summed Bernoulli NLL per image [B], nats.
  Tensor kl;                 ///< Summed Gaussian KL per image [B], nats.
  Tensor loss;               ///< Mean reconstruction_nll+beta*kl, scalar nats/image.
  Tensor unweighted_loss;    ///< Mean reconstruction_nll+kl, scalar negative ELBO estimate.
};

/// @brief Evaluate a beta-weighted diagonal-Gaussian VAE with caller-supplied posterior noise.
///
/// Architecture: features->hidden ReLU->separate linear mu/log_variance heads,
/// sample z=mu+exp(log_variance/2)*epsilon, then GaussianVaeDecode(). Names/order
/// are encoder/W,b; mu/W,b; log_variance/W,b; decoder/W,b; output/W,b. Ordinary
/// Glorot-uniform weights and zero biases give10 leaves/51416 default scalars.
/// Pixel NLL and latent KL are SUMMED, then images are averaged. beta=1 is a
/// single-draw negative ELBO; beta>1 strengthens KL but does not guarantee
/// disentanglement. No annealing, sampling, training, clipping or factor labels
/// are hidden. Independent explicit noise banks are the caller's responsibility.
///
/// Finite unit-range images and finite noise must match store Float32/64 dtype.
/// Continuous image, epsilon and all leaf derivatives are retained. Input/beta
/// checks precede creation; incompatible stored leaves/nonfinite computed states
/// can fail later. beta must remain finite positive when cast to store dtype.
/// No inputs mutated; output owns lifetime. Work O(B*hidden*(features+latent)),
/// activation storage O(B*(features+hidden+latent)); synchronize shared stores.
/// @param parameters Mutable floating parameter store.
/// @param image Finite nonempty [B,features] pixel rows in [0,1].
/// @param epsilon Explicit finite [B,latent] standard-normal noise, same dtype.
/// @param beta Finite positive KL weight, default1.
/// @param features Positive pixel width, default784.
/// @param latent Positive latent width, default4.
/// @param hidden Positive encoder/decoder ReLU width, default32.
/// @return Owning nine-field GaussianVaeOutput including weighted/unweighted losses.
/// @throws std::invalid_argument For invalid inputs/options or incompatible leaves;
///         nonfinite computed posterior/logits may fail validation later.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(2237);
/// const auto
/// noise=n::cast(n::select(n::models::GaussianNoise({2,4},1,2252),0,0),n::DType::Float32); const
/// auto out=n::models::BetaVariationalAutoencoder(p,n::full({2,784},.2),noise,4); const auto
/// gradients=n::grad(out.loss,p.parameters()); // all10 leaves; no update is hidden.
/// @endcode
/// @par References
/// - Kingma and Welling (2013/2014), https://arxiv.org/abs/1312.6114 .
/// - Higgins et al. (2017), beta-VAE: Learning Basic Visual Concepts with a
///   Constrained Variational Framework, https://openreview.net/forum?id=Sy2fzU9gl .
///   Uses the migrated small ReLU/Bernoulli model, not the paper's benchmark metric.
/// @see GaussianVaeOutput, GaussianReparameterize(), DiagonalGaussianKl(), GaussianVaeDecode(),
/// BernoulliPixelNll()
inline GaussianVaeOutput BetaVariationalAutoencoder(
    ParameterStore& parameters,
    const Tensor&   image,
    const Tensor&   epsilon,
    double          beta     = 1,
    std::int64_t    features = 784,
    std::int64_t    latent   = 4,
    std::int64_t    hidden   = 32) {
  aae_detail::Widths(features, latent, hidden);
  aae_detail::Matrix(image, features, parameters.dtype(), true);
  aae_detail::Matrix(epsilon, latent, parameters.dtype(), false);
  if (image.size(0) != epsilon.size(0) || !std::isfinite(beta) || beta <= 0 ||
      (parameters.dtype() == DType::Float32 && (!std::isfinite(float(beta)) || float(beta) <= 0)))
    throw std::invalid_argument("VAE requires matching batches and representable positive beta");
  GaussianVaeOutput out;
  const auto        h    = dense(parameters, "encoder", image, hidden, Activation::Relu);
  out.mu                 = dense(parameters, "mu", h, latent);
  out.log_variance       = dense(parameters, "log_variance", h, latent);
  out.z                  = GaussianReparameterize(out.mu, out.log_variance, epsilon);
  out.logits             = GaussianVaeDecode(parameters, out.z, features, latent, hidden);
  out.probability        = sigmoid(out.logits);
  out.reconstruction_nll = BernoulliPixelNll(out.logits, image);
  out.kl                 = DiagonalGaussianKl(out.mu, out.log_variance);
  out.unweighted_loss    = mean(out.reconstruction_nll + out.kl, {0});
  out.loss               = mean(out.reconstruction_nll + beta * out.kl, {0});
  return out;
}
} // namespace datamunge::nn::models
