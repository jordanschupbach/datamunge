#pragma once
/// @file importance_weighted_autoencoder.hpp
/// @brief Explicit sampled-density IWAE bounds, importance diagnostics and Gaussian forwards.
#include <limits>

#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/gaussian_autoencoder.hpp"

namespace datamunge::nn::models {
/// @brief Reduce sample-major log weights with a numerically shifted log-mean-exp.
///
/// For finite nonempty Float32/64[K,B] input, returns log(sum_k exp(w[k,b]))-log(K).
/// The column maximum is gradient-stopped; all log-weight paths remain live,
/// including second derivatives. Float32 rounds log(K) to float before subtracting,
/// preserving the migrated calculation; Float64 uses full precision. K=1 returns
/// the input row exactly. Equal samples give the same value up to reduction rounding.
/// Finite extreme positive/negative weights avoid raw-exponential overflow, although
/// arithmetic overflow is not repaired. Strides are accepted. No RNG, mutation or
/// shared state; result owns lifetime. Work/storage O(K*B).
/// @param log_weights Finite matching floating log weights[K,B], K,B positive.
/// @return Owning differentiable per-case log mean importance weight[B], in nats.
/// @throws std::invalid_argument For invalid shape/dtype or nonfinite weights.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// const auto bound=n::models::IwaeLogMeanExp(n::tensor({1000.,1001.,999.},{3,1}));
/// // Approximately1000.308994; raw exp(1000) would overflow Float64.
/// @endcode
/// @par References
/// - Burda, Grosse and Salakhutdinov (2015/2016), Importance Weighted Autoencoders,
///   https://arxiv.org/abs/1509.00519 . Uses the multi-sample importance bound.
/// @see ImportanceWeightedBound(), ImportanceWeightedAutoencoder()
inline Tensor IwaeLogMeanExp(const Tensor& log_weights) {
  if (!log_weights.defined() || log_weights.dim() != 2)
    throw std::invalid_argument("IWAE log weights require a floating sample-by-case matrix");
  aae_detail::Matrix(log_weights, log_weights.size(1), log_weights.dtype(), false);
  const auto maximum     = stop_gradient(max(log_weights, {0}, true));
  double     log_samples = std::log(double(log_weights.size(0)));
  if (log_weights.dtype() == DType::Float32)
    log_samples = double(float(log_samples));
  return squeeze(maximum + log(sum(exp(log_weights - maximum), {0}, true)), {0}) - log_samples;
}

/// @brief Own differentiable sampled IWAE objectives and importance-weight diagnostics.
///
/// Fields own lifetime and retain autograd history. No fitting state, RNG, analytic
/// KL substitution or gradient reweighting is hidden. A finite sampled bound is
/// not an exact likelihood; a particular nested draw need not improve with K.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// const auto out=n::models::ImportanceWeightedBound(n::zeros({6,2}),n::zeros({6,2}),
///     n::ones({6,2}),n::full({6},3.),3);
/// // Identical standard-normal q and p give negative_bound=3 and effective_samples=3.
/// @endcode
/// @see ImportanceWeightedBound(), ImportanceWeightedAutoencoderOutput
struct ImportanceWeightedOutput {
  Tensor log_weights;        ///< Owning sampled log p(x,z)-log q(z|x)[K,B], nats.
  Tensor normalized_weights; ///< Owning sample-axis normalized weights[K,B], summing to one.
  Tensor negative_bound;     ///< Owning per-case negative log-mean-exp[B], nats.
  Tensor effective_samples;  ///< Owning 1/sum_k(normalized_weights^2)[B], ideally in[1,K].
  Tensor loss;               ///< Owning scalar mean of negative_bound over B cases.
};

/// @brief Form the sampled Gaussian IWAE bound and effective sample size from explicit endpoints.
///
/// The sample-major posterior endpoints are[KB,L]; pixel NLL is[KB]. Computes
/// logp=-.5*sum(z^2+log(2*pi)), logq=-.5*sum((z-mu)^2*exp(-lv)+lv+log(2*pi)),
/// logw=logp-NLL-logq, then reshapes[K,B] and reduces IwaeLogMeanExp. The source
/// parenthesization and Float32-rounded log(2*pi)/log(K) are preserved; Float64
/// uses full precision. This uses sampled densities, not analytic Gaussian KL.
/// Ordinary full pathwise gradients pass through mu,lv,z,NLL and all weights;
/// normalized weights are diagnostics, with no detachment or DReG replacement.
///
/// Inputs must be finite same-dtype nonempty Float32/64 with matching Gaussian
/// shapes and rows divisible by samples. Finite signed NLL is accepted for general
/// log-density use. Strides accepted; inputs unchanged. No noise/optimizer/state.
/// All outputs own lifetime. Work/storage O(K*B*L); computed nonfinite values may
/// fail weight validation, and are not clipped or otherwise sanitized.
/// @param mu Finite posterior means[KB,L] in sample-major order.
/// @param log_variance Finite matching posterior log variances[KB,L].
/// @param z Explicit finite matching posterior samples[KB,L], retaining live paths.
/// @param nll Finite same-dtype reconstruction negative log densities[KB].
/// @param samples Positive K dividing KB exactly.
/// @return Owning five-field objective/weight/ESS result; scalar mean is over cases.
/// @throws std::invalid_argument For invalid shape/dtype/values/K or nonfinite computed log
/// weights.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// auto z=n::ones({6,2}).requires_grad_();
/// const auto out=n::models::ImportanceWeightedBound(n::zeros({6,2}),n::zeros({6,2}),
///     z,n::full({6},3.),3);
/// const auto derivative=n::grad(out.loss,{z});
/// // q=p here, so sampled log-density terms cancel and every derivative is zero.
/// @endcode
/// @par References
/// - Burda, Grosse and Salakhutdinov (2015/2016), Importance Weighted Autoencoders,
///   https://arxiv.org/abs/1509.00519 . Uses ordinary reparameterized gradients.
/// @see ImportanceWeightedOutput, IwaeLogMeanExp(), GaussianReparameterize()
inline ImportanceWeightedOutput ImportanceWeightedBound(
    const Tensor& mu,
    const Tensor& log_variance,
    const Tensor& z,
    const Tensor& nll,
    std::int64_t  samples) {
  gaussian_detail::Match(mu, log_variance);
  gaussian_detail::Match(mu, z);
  if (samples <= 0 || mu.size(0) % samples || !nll.defined() || nll.dim() != 1 ||
      nll.size(0) != mu.size(0) || nll.dtype() != mu.dtype())
    throw std::invalid_argument(
        "IWAE requires positive K, sample-major Gaussian rows and matching NLL");
  for (double value : nll.contiguous().to_vector<double>())
    if (!std::isfinite(value))
      throw std::invalid_argument("IWAE NLL must be finite");
  double log2pi = std::log(2 * 3.14159265358979323846);
  if (mu.dtype() == DType::Float32)
    log2pi = double(float(log2pi));
  const auto logp = -0.5 * sum(square(z) + log2pi, {1});
  const auto logq = -0.5 * sum(square(z - mu) * exp(-log_variance) + log_variance + log2pi, {1});
  ImportanceWeightedOutput out;
  out.log_weights         = reshape(logp - nll - logq, {samples, -1});
  out.negative_bound      = -IwaeLogMeanExp(out.log_weights);
  out.loss                = mean(out.negative_bound, {0});
  const auto maximum      = stop_gradient(max(out.log_weights, {0}, true));
  const auto unnormalized = exp(out.log_weights - maximum);
  out.normalized_weights  = unnormalized / sum(unnormalized, {0}, true);
  out.effective_samples   = 1.0 / sum(square(out.normalized_weights), {0});
  return out;
}

/// @brief Own eleven sample-major Gaussian IWAE endpoints without fitting or sampling state.
///
/// All fields own lifetime and preserve autograd history. Sample-major rows are
/// flattened for Gaussian/decoder endpoints; bound/ESS have B rows, weights[K,B].
/// Analytic kl is diagnostic only and does not enter loss. Decode(mu) is not the
/// posterior predictive mean. Caller owns supplied noise and any optimizer.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(2237);
/// const auto out=n::models::ImportanceWeightedAutoencoder(p,n::full({2,784},.2),
///     n::zeros({5,2,4}));
/// // probability[10,784], log_weights[5,2], negative_bound[2], scalar loss.
/// @endcode
/// @see ImportanceWeightedAutoencoder(), ImportanceWeightedOutput, GaussianVaeDecode()
struct ImportanceWeightedAutoencoderOutput {
  Tensor mu;                 ///< Owning posterior means[KB,L], re-encoded per draw.
  Tensor log_variance;       ///< Owning posterior log variances[KB,L].
  Tensor z;                  ///< Owning explicitly reparameterized latent samples[KB,L].
  Tensor probability;        ///< Owning decoder sigmoid probabilities[KB,F], not sampled pixels.
  Tensor nll;                ///< Owning per-draw summed Bernoulli pixel NLL[KB], nats.
  Tensor kl;                 ///< Owning analytic per-row standard-normal KL[KB], diagnostic only.
  Tensor loss;               ///< Owning scalar mean per-case sampled negative IWAE bound, nats.
  Tensor negative_bound;     ///< Owning per-case sampled negative bound[B], nats.
  Tensor log_weights;        ///< Owning sampled density ratios[K,B], nats.
  Tensor normalized_weights; ///< Owning sample-normalized importance weights[K,B].
  Tensor effective_samples;  ///< Owning effective sample sizes[B].
};

/// @brief Evaluate a multi-sample Gaussian/Bernoulli IWAE with explicit noise and ordinary
/// gradients.
///
/// Tiles image[B,F] K times sample-major, flattens epsilon[K,B,L] to[KB,L] and
/// calls BetaVariationalAutoencoder with beta=1, then ImportanceWeightedBound.
/// Encoding is repeated on the tiled matrix to preserve source GEMM rounding.
/// Only sampled latent codes enter the ReLU decoder; no image bypass exists.
/// Ten Glorot/zero-bias leaves use encoder,mu,log_variance,decoder,output names;
/// defaults F784,L4,H32 have51,416 scalars. GaussianVaeDecode reuses decoder leaves
/// for image-free prior decoding. K=1 is the sampled negative ELBO, not the
/// analytic-KL VAE estimator, although their expectations agree.
///
/// Every parameter/image/noise path remains live. No fit, optimizer, checkpoint
/// selection, gradient estimator substitution or new random draws are hidden
/// beyond lazy initialization. Float32 matches source log constants; Float64
/// retains full precision. Validates finite pixels[0,1], noise/dtype/shape and
/// representable tiled/weight products before creating leaves. Incompatible
/// existing leaves and nonfinite computed posteriors may fail later. Strides are
/// accepted, inputs unchanged; all results own lifetime. Synchronize shared stores.
/// Work O(K*B*H*(F+L)); activation storage O(K*B*(F+H+L)).
/// @param parameters Mutable matching floating store; original seed2237.
/// @param image Finite Float32/64 pixel matrix[B,features], B>0, values in[0,1].
/// @param epsilon Finite matching explicit standard-normal noise[K,B,latent], K>0.
/// @param features Positive pixel width F, default784.
/// @param latent Positive code width L, default4.
/// @param hidden Positive encoder/decoder ReLU width H, default32.
/// @return Owning eleven-field output with ordinary fully pathwise sampled IWAE loss.
/// @throws std::invalid_argument For invalid input/options/overflow or incompatible leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(2237);
/// const auto out=n::models::ImportanceWeightedAutoencoder(p,n::full({2,784},.2),
///     n::ones({5,2,4}));
/// n::OptimizerConfig config;
/// config.kind=n::OptimizerKind::Adam;
/// n::Optimizer optimizer(p,.003,config);
/// optimizer.step(n::grad(out.loss,p.parameters()));
/// // One visible update; no optimization is hidden in the forward function.
/// @endcode
/// @par References
/// - Burda, Grosse and Salakhutdinov (2015/2016), Importance Weighted Autoencoders,
///   https://arxiv.org/abs/1509.00519 . Uses a small migrated educational architecture,
///   not the paper's benchmark networks or a doubly reparameterized gradient estimator.
/// @see ImportanceWeightedAutoencoderOutput, ImportanceWeightedBound(), GaussianVaeDecode()
inline ImportanceWeightedAutoencoderOutput ImportanceWeightedAutoencoder(
    ParameterStore& parameters,
    const Tensor&   image,
    const Tensor&   epsilon,
    std::int64_t    features = 784,
    std::int64_t    latent   = 4,
    std::int64_t    hidden   = 32) {
  aae_detail::Widths(features, latent, hidden);
  aae_detail::Matrix(image, features, parameters.dtype(), true);
  if (!epsilon.defined() || epsilon.dim() != 3 || epsilon.dtype() != image.dtype() ||
      epsilon.size(0) <= 0 || epsilon.size(1) != image.size(0) || epsilon.size(2) != latent)
    throw std::invalid_argument(
        "IWAE noise must have matching positive K,B,L axes and floating dtype");
  const auto limit = std::numeric_limits<std::int64_t>::max();
  const auto k = epsilon.size(0), b = image.size(0);
  if (k > limit / b || k * b > limit / std::max(features, std::max(latent, hidden)) ||
      features > limit / hidden || latent > limit / hidden)
    throw std::invalid_argument("IWAE tiled or parameter width overflow");
  const auto noise = reshape(epsilon.contiguous(), {k * b, latent});
  aae_detail::Matrix(noise, latent, image.dtype(), false);
  const auto tiled =
      reshape(tile(reshape(image.contiguous(), {1, b, features}), {k, 1, 1}), {k * b, features});
  const auto net = BetaVariationalAutoencoder(
      parameters, tiled.contiguous(), noise, 1, features, latent, hidden);
  const auto bound =
      ImportanceWeightedBound(net.mu, net.log_variance, net.z, net.reconstruction_nll, k);
  return {
      net.mu,
      net.log_variance,
      net.z,
      net.probability,
      net.reconstruction_nll,
      net.kl,
      bound.loss,
      bound.negative_bound,
      bound.log_weights,
      bound.normalized_weights,
      bound.effective_samples};
}
} // namespace datamunge::nn::models
