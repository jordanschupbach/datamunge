#pragma once
/// @file convolutional_variational_autoencoder.hpp
/// @brief Spatial Gaussian VAE endpoints with an explicit sampled-code-only decoder.
#include <limits>

#include "datamunge/nn/gaussian_autoencoder.hpp"
#include "datamunge/nn/image_ops.hpp"

namespace datamunge::nn::models {
/// @brief Internal geometry, pixel validation and convolution options for the spatial VAE.
namespace conv_vae_detail {
/// @brief Validate geometry and return the flattened quarter-resolution feature width.
/// @param height Positive image height divisible by four.
/// @param width Positive image width divisible by four.
/// @param latent Positive Gaussian code width.
/// @param channels Positive first encoder/decoder hidden channel count.
/// @param base_channels Positive quarter-resolution channel count.
/// @return (height/4)*(width/4)*base_channels, representable in int64.
/// @throws std::invalid_argument For invalid dimensions or product overflow.
inline std::int64_t Geometry(
    std::int64_t height,
    std::int64_t width,
    std::int64_t latent,
    std::int64_t channels,
    std::int64_t base_channels) {
  if (height <= 0 || width <= 0 || height % 4 || width % 4 || latent <= 0 || channels <= 0 ||
      base_channels <= 0 || height > std::numeric_limits<std::int64_t>::max() / width)
    throw std::invalid_argument(
        "spatial VAE requires positive representable multiples-of-four geometry");
  const auto area = (height / 4) * (width / 4);
  if (area > std::numeric_limits<std::int64_t>::max() / base_channels)
    throw std::invalid_argument("spatial VAE flattened width overflow");
  return area * base_channels;
}
/// @brief Require finite unit-range nonempty grayscale NHWC images in computation dtype.
/// @param image Floating [B,H,W,1] pixels, B>0.
/// @param type Matching Float32/64 store dtype.
/// @throws std::invalid_argument For malformed shape/dtype or invalid pixel values.
inline void Image(const Tensor& image, DType type) {
  if (!image.defined() || image.dim() != 4 || image.size(0) <= 0 || image.size(3) != 1 ||
      (type != DType::Float32 && type != DType::Float64) || image.dtype() != type)
    throw std::invalid_argument(
        "spatial VAE requires matching nonempty Float32/64 NHWC grayscale images");
  for (double x : image.contiguous().to_vector<double>())
    if (!std::isfinite(x) || x < 0 || x > 1)
      throw std::invalid_argument("spatial VAE pixels must be finite in [0,1]");
}
/// @brief Build same-padded stride-one He-uniform ReLU convolution options with zero bias.
/// @return Independent layer options; no parameters or random values are created here.
inline ConvLayerOptions Relu() {
  ConvLayerOptions result;
  result.init       = init::he_uniform();
  result.activation = Activation::Relu;
  return result;
}
} // namespace conv_vae_detail

/// @brief Own all decoder stages from a Gaussian sample to Bernoulli pixel probabilities.
///
/// Every field owns its lifetime and continuous autograd history. The decoder has
/// no image, target, encoder feature or skip input and stores no RNG/optimizer.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(2011);
/// const auto out=n::models::ConvolutionalVaeDecode(p,n::zeros({2,2}));
/// // out.decoder_base[2,7,7,4]; probability[2,784]; six leaves/957 scalars.
/// @endcode
/// @see ConvolutionalVaeDecode()
struct ConvolutionalVaeDecoderOutput {
  Tensor decoder_base;   ///< Projected/ReLU spatial base [B,H/4,W/4,base_channels].
  Tensor decoder1;       ///< First repeated/convolved ReLU map [B,H/2,W/2,channels].
  Tensor spatial_logits; ///< Final repeated/convolved log-odds [B,H,W,1].
  Tensor logits;         ///< Same log-odds flattened [B,H*W].
  Tensor probability;    ///< Sigmoid decoder probabilities [B,H*W], not sampled pixels.
};

/// @brief Decode explicit Gaussian-latent codes through projection, nearest repetition and
/// convolutions.
///
/// Projects code with decoder_projection/W,b to (H/4)*(W/4)*base_channels ReLU
/// values, reshapes NHWC, repeats each spatial value2x2, applies same-padded3x3
/// base_channels->channels ReLU, repeats2x2 again and applies same-padded3x3
/// channels->1 linear logits. Names/order: decoder_projection/W,b; decoder1/W,b;
/// decoder2/W,b. Projection uses ordinary Glorot-uniform weights even with ReLU;
/// decoder1 uses He uniform, decoder2 Glorot uniform, all biases zero. Default
/// six leaves/957 scalars. This is neither transposed convolution nor a spatial
/// feature bypass. Signed/zero finite codes are accepted; only code enters.
///
/// Positive H/W multiples of four and flattened widths are checked before leaf
/// creation. Strided Float32/64 codes match the store and retain code/parameter
/// gradients. Returned fields own lifetime; no input mutation, RNG, fitting,
/// clipping or implicit detachment. Incompatible stored leaves can fail later;
/// nonfinite parameters/computed values are not sanitized. Work/storage scale
/// linearly with B*H*W for fixed channel/latent widths. Synchronize shared stores.
/// @param parameters Mutable Float32/64 store containing/creating decoder leaves.
/// @param code Finite nonempty [B,latent] code matching store dtype.
/// @param height Positive output height divisible by four, default28.
/// @param width Positive output width divisible by four, default28.
/// @param latent Positive code width, default2.
/// @param channels Positive first decoder hidden channels, default8.
/// @param base_channels Positive quarter-resolution channels, default4.
/// @return Owning five-field decoder endpoints, with flattened probabilities.
/// @throws std::invalid_argument For invalid geometry/code/dtype or incompatible leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(2011);
/// const auto out=n::models::ConvolutionalVaeDecode(p,n::ones({1,1}),4,4,1,2,1);
/// // Spatial logits[1,4,4,1], flattened logits[1,16], six leaves/41 scalars.
/// @endcode
/// @par References
/// - Kingma and Welling (2013/2014), Auto-Encoding Variational Bayes,
///   https://arxiv.org/abs/1312.6114 . This small repeat/convolution teaching
///   architecture is repository-specific, not a benchmark configuration.
/// - He, Zhang, Ren and Sun (2015), Delving Deep into Rectifiers,
///   https://arxiv.org/abs/1502.01852 , for the ReLU convolution initialization.
/// @see ConvolutionalVariationalAutoencoder(), upsample_nearest()
inline ConvolutionalVaeDecoderOutput ConvolutionalVaeDecode(
    ParameterStore& parameters,
    const Tensor&   code,
    std::int64_t    height        = 28,
    std::int64_t    width         = 28,
    std::int64_t    latent        = 2,
    std::int64_t    channels      = 8,
    std::int64_t    base_channels = 4) {
  const auto encoded = conv_vae_detail::Geometry(height, width, latent, channels, base_channels);
  aae_detail::Matrix(code, latent, parameters.dtype(), false);
  ConvolutionalVaeDecoderOutput out;
  out.decoder_base = reshape(
      dense(parameters, "decoder_projection", code, encoded, Activation::Relu),
      {code.size(0), height / 4, width / 4, base_channels});
  out.decoder1 = conv2d(
      parameters,
      "decoder1",
      upsample_nearest(out.decoder_base, 2),
      channels,
      3,
      conv_vae_detail::Relu());
  out.spatial_logits =
      conv2d(parameters, "decoder2", upsample_nearest(out.decoder1, 2), 1, 3, ConvLayerOptions{});
  out.logits      = reshape(out.spatial_logits, {code.size(0), height * width});
  out.probability = sigmoid(out.logits);
  return out;
}

/// @brief Own Gaussian posterior endpoints and all seven spatial encoder/decoder stages.
///
/// Inherits nine owning GaussianVaeOutput fields; loss and unweighted_loss are
/// the same beta1 scalar. Encoder pool2 is exposed for inspection only; z is the
/// sole bridge consumed by the decoder. Every continuous tensor retains autograd
/// history. No optimizer, fitting state or RNG is stored. Decode(mu) is not the
/// posterior predictive mean.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(2011);
/// const auto
/// out=n::models::ConvolutionalVariationalAutoencoder(p,n::zeros({2,28,28,1}),n::zeros({2,2}));
/// // pool2[2,7,7,4] is reduced to two Gaussian coordinates before decoding.
/// @endcode
/// @see ConvolutionalVariationalAutoencoder(), GaussianVaeOutput
struct ConvolutionalVaeOutput : GaussianVaeOutput {
  Tensor conv1;          ///< First ReLU convolution [B,H,W,channels].
  Tensor pool1;          ///< Nonoverlapping2x2 mean pool [B,H/2,W/2,channels].
  Tensor conv2;          ///< Second ReLU convolution [B,H/2,W/2,base_channels].
  Tensor pool2;          ///< Second mean pool [B,H/4,W/4,base_channels], not a decoder skip input.
  Tensor decoder_base;   ///< ReLU-projected Gaussian sample [B,H/4,W/4,base_channels].
  Tensor decoder1;       ///< First repeat/convolution decoder stage [B,H/2,W/2,channels].
  Tensor spatial_logits; ///< Final spatial log-odds [B,H,W,1].
};

/// @brief Evaluate a spatial diagonal-Gaussian/Bernoulli VAE with explicit posterior noise and no
/// skips.
///
/// Encoder: same-padded3x3 grayscale->channels ReLU, nonoverlapping2x2 average
/// pool, same-padded3x3 channels->base_channels ReLU, another average pool, then
/// separate linear Gaussian mean/log-variance heads of width latent. Samples use
/// GaussianReparameterize(); ONLY z enters ConvolutionalVaeDecode(). The pooled
/// feature map is never forwarded as a deterministic decoder input. Fourteen
/// default leaves/2117 scalars have names/order encoder1/W,b; encoder2/W,b;
/// mu/W,b; log_variance/W,b; decoder_projection/W,b; decoder1/W,b; decoder2/W,b.
/// ReLU convolutions use He uniform; dense heads, ReLU dense projection and output
/// convolution use ordinary Glorot uniform; all biases zero.
///
/// Bernoulli NLL sums H*W pixels and analytic Gaussian KL sums latent coordinates,
/// then images are averaged: loss=mean(NLL+KL), beta1. Binary targets give a
/// one-draw negative-ELBO estimate; soft [0,1] targets give cross-entropy, not a
/// continuous-observation likelihood density. Fixed N(0,I) prior, no annealing,
/// hidden sampling, labels, fitting, validation selection or clipping. Strides
/// and continuous image/epsilon/leaf gradients are retained. Input checks precede
/// creation; incompatible leaves/nonfinite computed states may fail later.
/// Returned fields own lifetime; inputs unchanged. Work/storage are linear in
/// B*H*W for fixed latent/channel widths. Shared stores require synchronization.
/// @param parameters Mutable floating store matching image/epsilon dtype.
/// @param image Finite nonempty Float32/64 NHWC [B,H,W,1] pixels in [0,1]; H/W positive multiples
/// of four.
/// @param epsilon Explicit finite [B,latent] standard-normal noise, same dtype/batch.
/// @param latent Positive Gaussian code width, default2.
/// @param channels Positive first encoder/decoder hidden channels, default8.
/// @param base_channels Positive quarter-resolution channels, default4.
/// @return Owning sixteen-field Gaussian/spatial forward endpoints.
/// @throws std::invalid_argument For invalid input/geometry/dtype or incompatible leaves;
///         nonfinite computed posterior/logits can fail validation later.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(2011);
/// const auto
/// out=n::models::ConvolutionalVariationalAutoencoder(p,n::full({2,28,28,1},.2),n::zeros({2,2}));
/// const auto gradients=n::grad(out.loss,p.parameters()); //14 gradients; no update is hidden.
/// @endcode
/// @par References
/// - Kingma and Welling (2013/2014), https://arxiv.org/abs/1312.6114 .
/// - He et al. (2015), https://arxiv.org/abs/1502.01852 . This small pooled
///   repeat-decoder architecture matches the migrated source, not a published benchmark.
/// @see ConvolutionalVaeOutput, ConvolutionalVaeDecode(), GaussianReparameterize(),
/// DiagonalGaussianKl(), BernoulliPixelNll()
inline ConvolutionalVaeOutput ConvolutionalVariationalAutoencoder(
    ParameterStore& parameters,
    const Tensor&   image,
    const Tensor&   epsilon,
    std::int64_t    latent        = 2,
    std::int64_t    channels      = 8,
    std::int64_t    base_channels = 4) {
  conv_vae_detail::Image(image, parameters.dtype());
  const auto height = image.size(1), width = image.size(2);
  const auto encoded = conv_vae_detail::Geometry(height, width, latent, channels, base_channels);
  aae_detail::Matrix(epsilon, latent, parameters.dtype(), false);
  if (epsilon.size(0) != image.size(0))
    throw std::invalid_argument("spatial VAE batch sizes must match");
  ConvolutionalVaeOutput out;
  out.conv1 = conv2d(parameters, "encoder1", image, channels, 3, conv_vae_detail::Relu());
  out.pool1 = avg_pool2d(out.conv1, 2, 2);
  out.conv2 = conv2d(parameters, "encoder2", out.pool1, base_channels, 3, conv_vae_detail::Relu());
  out.pool2 = avg_pool2d(out.conv2, 2, 2);
  const auto flat  = reshape(out.pool2, {image.size(0), encoded});
  out.mu           = dense(parameters, "mu", flat, latent);
  out.log_variance = dense(parameters, "log_variance", flat, latent);
  out.z            = GaussianReparameterize(out.mu, out.log_variance, epsilon);
  const auto decoded =
      ConvolutionalVaeDecode(parameters, out.z, height, width, latent, channels, base_channels);
  out.decoder_base   = decoded.decoder_base;
  out.decoder1       = decoded.decoder1;
  out.spatial_logits = decoded.spatial_logits;
  out.logits         = decoded.logits;
  out.probability    = decoded.probability;
  out.reconstruction_nll =
      BernoulliPixelNll(out.logits, reshape(image, {image.size(0), height * width}));
  out.kl              = DiagonalGaussianKl(out.mu, out.log_variance);
  out.loss            = mean(out.reconstruction_nll + out.kl, {0});
  out.unweighted_loss = out.loss;
  return out;
}
} // namespace datamunge::nn::models
