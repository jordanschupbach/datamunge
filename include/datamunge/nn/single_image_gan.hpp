#pragma once
/// @file single_image_gan.hpp
/// @brief Explicit single-image residual patch game, seeded noise and detached scale amplitude.
#include "datamunge/nn/autoencoder_models.hpp"
#include "datamunge/nn/gan_models.hpp"
#include "datamunge/nn/super_resolution_models.hpp"

namespace datamunge::nn::models {
/// @brief Internal positive RGB geometry and explicit scalar-amplitude contracts.
namespace single_image_detail {
/// @brief Validate positive floating RGB rectangles and representable hidden maps.
/// @param image Finite CPU[N,H,W,3], positive axes and arbitrary strides.
/// @param dtype Required Float32/64 computation dtype.
/// @throws std::invalid_argument For wrong geometry, dtype, nonfinite values or products.
/// @throws std::logic_error For unsupported storage.
inline void Image(const Tensor& image, DType dtype) {
  gan_detail::Finite(image);
  if (image.dim() != 4 || image.size(3) != 3 || image.dtype() != dtype ||
      image.numel() / 3 > std::numeric_limits<std::int64_t>::max() / 4)
    throw std::invalid_argument("Single-image GAN needs matching positive floating RGB rectangles");
}
/// @brief Require a finite matching scalar or one-element source amplitude.
/// @param amplitude Finite CPU scalar or[1], signed or zero allowed.
/// @param dtype Required image/store dtype.
/// @throws std::invalid_argument For wrong geometry, dtype or nonfinite value.
/// @throws std::logic_error For unsupported storage.
inline void Amplitude(const Tensor& amplitude, DType dtype) {
  gan_detail::Finite(amplitude);
  if (amplitude.dtype() != dtype ||
      !(amplitude.dim() == 0 || amplitude.shape() == Shape({1})))
    throw std::invalid_argument("Single-image amplitude must be a matching scalar or[1]");
}
/// @brief Check matching previous/noise rectangles before any parameter creation.
/// @param previous Finite positive RGB CPU rectangle.
/// @param noise Matching explicit noise tensor, signed values allowed.
/// @param dtype Required image/store dtype.
/// @throws std::invalid_argument For incompatible geometry, dtype, products or values.
/// @throws std::logic_error For unsupported storage.
inline void Pair(const Tensor& previous, const Tensor& noise, DType dtype) {
  Image(previous, dtype);
  Image(noise, dtype);
  if (previous.shape() != noise.shape())
    throw std::invalid_argument("Single-image previous and noise shapes must agree");
}
} // namespace single_image_detail

/// @brief Draw one explicit Gaussian stream, optionally sharing each pixel across RGB.
///
/// Uses GaussianNoise's local raw MT19937 midpoint Box-Muller stream in NHWC
/// order. Shared mode draws[N,H,W,1], casts once, then tiles the last axis;
/// independent mode draws[N,H,W,3] in a single stream. Default Float32 matches
/// the source SinGAN pyramid. Returns owned detached data; no global RNG,
/// parameter store, optimizer, fitted state or input graph is retained.
/// Work/storage O(rows*height*width). Concurrent calls have independent local RNGs.
/// @param rows Positive number of images N.
/// @param height Positive spatial height H.
/// @param width Positive spatial width W.
/// @param shared_rgb True for coarse-scale identical RGB channels; false for independent RGB.
/// @param seed Local uint32 MT seed, used without a global stream.
/// @param dtype Float32(default) or Float64 result computation type.
/// @return Owning detached finite CPU[N,H,W,3] Gaussian noise.
/// @throws std::invalid_argument For nonpositive axes, product overflow or nonfloating dtype.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// auto z=dm::nn::models::SingleImageNoise(1,16,16,true,7411);
/// // z[1,16,16,3]; all three channels at each pixel are identical.
/// @endcode
/// @par References
/// - Rott Shaham, Dekel and Michaeli (2019), SinGAN: Learning a Generative Model
/// from a Single Natural Image, https://arxiv.org/abs/1905.01164 . Source teaching
/// example uses three small scales and this explicit Gaussian stream.
/// @see GaussianNoise(), SingleImageGenerate(), SingleImageAmplitude()
inline Tensor SingleImageNoise(
    std::int64_t rows, std::int64_t height, std::int64_t width, bool shared_rgb,
    std::uint32_t seed, DType dtype = DType::Float32) {
  const auto limit = std::numeric_limits<std::int64_t>::max();
  if (rows <= 0 || height <= 0 || width <= 0 || rows > limit / height ||
      rows * height > limit / width || rows * height * width > limit / 3 ||
      (dtype != DType::Float32 && dtype != DType::Float64))
    throw std::invalid_argument("Single-image noise needs positive representable floating RGB axes");
  Tensor z = cast(GaussianNoise({height, width, shared_rgb ? 1 : 3}, rows, seed), dtype);
  return shared_rgb ? tile(z, {1, 1, 1, 3}) : z;
}

/// @brief Measure a detached fixed scale amplitude from reconstruction RMS error.
///
/// Computes factor*sqrt(sum((Double(previous)-Double(real))^2)/numel) and
/// casts once to the matching input dtype. The owning result has shape[1],
/// preserving the source example's one-element scalar convention. Inputs are
/// finite matching positive RGB CPU rectangles with arbitrary strides and signed
/// pixels. Both image paths are stopped before arithmetic. Factor defaults .1;
/// zero is allowed. No fitting, sampling, moving state or store mutation occurs.
/// Work/storage O(real.numel()); independent calls share no mutable state.
/// @param real Finite floating CPU[N,H,W,3] target at this scale.
/// @param previous_reconstruction Matching frozen lower-scale reconstruction already resized.
/// @param factor Finite nonnegative RMS multiplier, default .1.
/// @return Owning detached finite[1] amplitude in real's Float32/64 dtype.
/// @throws std::invalid_argument For incompatible inputs, invalid factor or nonfinite result.
/// @throws std::logic_error For unsupported storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// auto a=n::models::SingleImageAmplitude(n::ones({1,2,3,3}),n::zeros({1,2,3,3}));
/// // a[1] contains approximately .1; no image gradient path is retained.
/// @endcode
/// @par References
/// - Rott Shaham, Dekel and Michaeli (2019), SinGAN: Learning a Generative Model
/// from a Single Natural Image, https://arxiv.org/abs/1905.01164 . Coarse amplitude
/// is separately chosen as one by the caller, rather than measured here.
/// @see SingleImageNoise(), SingleImageGenerate()
inline Tensor SingleImageAmplitude(
    const Tensor& real, const Tensor& previous_reconstruction, double factor = .1) {
  single_image_detail::Pair(real, previous_reconstruction, real.dtype());
  if (!std::isfinite(factor) || factor < 0)
    throw std::invalid_argument("Single-image amplitude factor must be finite and nonnegative");
  NoGradGuard guard;
  const Tensor delta = cast(previous_reconstruction.detach(), DType::Float64) -
                       cast(real.detach(), DType::Float64);
  Tensor result = reshape(cast(factor * sqrt(sum(square(delta)) / double(delta.numel())),
                               real.dtype()), {1});
  gan_detail::Finite(result);
  return result;
}

/// @brief Own the five live boundaries of a fully convolutional residual generator.
///
/// Flat Tensor handles retain storage and applicable previous/noise/amplitude/
/// weight graphs after the parent or parameter store is destroyed. No saved
/// statistics, optimizer, target image or hidden noise stream is retained.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore g(7307);
/// auto o=n::models::SingleImageGenerate(g,n::zeros({1,5,7,3}),n::ones({1,5,7,3}),n::ones({1}));
/// // input/image/residual[1,5,7,3]; head/body[1,5,7,4].
/// @endcode
/// @par References
/// - Rott Shaham, Dekel and Michaeli (2019), SinGAN: Learning a Generative Model
/// from a Single Natural Image, https://arxiv.org/abs/1905.01164 . Compact source
/// variant uses width4 and three biased convolutions without normalization.
/// @see SingleImageGenerate(), SingleImageGanOutput
struct SingleImageGenerated {
  Tensor input;    ///< Owning live previous+amplitude*noise[N,H,W,3].
  Tensor head;     ///< Owning live first3x3 leaky(.2) feature[N,H,W,4].
  Tensor body;     ///< Owning live second3x3 leaky(.2) feature[N,H,W,4].
  Tensor residual; ///< Owning live tanh third3x3 correction[N,H,W,3], bounded in[-1,1].
  Tensor image;    ///< Owning live previous+residual[N,H,W,3], unbounded signed output.
};

/// @brief Generate previous plus a learned tanh correction from explicit scaled noise.
///
/// Matching finite signed Float32/64 CPU previous/noise[N,H,W,3], positive axes,
/// arbitrary strides/rectangles. Amplitude is a matching finite scalar or[1],
/// signed or zero allowed. Validate supplied inputs before creating leaves.
/// Three biased3x3 SAME stride1 convolutions3->4->4->3 use source fan-in uniform
/// kernels; two leaky(.2) activations then tanh. Source seed7307 creates371 values
/// in6 named leaves, head/body/output W/b. Output is previous+tanh(correction),
/// without clipping, normalization, pooling, dropout or batch coupling. Only
/// missing leaves initialize; no later RNG, existing-leaf write, optimizer,
/// sampling or hidden no-grad. Applicable first/higher paths remain live through
/// the input mixture and residual skip. No target image is required. Work/storage
/// O(N*H*W) for fixed width; synchronize shared mutable parameter stores.
/// @param generator Caller-owned floating G store, source seed7307.
/// @param previous Finite matching CPU[N,H,W,3] previous scale already resized.
/// @param noise Explicit matching finite noise, never drawn internally.
/// @param amplitude Matching finite scalar or[1] scale multiplier.
/// @return Owning SingleImageGenerated with all five actual boundaries.
/// @throws std::invalid_argument For incompatible input geometry/dtype/products/values or leaves.
/// @throws std::logic_error For unsupported storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore g(7307);
/// auto previous=n::zeros({2,20,24,3});auto z=n::models::SingleImageNoise(2,20,24,true,7411);
/// auto o=n::models::SingleImageGenerate(g,previous,z,n::ones({1}));
/// // Fully convolutional image[2,20,24,3]; G owns371 values,6 leaves.
/// @endcode
/// @par References
/// - Rott Shaham, Dekel and Michaeli (2019), SinGAN: Learning a Generative Model
/// from a Single Natural Image, https://arxiv.org/abs/1905.01164 . This preserves
/// the compact source residual generator, rather than the full paper architecture.
/// @see SingleImageGenerated, SingleImageDiscriminate(), SingleImageGan(), SRConv()
inline SingleImageGenerated SingleImageGenerate(
    ParameterStore& generator, const Tensor& previous, const Tensor& noise,
    const Tensor& amplitude) {
  single_image_detail::Pair(previous, noise, generator.dtype());
  single_image_detail::Amplitude(amplitude, generator.dtype());
  SingleImageGenerated o;
  o.input = previous + noise * amplitude;
  o.head = leaky_relu(SRConv(generator, "head", o.input, 3, 4, 3), .2);
  o.body = leaky_relu(SRConv(generator, "body", o.head, 4, 4, 3), .2);
  o.residual = tanh(SRConv(generator, "output", o.body, 4, 3, 3));
  o.image = previous + o.residual;
  gan_detail::Finite(o.image);
  return o;
}

/// @brief Score local RGB patches through three biased SAME convolutions.
///
/// Finite signed matching Float32/64 CPU image[N,H,W,3], positive rectangular
/// axes and arbitrary strides. Three fan-in uniform3x3 stride1 convolutions
/// 3->4->4->1 with two leaky(.2) activations produce raw[N,H,W,1] scores,
/// nominal7x7 receptive field with zero padding at boundaries. Source seed7309
/// creates297 values in6 leaves, head/body/output W/b. Image only: no noise,
/// amplitude, target, pooling, sigmoid, normalization or batch coupling. Inputs
/// validate before leaves; stored-leaf/math errors may follow initialization.
/// Only missing leaves initialize; no existing-leaf writes, later RNG, optimizer
/// or hidden no-grad. Owning output retains applicable first/higher image/weight
/// paths. Fixed-width work/storage O(N*H*W); synchronize shared mutable stores.
/// @param discriminator Caller-owned matching floating D store, source seed7309.
/// @param image Finite matching CPU[N,H,W,3] signed RGB image batch.
/// @return Owning live raw patch scores[N,H,W,1], with no probability transform.
/// @throws std::invalid_argument For incompatible geometry/dtype/products/values or leaves.
/// @throws std::logic_error For unsupported storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore d(7309);
/// auto score=n::models::SingleImageDiscriminate(d,n::zeros({2,30,36,3}));
/// // score[2,30,36,1]; D owns297 values,6 leaves.
/// @endcode
/// @par References
/// - Rott Shaham, Dekel and Michaeli (2019), SinGAN: Learning a Generative Model
/// from a Single Natural Image, https://arxiv.org/abs/1905.01164 . Compact source
/// critic uses width4 and raw local least-squares scores.
/// @see SingleImageGenerate(), SingleImageGan()
inline Tensor SingleImageDiscriminate(ParameterStore& discriminator, const Tensor& image) {
  single_image_detail::Image(image, discriminator.dtype());
  Tensor h = leaky_relu(SRConv(discriminator, "head", image, 3, 4, 3), .2);
  h = leaky_relu(SRConv(discriminator, "body", h, 4, 4, 3), .2);
  Tensor score = SRConv(discriminator, "output", h, 4, 1, 3);
  gan_detail::Finite(score);
  return score;
}

/// @brief Own both generator branches, local scores and the four single-scale losses.
///
/// Eight flat owning Tensor handles retain storage/graphs after parent/store
/// collection. Generator is full fake-to-one LSGAN plus10*MSE reconstruction;
/// discriminator is half the sum of real-to-one and fake-to-zero means. No
/// detach is hidden in these outputs; caller owns gradient groups and writes.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore g(7307),d(7309);
/// auto x=n::zeros({1,4,6,3});auto o=n::models::SingleImageGan(g,d,x,x,x,x,x,n::ones({1}));
/// // Eight owning fields; g_loss=adversarial_loss+10*reconstruction_loss.
/// @endcode
/// @par References
/// - Rott Shaham, Dekel and Michaeli (2019), SinGAN: Learning a Generative Model
/// from a Single Natural Image, https://arxiv.org/abs/1905.01164 . Source teaching
/// variant substitutes least-squares losses for the paper's WGAN-GP.
/// @see SingleImageGan(), SingleImageGenerated
struct SingleImageGanOutput {
  Tensor fake;                ///< Owning live random branch image[Nfake,H,W,3].
  Tensor reconstruction;      ///< Owning live fixed-noise branch image[Nreal,H,W,3].
  Tensor real_score;          ///< Owning live image-only D(real)[Nreal,H,W,1].
  Tensor fake_score;          ///< Owning live image-only D(fake)[Nfake,H,W,1].
  Tensor d_loss;              ///< Owning live scalar .5*(mean((real_score-1)^2)+mean(fake_score^2)).
  Tensor g_loss;              ///< Owning live scalar adversarial_loss+10*reconstruction_loss.
  Tensor adversarial_loss;    ///< Owning live scalar mean((fake_score-1)^2), with no half factor.
  Tensor reconstruction_loss; ///< Owning live scalar mean((reconstruction-real)^2), over all RGB values.
};

/// @brief Evaluate one explicit single-image least-squares game with shared reconstruction weights.
///
/// Distinct matching Float32/64 G/D stores; finite positive RGB CPU rectangles
/// with common H/W, arbitrary strides. Previous/noise have identical geometry;
/// rec_previous/rec_noise exactly match real. Random and real batch counts may
/// differ; each score pool has its own mean. Matching scalar or[1] amplitude is
/// shared by both G calls. Validate all supplied inputs before parameter leaves.
/// Random fake and fixed reconstruction share the same371 G values; image-only
/// D(real)/D(fake) share297 D values. Full G LSGAN+10MSE and half D LSGAN match
/// the source. No gradient penalty, normalization, target-conditioned critic,
/// clipping, stop-gradient, fitted state, optimizer or sampling is hidden.
/// First/higher input/weight paths remain live, including both G uses. Caller
/// must freeze/detach lower scales and measure fixed amplitudes separately, then
/// take D followed by a fresh G capture. Only missing leaves initialize; inputs
/// and existing leaves are unchanged. Work/storage O((Nreal+Nfake)*H*W);
/// synchronize shared mutable stores. Returned Tensor fields own graph/storage.
/// @param generator Caller-owned G store, source seed7307.
/// @param discriminator Distinct matching D store, source seed7309.
/// @param real Finite matching CPU[Nreal,H,W,3] reconstruction target and D real input.
/// @param previous Finite matching CPU[Nfake,H,W,3] random lower-scale draw already resized.
/// @param noise Matching explicit random noise, shared RGB at the coarsest source scale.
/// @param rec_previous Finite matching real-shaped frozen reconstruction already resized.
/// @param rec_noise Real-shaped explicit fixed noise; source uses zeros above the coarse scale.
/// @param amplitude Matching finite scalar or[1], signed or zero allowed.
/// @return Owning eight-field SingleImageGanOutput with two images, two scores and four losses.
/// @throws std::invalid_argument For aliased stores, incompatible inputs/dtypes/values/products or leaves.
/// @throws std::logic_error For unsupported storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore g(7307),d(7309);
/// auto x=n::zeros({1,4,6,3});auto z=n::models::SingleImageNoise(1,4,6,true,7411);
/// auto o=n::models::SingleImageGan(g,d,x,x,z,x,z,n::ones({1}));
/// // Capture o.d_loss for D; after its write, re-evaluate and capture o.g_loss for G.
/// @endcode
/// @par References
/// - Rott Shaham, Dekel and Michaeli (2019), SinGAN: Learning a Generative Model
/// from a Single Natural Image, https://arxiv.org/abs/1905.01164 . Compact source
/// variant has three scales, width4, LSGAN and a fixed10-times reconstruction weight.
/// @see SingleImageGanOutput, SingleImageGenerate(), SingleImageDiscriminate(), SingleImageAmplitude()
inline SingleImageGanOutput SingleImageGan(
    ParameterStore& generator, ParameterStore& discriminator, const Tensor& real,
    const Tensor& previous, const Tensor& noise, const Tensor& rec_previous,
    const Tensor& rec_noise, const Tensor& amplitude) {
  if (&generator == &discriminator || generator.dtype() != discriminator.dtype())
    throw std::invalid_argument("Single-image GAN requires distinct matching G/D stores");
  const auto dtype = generator.dtype();
  single_image_detail::Image(real, dtype);
  single_image_detail::Pair(previous, noise, dtype);
  single_image_detail::Pair(rec_previous, rec_noise, dtype);
  single_image_detail::Amplitude(amplitude, dtype);
  if (real.shape() != rec_previous.shape() || real.size(1) != previous.size(1) ||
      real.size(2) != previous.size(2))
    throw std::invalid_argument("Single-image game needs aligned real/reconstruction and common spatial axes");
  SingleImageGanOutput o;
  o.fake = SingleImageGenerate(generator, previous, noise, amplitude).image;
  o.reconstruction = SingleImageGenerate(generator, rec_previous, rec_noise, amplitude).image;
  o.real_score = SingleImageDiscriminate(discriminator, real);
  o.fake_score = SingleImageDiscriminate(discriminator, o.fake);
  o.adversarial_loss = mean(square(o.fake_score - 1.0));
  o.reconstruction_loss = mean(square(o.reconstruction - real));
  o.d_loss = .5 * (mean(square(o.real_score - 1.0)) + mean(square(o.fake_score)));
  o.g_loss = o.adversarial_loss + 10.0 * o.reconstruction_loss;
  gan_detail::Finite(o.d_loss);
  gan_detail::Finite(o.g_loss);
  return o;
}
} // namespace datamunge::nn::models
