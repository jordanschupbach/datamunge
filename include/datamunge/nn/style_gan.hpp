#pragma once
/// @file style_gan.hpp
/// @brief Explicit8x8 AdaIN/modulated generators, supplied spatial noise and source least-squares
/// game.
#include <array>

#include "datamunge/nn/gan_models.hpp"
#include "datamunge/nn/image_ops.hpp"
#include "datamunge/nn/modulated_conv.hpp"
#include "datamunge/nn/pixel_rms_norm.hpp"
#include "datamunge/nn/spatial_instance_norm.hpp"

namespace datamunge::nn::models {
/// @brief Internal fixed source image/noise geometry and source-ordered styled maps.
namespace style_detail {
/// @brief Require finite matching signed grayscale8x8 rows and representable critic activations.
/// @param image Matching floating CPU[N,8,8,1], positive N.
/// @param dtype Required Float32/64 dtype.
/// @throws std::invalid_argument For invalid inputs/products.
/// @throws std::logic_error For non-CPU storage.
inline void Image(const Tensor& image, DType dtype) {
  gan_detail::Finite(image);
  if (image.dim() != 4 || image.shape() != Shape({image.size(0), 8, 8, 1}) ||
      image.dtype() != dtype || image.size(0) > std::numeric_limits<std::int64_t>::max() / 128)
    throw std::invalid_argument("Style critic needs matching positive8x8 grayscale rows");
}
/// @brief Validate all fixed synthesis inputs and optional first-layer style before leaves.
/// @param p Floating generator owner.
/// @param w0 Matching first-layer styles[N,8].
/// @param w1 Matching second-layer/RGB styles[N,8].
/// @param n4 Matching supplied spatial noise[N,4,4,1].
/// @param n8 Matching supplied spatial noise[N,8,8,1].
/// @param version2 Select modulated/demodulated rather than AdaIN maps.
/// @param override Optional matching first-layer channel gains[N,4], modulated variant only.
/// @throws std::invalid_argument For invalid inputs/alignment/override/products.
/// @throws std::logic_error For non-CPU storage.
inline void Inputs(
    ParameterStore& p,
    const Tensor&   w0,
    const Tensor&   w1,
    const Tensor&   n4,
    const Tensor&   n8,
    bool            version2,
    const Tensor&   override) {
  gan_detail::Matrix(w0, 8, p.dtype());
  gan_detail::Matrix(w1, 8, p.dtype());
  gan_detail::Finite(n4);
  gan_detail::Finite(n8);
  const auto n = w0.size(0);
  if (w1.size(0) != n || n4.shape() != Shape({n, 4, 4, 1}) || n8.shape() != Shape({n, 8, 8, 1}) ||
      n4.dtype() != p.dtype() || n8.dtype() != p.dtype() ||
      n > std::numeric_limits<std::int64_t>::max() / 256)
    throw std::invalid_argument(
        "Style synthesis needs aligned W/noise geometry and matching dtypes");
  if (override.defined()) {
    if (!version2)
      throw std::invalid_argument("First style override requires the modulated variant");
    gan_detail::Matrix(override, 4, p.dtype());
    if (override.size(0) != n)
      throw std::invalid_argument("First style override rows must match W");
  }
}
/// @brief Return a source-ordered biased-normal kernel with a learned affine style and live
/// demodulation.
/// @param p Generator owner.
/// @param name Fixed layer4/layer8/rgb source prefix.
/// @param x Four-channel source activation.
/// @param w Mapped styles[N,8].
/// @param kernel Odd kernel extent,3 or1.
/// @param out Output channels,4 or1.
/// @param demodulate Source true for feature layers, false for RGB.
/// @param override Optional first-layer channel gains, otherwise learned affine(w)+1.
/// @return Owning array(image,style,demodulator), with all applicable paths live.
/// @throws std::invalid_argument For incompatible leaves or modulated computation errors.
inline std::array<Tensor, 3> ModConv(
    ParameterStore&    p,
    const std::string& name,
    const Tensor&      x,
    const Tensor&      w,
    std::int64_t       kernel,
    std::int64_t       out,
    bool               demodulate,
    const Tensor&      override = Tensor()) {
  auto weight = p.get(name + "/W", {kernel, kernel, 4, out}, init::normal(.02));
  auto style  = dense(p, name + "/style", w, 4) + 1.;
  if (override.defined())
    style = override;
  auto result = modulated_conv2d(x, weight, style, demodulate);
  return {result.image, style, result.demodulator};
}
/// @brief Compute one source3x3 styled/noisy/leaky layer, with AdaIN or kernel demodulation.
/// @param p Generator owner.
/// @param name Fixed layer4/layer8 source prefix.
/// @param x Validated four-channel activation.
/// @param w Mapped styles[N,8].
/// @param noise Supplied signed one-channel spatial noise, broadcast through learned strengths.
/// @param version2 Select modulated weights and no instance normalization.
/// @param override Optional first-layer channel gains for version2 only.
/// @return Owning array(image,raw style,demodulator), with source scalar1 for AdaIN demodulator.
/// @throws std::invalid_argument For incompatible leaves or primitive errors.
inline std::array<Tensor, 3> Layer(
    ParameterStore&    p,
    const std::string& name,
    const Tensor&      x,
    const Tensor&      w,
    const Tensor&      noise,
    bool               version2,
    const Tensor&      override = Tensor()) {
  Tensor pre, style, demod;
  if (version2) {
    auto layer = ModConv(p, name, x, w, 3, 4, true, override);
    pre        = layer[0];
    style      = layer[1];
    demod      = layer[2];
  } else {
    auto weight = p.get(name + "/W", {3, 3, 4, 4}, init::normal(.02));
    pre         = conv2d(x, weight);
    style       = dense(p, name + "/style", w, 8);
    demod       = scalar_like(1., pre);
  }
  auto strength = p.get(name + "/noise_strength", {4}, init::zeros());
  auto bias     = p.get(name + "/b", {4}, init::zeros());
  auto h        = leaky_relu((pre + noise * strength) + bias, .2);
  if (!version2) {
    const auto n     = style.size(0);
    auto       scale = slice(style, {0, 0}, {n, 4}) + 1.;
    auto       shift = slice(style, {0, 4}, {n, 4});
    h = spatial_instance_normalize(h) * reshape(scale, {n, 1, 1, 4}) + reshape(shift, {n, 1, 1, 4});
  }
  return {h, style, demod};
}
} // namespace style_detail

/// @brief Map source latent8-vectors through pixel RMS and two independent affine/leaky maps.
///
/// Finite signed matching Float32/64 CPU z[N,8], N>0, strides allowed. Divide
/// by last-axis RMS with double(1e-8f) guard, then dense8/leaky.2 twice. Source
/// names mapping1/W,b then mapping2/W,b, native Glorot kernels and zero biases;
/// 144 values in4 leaves. Source seed6607. Initializes only missing leaves, no
/// existing-leaf writes, fitting, optimizer, hidden no-grad or later RNG.
/// Call mapping before synthesis to preserve the original full-G draw order;
/// StyleGan does so automatically. Applicable first/higher z/weight paths live;
/// owning output retains graph/storage. Inputs unchanged. O(N) work/storage for
/// fixed maps; synchronize stores. Stored-leaf/math errors can follow new leaves.
/// @param generator Caller-owned floating G parameter store, source seed6607.
/// @param z Finite signed matching CPU[N,8] latent bank, positive N.
/// @return Owning live mapped W[N,8], without truncation or moving average state.
/// @throws std::invalid_argument For invalid inputs/leaves or nonfinite RMS/map output.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// dm::nn::ParameterStore g(6607);
/// auto w=dm::nn::models::StyleGanMap(g,dm::nn::ones({2,8}));
/// // w is[2,8];144 mapping values, no Z-independent learned noise or target input.
/// @endcode
/// @par References
/// - Karras,Laine and Aila (2018/2019), https://arxiv.org/abs/1812.04948 .
///   This repository uses two8-unit mapping layers and its source-rounded RMS guard.
/// @see StyleGanSynthesize(), StyleGan(), pixel_rms_normalize()
inline Tensor StyleGanMap(ParameterStore& generator, const Tensor& z) {
  gan_detail::Matrix(z, 8, generator.dtype());
  auto mapped = dense(
      generator,
      "mapping2",
      dense(generator, "mapping1", pixel_rms_normalize(z), 8, Activation::LeakyRelu),
      8,
      Activation::LeakyRelu);
  gan_detail::Finite(mapped);
  return mapped;
}

/// @brief Own synthesized pixels, all synthesis boundaries, styles and source demodulator fields.
///
/// Nine flat fields retain storage and applicable W/noise/weight paths after
/// parent/store collection. AdaIN style fields are raw scale-delta/shift[N,8];
/// modulated fields are actual affine(w)+1 channel gains[N,4]. AdaIN demodulators
/// are scalar1, while modulated demodulators are[N,4]. No fitting or state flag.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore g(6607);
/// auto w=n::models::StyleGanMap(g,n::ones({2,8}));
/// auto o=n::models::StyleGanSynthesize(g,w,w,n::zeros({2,4,4,1}),n::zeros({2,8,8,1}),true);
/// // first is[2,4,4,4],second[2,8,8,4],image[2,8,8,1]; first_style is[2,4].
/// @endcode
/// @par References
/// - Karras et al. (2019/2020), https://arxiv.org/abs/1912.04958 .
/// @see StyleGanSynthesize(), StyleGanOutput
struct StyleGanSynthesis {
  Tensor image;             ///< Owning live sigmoid image[N,8,8,1].
  Tensor tiled_constant;    ///< Owning live tile of learned constant[N,4,4,4], source leaf
                            ///< shape[1,4,4,4].
  Tensor first;             ///< Owning live first styled/noisy feature map[N,4,4,4].
  Tensor upsampled;         ///< Owning live nearest-up first map[N,8,8,4].
  Tensor second;            ///< Owning live second styled/noisy feature map[N,8,8,4].
  Tensor first_style;       ///< Owning raw affine scale-delta/shift[N,8] for AdaIN, actual channel
                            ///< gains[N,4] for version2.
  Tensor second_style;      ///< Owning second layer's raw affine[N,8] or actual channel gains[N,4].
  Tensor first_demodulator; ///< Owning live[N,4] factors for version2; owning scalar1 for AdaIN.
  Tensor second_demodulator; ///< Owning live[N,4] factors for version2; owning scalar1 for AdaIN.
};

/// @brief Synthesize8x8 pixels from per-layer W styles and caller-supplied spatial noise only.
///
/// Finite signed matching Float32/64 CPU w0/w1[N,8], noise4[N,4,4,1],noise8
///[N,8,8,1], positive aligned N, strides allowed. Learn constant[1,4,4,4] normal
/// .02, tile N, styled3x3 layer4, nearest up2, styled3x3 layer8,1x1 RGB sigmoid.
/// AdaIN(default): plain normal conv, raw dense(w,8), zero noise strengths/bias,
/// leaky((conv+noise*strength)+bias,.2), population spatial normalization, then
/// affine(1+scale_delta,shift). RGB is biased normal plain1x1. Version2: learned
/// dense(w,4)+1 gains, per-example modulated/demodulated feature kernels, same
/// explicit noisy/leaky grouping, no AdaIN; RGB is modulated without demodulation
/// using w1, then bias/sigmoid. Source guards are double(1e-8f); taps accumulate
/// row-major batched matmuls. All normal kernels.02, affine styles native Glorot.
///
/// Call StyleGanMap first for original full-G initialization order. Synthesis
/// alone creates517/13(AdaIN) or481/15(version2) values/leaves; mapping adds144/4
/// for source totals661/17 or625/19. Existing leaves unchanged. Never retain a
/// variant flag; using incompatible variant leaves raises an error. Optional
/// first_style_override[N,4] is version2 only: replaces first actual gains while
/// still creating/computing its unused affine parameters, as the original control.
/// All nine fields own applicable first/higher W/noise/weight graphs. No hidden
/// no-grad, random noise, truncation, moving W average, fitting or optimizer write.
/// Inputs unchanged. Validate all supplied geometry before leaves; stored-leaf/
/// math failures may follow initialization. O(N) fixed-map work/storage; synchronize.
/// @param generator Caller-owned matching floating G store, source seed6607.
/// @param w0 First layer mapped styles[N,8], independent from w1 for mixing.
/// @param w1 Second layer and version2 RGB mapped styles[N,8].
/// @param noise4 Explicit finite signed matching spatial noise[N,4,4,1].
/// @param noise8 Explicit finite signed matching spatial noise[N,8,8,1].
/// @param version2 True selects weight modulation/demodulation; false(default) selects AdaIN.
/// @param first_style_override Optional matching first-layer channel gains[N,4] for version2;
/// undefined(default) uses affine(w0)+1.
/// @return Owning StyleGanSynthesis with live image/boundary/style/demodulator fields.
/// @throws std::invalid_argument For invalid inputs/products/variant/override/leaves or nonfinite
/// computation.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore g(6607);
/// auto w=n::models::StyleGanMap(g,n::ones({1,8}));
/// auto o=n::models::StyleGanSynthesize(g,w,w,n::zeros({1,4,4,1}),n::zeros({1,8,8,1}));
/// // First/second AdaIN styles are[1,8],demodulators scalar1; source G has661 values.
/// @endcode
/// @par References
/// - Karras et al. (2018/2019), https://arxiv.org/abs/1812.04948 ;
///   Karras et al. (2019/2020), https://arxiv.org/abs/1912.04958 . Compact two-layer
///   teaching variants, without progressive growth, path-length regularization or EMA.
/// @see StyleGanSynthesis, StyleGanMap(), modulated_conv2d(), spatial_instance_normalize()
inline StyleGanSynthesis StyleGanSynthesize(
    ParameterStore& generator,
    const Tensor&   w0,
    const Tensor&   w1,
    const Tensor&   noise4,
    const Tensor&   noise8,
    bool            version2             = false,
    const Tensor&   first_style_override = Tensor()) {
  style_detail::Inputs(generator, w0, w1, noise4, noise8, version2, first_style_override);
  StyleGanSynthesis o;
  auto              constant = generator.get("constant", {1, 4, 4, 4}, init::normal(.02));
  o.tiled_constant           = tile(constant, {w0.size(0), 1, 1, 1});
  auto first                 = style_detail::Layer(
      generator, "layer4", o.tiled_constant, w0, noise4, version2, first_style_override);
  o.first             = first[0];
  o.first_style       = first[1];
  o.first_demodulator = first[2];
  o.upsampled         = upsample_nearest(o.first, 2);
  auto second         = style_detail::Layer(generator, "layer8", o.upsampled, w1, noise8, version2);
  o.second            = second[0];
  o.second_style      = second[1];
  o.second_demodulator = second[2];
  if (version2) {
    auto rgb = style_detail::ModConv(generator, "rgb", o.second, w1, 1, 1, false);
    o.image  = sigmoid(rgb[0] + generator.get("rgb/b", {1}, init::zeros()));
  } else {
    generator.get("rgb/W", {1, 1, 4, 1}, init::normal(.02));
    o.image = conv2d(
        generator, "rgb", o.second, 1, 1, ConvLayerOptions{.activation = Activation::Sigmoid});
  }
  gan_detail::Finite(o.image);
  return o;
}

/// @brief Score8x8 grayscale images with the source493-value raw least-squares critic.
///
/// Finite signed matching Float32/64 CPU[N,8,8,1], positive N, strides allowed.
/// Biased normal(.02) SAME4x4/stride2 conv1->4/leaky.2,3x3/stride1 conv4->8/
/// leaky.2, flatten128, native Glorot dense128->1 raw score. Source seed6619;
/// 493 values in6 leaves. Only missing leaves initialize; no subsequent RNG,
/// sigmoid, batch state, hidden no-grad, fitting or optimizer writes. Applicable
/// first/higher image/weight paths live, owning output/graph. Input unchanged.
/// Validate input before leaves; later stored-leaf/math errors may follow earlier
/// initialization. O(N) work/storage for fixed map; synchronize stores.
/// @param critic Caller-owned floating critic store, source seed6619.
/// @param image Finite signed matching CPU[N,8,8,1], positive N.
/// @return Owning live raw scores[N,1].
/// @throws std::invalid_argument For invalid input/products/leaves or nonfinite score.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// dm::nn::ParameterStore d(6619);
/// auto score=dm::nn::models::StyleGanDiscriminate(d,dm::nn::ones({2,8,8,1}));
/// // Raw[2,1],493 values/6 tensors, no sigmoid or style/noise input.
/// @endcode
/// @par References
/// - Karras et al. (2018/2019), https://arxiv.org/abs/1812.04948 . This repository's
///   small fixed critic is shared by both teaching variants, not a paper benchmark.
/// @see StyleGan(), StyleGanSynthesize()
inline Tensor StyleGanDiscriminate(ParameterStore& critic, const Tensor& image) {
  style_detail::Image(image, critic.dtype());
  critic.get("down4/W", {4, 4, 1, 4}, init::normal(.02));
  ConvLayerOptions down;
  down.stride = 2;
  auto h      = leaky_relu(conv2d(critic, "down4", image, 4, 4, down), .2);
  critic.get("conv/W", {3, 3, 4, 8}, init::normal(.02));
  h          = leaky_relu(conv2d(critic, "conv", h, 8, 3), .2);
  auto score = dense(critic, "score", reshape(h, {-1, 128}), 1);
  gan_detail::Finite(score);
  return score;
}

/// @brief Own the ten complete source image/mapping/style/demodulator/score/loss fields.
///
/// Flat Tensor fields retain storage/applicable graphs after parent/store
/// collection. Variant-specific style/demodulator shapes match StyleGanSynthesis.
/// No capture, optimizer, persistent variant flag, fitting or mutable batch state.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore g(6607),d(6619);
/// auto
/// o=n::models::StyleGan(g,d,n::zeros({2,8,8,1}),n::ones({3,8}),n::zeros({3,4,4,1}),n::zeros({3,8,8,1}),true);
/// // Real/fake pools may differ: real_score[2,1],fake[3,8,8,1],mapped[3,8].
/// @endcode
/// @par References
/// - Karras et al. (2019/2020), https://arxiv.org/abs/1912.04958 .
/// @see StyleGan(), StyleGanSynthesis
struct StyleGanOutput {
  Tensor fake;               ///< Owning live sigmoid generated images[M,8,8,1].
  Tensor mapped;             ///< Owning live pixel-normalized/two-layer latent map[M,8].
  Tensor first_style;        ///< Owning live first raw affine[M,8](AdaIN) or actual channel
                             ///< gains[M,4](version2).
  Tensor second_style;       ///< Owning live second raw affine[M,8] or actual channel gains[M,4].
  Tensor first_demodulator;  ///< Owning scalar1(AdaIN) or live[M,4] first output-channel factors.
  Tensor second_demodulator; ///< Owning scalar1(AdaIN) or live[M,4] second output-channel factors.
  Tensor real_score;         ///< Owning live raw critic scores[N,1].
  Tensor fake_score;         ///< Owning live raw critic scores[M,1].
  Tensor d_loss;             ///< Owning half*(mean((real_score-1)^2)+mean(fake_score^2)).
  Tensor g_loss;             ///< Owning full mean((fake_score-1)^2), without half.
};

/// @brief Evaluate either styled least-squares game with explicit latent/noise inputs and no
/// fitting.
///
/// Distinct same-dtype Float32/64 G/D stores; finite signed CPU real[N,8,8,1],
/// z[M,8],noise4[M,4,4,1],noise8[M,8,8,1], N,M>0, strides allowed. Real pool
/// count may differ; latent/noise rows must align. Map z first, synthesize with
/// that same W at both layers, score real/fake separately. Source order/names
/// initialize G661/17(AdaIN) or625/19(version2) values/leaves as documented by
/// StyleGanSynthesize, then D493/6. Source seeds6607/6619; independent stores are
/// required for the two variant trials. No hidden style mixing, random noise,
/// truncation, W average, EMA or extra penalty. G is full fake-to1 MSE; D half
/// the sum of real-to1/fake-to0 means. All applicable first/higher real,z,noise
/// and parameter paths remain live; G gradients pass through current D.
///
/// No capture, optimizer, hidden no-grad, training loop or variant state. Caller
/// captures D/steps D then makes a fresh G forward through updated D with the
/// same supplied latent/noise batch and steps G alone. Existing leaves/inputs
/// unchanged; only missing parameters initialize, no later RNG. Owning fields
/// retain graph/storage. Validate stores/all inputs before leaves; stored-leaf/
/// math errors may follow initialization. O(N+M) fixed-map work/storage, including
/// per-example modulated kernels for version2; synchronize shared stores.
/// @param generator Distinct matching floating G store, source seed6607.
/// @param critic Distinct matching floating D store, source seed6619.
/// @param real Finite signed matching CPU[N,8,8,1] observed images, positive N.
/// @param z Finite signed matching CPU[M,8] latent bank, positive M.
/// @param noise4 Explicit finite signed matching CPU[M,4,4,1] spatial noise.
/// @param noise8 Explicit finite signed matching CPU[M,8,8,1] spatial noise.
/// @param version2 True selects modulated/demodulated synthesis; false(default) selects AdaIN.
/// @return Owning StyleGanOutput with ten complete live source fields.
/// @throws std::invalid_argument For invalid/aliased stores, inputs, products, leaves or nonfinite
/// loss.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore g(6607),d(6619);
/// auto
/// o=n::models::StyleGan(g,d,n::zeros({2,8,8,1}),n::ones({2,8}),n::zeros({2,4,4,1}),n::zeros({2,8,8,1}));
/// auto gradient=n::grad(o.d_loss,d.parameters());
/// n::OptimizerConfig c;c.kind=n::OptimizerKind::Adam;c.beta1=.5;
/// n::Optimizer adam(d,.0002,c);adam.step(gradient); // caller then makes a fresh G forward.
/// @endcode
/// @par References
/// - Karras,Laine and Aila (2018/2019), https://arxiv.org/abs/1812.04948 ;
///   Karras et al. (2019/2020), https://arxiv.org/abs/1912.04958 . These are compact
///   fixed8x8 teaching variants with least-squares objectives, not full paper training.
/// @see StyleGanOutput, StyleGanMap(), StyleGanSynthesize(), StyleGanDiscriminate()
inline StyleGanOutput StyleGan(
    ParameterStore& generator,
    ParameterStore& critic,
    const Tensor&   real,
    const Tensor&   z,
    const Tensor&   noise4,
    const Tensor&   noise8,
    bool            version2 = false) {
  if (&generator == &critic || generator.dtype() != critic.dtype())
    throw std::invalid_argument("StyleGAN needs distinct matching stores");
  style_detail::Image(real, generator.dtype());
  style_detail::Inputs(generator, z, z, noise4, noise8, version2, Tensor());
  StyleGanOutput o;
  o.mapped            = StyleGanMap(generator, z);
  auto g              = StyleGanSynthesize(generator, o.mapped, o.mapped, noise4, noise8, version2);
  o.fake              = g.image;
  o.first_style       = g.first_style;
  o.second_style      = g.second_style;
  o.first_demodulator = g.first_demodulator;
  o.second_demodulator = g.second_demodulator;
  o.real_score         = StyleGanDiscriminate(critic, real);
  o.fake_score         = StyleGanDiscriminate(critic, o.fake);
  o.d_loss             = .5 * (mean(square(o.real_score - 1.)) + mean(square(o.fake_score)));
  o.g_loss             = mean(square(o.fake_score - 1.));
  gan_detail::Finite(o.d_loss);
  gan_detail::Finite(o.g_loss);
  return o;
}
} // namespace datamunge::nn::models
