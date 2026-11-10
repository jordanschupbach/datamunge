#pragma once
/// @file pix2pix.hpp
/// @brief Paired28x28 U-Net, conditional49-patch discriminator and explicit logistic/L1 objectives.
#include "datamunge/nn/gan_models.hpp"

namespace datamunge::nn::models {
/// @brief Internal fixed grayscale geometry and source-compatible biased normal convolutions.
namespace pix_detail {
/// @brief Require matching finite signed grayscale images and representable activation products.
/// @param image Finite floating CPU[N,28,28,1], positive N, strides allowed.
/// @param dtype Required Float32/64 computation dtype.
/// @throws std::invalid_argument For incompatible dtype/geometry/values or product overflow.
/// @throws std::logic_error For non-CPU storage.
inline void Image(const Tensor& image, DType dtype) {
  gan_detail::Finite(image);
  if (image.dim() != 4 || image.shape() != Shape({image.size(0), 28, 28, 1}) ||
      image.dtype() != dtype || image.size(0) > std::numeric_limits<std::int64_t>::max() / 2352)
    throw std::invalid_argument("Pix2Pix needs matching positive28x28 grayscale NHWC images");
}
/// @brief Apply a biased SAME normal(.02) map, with optional fused activation/stride-two transpose.
/// @param p Floating architecture parameter owner.
/// @param name Fixed source layer prefix.
/// @param x Validated matching NHWC activation.
/// @param out Positive output channel width.
/// @param kernel Fixed3x3 or4x4 kernel extent.
/// @param stride Fixed1 or2 shared spatial stride.
/// @param activation Source activation, default None.
/// @param transpose Use learned transposed convolution when true, default false.
/// @return Owning live layer activation; new kernels use source names/order/normal initializer.
/// @throws std::invalid_argument For incompatible stored leaves or primitive errors.
inline Tensor Conv(
    ParameterStore&    p,
    const std::string& name,
    const Tensor&      x,
    std::int64_t       out,
    std::int64_t       kernel,
    std::int64_t       stride,
    Activation         activation = Activation::None,
    bool               transpose  = false) {
  const auto in = x.shape().back();
  p.get(
      name + "/W",
      transpose ? Shape{kernel, kernel, out, in} : Shape{kernel, kernel, in, out},
      init::normal(.02));
  ConvLayerOptions options;
  options.stride     = stride;
  options.activation = activation;
  return transpose ? conv2d_transpose(p, name, x, out, kernel, options)
                   : conv2d(p, name, x, out, kernel, options);
}
} // namespace pix_detail

/// @brief Own every U-Net encoder/decoder boundary and both genuine concatenation skips.
///
/// All fields retain storage/applicable source and parameter graphs after parent
/// or store collection. No inference/training flag, batch state or callback.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// dm::nn::ParameterStore p(6101);
/// auto o=dm::nn::models::Pix2PixGenerate(p,dm::nn::ones({2,28,28,1}));
/// // joined14 is[2,14,14,8]; joined28 is[2,28,28,3]; image is[2,28,28,1].
/// @endcode
/// @par References
/// - Isola et al. (2017), https://arxiv.org/abs/1611.07004 ; compact teaching U-Net.
/// @see Pix2PixGenerate(), Pix2PixOutput
struct Pix2PixGenerated {
  Tensor down14;   ///< Owning live first encoder ReLU[N,14,14,4].
  Tensor down7;    ///< Owning live second encoder ReLU[N,7,7,8].
  Tensor up14;     ///< Owning live first learned transpose/ReLU[N,14,14,4].
  Tensor joined14; ///< Owning live concat(up14,down14), in that channel order[N,14,14,8].
  Tensor up28;     ///< Owning live second learned transpose/ReLU[N,28,28,2].
  Tensor joined28; ///< Owning live concat(up28,source), in that channel order[N,28,28,3].
  Tensor image;    ///< Owning live biased3x3 sigmoid image[N,28,28,1].
};

/// @brief Generate28x28 sigmoid pixels from source alone through two actual U-Net skips.
///
/// Finite matching Float32/64 CPU source[N,28,28,1], N>0, strides and signed
/// counterfactual values allowed without clipping. Parameter store seed6101
/// preserves the original run. SAME4x4/stride2 encoder maps1->4->8, ReLU;
/// learned transpose4x4/stride2 maps8->4 and then8->2 with ReLU. First concat
/// joins(up14,down14); second joins(up28,source), then3x3/stride1 sigmoid3->1.
/// All kernels are biased normal(.02), biases zero; source names/order create
/// 1390 values in10 leaves. No normalization, dropout, labels, target or noise.
///
/// Initializes missing leaves, otherwise preserves store values. No fitting,
/// optimizer write, RNG after initialization or implicit no-grad. Applicable
/// first/higher source/weight paths stay live and sum across both skips.
/// Outputs own graph/storage; input unchanged. Validate input before creating
/// leaves; incompatible stored leaves/nonfinite math can fail after earlier
/// initialization. O(N) work/storage for fixed architecture; synchronize stores.
/// @param generator Caller-owned floating generator store, source seed6101.
/// @param source Finite matching signed CPU[N,28,28,1] condition, positive N.
/// @return Owning Pix2PixGenerated with both real skip joins and sigmoid image.
/// @throws std::invalid_argument For incompatible geometry/dtype/values/leaves or nonfinite output.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// dm::nn::ParameterStore g(6101);
/// auto image=dm::nn::models::Pix2PixGenerate(g,dm::nn::zeros({1,28,28,1})).image;
/// // A blank condition is valid; target images are not generator inputs.
/// @endcode
/// @par References
/// - Isola,Zhu,Zhou and Efros (2017), Image-to-Image Translation with Conditional
///   Adversarial Networks, https://arxiv.org/abs/1611.07004 . Compact two-level
///   teaching variant without normalization or dropout.
/// @see Pix2PixGenerated, Pix2PixDiscriminate(), nn::quantized_sobel_edges()
inline Pix2PixGenerated Pix2PixGenerate(ParameterStore& generator, const Tensor& source) {
  pix_detail::Image(source, generator.dtype());
  Pix2PixGenerated o;
  o.down14   = pix_detail::Conv(generator, "down14", source, 4, 4, 2, Activation::Relu);
  o.down7    = pix_detail::Conv(generator, "down7", o.down14, 8, 4, 2, Activation::Relu);
  o.up14     = pix_detail::Conv(generator, "up14", o.down7, 4, 4, 2, Activation::Relu, true);
  o.joined14 = concat({o.up14, o.down14}, 3);
  o.up28     = pix_detail::Conv(generator, "up28", o.joined14, 2, 4, 2, Activation::Relu, true);
  o.joined28 = concat({o.up28, source}, 3);
  o.image    = pix_detail::Conv(generator, "output", o.joined28, 1, 3, 1, Activation::Sigmoid);
  gan_detail::Finite(o.image);
  return o;
}

/// @brief Score each source/image pair with a conditional49-patch raw-logit discriminator.
///
/// Finite matching signed Float32/64 CPU source/image[N,28,28,1], positive N,
/// equal rows, strides allowed. concat(source,image) channels then biased SAME
/// 4x4/stride2 maps2->4->8, each followed by LeakyReLU(.2), and biased3x3/stride1
/// 8->1 raw output[N,7,7,1]. All kernels normal(.02), biases zero, source names/
/// order; source seed6113 creates725 values in6 leaves. Each interior logit has
/// an18x18 nominal receptive field; borders include zero padding. No sigmoid,
/// pixel normalization or batch-dependent state. Applicable first/higher paths
/// through both inputs and weights remain live; no hidden no-grad or stop.
///
/// Only missing leaves initialize; no fitting/optimizer write or later RNG.
/// Inputs unchanged; owning output retains graph/storage. Validate both inputs
/// and alignment before leaves; stored-leaf/math failures may follow earlier
/// initialization. O(N) work/storage for fixed architecture; synchronize stores.
/// @param discriminator Caller-owned floating D store, source seed6113.
/// @param source Finite matching signed CPU[N,28,28,1] condition.
/// @param image Finite same-shape/dtype CPU candidate image; target is not a G input.
/// @return Owning live raw logits[N,7,7,1], without reduction or sigmoid.
/// @throws std::invalid_argument For incompatible inputs/alignment/leaves or nonfinite output.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// dm::nn::ParameterStore d(6113);auto x=dm::nn::ones({2,28,28,1});
/// auto logits=dm::nn::models::Pix2PixDiscriminate(d,x,x);
/// // logits is[2,7,7,1]:49 conditional raw scores per example.
/// @endcode
/// @par References
/// - Isola et al. (2017), https://arxiv.org/abs/1611.07004 ; compact conditional PatchGAN.
/// @see Pix2PixGenerate(), Pix2Pix(), LogisticGanObjectives()
inline Tensor
Pix2PixDiscriminate(ParameterStore& discriminator, const Tensor& source, const Tensor& image) {
  pix_detail::Image(source, discriminator.dtype());
  pix_detail::Image(image, discriminator.dtype());
  if (source.shape() != image.shape())
    throw std::invalid_argument("Pix2Pix pair rows must match");
  auto down14 = leaky_relu(
      pix_detail::Conv(discriminator, "down14", concat({source, image}, 3), 4, 4, 2), .2);
  auto down7  = leaky_relu(pix_detail::Conv(discriminator, "down7", down14, 8, 4, 2), .2);
  auto logits = pix_detail::Conv(discriminator, "output", down7, 1, 3, 1);
  gan_detail::Finite(logits);
  return logits;
}

/// @brief Own generated pixels, two raw patch maps, and explicit paired adversarial/L1 losses.
///
/// All fields retain applicable source/target/weight paths and storage after
/// parent/store collection. No training schedule, captured derivative or update.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// dm::nn::ParameterStore g(6101),d(6113);auto x=dm::nn::ones({1,28,28,1});
/// auto o=dm::nn::models::Pix2Pix(g,d,x,x);
/// // generator_loss is g_adversarial+100*l1; discriminator_loss is a sum of patch means.
/// @endcode
/// @par References
/// - Isola et al. (2017), https://arxiv.org/abs/1611.07004 .
/// @see Pix2Pix(), Pix2PixGenerated
struct Pix2PixOutput {
  Tensor fake;               ///< Owning live generated sigmoid image[N,28,28,1].
  Tensor real_logit;         ///< Owning live D(source,target) raw patch map[N,7,7,1].
  Tensor fake_logit;         ///< Owning live D(source,fake) raw patch map[N,7,7,1].
  Tensor g_adversarial;      ///< Owning scalar mean(softplus(-fake_logit)), over all rows/patches.
  Tensor l1;                 ///< Owning scalar mean(abs(fake-target)), over all rows/pixels.
  Tensor discriminator_loss; ///< Owning scalar mean(softplus(-real))+mean(softplus(fake)), without
                             ///< half.
  Tensor generator_loss;     ///< Owning scalar g_adversarial+l1_weight*l1, default weight100.
};

/// @brief Evaluate the paired Pix2Pix game with pure forwards and caller-owned update selection.
///
/// Distinct same-dtype Float32/64 G/D stores, finite matching signed CPU target/
/// source[N,28,28,1], positive N. G reads source alone; D reads concat(source,
/// candidate). Patch means reduce all N*49 raw logits. D loss is sum of real
/// and fake logistic means, without half; G is non-saturating patch mean plus
/// l1_weight(default100)*mean absolute paired pixel error. L1 uses the core
/// abs subgradient0 at equality. Weight must be finite, nonnegative and finite
/// after computation-dtype rounding; zero retains the diagnostic L1 field.
///
/// No fitting, optimizer, batch state, implicit no-grad or derivative capture.
/// Initialize G1390/10 then D725/6 leaves in source order; existing values are
/// unchanged. All applicable first/higher paths are live, including through D
/// during a G gradient. Caller captures D partials/steps D, then makes a fresh
/// forward for G through updated D; selecting stores determines writes. Inputs
/// unchanged; owning fields retain graphs/storage. Validate stores, inputs,
/// alignment and weight before leaves; stored-leaf/math failure may follow
/// earlier initialization. O(N) work/storage; synchronize shared stores.
/// @param generator Distinct matching floating G store, source seed6101.
/// @param discriminator Distinct matching floating D store, source seed6113.
/// @param target Finite signed matching CPU[N,28,28,1] paired target.
/// @param source Finite signed same-shape/dtype CPU condition; no labels/noise needed.
/// @param l1_weight Nonnegative finite representable paired L1 multiplier, default100.
/// @return Owning Pix2PixOutput with seven live image/logit/objective fields.
/// @throws std::invalid_argument For invalid inputs/stores/alignment/weight/leaves or nonfinite
/// loss.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore g(6101),d(6113);auto x=n::ones({1,28,28,1});
/// auto o=n::models::Pix2Pix(g,d,x,x);auto gradient=n::grad(o.discriminator_loss,d.parameters());
/// n::OptimizerConfig c;c.kind=n::OptimizerKind::Adam;c.beta1=.5;
/// n::Optimizer adam(d,.0002,c);adam.step(gradient);
/// auto fresh=n::models::Pix2Pix(g,d,x,x); // capture G partials from this updated D.
/// @endcode
/// @par References
/// - Isola,Zhu,Zhou and Efros (2017), Image-to-Image Translation with Conditional
///   Adversarial Networks, https://arxiv.org/abs/1611.07004 . Compact two-level
///   teaching architecture with paired weight100 and explicit D-then-fresh-G.
/// @see Pix2PixGenerate(), Pix2PixDiscriminate(), LogisticGanObjectives(),
/// nn::quantized_sobel_edges()
inline Pix2PixOutput Pix2Pix(
    ParameterStore& generator,
    ParameterStore& discriminator,
    const Tensor&   target,
    const Tensor&   source,
    double          l1_weight = 100) {
  if (&generator == &discriminator || generator.dtype() != discriminator.dtype())
    throw std::invalid_argument("Pix2Pix needs distinct matching stores");
  pix_detail::Image(target, generator.dtype());
  pix_detail::Image(source, generator.dtype());
  const double rounded = generator.dtype() == DType::Float32 ? double(float(l1_weight)) : l1_weight;
  if (target.shape() != source.shape() || !std::isfinite(l1_weight) || l1_weight < 0 ||
      !std::isfinite(rounded))
    throw std::invalid_argument("Pix2Pix needs paired rows and a finite nonnegative L1 weight");
  Pix2PixOutput o;
  o.fake       = Pix2PixGenerate(generator, source).image;
  o.real_logit = Pix2PixDiscriminate(discriminator, source, target);
  o.fake_logit = Pix2PixDiscriminate(discriminator, source, o.fake);
  auto losses =
      LogisticGanObjectives(reshape(o.real_logit, {-1, 1}), reshape(o.fake_logit, {-1, 1}));
  o.g_adversarial      = losses.generator;
  o.l1                 = mean(abs(o.fake - target));
  o.discriminator_loss = losses.discriminator;
  o.generator_loss     = o.g_adversarial + l1_weight * o.l1;
  gan_detail::Finite(o.generator_loss);
  return o;
}
} // namespace datamunge::nn::models
