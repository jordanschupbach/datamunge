#pragma once
/// @file dcgan.hpp
/// @brief Fixed 28x28 DCGAN maps, explicit joint-D/fresh-G phases and detached BN captures.
#include "datamunge/nn/captured_affine_batch_norm.hpp"
#include "datamunge/nn/gan_models.hpp"

namespace datamunge::nn::models {
/// @brief Internal fixed image geometry, store separation and phase-row validation.
namespace dcgan_detail {
/// @brief Validate finite signed matching grayscale NHWC images and activation products.
/// @param image Finite Float32/64 CPU[N,28,28,1], N>0, strides allowed.
/// @param dtype Required computation dtype.
/// @throws std::invalid_argument For wrong geometry/dtype/values or product overflow.
/// @throws std::logic_error For non-CPU storage.
inline void Image(const Tensor& image, DType dtype) {
  gan_detail::Finite(image);
  if (image.dim() != 4 || image.shape() != Shape({image.size(0), 28, 28, 1}) ||
      image.dtype() != dtype || image.size(0) > std::numeric_limits<std::int64_t>::max() / 1568)
    throw std::invalid_argument("DCGAN needs matching positive grayscale NHWC images");
}
/// @brief Require distinct matching affine and moving stores.
/// @param p Floating parameter owner.
/// @param s Distinct matching saved-moment owner.
/// @throws std::invalid_argument For aliasing or dtype mismatch.
inline void Pair(ParameterStore& p, ParameterStore& s) {
  if (&p == &s || p.dtype() != s.dtype())
    throw std::invalid_argument("DCGAN parameter and state stores must be distinct and matching");
}
/// @brief Validate four distinct matching stores and aligned real/noise rows before leaves.
/// @param g Generator owner.
/// @param d Discriminator owner.
/// @param gs Generator saved moments.
/// @param ds Discriminator saved moments.
/// @param image Finite matching real image[N,28,28,1], units0..1 without clipping.
/// @param noise Finite matching noise[N,16].
/// @throws std::invalid_argument For store aliases, mismatches or invalid inputs/products.
/// @throws std::logic_error For non-CPU storage.
inline void Inputs(
    ParameterStore& g,
    ParameterStore& d,
    ParameterStore& gs,
    ParameterStore& ds,
    const Tensor&   image,
    const Tensor&   noise) {
  Pair(g, gs);
  Pair(d, ds);
  if (&g == &d || &g == &ds || &d == &gs || &gs == &ds || g.dtype() != d.dtype())
    throw std::invalid_argument("DCGAN needs four distinct matching stores");
  Image(image, g.dtype());
  gan_detail::Matrix(noise, 16, g.dtype());
  if (image.size(0) != noise.size(0))
    throw std::invalid_argument("DCGAN phase real/noise rows must match");
}
} // namespace dcgan_detail

/// @brief Own signed generated images and optional detached BN captures for both generator layers.
///
/// image[N,28,28,1] retains applicable first/higher noise/parameter graphs.
/// Four moment vectors are detached, undefined on saved-statistics calls.
/// Fields retain storage after parent/store collection; no update callback retained.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// dm::nn::ParameterStore p(5003),s(0);
/// auto out=dm::nn::models::DcganGenerate(p,s,dm::nn::zeros({2,16}),true);
/// // out.image is signed tanh pixels; stem_mean[8], up14_mean[4] are detached.
/// @endcode
/// @par References
/// - Radford, Metz and Chintala (2015/2016), https://arxiv.org/abs/1511.06434 .
/// @see DcganGenerate(), DcganPhaseOutput
struct DcganGenerated {
  Tensor image;         ///< Owning live signed tanh pixels[N,28,28,1].
  Tensor stem_mean;     ///< Owning detached stem batch mean[8], undefined in inference.
  Tensor stem_variance; ///< Owning detached stem unbiased variance[8], undefined in inference.
  Tensor up14_mean;     ///< Owning detached up14 batch mean[4], undefined in inference.
  Tensor up14_variance; ///< Owning detached up14 unbiased variance[4], undefined in inference.
};

/// @brief Generate signed 28x28 pixels through projection, fused BN and two learned transposed
/// maps.
///
/// Finite matching Float32/64 CPU noise[N,16], N>0, strides allowed. Distinct
/// matching parameter/state stores. Bias-free normal(.02) projection16->392
/// reshapes to7x7x8, fused affine BN/ReLU, SAME4x4 stride2 transpose8->4,
/// fused affine BN/ReLU, SAME4x4 stride2 transpose4->1 with bias and tanh.
/// Creation order/names preserve source initialization:6873 values/eight tensors;
/// moving24 values/four tensors. Default saved-statistics path; batch mode
/// returns detached mean/unbiased variance captures using Double epsilon.001.
/// No moving write after initialization, RNG beyond new parameter creation,
/// optimizer or fitting. Applicable first/higher noise/weight paths live; saved
/// and captured moments stop. Outputs own lifetime, inputs unchanged. Finite
/// signed noise and positive representable intermediate products required;
/// stored-leaf/math errors may follow earlier new leaves. Work/storage O(N)
/// for this fixed architecture; synchronize shared stores, exclude state from optimizers.
/// @param generator Mutable matching floating generator parameter owner.
/// @param moving Distinct matching moving BN owner, unchanged after initialization.
/// @param noise Finite signed matching CPU[N,16] latent bank, N>0.
/// @param batch_statistics Fused batch BN/captures when true; saved BN when false(default).
/// @return Owning DcganGenerated with signed pixels and optional detached moments.
/// @throws std::invalid_argument For invalid geometry/stores, incompatible leaves or nonfinite
/// computation.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore g(5003),s(0);
/// auto out=n::models::DcganGenerate(g,s,n::ones({2,16}));
/// auto unit_pixels=(out.image+1.)*.5; // explicit display conversion, no clipping.
/// @endcode
/// @par References
/// - Radford, Metz and Chintala (2015/2016), Unsupervised Representation Learning
///   with Deep Convolutional Generative Adversarial Networks, https://arxiv.org/abs/1511.06434 .
///   This source uses a compact8/4-channel teaching architecture.
/// @see DcganGenerated, DcganDiscriminate(), capture_affine_batch_norm()
inline DcganGenerated DcganGenerate(
    ParameterStore& generator,
    ParameterStore& moving,
    const Tensor&   noise,
    bool            batch_statistics = false) {
  dcgan_detail::Pair(generator, moving);
  gan_detail::Matrix(noise, 16, generator.dtype());
  if (noise.size(0) > std::numeric_limits<std::int64_t>::max() / 784)
    throw std::invalid_argument("DCGAN activation product overflow");
  const auto normal = init::normal(.02);
  auto h = reshape(matmul(noise, generator.get("projection/W", {16, 392}, normal)), {-1, 7, 7, 8});
  auto stem = capture_affine_batch_norm(generator, moving, "stem_bn", h, batch_statistics);
  h         = relu(stem.value);
  ConvLayerOptions up;
  up.stride = 2;
  up.bias   = false;
  generator.get("up14/W", {4, 4, 4, 8}, normal);
  h             = conv2d_transpose(generator, "up14", h, 4, 4, up);
  auto second   = capture_affine_batch_norm(generator, moving, "up14_bn", h, batch_statistics);
  h             = relu(second.value);
  up.bias       = true;
  up.activation = Activation::Tanh;
  generator.get("output/W", {4, 4, 1, 4}, normal);
  h = conv2d_transpose(generator, "output", h, 1, 4, up);
  gan_detail::Finite(h);
  return {h, stem.mean, stem.unbiased_variance, second.mean, second.unbiased_variance};
}

/// @brief Own discriminator raw logits and optional detached joint-batch moments.
///
/// logit[N,1] retains applicable first/higher image/parameter graphs. mean and
/// variance[8] are detached and undefined on saved-statistics calls. Fields own
/// storage after parent/store collection; no optimizer or state write retained.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// dm::nn::ParameterStore p(5011),s(0);
/// auto out=dm::nn::models::DcganDiscriminate(p,s,dm::nn::zeros({2,28,28,1}),true);
/// // out.logit is[2,1]; no sigmoid and no saved-moment write.
/// @endcode
/// @par References
/// - Radford et al. (2015/2016), https://arxiv.org/abs/1511.06434 .
/// @see DcganDiscriminate(), DcganDiscriminatorPhase()
struct DcganDiscriminated {
  Tensor logit;    ///< Owning live raw logits[N,1], without sigmoid.
  Tensor mean;     ///< Owning detached down7 batch mean[8], undefined in inference.
  Tensor variance; ///< Owning detached down7 unbiased variance[8], undefined in inference.
};

/// @brief Score signed 28x28 images through strided convolutions, fused BN and a raw dense head.
///
/// Finite matching Float32/64 CPU image[N,28,28,1], positive N, strides allowed;
/// two distinct matching stores. SAME4x4 stride2 conv1->4 with bias/LeakyReLU(.2),
/// bias-free SAME4x4 stride2 conv4->8, fused affine BN/LeakyReLU(.2), flatten392
/// and biased raw dense1. Kernels normal(.02), source names/order;989 values/seven
/// tensors, moving16 values/two tensors. Batch flag false(default) reads detached
/// saved moments; true captures detached mean/sample variance with epsilon.001.
/// No sigmoid, joint-pool construction, G call, moving write after initialization,
/// optimizer or fitting. Applicable first/higher continuous paths live; captures
/// and saved moments stop. Owning outputs; inputs unchanged. Invalid geometry/
/// products checked before leaves; stored-leaf/nonfinite math may fail later.
/// Work/storage O(N); synchronize stores and exclude state from optimizers.
/// @param discriminator Mutable matching floating discriminator owner.
/// @param moving Distinct matching saved BN owner, unchanged after initialization.
/// @param image Finite signed matching CPU[N,28,28,1], N>0.
/// @param batch_statistics Fused batch BN/captures when true; saved BN when false(default).
/// @return Owning DcganDiscriminated with raw logits and optional detached moments.
/// @throws std::invalid_argument For invalid geometry/stores, incompatible leaves or nonfinite
/// computation.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore d(5011),s(0);
/// auto out=n::models::DcganDiscriminate(d,s,n::ones({2,28,28,1}));
/// auto probability=n::sigmoid(out.logit); // optional diagnostic, never the loss input.
/// @endcode
/// @par References
/// - Radford et al. (2015/2016), https://arxiv.org/abs/1511.06434 .
/// @see DcganDiscriminated, DcganDiscriminatorPhase(), DcganGeneratorPhase()
inline DcganDiscriminated DcganDiscriminate(
    ParameterStore& discriminator,
    ParameterStore& moving,
    const Tensor&   image,
    bool            batch_statistics = false) {
  dcgan_detail::Pair(discriminator, moving);
  dcgan_detail::Image(image, discriminator.dtype());
  const auto       normal = init::normal(.02);
  ConvLayerOptions down;
  down.stride = 2;
  discriminator.get("down14/W", {4, 4, 1, 4}, normal);
  auto h    = leaky_relu(conv2d(discriminator, "down14", image, 4, 4, down), .2);
  down.bias = false;
  discriminator.get("down7/W", {4, 4, 4, 8}, normal);
  h          = conv2d(discriminator, "down7", h, 8, 4, down);
  auto bn    = capture_affine_batch_norm(discriminator, moving, "down7_bn", h, batch_statistics);
  h          = leaky_relu(bn.value, .2);
  auto w     = discriminator.get("output/W", {392, 1}, normal),
       b     = discriminator.get("output/b", {1}, init::zeros());
  auto logit = bias_add(matmul(reshape(h, {-1, 392}), w), b);
  gan_detail::Finite(logit);
  return {logit, bn.mean, bn.unbiased_variance};
}

/// @brief Own one DCGAN phase objective, signed fake pixels, logits and detached moment captures.
///
/// D phase stops G fake/weights, retains real/D paths, and captures both G and D
/// moments; G phase retains G/noise/D-weight paths, defines no real_logit/down7
/// moments and uses saved D BN. Choose only the intended parameter group's
/// gradient. Every field owns lifetime; no store/optimizer/update callback retained.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore g(5003),d(5011),gs(0),ds(0);
/// auto out=n::models::DcganDiscriminatorPhase(g,d,gs,ds,n::zeros({2,28,28,1}),n::ones({2,16}));
/// // out.fake has no G path; out.down7_variance is a detached sample variance.
/// @endcode
/// @par References
/// - Radford et al. (2015/2016), https://arxiv.org/abs/1511.06434 .
/// @see DcganDiscriminatorPhase(), DcganGeneratorPhase(), batch_norm_moving_values()
struct DcganPhaseOutput {
  Tensor fake;           ///< Owning signed fake[N,28,28,1]; stopped for D, live for G.
  Tensor real_logit;     ///< Owning live real[N,1] logits in D phase; undefined for G phase.
  Tensor fake_logit;     ///< Owning live fake[N,1] logits under this phase's D BN convention.
  Tensor loss;           ///< Owning scalar logistic D or non-saturating G phase objective.
  Tensor stem_mean;      ///< Owning detached G stem batch mean[8].
  Tensor stem_variance;  ///< Owning detached G stem unbiased variance[8].
  Tensor up14_mean;      ///< Owning detached G up14 batch mean[4].
  Tensor up14_variance;  ///< Owning detached G up14 unbiased variance[4].
  Tensor down7_mean;     ///< Owning detached D joint batch mean[8]; undefined for G phase.
  Tensor down7_variance; ///< Owning detached D joint unbiased variance[8]; undefined for G phase.
};

/// @brief Compute the D objective with stopped batch-BN G and one jointly normalized real/fake D
/// batch.
///
/// Four distinct matching Float32/64 stores, finite real image[N,28,28,1] in
/// unit-pixel coordinates (no range clipping) and noise[N,16], positive matching
/// rows. Maps real to2*image-1, stops G under NoGradGuard, concatenates real then
/// fake, runs D batch BN once and returns mean softplus(-real)+mean softplus(fake).
/// No existing moments or weights are written. Commit only D captures using
/// batch_norm_moving_values(...,.9), then apply the selected D gradient. Captures
/// are detached; applicable real/D first/higher paths remain live, all G paths
/// stop. Initialization, geometry, failure and concurrency contracts follow the
/// component maps; all row/store checks precede leaves. O(N) work/storage.
/// @param generator Mutable matching G owner, reused without gradient recording.
/// @param discriminator Mutable matching D owner, intended gradient group.
/// @param g_moving Distinct matching G saved moments, unchanged after initialization.
/// @param d_moving Distinct matching D saved moments, unchanged after initialization.
/// @param image Finite matching real CPU[N,28,28,1], units0..1 without enforcement.
/// @param noise Finite signed matching CPU[N,16], N>0.
/// @return Owning DcganPhaseOutput with stopped fake, joint-D logits/loss and detached captures.
/// @throws std::invalid_argument For invalid inputs/stores or nonfinite computation.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore g(5003),d(5011),gs(0),ds(0);
/// auto out=n::models::DcganDiscriminatorPhase(g,d,gs,ds,n::zeros({2,28,28,1}),n::ones({2,16}));
/// auto gradients=n::grad(out.loss,d.parameters()); // no optimizer/moving write here.
/// @endcode
/// @par References
/// - Goodfellow et al. (2014), Generative Adversarial Nets, https://arxiv.org/abs/1406.2661 .
/// - Radford et al. (2015/2016), https://arxiv.org/abs/1511.06434 .
/// @see DcganGeneratorPhase(), DcganPhaseOutput, batch_norm_moving_values()
inline DcganPhaseOutput DcganDiscriminatorPhase(
    ParameterStore& generator,
    ParameterStore& discriminator,
    ParameterStore& g_moving,
    ParameterStore& d_moving,
    const Tensor&   image,
    const Tensor&   noise) {
  dcgan_detail::Inputs(generator, discriminator, g_moving, d_moving, image, noise);
  DcganGenerated g;
  {
    NoGradGuard guard;
    g = DcganGenerate(generator, g_moving, noise, true);
  }
  auto d = DcganDiscriminate(discriminator, d_moving, concat({image * 2. - 1., g.image}, 0), true);
  auto n = image.size(0);
  auto real = slice_axis(d.logit, 0, 0, n), fake = slice_axis(d.logit, 0, n, 2 * n);
  auto loss = LogisticGanObjectives(real, fake).discriminator;
  return {
      g.image,
      real,
      fake,
      loss,
      g.stem_mean,
      g.stem_variance,
      g.up14_mean,
      g.up14_variance,
      d.mean,
      d.variance};
}

/// @brief Compute fresh non-saturating G loss with G batch BN and the current D saved BN.
///
/// Same fixed geometry/four-store contract as DcganDiscriminatorPhase(). G uses
/// fused batch statistics and returns detached G captures; D uses detached saved
/// statistics without capturing or updating them. G=-mean log sigmoid(fake_logit),
/// evaluated as mean softplus(-fake_logit). Call after the D capture commit and D
/// optimizer write. Select only G gradients; commit both G layers using explicit
/// Double .9 retention, then apply G Adam. The D weight path remains available,
/// so group selection is the caller's responsibility. No implicit freeze, fit,
/// optimizer or existing-state mutation. Applicable first/higher continuous paths
/// live; saved/captured moments stop. Checks precede leaves, owning output fields,
/// O(N) work/storage, synchronize shared stores.
/// @param generator Mutable matching G owner, intended gradient group.
/// @param discriminator Mutable matching current D owner, not updated here.
/// @param g_moving Distinct matching G saved moments, unchanged after initialization.
/// @param d_moving Distinct matching D saved moments, unchanged after initialization.
/// @param image Finite matching real CPU[N,28,28,1], used for input/row validation only.
/// @param noise Finite signed matching CPU[N,16], N>0.
/// @return Owning DcganPhaseOutput with live fake/G loss and detached G captures; real/D captures
/// undefined.
/// @throws std::invalid_argument For invalid inputs/stores or nonfinite computation.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore g(5003),d(5011),gs(0),ds(0);
/// auto out=n::models::DcganGeneratorPhase(g,d,gs,ds,n::zeros({2,28,28,1}),n::ones({2,16}));
/// auto gradients=n::grad(out.loss,g.parameters()); // commit G captures, then step explicitly.
/// @endcode
/// @par References
/// - Goodfellow et al. (2014), https://arxiv.org/abs/1406.2661 , non-saturating G objective.
/// - Radford et al. (2015/2016), https://arxiv.org/abs/1511.06434 .
/// @see DcganDiscriminatorPhase(), DcganPhaseOutput, batch_norm_moving_values()
inline DcganPhaseOutput DcganGeneratorPhase(
    ParameterStore& generator,
    ParameterStore& discriminator,
    ParameterStore& g_moving,
    ParameterStore& d_moving,
    const Tensor&   image,
    const Tensor&   noise) {
  dcgan_detail::Inputs(generator, discriminator, g_moving, d_moving, image, noise);
  auto g    = DcganGenerate(generator, g_moving, noise, true);
  auto d    = DcganDiscriminate(discriminator, d_moving, g.image);
  auto loss = mean(softplus(-d.logit));
  gan_detail::Finite(loss);
  return {
      g.image,
      Tensor{},
      d.logit,
      loss,
      g.stem_mean,
      g.stem_variance,
      g.up14_mean,
      g.up14_variance,
      Tensor{},
      Tensor{}};
}

/// @brief Own saved-BN unit pixels, raw/probability D scores and explicit logistic objectives.
///
/// Seven fields own storage and applicable graphs after parent/store collection.
/// No moving update, coarse-pixel diagnostic, optimizer, fitting or checkpoint hidden.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore g(5003),d(5011),gs(0),ds(0);
/// auto out=n::models::Dcgan(g,d,gs,ds,n::zeros({2,28,28,1}),n::ones({2,16}));
/// // out.fake is[2,784] in unit-pixel coordinates; losses consume raw logits.
/// @endcode
/// @par References
/// - Radford et al. (2015/2016), https://arxiv.org/abs/1511.06434 .
/// @see Dcgan(), DcganPhaseOutput
struct DcganOutput {
  Tensor fake;             ///< Owning generated unit pixels[N,784], converted from signed tanh.
  Tensor real_logit;       ///< Owning live real raw logits[N,1].
  Tensor fake_logit;       ///< Owning live generated raw logits[N,1].
  Tensor real_probability; ///< Owning sigmoid(real_logit)[N,1], diagnostic only.
  Tensor fake_probability; ///< Owning sigmoid(fake_logit)[N,1], diagnostic only.
  Tensor d_loss;           ///< Owning logistic discriminator scalar from raw logits.
  Tensor g_loss;           ///< Owning non-saturating generator scalar from raw logits.
};

/// @brief Evaluate both saved-BN DCGAN maps with raw logistic losses and unit-pixel generation.
///
/// Four distinct matching floating stores and aligned finite real[N,28,28,1]/
/// noise[N,16], N>0, under the phase input contract. G uses saved BN; D runs once
/// on concat(2*real-1,signed_fake) using saved BN, so rows do not share batch
/// statistics. Returns fake[N,784]=(signed_fake+1)*.5, raw/probability scores,
/// D=mean softplus(-real)+mean softplus(fake), G=mean softplus(-fake). No NoGrad
/// guard is hidden: applicable first/higher continuous paths live, saved moments
/// stop. No existing-state write, training selection, optimizer, RNG except new
/// leaves, checkpoint or O(N^2) MMD. Compute GanCoarsePixels()/ImqMmd() separately.
/// Owning output fields, inputs unchanged, O(N) work/storage; synchronize stores.
/// Checks precede leaves, while stored-leaf/math errors may occur later.
/// @param generator Mutable matching floating G owner.
/// @param discriminator Mutable matching floating D owner.
/// @param g_moving Distinct matching G saved moments.
/// @param d_moving Distinct matching D saved moments.
/// @param image Finite matching real CPU[N,28,28,1], units0..1 without enforcement.
/// @param noise Finite signed matching CPU[N,16], positive matching rows.
/// @return Owning seven-field DcganOutput using saved statistics in both networks.
/// @throws std::invalid_argument For invalid inputs/stores or nonfinite computation.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore g(5003),d(5011),gs(0),ds(0);
/// auto out=n::models::Dcgan(g,d,gs,ds,n::zeros({2,28,28,1}),n::ones({2,16}));
/// auto features=n::models::GanCoarsePixels(out.fake); // separate diagnostic input.
/// @endcode
/// @par References
/// - Radford et al. (2015/2016), https://arxiv.org/abs/1511.06434 .
/// - Goodfellow et al. (2014), https://arxiv.org/abs/1406.2661 .
/// @see DcganOutput, DcganGenerate(), DcganDiscriminatorPhase(), DcganGeneratorPhase()
inline DcganOutput Dcgan(
    ParameterStore& generator,
    ParameterStore& discriminator,
    ParameterStore& g_moving,
    ParameterStore& d_moving,
    const Tensor&   image,
    const Tensor&   noise) {
  dcgan_detail::Inputs(generator, discriminator, g_moving, d_moving, image, noise);
  auto g    = DcganGenerate(generator, g_moving, noise);
  auto d    = DcganDiscriminate(discriminator, d_moving, concat({image * 2. - 1., g.image}, 0));
  auto n    = image.size(0);
  auto real = slice_axis(d.logit, 0, 0, n), fake = slice_axis(d.logit, 0, n, 2 * n);
  auto loss   = LogisticGanObjectives(real, fake);
  auto pixels = reshape((g.image + 1.) * .5, {-1, 784});
  gan_detail::Finite(pixels);
  return {pixels, real, fake, sigmoid(real), sigmoid(fake), loss.discriminator, loss.generator};
}
} // namespace datamunge::nn::models
