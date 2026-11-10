#pragma once
/// @file progressive_gan.hpp
/// @brief Eight-store 4x4-to-8x8 progressive GAN with actual old/new fade paths and explicit stage
/// activity.
#include <initializer_list>

#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/gan_models.hpp"
#include "datamunge/nn/image_ops.hpp"
#include "datamunge/nn/pixel_rms_norm.hpp"

namespace datamunge::nn::models {
/// @brief Internal distinct-store, scalar-fade, fixed-shape and normal-kernel feature validation.
namespace progressive_detail {
/// @brief Require nonempty distinct matching Float32/64 store pointers before creating any leaves.
/// @param stores Nonempty nonnull parameter owners with a common floating dtype.
/// @throws std::invalid_argument For empty/null/aliased/nonfloating/mismatched stores.
inline void Stores(std::initializer_list<ParameterStore*> stores) {
  if (stores.size() == 0)
    throw std::invalid_argument("Progressive stores cannot be empty");
  std::vector<ParameterStore*> seen;
  DType                        dtype{};
  for (auto* p : stores) {
    if (!p || (p->dtype() != DType::Float32 && p->dtype() != DType::Float64))
      throw std::invalid_argument("Progressive stores must be floating and nonnull");
    if (seen.empty())
      dtype = p->dtype();
    if (p->dtype() != dtype || std::find(seen.begin(), seen.end(), p) != seen.end())
      throw std::invalid_argument("Progressive stores must be distinct and dtype matched");
    seen.push_back(p);
  }
}
/// @brief Require one finite live-or-constant fade coefficient in the matching dtype.
/// @param alpha Float32/64 CPU scalar or[1] coefficient in[0,1]; both source forms are accepted.
/// @param dtype Required computation dtype.
/// @throws std::invalid_argument For wrong dtype/rank, nonfinite value or invalid range.
/// @throws std::logic_error For non-CPU storage.
inline void Alpha(const Tensor& alpha, DType dtype) {
  gan_detail::Finite(alpha);
  if ((alpha.dim() != 0 && alpha.shape() != Shape({1})) || alpha.dtype() != dtype ||
      alpha.item() < 0 || alpha.item() > 1)
    throw std::invalid_argument(
        "Progressive alpha must be a matching scalar or[1] coefficient in[0,1]");
}
/// @brief Require finite signed grayscale8x8 NHWC rows and representable intermediate products.
/// @param image Finite matching floating CPU[N,8,8,1], N>0, strides allowed.
/// @param dtype Required computation dtype.
/// @throws std::invalid_argument For incompatible geometry/dtype/values or overflowing N*256.
/// @throws std::logic_error For non-CPU storage.
inline void Image(const Tensor& image, DType dtype) {
  gan_detail::Finite(image);
  if (image.dim() != 4 || image.shape() != Shape({image.size(0), 8, 8, 1}) ||
      image.dtype() != dtype || image.size(0) > std::numeric_limits<std::int64_t>::max() / 256)
    throw std::invalid_argument("Progressive image needs positive matching8x8 NHWC rows");
}
/// @brief Apply a biased normal(.02) SAME stride-one convolution with source names/order.
/// @param p Floating parameter owner.
/// @param name Fixed nonempty architecture prefix.
/// @param x Validated matching NHWC activation.
/// @param out Required positive output channel count,1 or4.
/// @param kernel Required1x1 or3x3 square extent.
/// @return Owning live biased convolution, without activation or normalization.
/// @throws std::invalid_argument For incompatible stored leaves or primitive errors.
inline Tensor Conv(
    ParameterStore&    p,
    const std::string& name,
    const Tensor&      x,
    std::int64_t       out,
    std::int64_t       kernel) {
  p.get(name + "/W", {kernel, kernel, x.shape().back(), out}, init::normal(.02));
  return conv2d(p, name, x, out, kernel);
}
/// @brief Apply the fixed four-channel3x3 normal convolution and LeakyReLU(.2), optionally
/// pixel-normalized.
/// @param p Floating architecture-group owner.
/// @param name Source conv/conv1/conv2 prefix.
/// @param x Matching four-channel NHWC input.
/// @param normalize Apply channel RMS division after the activation for G only.
/// @return Owning live four-channel features, without any state/optimizer write.
/// @throws std::invalid_argument For incompatible leaves or nonfinite RMS computation.
inline Tensor Feature(ParameterStore& p, const std::string& name, const Tensor& x, bool normalize) {
  auto h = leaky_relu(Conv(p, name, x, 4, 3), .2);
  return normalize ? pixel_rms_normalize(h) : h;
}
} // namespace progressive_detail

/// @brief Return the source's active optimizer-group indices for one progressive stage and network.
///
/// stage0(stable4) selects G{0,1}; stage1(fade8) G{0,1,2,3}; stage2(stable8)
/// G{0,2,3}. D uses those indices plus4. Activity is a stage decision, not a test
/// of alpha: old RGB groups remain active at the final alpha1 fade batch and
/// their Adam moments may still move weights with a zero current gradient.
/// Inactive groups receive no optimizer step/counter advance. Returns an owning
/// ordered vector without reading stores, fitting or mutating anything; O(1).
/// @param stage Stage integer0,1 or2, rejecting all other values.
/// @param generator Select G indices when true(default), D indices when false.
/// @return Owning group-index vector in source application order.
/// @throws std::invalid_argument For a stage outside0..2.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const auto active=dm::nn::models::ProgressiveActiveGroups(1,false);
/// // active is{4,5,6,7}, including old D RGB at the last fade batch.
/// @endcode
/// @par References
/// - Karras et al. (2017/2018), https://arxiv.org/abs/1710.10196 . This repository's
///   three-stage/eight-optimizer schedule is a small teaching variant.
/// @see ProgressiveGan(), ProgressiveGanGenerate(), ProgressiveGanDiscriminate()
inline std::vector<int> ProgressiveActiveGroups(int stage, bool generator = true) {
  if (stage < 0 || stage > 2)
    throw std::invalid_argument("Progressive stage must be0,1 or2");
  auto groups = stage == 0   ? std::vector<int>{0, 1}
                : stage == 1 ? std::vector<int>{0, 1, 2, 3}
                             : std::vector<int>{0, 2, 3};
  if (!generator)
    for (auto& k : groups)
      k += 4;
  return groups;
}

/// @brief Own old4/new8 sigmoid generator images and their live post-sigmoid fade.
///
/// All three fields retain storage/applicable input,alpha and parameter graphs
/// after parent/store collection. Both paths always evaluate, including endpoints;
/// no stage selection, optimizer or frozen-parameter policy is retained.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore a(6401),b(6407),c(6413),d(6421);
/// auto z=n::ones({2,16});auto
/// out=n::models::ProgressiveGanGenerate(a,b,c,d,z,n::scalar_like(.5,z));
/// // old4 is[2,4,4,1]; new8/image are[2,8,8,1].
/// @endcode
/// @par References
/// - Karras et al. (2017/2018), https://arxiv.org/abs/1710.10196 .
/// @see ProgressiveGanGenerate(), ProgressiveGanOutput
struct ProgressiveGanGenerated {
  Tensor old4;  ///< Owning live old RGB sigmoid image[N,4,4,1] at current shared-core weights.
  Tensor new8;  ///< Owning live new RGB sigmoid image[N,8,8,1].
  Tensor image; ///< Owning alpha*new8+(1-alpha)*nearest_up(old4)[N,8,8,1].
};

/// @brief Generate genuine old/new RGB images with shared pixel-normalized features and a scalar
/// fade.
///
/// Four distinct matching Float32/64 CPU stores; finite noise[N,16], N>0 and
/// representable N*256; matching scalar or[1] alpha in[0,1], optionally differentiable.
/// Core stem16->64 tanh reshapes4x4x4, channel RMS division, normal3x3 conv/leaky
/// .2/RMS. Old1x1 sigmoid RGB maps4->1. New block nearest-upsamples shared features
/// then applies two normal3x3 conv/leaky/RMS maps; new1x1 sigmoid RGB maps4->1.
/// All convs are biased normal(.02), stem uses native dense Glorot/zeros, names
/// and creation order preserve the source. Group counts1236,5,296,5 (12 leaves).
/// RMS epsilon=double(1e-8f), explicit division/sqrt, no learned normalization.
///
/// Always evaluates both branches; fades sigmoid pixels, not logits. alpha0 is
/// exactly nearest-up old4; alpha1 exactly new8. First/higher noise,alpha and
/// parameter paths remain live. No fitting, active-group decision, optimizer,
/// clipping or existing-leaf mutation. Inputs unchanged, owning outputs. Validate
/// input/store geometry before leaves; incompatible stored leaves/nonfinite math
/// may fail later after earlier initialization. O(N) work/storage plus validation;
/// synchronize shared stores. Frozen old-RGB parameters do not freeze the shared core.
/// @param g_core Group0 shared stem/conv owner, source seed6401.
/// @param g_old_rgb Distinct group1 old RGB owner, source seed6407.
/// @param g_new_block Distinct group2 new two-convolution owner, source seed6413.
/// @param g_new_rgb Distinct group3 new RGB owner, source seed6421.
/// @param noise Finite signed matching CPU[N,16], positive N.
/// @param alpha Finite matching CPU scalar or[1] coefficient in[0,1], with optional live fade path.
/// @return Owning ProgressiveGanGenerated with both paths and the blended8x8 image.
/// @throws std::invalid_argument For invalid inputs/stores/products/leaves or nonfinite
/// computation.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore a(6401),b(6407),c(6413),d(6421);auto z=n::ones({2,16});
/// auto out=n::models::ProgressiveGanGenerate(a,b,c,d,z,n::scalar_like(0,z));
/// // out.image equals nearest-up old4; new8 was still evaluated and initialized.
/// @endcode
/// @par References
/// - Karras,Aila,Laine and Lehtinen (2017/2018), https://arxiv.org/abs/1710.10196 .
///   This source omits equalized learning rate/minibatch std and uses a tiny8x8 model.
/// @see ProgressiveGanGenerated, ProgressiveGanDiscriminate(), pixel_rms_normalize()
inline ProgressiveGanGenerated ProgressiveGanGenerate(
    ParameterStore& g_core,
    ParameterStore& g_old_rgb,
    ParameterStore& g_new_block,
    ParameterStore& g_new_rgb,
    const Tensor&   noise,
    const Tensor&   alpha) {
  progressive_detail::Stores({&g_core, &g_old_rgb, &g_new_block, &g_new_rgb});
  gan_detail::Matrix(noise, 16, g_core.dtype());
  progressive_detail::Alpha(alpha, g_core.dtype());
  if (noise.size(0) > std::numeric_limits<std::int64_t>::max() / 256)
    throw std::invalid_argument("Progressive activation product overflow");
  auto h = pixel_rms_normalize(
      reshape(dense(g_core, "stem", noise, 64, Activation::Tanh), {-1, 4, 4, 4}));
  h              = progressive_detail::Feature(g_core, "conv", h, true);
  auto old       = sigmoid(progressive_detail::Conv(g_old_rgb, "rgb", h, 1, 1));
  auto high      = progressive_detail::Feature(g_new_block, "conv1", upsample_nearest(h, 2), true);
  high           = progressive_detail::Feature(g_new_block, "conv2", high, true);
  auto generated = sigmoid(progressive_detail::Conv(g_new_rgb, "rgb", high, 1, 1));
  auto image     = alpha * generated + (1. - alpha) * upsample_nearest(old, 2);
  gan_detail::Finite(old);
  gan_detail::Finite(generated);
  gan_detail::Finite(image);
  return {old, generated, image};
}

/// @brief Own both discriminator feature paths, their pre-core fade, shared features and raw score.
///
/// old_path/new_path/mixed/base are[N,4,4,4]; score[N,1]. All fields own storage
/// and applicable first/higher image,alpha and parameter graphs after parent/store
/// collection. No stage, optimizer, fitting or update callback is retained.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore a(6427),b(6449),c(6451),d(6469);auto x=n::ones({2,8,8,1});
/// auto out=n::models::ProgressiveGanDiscriminate(a,b,c,d,x,n::scalar_like(.5,x));
/// // mixed is a4x4 feature blend, before the shared core/score head.
/// @endcode
/// @par References
/// - Karras et al. (2017/2018), https://arxiv.org/abs/1710.10196 .
/// @see ProgressiveGanDiscriminate(), ProgressiveGanOutput
struct ProgressiveGanDiscriminated {
  Tensor old_path; ///< Owning leaky RGB features from avg-pooled input[N,4,4,4].
  Tensor new_path; ///< Owning avg-pooled new RGB/two-convolution features[N,4,4,4].
  Tensor mixed;    ///< Owning alpha*new_path+(1-alpha)*old_path[N,4,4,4].
  Tensor base;     ///< Owning shared core leaky convolution of mixed features[N,4,4,4].
  Tensor score;    ///< Owning live raw dense scores[N,1], without sigmoid.
};

/// @brief Discriminate8x8 images through faded old/new feature paths before a shared raw-score
/// core.
///
/// Four distinct matching Float32/64 CPU stores, finite signed image[N,8,8,1],
/// N>0, strides and representable N*256; matching scalar or[1] alpha in[0,1]. Old path
/// avg-pools input to4x4, applies1x1 RGB1->4/leaky.2. New path applies1x1 RGB1->4/
/// leaky, two normal3x3 conv/leaky maps at8x8, then avg-pools. Fades these4x4
/// feature maps before shared3x3 conv/leaky and flatten64->1 biased dense score.
/// All convs biased normal(.02); score dense Glorot/zeros; source names/order.
/// Group counts213,8,296,8 (12 leaves). No pixel normalization in D.
///
/// Both branches always evaluate, including endpoints. First/higher image,alpha
/// and parameter paths live. No G call, sigmoid probability, stage activity,
/// optimizer, fit or existing-leaf mutation. Inputs unchanged; output fields own
/// lifetime. Checks precede leaves; existing-leaf/nonfinite math errors may occur
/// later after initialization. O(N) work/storage; synchronize shared stores.
/// @param d_core Group4 shared conv/score owner, source seed6427.
/// @param d_old_rgb Distinct group5 old RGB owner, source seed6449.
/// @param d_new_block Distinct group6 new two-convolution owner, source seed6451.
/// @param d_new_rgb Distinct group7 new RGB owner, source seed6469.
/// @param image Finite signed matching CPU[N,8,8,1], positive N.
/// @param alpha Finite matching CPU scalar or[1] coefficient in[0,1], optionally live.
/// @return Owning five-field ProgressiveGanDiscriminated with both feature paths and raw score.
/// @throws std::invalid_argument For invalid inputs/stores/products/leaves or nonfinite
/// computation.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore a(6427),b(6449),c(6451),d(6469);auto x=n::ones({2,8,8,1});
/// auto out=n::models::ProgressiveGanDiscriminate(a,b,c,d,x,n::scalar_like(1,x));
/// // out.mixed equals out.new_path; the old path remains an initialized endpoint.
/// @endcode
/// @par References
/// - Karras et al. (2017/2018), https://arxiv.org/abs/1710.10196 .
/// @see ProgressiveGanDiscriminated, ProgressiveGanGenerate(), ProgressiveGan()
inline ProgressiveGanDiscriminated ProgressiveGanDiscriminate(
    ParameterStore& d_core,
    ParameterStore& d_old_rgb,
    ParameterStore& d_new_block,
    ParameterStore& d_new_rgb,
    const Tensor&   image,
    const Tensor&   alpha) {
  progressive_detail::Stores({&d_core, &d_old_rgb, &d_new_block, &d_new_rgb});
  progressive_detail::Image(image, d_core.dtype());
  progressive_detail::Alpha(alpha, d_core.dtype());
  auto old =
      leaky_relu(progressive_detail::Conv(d_old_rgb, "rgb", avg_pool2d(image, 2, 2), 4, 1), .2);
  auto high  = leaky_relu(progressive_detail::Conv(d_new_rgb, "rgb", image, 4, 1), .2);
  high       = progressive_detail::Feature(d_new_block, "conv1", high, false);
  high       = progressive_detail::Feature(d_new_block, "conv2", high, false);
  auto down  = avg_pool2d(high, 2, 2);
  auto mixed = alpha * down + (1. - alpha) * old;
  auto base  = progressive_detail::Feature(d_core, "conv", mixed, false);
  auto score = dense(d_core, "score", reshape(base, {-1, 64}), 1);
  for (const auto& x : {old, down, mixed, base, score})
    gan_detail::Finite(x);
  return {old, down, mixed, base, score};
}

/// @brief Own seven progressive image/score/loss fields without stage or optimizer policy.
///
/// fake/new8[N,8,8,1],old4[N,4,4,1],real_score/fake_score[N,1] and scalar d_loss/
/// g_loss retain storage/applicable graphs after parent/store collection. All
/// eight groups always evaluate, independently of which optimizers the caller
/// activates. No counter, optimizer, fitting callback or freeze policy is retained.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore a(6401),b(6407),c(6413),d(6421),e(6427),f(6449),g(6451),h(6469);
/// auto z=n::ones({2,16}),x=n::ones({2,8,8,1});
/// auto out=n::models::ProgressiveGan(a,b,c,d,e,f,g,h,x,z,n::scalar_like(.5,z));
/// // d_loss has factor.5; the source g_loss has NO half factor.
/// @endcode
/// @par References
/// - Karras et al. (2017/2018), https://arxiv.org/abs/1710.10196 .
/// @see ProgressiveGan(), ProgressiveGanGenerated, ProgressiveGanDiscriminated
struct ProgressiveGanOutput {
  Tensor fake;       ///< Owning live faded image[N,8,8,1].
  Tensor old4;       ///< Owning live old sigmoid endpoint[N,4,4,1] at current shared-core weights.
  Tensor new8;       ///< Owning live new sigmoid endpoint[N,8,8,1].
  Tensor real_score; ///< Owning live real raw scores[N,1].
  Tensor fake_score; ///< Owning live generated raw scores[N,1].
  Tensor d_loss;     ///< Owning .5*(mean(square(real-1))+mean(square(fake))) scalar.
  Tensor g_loss; ///< Owning mean(square(fake-1)) scalar, without the standard LSGAN half factor.
};

/// @brief Evaluate the eight-store progressive GAN with both actual fades and source-scaled
/// least-squares losses.
///
/// Eight distinct matching Float32/64 CPU stores, aligned finite image[N,8,8,1]/
/// noise[N,16], positive N with representable intermediates, and matching scalar or[1]
/// alpha in[0,1]. Uses ProgressiveGanGenerate()/Discriminate() without pruning
/// either path at endpoints. D=.5*(mean((real_score-1)^2)+mean(fake_score^2));
/// G=mean((fake_score-1)^2), deliberately NO half factor, preserving this source's
/// update scale rather than LeastSquaresGanObjectives()'s G convention.
///
/// All applicable first/higher continuous paths live, including alpha. No hidden
/// gradient capture, freeze, optimizer, stage selection, fitting or checkpoint.
/// Use ProgressiveActiveGroups() to choose groups; capture ALL active D gradients
/// from ONE forward before any D write, then recompute a fresh G forward through
/// updated D and capture ALL active G gradients before G writes. Retain the graph
/// across per-group gradient calls; convert unused partials to explicit zeros.
/// Inactive groups receive no Adam step/counter advance. Old RGB groups remain
/// active at the final alpha1 fade batch; current zero gradients can still yield
/// moment-driven writes. Freezing old RGB parameters does not freeze shared cores.
///
/// Full counts1236,5,296,5,213,8,296,8 total2067 values/24 leaves. Stage-selected
/// optimized counts1462(stable4),2067(fade),2054(stable8). Input/store checks
/// precede leaves; stored-leaf/nonfinite math errors may follow initialization.
/// Owning output fields; inputs unchanged, O(N) work/storage; synchronize stores.
/// @param g_core Group0 G stem/shared conv owner.
/// @param g_old_rgb Distinct group1 G old RGB owner.
/// @param g_new_block Distinct group2 G new block owner.
/// @param g_new_rgb Distinct group3 G new RGB owner.
/// @param d_core Distinct group4 D shared conv/score owner.
/// @param d_old_rgb Distinct group5 D old RGB owner.
/// @param d_new_block Distinct group6 D new block owner.
/// @param d_new_rgb Distinct group7 D new RGB owner.
/// @param image Finite signed matching real CPU[N,8,8,1], N>0.
/// @param noise Finite signed matching CPU[N,16], same positive N.
/// @param alpha Finite matching CPU scalar or[1] coefficient in[0,1], optionally live.
/// @return Owning seven-field ProgressiveGanOutput with both endpoint images and raw least-squares
/// objectives.
/// @throws std::invalid_argument For invalid geometry/stores/products/leaves or nonfinite
/// computation.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore a(6401),b(6407),c(6413),d(6421),e(6427),f(6449),g(6451),h(6469);
/// auto z=n::ones({2,16}),x=n::ones({2,8,8,1});
/// auto out=n::models::ProgressiveGan(a,b,c,d,e,f,g,h,x,z,n::scalar_like(.5,z));
/// n::GradOptions options;options.retain_graph=true;options.allow_unused=true;
/// auto first_D=n::grad(out.d_loss,e.parameters(),options);
/// // Capture other active D groups BEFORE any optimizer writes, then a fresh G forward.
/// @endcode
/// @par References
/// - Karras et al. (2017/2018), https://arxiv.org/abs/1710.10196 .
/// - Mao et al. (2016/2017), Least Squares Generative Adversarial Networks,
///   https://arxiv.org/abs/1611.04076 . This source's G scale is1, not.5.
/// @see ProgressiveGanOutput, ProgressiveActiveGroups(), ProgressiveGanGenerate(),
/// ProgressiveGanDiscriminate()
inline ProgressiveGanOutput ProgressiveGan(
    ParameterStore& g_core,
    ParameterStore& g_old_rgb,
    ParameterStore& g_new_block,
    ParameterStore& g_new_rgb,
    ParameterStore& d_core,
    ParameterStore& d_old_rgb,
    ParameterStore& d_new_block,
    ParameterStore& d_new_rgb,
    const Tensor&   image,
    const Tensor&   noise,
    const Tensor&   alpha) {
  progressive_detail::Stores(
      {&g_core,
       &g_old_rgb,
       &g_new_block,
       &g_new_rgb,
       &d_core,
       &d_old_rgb,
       &d_new_block,
       &d_new_rgb});
  progressive_detail::Image(image, g_core.dtype());
  gan_detail::Matrix(noise, 16, g_core.dtype());
  progressive_detail::Alpha(alpha, g_core.dtype());
  if (image.size(0) != noise.size(0))
    throw std::invalid_argument("Progressive real/noise rows must align");
  auto generated = ProgressiveGanGenerate(g_core, g_old_rgb, g_new_block, g_new_rgb, noise, alpha);
  auto real =
      ProgressiveGanDiscriminate(d_core, d_old_rgb, d_new_block, d_new_rgb, image, alpha).score;
  auto fake =
      ProgressiveGanDiscriminate(d_core, d_old_rgb, d_new_block, d_new_rgb, generated.image, alpha)
          .score;
  auto dl = .5 * (mean(square(real - 1.)) + mean(square(fake)));
  auto gl = mean(square(fake - 1.));
  gan_detail::Finite(dl);
  gan_detail::Finite(gl);
  return {generated.image, generated.old4, generated.new8, real, fake, dl, gl};
}
} // namespace datamunge::nn::models
