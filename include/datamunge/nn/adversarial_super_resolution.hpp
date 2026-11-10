#pragma once
/// @file adversarial_super_resolution.hpp
/// @brief Explicit compact SRGAN/ESRGAN paths, frozen VGG features and captured BN moments.
#include "datamunge/nn/captured_batch_norm.hpp"
#include "datamunge/nn/image_ops.hpp"
#include "datamunge/nn/perceptual_models.hpp"
#include "datamunge/nn/relativistic_gan.hpp"
#include "datamunge/nn/super_resolution_models.hpp"
namespace datamunge::nn::models {
/// @brief Internal fixed RGB contracts and source-compatible compact SR primitives.
namespace adversarial_sr_detail {
/// @brief Require matching finite positive RGB rows at a fixed source side.
/// @param image Floating CPU[N,side,side,3], positive N.
/// @param side Required16 or32 spatial extent.
/// @param dtype Required Float32/64 computation dtype.
/// @throws std::invalid_argument For invalid shape/dtype/values/products.
/// @throws std::logic_error For non-CPU storage.
inline void Image(const Tensor& image, std::int64_t side, DType dtype) {
  gan_detail::Finite(image);
  if (image.dim() != 4 || image.shape() != Shape({image.size(0), side, side, 3}) ||
      image.dtype() != dtype || image.size(0) > std::numeric_limits<std::int64_t>::max() / 4096)
    throw std::invalid_argument("Adversarial SR needs matching positive fixed RGB geometry");
}
/// @brief Require independent matching floating G and saved-statistic owners.
/// @param g Learned generator owner.
/// @param s Saved BN owner, distinct even on the empty enhanced path.
/// @throws std::invalid_argument For aliasing or dtype mismatch.
inline void Stores(ParameterStore& g, ParameterStore& s) {
  if (&g == &s || g.dtype() != s.dtype())
    throw std::invalid_argument("Adversarial SR needs distinct matching G/BN stores");
}
/// @brief Check frozen first-convolution weights and bias against the input dtype.
/// @param w Finite CPU[3,3,3,64] frozen kernel.
/// @param b Finite CPU[64] frozen bias.
/// @param dtype Required input Float32/64 dtype.
/// @throws std::invalid_argument For incompatible shapes/dtypes/values.
/// @throws std::logic_error For non-CPU storage.
inline void Features(const Tensor& w, const Tensor& b, DType dtype) {
  gan_detail::Finite(w);
  gan_detail::Finite(b);
  if (w.shape() != Shape({3, 3, 3, 64}) || b.shape() != Shape({64}) || w.dtype() != dtype ||
      b.dtype() != dtype)
    throw std::invalid_argument("Frozen SR features need matching1792 first-convolution values");
}
/// @brief Apply source fan-in uniform SAME convolution with explicit stride and bias.
/// @param p Learned owner.
/// @param name Fixed source layer prefix.
/// @param x Validated matching NHWC feature map.
/// @param out Fixed positive output channels.
/// @param kernel Fixed3 or4 square extent.
/// @param stride Fixed1(default) or2.
/// @param bias Create/use zero bias when true(default), otherwise no bias leaf.
/// @return Owning live biased/unbiased feature map; source float fan-in guard preserved.
/// @throws std::invalid_argument For incompatible leaves or primitive errors.
inline Tensor Conv(
    ParameterStore&    p,
    const std::string& name,
    const Tensor&      x,
    std::int64_t       out,
    std::int64_t       kernel,
    std::int64_t       stride = 1,
    bool               bias   = true) {
  auto in = x.shape().back();
  p.get(
      name + "/W",
      {kernel, kernel, in, out},
      init::uniform(1 / std::sqrt(float(kernel * kernel * in))));
  ConvLayerOptions o;
  o.stride = stride;
  o.bias   = bias;
  return conv2d(p, name, x, out, kernel, o);
}
/// @brief Apply four learned channel PReLU slopes with nonnegative identity branch.
/// @param p Learned G owner.
/// @param name Fixed source prefix.
/// @param x Four-channel activation.
/// @return Owning live where(x>=0,x,x*alpha), alpha starts.25.
/// @throws std::invalid_argument For incompatible leaves or primitive errors.
inline Tensor Prelu(ParameterStore& p, const std::string& name, const Tensor& x) {
  auto a = p.get(name + "/alpha", {4}, init::constant(.25));
  return where(greater_equal(x, 0.), x, x * a);
}
/// @brief Normalize with live source batch moments or stopped saved statistics, then affine.
/// @param p Learned gamma/beta owner, source initialization before BN state.
/// @param stats Saved moving-mean/variance owner, no writes beyond missing leaves.
/// @param means Output metadata list of detached per-layer batch means.
/// @param variances Output metadata list of detached unbiased variances.
/// @param name Fixed source BN prefix.
/// @param x Four-channel activation.
/// @param training Select batch captures when true, saved statistics otherwise.
/// @return Owning live normalized*gamma+beta with explicit source division/sqrt.
/// @throws std::invalid_argument For invalid saved moments/leaves or primitive errors.
inline Tensor BatchNorm(
    ParameterStore&      p,
    ParameterStore&      stats,
    std::vector<Tensor>& means,
    std::vector<Tensor>& variances,
    const std::string&   name,
    const Tensor&        x,
    bool                 training) {
  auto gamma   = p.get(name + "/gamma", {4}, init::ones()),
       beta    = p.get(name + "/beta", {4}, init::zeros());
  auto capture = capture_batch_norm(stats, name, x, training);
  if (training) {
    means.push_back(capture.mean.detach());
    variances.push_back(capture.unbiased_variance.detach());
  }
  return capture.value * gamma + beta;
}
/// @brief Apply source two-convolution BN/PReLU residual map without convolution biases.
/// @param p Learned owner.
/// @param stats Saved-statistic owner.
/// @param means Detached capture list appended in source forward order.
/// @param variances Detached unbiased variance list, same order.
/// @param name Fixed residual0 or residual1 prefix.
/// @param x Four-channel LR feature map.
/// @param training Use live batch moments when true, stopped saved moments otherwise.
/// @return Owning live input+second normalized convolution.
/// @throws std::invalid_argument For incompatible leaves/state or primitive errors.
inline Tensor Residual(
    ParameterStore&      p,
    ParameterStore&      stats,
    std::vector<Tensor>& means,
    std::vector<Tensor>& variances,
    const std::string&   name,
    const Tensor&        x,
    bool                 training) {
  auto h = Conv(p, name + "/conv1", x, 4, 3, 1, false);
  h      = BatchNorm(p, stats, means, variances, name + "/bn1", h, training);
  h      = Prelu(p, name + "/prelu", h);
  h      = Conv(p, name + "/conv2", h, 4, 3, 1, false);
  return x + BatchNorm(p, stats, means, variances, name + "/bn2", h, training);
}
/// @brief Apply four growth-two leaky dense convolutions and one scaled four-channel tail.
/// @param p Learned owner.
/// @param name Fixed RRDB dense-block prefix.
/// @param x Four-channel LR feature map.
/// @return Owning live input+.2*conv5(concat(input,all four growth maps)).
/// @throws std::invalid_argument For incompatible leaves or primitive errors.
inline Tensor DenseBlock(ParameterStore& p, const std::string& name, const Tensor& x) {
  std::vector<Tensor> features{x};
  for (int i = 0; i < 4; ++i) {
    auto joined = features.size() == 1 ? x : concat(features, 3);
    features.push_back(
        leaky_relu(Conv(p, name + "/conv" + std::to_string(i + 1), joined, 2, 3), .2));
  }
  return x + .2 * Conv(p, name + "/conv5", concat(features, 3), 4, 3);
}
} // namespace adversarial_sr_detail
/// @brief Own compact SR pixels, architecture boundaries and detached optional five-layer BN
/// captures.
///
/// Nine flat Tensor fields retain storage/applicable graphs after parent/store
/// collection. image=bicubic+residual is unbounded. SRGAN packed[N,16,16,16]
/// and shuffled[N,32,32,4] expose CRD order. ESRGAN packed/shuffled both alias the
/// post-HR-conv leaky[N,32,32,4] boundary. Batch SRGAN captures means/variances[5,4]
/// in order residual0/bn1,bn2,residual1/bn1,bn2,body_bn; detached and undefined in
/// inference or enhanced mode. No EMA write, optimizer, fitting or RNG after init.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore g(7001),stats(0);
/// auto
/// o=n::models::AdversarialSRGenerate(g,stats,n::zeros({2,16,16,3}),n::zeros({2,32,32,3}),false,true);
/// // image[2,32,32,3],moments_mean/moments_unbiased_variance[5,4]; state unchanged.
/// @endcode
/// @par References
/// - Ledig et al.(2017), Photo-Realistic Single Image Super-Resolution Using a Generative
/// Adversarial Network, https://arxiv.org/abs/1609.04802 ; Wang et al.(2018), ESRGAN: Enhanced
/// Super-Resolution Generative Adversarial Networks, https://arxiv.org/abs/1809.00219 . Fixed-width
/// x2 teaching adaptations with a bicubic image shortcut, not pretrained benchmark models.
/// @see AdversarialSRGenerate(), AdversarialSR()
struct AdversarialSRGenerated {
  Tensor image;    ///< Owning live bicubic+residual[N,32,32,3], no clipping.
  Tensor residual; ///< Owning live learned image residual[N,32,32,3].
  Tensor packed;   ///< Owning live packed CRD feature[N,16,16,16] for SRGAN, HR feature for ESRGAN.
  Tensor shuffled; ///< Owning live CRD shuffled[N,32,32,4] for SRGAN, HR feature for ESRGAN.
  Tensor moments_mean; ///< Owning detached training SRGAN means[5,4], otherwise undefined.
  Tensor moments_unbiased_variance; ///< Owning detached training SRGAN unbiased variances[5,4],
                                    ///< otherwise undefined.
  Tensor head;      ///< Owning live LR four-channel head after SRGAN PReLU, raw conv for ESRGAN.
  Tensor body;      ///< Owning live global-skip LR four-channel body boundary.
  Tensor upsampled; ///< Owning live HR four-channel upsampled activation, before ESRGAN HR conv.
};
/// @brief Generate compact x2 SRGAN/ESRGAN residual pixels using explicit bicubic input and BN
/// mode.
///
/// Finite signed matching Float32/64 CPU lr[N,16,16,3],bicubic[N,32,32,3],
/// N>0 aligned, strides allowed; independent matching G/stats stores. Source seeds
/// 7001/0. G head3->4 fan-in-uniform conv. SRGAN(default): head PReLU, two
/// bias-free conv/BN/PReLU residual blocks, global body BN skip, up conv4->16,
/// CRD pixel shuffle2, PReLU4;1591 values/25 leaves and40 saved BN values/10 leaves.
/// ESRGAN: three sequential growth-two five-conv dense blocks with .2 internal
/// residuals, head+.2*last_block before body-tail conv/global head skip, nearest2,
/// up/HR leaky.2 convs;3511 values/40 leaves, no stats leaves. Both output kernels
/// normal(.001), output bias zero, add residual to supplied bicubic image without
/// clipping. Other kernels use source float fan-in uniform; names/order retained.
/// SRGAN batch_statistics=true uses live population BN moments, source-rounded
/// unbiased correction and detached captures in the five documented rows. False
/// uses stopped saved moments. Enhanced paths do not use BN; flag has no effect.
/// No implicit no-grad, EMA update, optimizer, fitting or later RNG. Applicable
/// first/higher LR,bicubic/parameter paths live, saved-state/capture paths stop.
/// Only missing leaves initialize; existing inputs/leaves unchanged. Validate inputs/
/// aliases before leaves; stored-state/math failures may follow initialization.
/// Owning fields retain graph/storage. O(N) fixed-map work/storage; synchronize stores.
/// @param generator Caller-owned floating G store, source seed7001.
/// @param stats Distinct same-dtype BN store, source seed0; exclude from optimizers.
/// @param lr Finite matching CPU[N,16,16,3] low-resolution images.
/// @param bicubic Finite aligned matching CPU[N,32,32,3] external image shortcut.
/// @param enhanced Select ESRGAN(true) instead of SRGAN(false default).
/// @param batch_statistics Select live batch moments/capture for SRGAN when true, saved statistics
/// when false(default).
/// @return Owning AdversarialSRGenerated with unbounded residual pixels and explicit captures.
/// @throws std::invalid_argument For invalid geometry, dtype, values, alignment, products, aliased
/// stores or incompatible leaves.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore g(7001),s(0);
/// auto lr=n::ones({1,16,16,3});auto bic=n::ones({1,32,32,3});
/// auto out=n::models::AdversarialSRGenerate(g,s,lr,bic,true);
/// // ESRGAN has3511 learned values and no BN state; no image clipping or fitting.
/// @endcode
/// @par References
/// - Ledig et al.(2017), Photo-Realistic Single Image Super-Resolution Using a Generative
/// Adversarial Network, https://arxiv.org/abs/1609.04802 ; Wang et al.(2018), ESRGAN: Enhanced
/// Super-Resolution Generative Adversarial Networks, https://arxiv.org/abs/1809.00219 . Fixed-width
/// x2 teaching adaptations with a bicubic image shortcut, not pretrained benchmark models.
/// @see AdversarialSRGenerated, SRPixelShuffle(), capture_batch_norm(), batch_norm_moving_values()
inline AdversarialSRGenerated AdversarialSRGenerate(
    ParameterStore& generator,
    ParameterStore& stats,
    const Tensor&   lr,
    const Tensor&   bicubic,
    bool            enhanced         = false,
    bool            batch_statistics = false) {
  using namespace adversarial_sr_detail;
  Stores(generator, stats);
  Image(lr, 16, generator.dtype());
  Image(bicubic, 32, generator.dtype());
  if (lr.size(0) != bicubic.size(0))
    throw std::invalid_argument("SR shortcut rows must match LR");
  AdversarialSRGenerated o;
  std::vector<Tensor>    means, variances;
  auto                   head = Conv(generator, "head", lr, 4, 3);
  Tensor                 h;
  if (enhanced) {
    h = head;
    for (int i = 0; i < 3; ++i)
      h = DenseBlock(generator, "rrdb/dense" + std::to_string(i), h);
    o.head      = head;
    h           = head + Conv(generator, "body_tail", head + .2 * h, 4, 3);
    o.body      = h;
    h           = resize_nearest(h, {32, 32});
    h           = leaky_relu(Conv(generator, "up", h, 4, 3), .2);
    o.upsampled = h;
    h           = leaky_relu(Conv(generator, "hr", h, 4, 3), .2);
    o.packed = o.shuffled = h;
  } else {
    head   = Prelu(generator, "head_prelu", head);
    o.head = head;
    h      = head;
    for (int i = 0; i < 2; ++i)
      h = Residual(
          generator, stats, means, variances, "residual" + std::to_string(i), h, batch_statistics);
    h = head + BatchNorm(
                   generator,
                   stats,
                   means,
                   variances,
                   "body_bn",
                   Conv(generator, "body_tail", h, 4, 3, 1, false),
                   batch_statistics);
    o.body      = h;
    o.packed    = Conv(generator, "up", h, 16, 3);
    o.shuffled  = SRPixelShuffle(o.packed, 4, 2);
    h           = Prelu(generator, "up_prelu", o.shuffled);
    o.upsampled = h;
    if (batch_statistics) {
      o.moments_mean              = stack(means, 0);
      o.moments_unbiased_variance = stack(variances, 0);
    }
  }
  generator.get("output/W", {3, 3, 4, 3}, init::normal(.001));
  o.residual = Conv(generator, "output", h, 3, 3);
  o.image    = bicubic + o.residual;
  gan_detail::Finite(o.image);
  return o;
}
/// @brief Score RGB HR images with the compact shared SR critic and explicit relative-head variant.
///
/// Finite signed matching Float32/64 CPU image[N,32,32,3], N>0, strides allowed.
/// Fan-in-uniform SAME head3->4/leaky.2,4x4/stride2 down4->8/leaky.2,down8->8/
/// leaky.2, spatial mean8, raw score1. Source seed7013. Default ordinary head is
/// native Glorot dense plus bias,1673 values/8 leaves. relative=true uses an
/// explicit bias-free8x1 kernel with source bound sqrt(6.f/9),1672/7 leaves.
/// No sigmoid, batch statistics, fitting, writes or hidden no-grad; only missing
/// leaves initialize, no later RNG. Applicable first/higher input/weight paths
/// live, owning scores retain graph/storage. Validate image before leaves;
/// stored-leaf/math errors may follow initialization. O(N) work/storage; synchronize stores.
/// @param critic Caller-owned floating D store, source seed7013.
/// @param image Finite matching CPU[N,32,32,3] raw candidate images.
/// @param relative Select source bias-free ESRGAN score head when true, ordinary dense
/// otherwise(default false).
/// @return Owning live raw scores[N,1].
/// @throws std::invalid_argument For invalid geometry, dtype, values, alignment, products, aliased
/// stores or incompatible leaves.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore d(7013);
/// auto score=n::models::AdversarialSRDiscriminate(d,n::zeros({2,32,32,3}),true);
/// // Raw[2,1],1672 values in7 leaves; no domain code or probability clipping.
/// @endcode
/// @par References
/// - Ledig et al.(2017), Photo-Realistic Single Image Super-Resolution Using a Generative
/// Adversarial Network, https://arxiv.org/abs/1609.04802 ; Wang et al.(2018), ESRGAN: Enhanced
/// Super-Resolution Generative Adversarial Networks, https://arxiv.org/abs/1809.00219 . Fixed-width
/// x2 teaching adaptations with a bicubic image shortcut, not pretrained benchmark models.
/// @see AdversarialSR(), RelativisticAverageGan()
inline Tensor
AdversarialSRDiscriminate(ParameterStore& critic, const Tensor& image, bool relative = false) {
  using namespace adversarial_sr_detail;
  Image(image, 32, critic.dtype());
  auto h      = leaky_relu(Conv(critic, "head", image, 4, 3), .2);
  h           = leaky_relu(Conv(critic, "down16", h, 8, 4, 2), .2);
  h           = leaky_relu(Conv(critic, "down8", h, 8, 4, 2), .2);
  auto pooled = mean(h, {1, 2});
  auto result =
      relative ? matmul(pooled, critic.get("score/W", {8, 1}, init::uniform(std::sqrt(6.f / 9))))
               : dense(critic, "score", pooled, 1);
  gan_detail::Finite(result);
  return result;
}
/// @brief Extract live unactivated frozen VGG19 first-convolution features with stopped
/// weights/bias.
///
/// Finite signed matching Float32/64 CPU rgb[N,H,W,3], positive axes, strides
/// allowed; weight[3,3,3,64],bias[64],1792 finite matching floating values. Uses
/// canonical VggImageNetInput: RGB*255 reversed to BGR minus source-Float32-rounded
/// ImageNet means, then SAME3x3 convolution plus bias, no ReLU or /255 here.
/// Both supplied weight/bias are detached for every derivative order; RGB paths
/// stay live. Caller supplies the actual pretrained values; geometry alone does
/// not certify provenance. No internal random feature substitution, storage mutation,
/// optimizer or fitting. Owning output retains graph/storage after inputs are
/// collected. O(N*H*W*3*64*9) work; products checked for int64 overflow. Synchronize
/// external writes; computed nonfinite output rejected.
/// @param rgb Finite signed floating CPU[N,H,W,3], all axes positive.
/// @param weight Finite matching frozen CPU[3,3,3,64] convolution values, detached on use.
/// @param bias Finite matching frozen CPU[64] bias values, detached on use.
/// @return Owning live unactivated feature tensor[N,H,W,64], frozen-weight paths stopped.
/// @throws std::invalid_argument For invalid geometry, dtype, values, alignment, products, aliased
/// stores or incompatible leaves.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// auto x=n::ones({1,2,2,3}).requires_grad_();
/// auto f=n::models::AdversarialSRFeatures(x,n::zeros({3,3,3,64}),n::ones({64}));
/// // Supplied zero-kernel fixture gives all1 features; real runs load official VGG values.
/// @endcode
/// @par References
/// - Simonyan and Zisserman(2015), Very Deep Convolutional Networks for Large-Scale Image
/// Recognition, https://arxiv.org/abs/1409.1556 ; Ledig et al.(2017),
/// https://arxiv.org/abs/1609.04802 ; Wang et al.(2018), https://arxiv.org/abs/1809.00219 . Source
/// first-layer content features are a compact teaching adaptation.
/// @see VggImageNetInput(), AdversarialSR()
inline Tensor AdversarialSRFeatures(const Tensor& rgb, const Tensor& weight, const Tensor& bias) {
  gan_detail::Finite(rgb);
  if (rgb.dim() != 4 || rgb.size(3) != 3 ||
      rgb.numel() / 3 > std::numeric_limits<std::int64_t>::max() / 64)
    throw std::invalid_argument("SR features need positive representable RGB geometry");
  adversarial_sr_detail::Features(weight, bias, rgb.dtype());
  auto value = conv2d(VggImageNetInput(rgb), weight.detach()) + bias.detach();
  gan_detail::Finite(value);
  return value;
}
/// @brief Own both SR generator paths, raw scores, feature/loss boundaries and detached BN
/// captures.
///
/// Twenty flat fields retain storage/applicable graphs beyond parent/store
/// collection. prediction/residual use saved-statistic inference; prediction_training
/// and all objectives use batch-statistic G. Target feature extraction stops before
/// ReLU/scaling. moments_mean/unbiased_variance[5,4] are detached on SRGAN path,
/// undefined for ESRGAN; row order matches AdversarialSRGenerated. No captured
/// partials, optimizer state, fitting, EMA write or mutable callback is retained.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore g(7001),d(7013),s(0);
/// auto
/// o=n::models::AdversarialSR(g,d,s,n::zeros({2,16,16,3}),n::zeros({2,32,32,3}),n::zeros({2,32,32,3}),n::zeros({3,3,3,64}),n::zeros({64}));
/// // Source-sized zero-feature fixture; actual report loads1792 official frozen values.
/// @endcode
/// @par References
/// - Ledig et al.(2017), Photo-Realistic Single Image Super-Resolution Using a Generative
/// Adversarial Network, https://arxiv.org/abs/1609.04802 ; Wang et al.(2018), ESRGAN: Enhanced
/// Super-Resolution Generative Adversarial Networks, https://arxiv.org/abs/1809.00219 . Fixed-width
/// x2 teaching adaptations with a bicubic image shortcut, not pretrained benchmark models.
/// @see AdversarialSR(), AdversarialSRGenerated
struct AdversarialSROutput {
  Tensor prediction; ///< Owning live saved-statistic inference image[N,32,32,3], no clipping.
  Tensor prediction_training;  ///< Owning live batch-statistic image[N,32,32,3], no clipping.
  Tensor residual;             ///< Owning live saved-statistic inference residual[N,32,32,3].
  Tensor real_score;           ///< Owning live raw target-image scores[N,1].
  Tensor fake_score;           ///< Owning live raw batch-statistic generated-image scores[N,1].
  Tensor fake_inference_score; ///< Owning live raw saved-statistic generated-image scores[N,1].
  Tensor d_loss; ///< Owning live ordinary logistic or half-scaled relativistic D objective.
  Tensor g_loss; ///< Owning live(1-phase)*pixel_loss+phase*gan_loss; all components evaluated.
  Tensor
      gan_loss; ///< Owning live content+source-rounded adversarial weight, plus ESRGAN pixel term.
  Tensor pixel_loss; ///< Owning live batch-image MSE(SRGAN) or meanL1(ESRGAN) against target.
  Tensor
      content_loss; ///< Owning live post-ReLU feature MSE(SRGAN) or pre-activation meanL1(ESRGAN).
  Tensor
      adversarial_loss; ///< Owning live ordinary nonsaturating or half-scaled relativistic G term.
  Tensor relative_real; ///< Owning live relative real score for ESRGAN, raw real score for SRGAN.
  Tensor relative_fake; ///< Owning live relative fake score for ESRGAN, raw fake score for SRGAN.
  Tensor
      prediction_feature; ///< Owning live source feature/ReLU selection divided by255[N,32,32,64].
  Tensor
      target_feature;  ///< Owning stopped target feature/ReLU selection divided by255[N,32,32,64].
  Tensor packed;       ///< Owning live source debug packed boundary from batch-statistic G.
  Tensor shuffled;     ///< Owning live source debug shuffled boundary from batch-statistic G.
  Tensor moments_mean; ///< Owning detached SRGAN batch means[5,4], undefined for ESRGAN.
  Tensor moments_unbiased_variance; ///< Owning detached SRGAN unbiased variances[5,4], undefined
                                    ///< for ESRGAN.
};
/// @brief Evaluate explicit compact SRGAN/ESRGAN pixel/content/adversarial phases without fitting.
///
/// Three distinct matching Float32/64 G,D,stats stores. Finite signed matching
/// CPU lr[N,16,16,3],target/bicubic[N,32,32,3], N>0 aligned; frozen matching1792
/// feature values as AdversarialSRFeatures. All supplied inputs/aliases/products
/// validate before leaves. Source seeds7001/7013/0. Both saved-statistic and batch-
/// statistic G paths always evaluate; stats unchanged beyond initialization.
/// G1591/25,D1673/8,BN40/10(default SRGAN), or G3511/40,D1672/7,no BN(ESRGAN).
/// Objectives read training G: SRGAN pixelMSE, post-ReLU featureMSE; ESRGAN pixelL1,
/// pre-activation featureL1. Both feature tensors divide by255 BEFORE loss. Frozen
/// weights/bias always stop; target extracted features stop, target pixel/real-score
/// paths remain live. SRGAN D is full real/fake logistic mean sum, G adversarial
/// is full nonsaturating mean. ESRGAN uses live relativistic means and half-scaled
/// D/G terms. gan_loss=content+double(float(.001 or .005))*adversarial;
/// ESRGAN adds double(float(.01))*pixel. g_loss=(1-phase)*pixel+phase*gan_loss,
/// phase=adversarial?1:0, computing all components even during warm-up.
/// No fitting, hidden no-grad, partial capture, weight/statistic write or later RNG.
/// Only missing leaves initialize. Caller owns16 pixel-only G steps, resets BOTH
/// Adam states without changing weights/stats, then D capture/write followed by a
/// fresh G capture/write on the same batch. After each G weight write, explicitly
/// commit BN moving candidates from the detached moments captured WITH that G
/// partial, momentum double(.9f). Never recompute moments after writing weights.
/// BN state and1792 frozen feature values are excluded from optimizers; no generator
/// weight EMA exists. Applicable first/higher paths remain live. Owning fields keep
/// storage/graphs; stored-leaf/state/math errors may follow initialization. O(N)
/// fixed-map work/storage, largest features[N,32,32,64]; synchronize stores.
/// @param generator Caller-owned floating G store, source seed7001.
/// @param critic Distinct same-dtype D store, source seed7013.
/// @param stats Distinct same-dtype saved BN store, source seed0; excluded from optimizers.
/// @param lr Finite matching CPU[N,16,16,3] low-resolution inputs.
/// @param target Finite aligned matching CPU[N,32,32,3] HR targets, never G inputs.
/// @param bicubic Finite aligned matching CPU[N,32,32,3] external LR-derived image shortcut.
/// @param feature_weight Finite matching CPU[3,3,3,64] actual frozen feature values.
/// @param feature_bias Finite matching CPU[64] actual frozen bias.
/// @param enhanced Select ESRGAN(true), otherwise SRGAN(false default).
/// @param adversarial Select content/adversarial G objective(true), otherwise pixel warm-up(false
/// default).
/// @return Owning AdversarialSROutput with all source fields and detached optional BN captures.
/// @throws std::invalid_argument For invalid geometry, dtype, values, alignment, products, aliased
/// stores or incompatible leaves.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore g(7001),d(7013),s(0);
/// auto lr=n::zeros({2,16,16,3}),hr=n::zeros({2,32,32,3});
/// auto out=n::models::AdversarialSR(g,d,s,lr,hr,hr,n::zeros({3,3,3,64}),n::zeros({64}),true,true);
/// auto partial=n::grad(out.d_loss,d.parameters());
/// // Caller applies D, reruns G, applies G, then commits same-capture BN moments if present.
/// @endcode
/// @par References
/// - Ledig et al.(2017), Photo-Realistic Single Image Super-Resolution Using a Generative
/// Adversarial Network, https://arxiv.org/abs/1609.04802 ; Wang et al.(2018), ESRGAN: Enhanced
/// Super-Resolution Generative Adversarial Networks, https://arxiv.org/abs/1809.00219 . Fixed-width
/// x2 teaching adaptations with a bicubic image shortcut, not pretrained benchmark models.
/// @see AdversarialSRGenerate(), AdversarialSRFeatures(), RelativisticAverageGan(),
/// batch_norm_moving_values()
inline AdversarialSROutput AdversarialSR(
    ParameterStore& generator,
    ParameterStore& critic,
    ParameterStore& stats,
    const Tensor&   lr,
    const Tensor&   target,
    const Tensor&   bicubic,
    const Tensor&   feature_weight,
    const Tensor&   feature_bias,
    bool            enhanced    = false,
    bool            adversarial = false) {
  using namespace adversarial_sr_detail;
  Stores(generator, stats);
  if (&generator == &critic || &critic == &stats || critic.dtype() != generator.dtype())
    throw std::invalid_argument("SR game needs three distinct matching stores");
  Image(lr, 16, generator.dtype());
  Image(target, 32, generator.dtype());
  Image(bicubic, 32, generator.dtype());
  Features(feature_weight, feature_bias, generator.dtype());
  if (target.size(0) != lr.size(0) || bicubic.size(0) != lr.size(0) ||
      lr.size(0) > std::numeric_limits<std::int64_t>::max() / 65536)
    throw std::invalid_argument("SR game rows/features must align and be representable");
  auto inference = AdversarialSRGenerate(generator, stats, lr, bicubic, enhanced, false),
       training  = AdversarialSRGenerate(generator, stats, lr, bicubic, enhanced, true);
  AdversarialSROutput o;
  o.prediction                = inference.image;
  o.prediction_training       = training.image;
  o.residual                  = inference.residual;
  o.packed                    = training.packed;
  o.shuffled                  = training.shuffled;
  o.moments_mean              = training.moments_mean;
  o.moments_unbiased_variance = training.moments_unbiased_variance;
  o.real_score                = AdversarialSRDiscriminate(critic, target, enhanced);
  o.fake_score                = AdversarialSRDiscriminate(critic, training.image, enhanced);
  o.fake_inference_score      = AdversarialSRDiscriminate(critic, inference.image, enhanced);
  auto predicted              = AdversarialSRFeatures(training.image, feature_weight, feature_bias),
       truth              = AdversarialSRFeatures(target, feature_weight, feature_bias).detach();
  o.prediction_feature    = (enhanced ? predicted : relu(predicted)) / 255.;
  o.target_feature        = (enhanced ? truth : relu(truth)) / 255.;
  auto difference         = training.image - target,
       feature_difference = o.prediction_feature - o.target_feature;
  o.pixel_loss            = enhanced ? mean(abs(difference)) : mean(square(difference));
  o.content_loss = enhanced ? mean(abs(feature_difference)) : mean(square(feature_difference));
  if (enhanced) {
    auto loss          = RelativisticAverageGan(o.real_score, o.fake_score);
    o.d_loss           = loss.discriminator;
    o.adversarial_loss = loss.generator;
    o.relative_real    = loss.real_relative;
    o.relative_fake    = loss.fake_relative;
  } else {
    o.d_loss           = mean(softplus(-o.real_score)) + mean(softplus(o.fake_score));
    o.adversarial_loss = mean(softplus(-o.fake_score));
    o.relative_real    = o.real_score;
    o.relative_fake    = o.fake_score;
  }
  o.gan_loss = o.content_loss + double(enhanced ? .005f : .001f) * o.adversarial_loss;
  if (enhanced)
    o.gan_loss = o.gan_loss + double(.01f) * o.pixel_loss;
  const double phase = adversarial ? 1 : 0;
  o.g_loss           = (1. - phase) * o.pixel_loss + phase * o.gan_loss;
  gan_detail::Finite(o.g_loss);
  gan_detail::Finite(o.d_loss);
  return o;
}
} // namespace datamunge::nn::models
