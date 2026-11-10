#pragma once
/// @file hierarchical_variational_autoencoder.hpp
/// @brief Spatial hierarchical Gaussian VAEs, residual posteriors and explicit BN state control.
#include <limits>

#include "datamunge/nn/gaussian_autoencoder.hpp"
#include "datamunge/nn/image_ops.hpp"

namespace datamunge::nn::models {
/// @brief Own the mean and log variance of a diagonal conditional Gaussian posterior.
///
/// Both fields own lifetime and retain the continuous paths used to construct them.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// const auto q=n::models::HierarchyPosterior(n::ones({2,3}),n::zeros({2,3}),
///     n::zeros({2,3}),n::zeros({2,3}),true);
/// // Zero residuals return the prior mean and log variance exactly.
/// @endcode
/// @see HierarchyPosterior(), ConditionalGaussianKl()
struct HierarchyGaussian {
  Tensor mu;           ///< Owning posterior mean[B,L].
  Tensor log_variance; ///< Owning posterior log variance[B,L].
};

/// @brief Construct direct Gaussian posterior heads or additive corrections to a learned prior.
///
/// residual=true returns(prior_mu+head_mu,prior_lv+head_lv); false returns the
/// supplied head tensors directly. This is additive residual parameterization,
/// not precision-weighted evidence/prior fusion. All four matrices must be finite
/// nonempty matching Float32/64[B,L]. Strides and live paths are retained, with no
/// clipping, RNG or input mutation. Direct mode has no prior gradient path.
/// Results own lifetime; O(B*L) work/storage, no shared state. Arithmetic overflow
/// is not repaired. Zero residuals make posterior and prior identical.
/// @param prior_mu Finite conditional prior mean[B,L].
/// @param prior_lv Finite matching conditional prior log variance[B,L].
/// @param head_mu Finite matching direct mean or residual mean correction.
/// @param head_lv Finite matching direct log variance or residual correction.
/// @param residual Add corrections to the prior when true, default true.
/// @return Owning posterior mean and log variance, retaining applicable continuous paths.
/// @throws std::invalid_argument For invalid shapes/dtypes/values.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// const auto q=n::models::HierarchyPosterior(n::ones({1,2}),n::zeros({1,2}),
///     n::full({1,2},.2),n::full({1,2},.1));
/// // Posterior mean1.2 and log variance.1 in both coordinates.
/// @endcode
/// @par References
/// - Vahdat and Kautz (2020), NVAE: A Deep Hierarchical Variational Autoencoder,
///   https://arxiv.org/abs/2007.03898 . Uses additive residual Gaussian parameters.
/// @see HierarchyGaussian, HierarchicalVariationalAutoencoder(), ConditionalGaussianKl()
inline HierarchyGaussian HierarchyPosterior(
    const Tensor& prior_mu,
    const Tensor& prior_lv,
    const Tensor& head_mu,
    const Tensor& head_lv,
    bool          residual = true) {
  gaussian_detail::Match(prior_mu, prior_lv);
  gaussian_detail::Match(prior_mu, head_mu);
  gaussian_detail::Match(prior_mu, head_lv);
  return residual ? HierarchyGaussian{prior_mu + head_mu, prior_lv + head_lv}
                  : HierarchyGaussian{head_mu, head_lv};
}

/// @brief Sum analytic KL between two diagonal conditional Gaussians over latent coordinates.
///
/// Computes0.5*sum((prior_lv-lv)+(exp(lv-prior_lv)+(mu-prior_mu)^2*exp(-prior_lv))-1,
/// axis1). All four finite nonempty matching Float32/64[B,L] matrices keep their
/// gradients and strides. The parenthesization preserves the migrated Float32
/// calculation. Identical inputs yield exact zero; negative roundoff is not clipped.
/// No sampling, coordinate averaging, variance clipping or shared state. Inputs
/// unchanged; output owns lifetime. O(B*L); computed exp overflow is not repaired.
/// @param mu Finite posterior mean[B,L].
/// @param lv Finite matching posterior log variance[B,L].
/// @param prior_mu Finite matching conditional prior mean[B,L].
/// @param prior_lv Finite matching conditional prior log variance[B,L].
/// @return Owning differentiable per-row conditional KL[B], nats.
/// @throws std::invalid_argument For invalid shapes/dtypes/values.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// const auto kl=n::models::ConditionalGaussianKl(n::ones({2,3}),n::zeros({2,3}),
///     n::zeros({2,3}),n::zeros({2,3}));
/// // KL is1.5 nats for each row.
/// @endcode
/// @par References
/// - Vahdat and Kautz (2020), https://arxiv.org/abs/2007.03898 , Gaussian hierarchy.
/// - Sonderby et al. (2016), Ladder Variational Autoencoders,
///   https://arxiv.org/abs/1602.02282 , conditional Gaussian inference.
/// @see DiagonalGaussianKl(), HierarchyPosterior()
inline Tensor ConditionalGaussianKl(
    const Tensor& mu,
    const Tensor& lv,
    const Tensor& prior_mu,
    const Tensor& prior_lv) {
  gaussian_detail::Match(mu, lv);
  gaussian_detail::Match(mu, prior_mu);
  gaussian_detail::Match(mu, prior_lv);
  return 0.5 *
         sum(((prior_lv - lv) + (exp(lv - prior_lv) + square(mu - prior_mu) * exp(-prior_lv))) -
                 1.0,
             {1});
}

/// @brief Shared geometry, state validation and decoder components for hierarchical forwards.
namespace hierarchy_detail {
/// @brief Validate grayscale geometry and return the lower spatial latent count.
/// @param height Positive multiple-of-four image height.
/// @param width Positive multiple-of-four image width.
/// @param channels Positive feature channels, safe for expansion and concatenation.
/// @param top_latent Positive top Gaussian width.
/// @return (height/4)*(width/4), with all image/feature products representable in int64.
/// @throws std::invalid_argument For invalid geometry or multiplication overflow.
inline std::int64_t
Geometry(std::int64_t height, std::int64_t width, std::int64_t channels, std::int64_t top_latent) {
  const auto limit = std::numeric_limits<std::int64_t>::max();
  if (height <= 0 || width <= 0 || height % 4 || width % 4 || channels <= 0 ||
      channels > limit / 2 || top_latent <= 0 || height > limit / width)
    throw std::invalid_argument(
        "hierarchy needs positive multiples-of-four geometry and safe widths");
  const auto area = (height / 4) * (width / 4);
  if (height * width > limit / channels || channels > limit / channels / 9 ||
      area > limit / channels || area > limit - top_latent || area * channels > limit / top_latent)
    throw std::invalid_argument("hierarchy feature or projection width overflow");
  return area;
}
/// @brief Require separate matching floating parameter/state stores and meaningful BN flags.
/// @param parameters Trainable weight store.
/// @param state BN moving-statistics store, excluded from optimization.
/// @param training Use batch moments when true.
/// @param update_stats Update moving statistics only on a training pass.
/// @throws std::invalid_argument For aliasing/mismatched stores or update without training.
inline void
Stores(ParameterStore& parameters, ParameterStore& state, bool training, bool update_stats) {
  if (&parameters == &state || parameters.dtype() != state.dtype() ||
      (parameters.dtype() != DType::Float32 && parameters.dtype() != DType::Float64) ||
      (!training && update_stats))
    throw std::invalid_argument(
        "hierarchy needs separate matching floating stores and training for BN updates");
}
/// @brief Validate finite nonempty NHWC feature maps without constraining their value range.
/// @param x Finite Float32/64[B,H,W,C], all dimensions positive.
/// @param type Matching computation dtype.
/// @throws std::invalid_argument For invalid shape/dtype/values.
inline void Spatial(const Tensor& x, DType type) {
  if (!x.defined() || x.dim() != 4 || x.dtype() != type ||
      (type != DType::Float32 && type != DType::Float64))
    throw std::invalid_argument("floating NHWC hierarchy features required");
  for (auto size : x.shape())
    if (size <= 0)
      throw std::invalid_argument("nonempty hierarchy feature axes required");
  for (double value : x.contiguous().to_vector<double>())
    if (!std::isfinite(value))
      throw std::invalid_argument("finite hierarchy features required");
}
/// @brief Apply the source Swish activation with unit sigmoid slope.
/// @param x Floating activation tensor.
/// @return Owning differentiable x*sigmoid(x).
inline Tensor Swish(const Tensor& x) {
  return x * sigmoid(x);
}
/// @brief Apply BN with source momentum.9,epsilon.001 and explicit update control.
/// @param p Trainable gamma/beta store.
/// @param state Moving-mean/variance store.
/// @param name Stable layer prefix.
/// @param x NHWC floating features.
/// @param training Use population batch moments in normalization.
/// @param update_stats Also update moving mean/unbiased variance when true.
/// @return Owning differentiable normalized values; inference detaches moving buffers.
inline Tensor Bn(
    ParameterStore&    p,
    ParameterStore&    state,
    const std::string& name,
    const Tensor&      x,
    bool               training,
    bool               update_stats) {
  BatchNormOptions o;
  o.momentum     = .9;
  o.eps          = 1e-3;
  o.update_stats = update_stats;
  return batch_norm(p, state, name, x, training, o);
}
} // namespace hierarchy_detail

/// @brief Apply the miniature NVAE expansion/depthwise/BN/Swish/projection/SE residual cell.
///
/// For C input channels: bias-free1x1 C->2C expansion, BN/Swish, depthwise3x3
/// multiplier1, BN/Swish, bias-free1x1 2C->C projection and BN. Spatial mean then
/// SE dense C->max(1,C/2) Swish->C sigmoid gates the project map. Returns
/// x+0.1*(project*gate). Names are cell/expand,expand_bn,depthwise,depth_bn,
/// project,project_bn,se_reduce,se_expand. Ordinary Glorot weights (depthwise
/// initialized per input channel), gamma1, beta0; default C4 gives13 leaves/
/// 198 scalars and six moving buffers/40 scalars. Use the same architecture per store.
///
/// training=false uses detached moving buffers; true uses batch moments without
/// moving updates unless update_stats=true. Moving mean/UNBIASED variance use
/// momentum.9; normalization uses population variance plus.001. For one scalar
/// observation per channel the returned population variance is zero. Initial
/// calls lazily create parameters/buffers even in inference. Existing weights
/// never change; only explicit training/update passes change existing buffers.
/// Caller excludes state from gradients and optimizer; checkpoint both stores.
/// Validation precedes leaf creation; incompatible leaves/nonfinite computed
/// values may fail later. Inputs unchanged, all continuous paths own lifetime.
/// Shared stores require synchronization; no noise, optimizer or fitting is hidden.
/// @param parameters Mutable floating trainable store.
/// @param state Separate same-dtype BN moving-statistics store.
/// @param x Finite nonempty same-dtype NHWC[B,H,W,C] features, C>0 with2C representable.
/// @param training Use batch moments when true, default false.
/// @param update_stats Update moving moments when true; requires training, default false.
/// @return Owning differentiable residual feature map with x's shape/dtype.
/// @throws std::invalid_argument For invalid inputs/stores/flags, channel overflow or incompatible
/// leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(2609),state(0);
/// const auto y=n::models::HierarchyNvaeCell(p,state,n::ones({2,7,7,4}),true,false);
/// // Batch moments are used; six moving buffers stay at their initial values.
/// @endcode
/// @par References
/// - Vahdat and Kautz (2020), https://arxiv.org/abs/2007.03898 . This is one small
///   cell with SE and residual scale.1, not the full deep/multiscale NVAE network.
/// @see HierarchicalVariationalAutoencoder(), batch_norm()
inline Tensor HierarchyNvaeCell(
    ParameterStore& parameters,
    ParameterStore& state,
    const Tensor&   x,
    bool            training     = false,
    bool            update_stats = false) {
  hierarchy_detail::Stores(parameters, state, training, update_stats);
  hierarchy_detail::Spatial(x, parameters.dtype());
  const auto channels = x.size(3);
  if (channels > std::numeric_limits<std::int64_t>::max() / 2 ||
      channels > std::numeric_limits<std::int64_t>::max() / channels / 2 ||
      channels > std::numeric_limits<std::int64_t>::max() / 18 ||
      x.numel() > std::numeric_limits<std::int64_t>::max() / 2)
    throw std::invalid_argument("NVAE expansion width overflow");
  ConvLayerOptions no_bias;
  no_bias.bias      = false;
  const auto expand = hierarchy_detail::Swish(
      hierarchy_detail::Bn(
          parameters,
          state,
          "cell/expand_bn",
          conv2d(parameters, "cell/expand", x, 2 * channels, 1, no_bias),
          training,
          update_stats));
  const auto depth = hierarchy_detail::Swish(
      hierarchy_detail::Bn(
          parameters,
          state,
          "cell/depth_bn",
          depthwise_conv2d(parameters, "cell/depthwise", expand, 1, 3, no_bias),
          training,
          update_stats));
  const auto project = hierarchy_detail::Bn(
      parameters,
      state,
      "cell/project_bn",
      conv2d(parameters, "cell/project", depth, channels, 1, no_bias),
      training,
      update_stats);
  const auto pooled = mean(project, {1, 2}),
             hidden = hierarchy_detail::Swish(dense(
                 parameters, "cell/se_reduce", pooled, std::max<std::int64_t>(1, channels / 2)));
  const auto gate   = reshape(
      sigmoid(dense(parameters, "cell/se_expand", hidden, channels)), {x.size(0), 1, 1, channels});
  return x + 0.1 * (project * gate);
}

namespace hierarchy_detail {
/// @brief Project a top code to quarter-resolution features and apply the selected cell.
/// @param p Trainable parameter store.
/// @param state Moving statistics for the NVAE variant.
/// @param top Top code[B,T], already validated.
/// @param nvae Select residual BN/SE cell rather than ordinary3x3 Swish.
/// @param training Use batch moments in the NVAE cell.
/// @param update_stats Also update moving moments.
/// @param height Image height, a positive multiple of four.
/// @param width Image width, a positive multiple of four.
/// @param channels Positive feature channel width.
/// @return Owning top feature[B,H/4,W/4,C].
inline Tensor TopFeature(
    ParameterStore& p,
    ParameterStore& state,
    const Tensor&   top,
    bool            nvae,
    bool            training,
    bool            update_stats,
    std::int64_t    height,
    std::int64_t    width,
    std::int64_t    channels) {
  const auto projected = reshape(
      dense(p, "top_projection", top, (height / 4) * (width / 4) * channels),
      {top.size(0), height / 4, width / 4, channels});
  return nvae ? HierarchyNvaeCell(p, state, projected, training, update_stats)
              : Swish(conv2d(p, "cell", projected, channels, 3));
}
/// @brief Evaluate the learned one-channel conditional-prior Gaussian heads.
/// @param p Trainable store sharing prior_mu/prior_lv1x1 convolution leaves.
/// @param feature Quarter-resolution top feature[B,Hq,Wq,C].
/// @return Owning means/log variances[B,Hq*Wq], preserving all top-feature paths.
inline HierarchyGaussian PriorHeads(ParameterStore& p, const Tensor& feature) {
  const auto n = feature.size(0), lower = feature.size(1) * feature.size(2);
  return {
      reshape(conv2d(p, "prior_mu", feature, 1, 1), {n, lower}),
      reshape(conv2d(p, "prior_lv", feature, 1, 1), {n, lower})};
}
/// @brief Decode explicit lower samples and top features into flat Bernoulli logits.
/// @param p Trainable combine/decoder14/output convolution store.
/// @param lower Lower sample[B,Hq*Wq], already validated.
/// @param top_feature Top feature[B,Hq,Wq,C], already validated.
/// @param height Original image height4*Hq.
/// @param width Original image width4*Wq.
/// @param channels Feature width C.
/// @return Owning logits[B,height*width], with both latent paths live.
inline Tensor Decoder(
    ParameterStore& p,
    const Tensor&   lower,
    const Tensor&   top_feature,
    std::int64_t    height,
    std::int64_t    width,
    std::int64_t    channels) {
  const auto lower_map = reshape(lower, {lower.size(0), height / 4, width / 4, 1});
  auto       h = Swish(conv2d(p, "combine", concat({top_feature, lower_map}, 3), channels, 1));
  h            = Swish(conv2d(p, "decoder14", upsample_nearest(h, 2), channels, 3));
  return reshape(
      conv2d(p, "output", upsample_nearest(h, 2), 1, 3), {lower.size(0), height * width});
}
} // namespace hierarchy_detail

/// @brief Decode explicit lower and top latent samples without an image or posterior encoder.
///
/// Projects the supplied top code to quarter-resolution features, applies the
/// selected ordinary/NVAE cell, combines the lower one-channel spatial code,
/// then repeat2/3x3 Swish/repeat2/3x3 logits and sigmoid. Both latent samples
/// are required; no image or encoder bypass. NVAE uses detached moving moments.
/// Same names/initializers as the full model, so restore BOTH parameter and BN
/// stores. Lazy inference can create missing leaves/buffers; existing values are
/// unchanged. Continuous code/parameter gradients stay live; no no-grad scope,
/// RNG, prior sampling or fitting is hidden. Inputs unchanged, output owns lifetime.
/// Validates geometry/stores/finite code rows before creation; incompatible leaves
/// or computed nonfinite states may fail later. Synchronize mutable stores.
/// @param parameters Mutable floating decoder/top-feature parameter store.
/// @param state Separate matching-dtype NVAE statistics store, unused by ordinary hierarchy.
/// @param lower Finite matching-dtype[B,(height/4)*(width/4)] lower sample.
/// @param top Finite matching-dtype[B,T] top sample, T>0.
/// @param nvae Select miniature NVAE cell when true, default false.
/// @param height Positive multiple-of-four output height, default28.
/// @param width Positive multiple-of-four output width, default28.
/// @param channels Positive feature width, default4.
/// @return Owning differentiable Bernoulli probabilities[B,height*width].
/// @throws std::invalid_argument For invalid inputs/geometry/stores or incompatible leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(2609),state(0);
/// const auto probability=n::models::HierarchyDecode(p,state,n::zeros({2,49}),n::zeros({2,2}));
/// // Decoder-only[2,784] probabilities; neither image nor encoder is needed.
/// @endcode
/// @see HierarchyPrior(), HierarchicalVariationalAutoencoder()
inline Tensor HierarchyDecode(
    ParameterStore& parameters,
    ParameterStore& state,
    const Tensor&   lower,
    const Tensor&   top,
    bool            nvae     = false,
    std::int64_t    height   = 28,
    std::int64_t    width    = 28,
    std::int64_t    channels = 4) {
  hierarchy_detail::Stores(parameters, state, false, false);
  if (!top.defined() || top.dim() != 2)
    throw std::invalid_argument("top hierarchy code must be a matrix");
  const auto area = hierarchy_detail::Geometry(height, width, channels, top.size(1));
  aae_detail::Matrix(top, top.size(1), parameters.dtype(), false);
  aae_detail::Matrix(lower, area, parameters.dtype(), false);
  if (lower.size(0) != top.size(0))
    throw std::invalid_argument("hierarchy code batches must match");
  const auto feature = hierarchy_detail::TopFeature(
      parameters, state, top, nvae, false, false, height, width, channels);
  return sigmoid(hierarchy_detail::Decoder(parameters, lower, feature, height, width, channels));
}

/// @brief Own all stages of an image-free ancestral sample from a learned Gaussian hierarchy.
///
/// Floating fields own lifetime and retain all top-code, lower-noise and parameter
/// paths. No noise is drawn internally; moving moments are used for the NVAE cell.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(2609),state(0);
/// const auto out=n::models::HierarchyPrior(p,state,n::zeros({2,2}),n::zeros({2,49}));
/// // lower_z equals the conditional prior mean for zero supplied lower noise.
/// @endcode
/// @see HierarchyPrior()
struct HierarchyPriorOutput {
  Tensor top_z;       ///< Supplied top sample[B,T], normally standard normal for generation.
  Tensor lower_noise; ///< Supplied conditional lower epsilon[B,Hq*Wq].
  Tensor prior_mu;    ///< Learned conditional prior mean[B,Hq*Wq].
  Tensor prior_lv;    ///< Learned conditional prior log variance[B,Hq*Wq].
  Tensor lower_z;     ///< prior_mu+exp(prior_lv/2)*lower_noise[B,Hq*Wq].
  Tensor probability; ///< Bernoulli decoder probabilities[B,height*width].
};

/// @brief Evaluate an ancestral prior draw using explicit top samples and conditional lower noise.
///
/// For supplied top z2, evaluate learned p(z1|z2), sample its lower Gaussian with
/// the supplied epsilon, then decode BOTH latent groups. No image/encoder input,
/// posterior substitution, RNG or fitting is used. The NVAE cell uses moving
/// statistics, detached from autograd; restore BOTH stores for trained inference.
/// Initial calls may create missing leaves/buffers; existing values are unchanged.
/// All continuous top/noise/parameter paths and strides are retained; outputs own
/// lifetime. Validates matching nonempty floating rows and geometry/stores before
/// creation; computed nonfinite values/incompatible leaves may fail later.
/// @param parameters Mutable matching floating generative parameter store.
/// @param state Separate matching-dtype moving-statistics store, unused for ordinary hierarchy.
/// @param top Finite[B,T] supplied top samples, T>0; normally independent N(0,I).
/// @param lower_noise Finite same-batch/dtype[B,(height/4)*(width/4)] supplied standard normals.
/// @param nvae Select miniature NVAE cell when true, default false.
/// @param height Positive multiple-of-four output height, default28.
/// @param width Positive multiple-of-four output width, default28.
/// @param channels Positive feature width, default4.
/// @return Owning six-field ancestral prior output, including both supplied inputs.
/// @throws std::invalid_argument For invalid inputs/geometry/stores or incompatible leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(2609),state(0);
/// const auto out=n::models::HierarchyPrior(p,state,n::ones({2,2}),n::ones({2,49}));
/// // Caller supplies both latent noise groups; no posterior image is needed.
/// @endcode
/// @par References
/// - Vahdat and Kautz (2020), https://arxiv.org/abs/2007.03898 , top-down Gaussian hierarchy.
/// @see HierarchyPriorOutput, HierarchyDecode(), HierarchicalVariationalAutoencoder()
inline HierarchyPriorOutput HierarchyPrior(
    ParameterStore& parameters,
    ParameterStore& state,
    const Tensor&   top,
    const Tensor&   lower_noise,
    bool            nvae     = false,
    std::int64_t    height   = 28,
    std::int64_t    width    = 28,
    std::int64_t    channels = 4) {
  hierarchy_detail::Stores(parameters, state, false, false);
  if (!top.defined() || top.dim() != 2)
    throw std::invalid_argument("top hierarchy code must be a matrix");
  const auto area = hierarchy_detail::Geometry(height, width, channels, top.size(1));
  aae_detail::Matrix(top, top.size(1), parameters.dtype(), false);
  aae_detail::Matrix(lower_noise, area, parameters.dtype(), false);
  if (top.size(0) != lower_noise.size(0))
    throw std::invalid_argument("ancestral hierarchy batches must match");
  const auto feature = hierarchy_detail::TopFeature(
      parameters, state, top, nvae, false, false, height, width, channels);
  const auto           prior = hierarchy_detail::PriorHeads(parameters, feature);
  HierarchyPriorOutput out;
  out.top_z       = top;
  out.lower_noise = lower_noise;
  out.prior_mu    = prior.mu;
  out.prior_lv    = prior.log_variance;
  out.lower_z     = GaussianReparameterize(prior.mu, prior.log_variance, lower_noise);
  out.probability =
      sigmoid(hierarchy_detail::Decoder(parameters, out.lower_z, feature, height, width, channels));
  return out;
}

/// @brief Own all nineteen spatial/posterior/prior/reconstruction/objective hierarchy endpoints.
///
/// Fields own lifetime and retain continuous paths. Neither greedy posterior
/// means nor decoded greedy means generally equal posterior predictive means.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(2609),state(0);
/// const auto out=n::models::HierarchicalVariationalAutoencoder(
///     p,state,n::zeros({2,784}),n::zeros({2,51}),true);
/// // Default NVAE:37 trainable tensors/2027 scalars plus six BN moving buffers.
/// @endcode
/// @see HierarchicalVariationalAutoencoder()
struct HierarchicalVaeOutput {
  Tensor encoder_feature; ///< Quarter-resolution encoder map[B,Hq,Wq,C].
  Tensor top_feature;     ///< Sampled-top-conditioned map[B,Hq,Wq,C].
  Tensor top_mu;          ///< Top Gaussian mean[B,T], dependent only on image.
  Tensor top_lv;          ///< Top Gaussian log variance[B,T], dependent only on image.
  Tensor top_z;           ///< Supplied-noise top sample[B,T].
  Tensor prior_mu;        ///< Learned lower prior mean conditioned on sampled top[B,Hq*Wq].
  Tensor prior_lv;        ///< Learned lower prior log variance[B,Hq*Wq].
  Tensor head_mu;         ///< Direct lower mean or residual correction[B,Hq*Wq].
  Tensor head_lv;         ///< Direct lower log variance or residual correction[B,Hq*Wq].
  Tensor lower_mu;        ///< Lower posterior mean[B,Hq*Wq], conditioned on image AND sampled top.
  Tensor lower_lv;        ///< Lower posterior log variance[B,Hq*Wq].
  Tensor lower_z;         ///< Supplied-noise lower sample[B,Hq*Wq].
  Tensor probability;     ///< Bernoulli decoder probabilities[B,height*width], not sampled pixels.
  Tensor nll;             ///< Summed pixel Bernoulli NLL/cross-entropy per image[B].
  Tensor kl_lower;        ///< Conditional lower Gaussian KL summed over Hq*Wq coordinates[B].
  Tensor kl_top;          ///< Standard-normal top Gaussian KL summed over T coordinates[B].
  Tensor loss;        ///< Scalar mean(nll+(kl_lower+kl_top)), unregularized negative ELBO estimate.
  Tensor regularizer; ///< Scalar smoothed two-prior-kernel penalty for NVAE, otherwise exact zero.
  Tensor train_loss;  ///< Scalar loss+regularizer; validation normally uses unregularized loss.
};

/// @brief Evaluate a spatial two-group Gaussian hierarchy or its miniature residual-posterior NVAE
/// variant.
///
/// Two3x3 Swish convolutions and average pools encode an image at quarter resolution.
/// Dense Gaussian heads produce global top z2; its feature map gives learned p(z1|z2).
/// An image/top feature fusion gives direct lower posterior heads (ordinary hierarchy)
/// or residual mean/log-variance corrections (NVAE). Explicit noise columns first
/// drive the spatial lower group, then the top group: default49 lower +2 top=51.
/// BOTH sampled latent groups reach the convolutional decoder. No image/encoder skip
/// bypasses sampling. Ordinary Glorot weights and zero biases; NVAE BN gamma1/beta0.
/// Defaults have26 leaves/1977 scalars and no state, or37/2027 with six BN buffers/40
/// scalars for NVAE. Use separate stores per architecture and restore weights AND state.
///
/// loss=mean(sum784 pixel NLL+(conditional lower KL+top KL)), preserving source
/// parenthesization. For non-default geometry use height*width pixels, Hq*Wq lower
/// coordinates and top_latent top coordinates. NVAE train_loss additionally adds
/// 1e-4*(sqrt(sum(prior_mu/W^2)+1e-12)+sqrt(sum(prior_lv/W^2)+1e-12)). Each prior
/// kernel is Cx1, so its spectral norm is its L2 norm before this smoothing. Only
/// these TWO kernels are penalized, not all NVAE kernels. No free bits, KL warmup,
/// variance clipping or posterior flows; this small adaptation is not full NVAE.
///
/// NVAE training uses batch moments; update_stats=true also updates moving means
/// and unbiased variances ONCE in this forward, before the caller's optimizer step.
/// Extra gradient/inspection passes use training=true,update_stats=false. Inference
/// uses detached moving moments. State is a separate store that the caller excludes
/// from gradients/Adam. Initial calls lazily create missing weights/buffers. Existing
/// weights never change during a forward; existing state changes only with explicit
/// BN updates. Ordinary hierarchy ignores both training flags after validating them.
///
/// Finite nonempty matching Float32/64 flat image and explicit noise matrices,
/// unit-range targets, safe multiples-of-four geometry and separate matching stores
/// are checked before leaf creation. Strides and all continuous image/noise/parameter
/// paths remain. Inputs unchanged; fields own lifetime. Incompatible leaves/computed
/// nonfinite values may fail later. No fitting, RNG or optimizer is hidden; synchronize
/// stores. Soft-target reconstruction is cross-entropy, binary targets give NLL.
/// @param parameters Mutable trainable floating store; seed2609 reproduces this example.
/// @param state Separate same-dtype BN moving-statistics store, seed0 in the example.
/// @param image Finite unit-range[B,height*width] pixels, B>0.
/// @param epsilon Finite same-dtype/batch[B,Hq*Wq+top_latent] explicit Gaussian noise.
/// @param nvae Select residual posterior and miniature BN/SE cell, default false.
/// @param training Use batch moments in NVAE when true, default false.
/// @param update_stats Update existing NVAE statistics when true; requires training, default false.
/// @param height Positive multiple-of-four image height, default28.
/// @param width Positive multiple-of-four image width, default28.
/// @param channels Positive encoder/top/decoder feature channels, default4.
/// @param top_latent Positive global Gaussian dimension, default2.
/// @return Owning nineteen-field forward with unregularized loss and separate train_loss.
/// @throws std::invalid_argument For invalid inputs/options/geometry/stores or incompatible leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(2609),state(0);
/// const auto out=n::models::HierarchicalVariationalAutoencoder(
///     p,state,n::full({2,784},.2),n::zeros({2,51}),true,true,false);
/// const auto gradients=n::grad(out.train_loss,p.parameters());
/// // Caller updates weights and explicitly chooses which forward updates BN state.
/// @endcode
/// @par References
/// - Vahdat and Kautz (2020), NVAE: A Deep Hierarchical Variational Autoencoder,
///   https://arxiv.org/abs/2007.03898 . Uses a two-group teaching adaptation.
/// - Sonderby et al. (2016), https://arxiv.org/abs/1602.02282 , Gaussian hierarchy;
///   this model uses direct/additive residual heads rather than precision fusion.
/// @see HierarchicalVaeOutput, HierarchyPosterior(), ConditionalGaussianKl(),
/// HierarchyNvaeCell(), HierarchyDecode(), HierarchyPrior()
inline HierarchicalVaeOutput HierarchicalVariationalAutoencoder(
    ParameterStore& parameters,
    ParameterStore& state,
    const Tensor&   image,
    const Tensor&   epsilon,
    bool            nvae         = false,
    bool            training     = false,
    bool            update_stats = false,
    std::int64_t    height       = 28,
    std::int64_t    width        = 28,
    std::int64_t    channels     = 4,
    std::int64_t    top_latent   = 2) {
  const auto area = hierarchy_detail::Geometry(height, width, channels, top_latent);
  hierarchy_detail::Stores(parameters, state, training, update_stats);
  aae_detail::Matrix(image, height * width, parameters.dtype(), true);
  aae_detail::Matrix(epsilon, area + top_latent, parameters.dtype(), false);
  if (image.size(0) != epsilon.size(0))
    throw std::invalid_argument("hierarchy image/noise batches must match");
  HierarchicalVaeOutput out;
  const auto            n = image.size(0);
  const auto            x = reshape(image, {n, height, width, 1});
  auto                  h = hierarchy_detail::Swish(conv2d(parameters, "encoder1", x, channels, 3));
  h                       = avg_pool2d(h, 2, 2);
  h                       = hierarchy_detail::Swish(conv2d(parameters, "encoder2", h, channels, 3));
  out.encoder_feature     = avg_pool2d(h, 2, 2);
  const auto flat         = reshape(out.encoder_feature, {n, area * channels});
  out.top_mu              = dense(parameters, "top_mu", flat, top_latent);
  out.top_lv              = dense(parameters, "top_lv", flat, top_latent);
  out.top_z               = GaussianReparameterize(
      out.top_mu, out.top_lv, slice_axis(epsilon, 1, area, area + top_latent));
  out.top_feature = hierarchy_detail::TopFeature(
      parameters, state, out.top_z, nvae, training, update_stats, height, width, channels);
  const auto prior       = hierarchy_detail::PriorHeads(parameters, out.top_feature);
  out.prior_mu           = prior.mu;
  out.prior_lv           = prior.log_variance;
  const auto posterior_h = hierarchy_detail::Swish(conv2d(
      parameters,
      "posterior_hidden",
      concat({out.encoder_feature, out.top_feature}, 3),
      channels,
      1));
  const auto heads       = conv2d(parameters, "posterior_heads", posterior_h, 2, 1);
  out.head_mu            = reshape(slice_axis(heads, 3, 0, 1), {n, area});
  out.head_lv            = reshape(slice_axis(heads, 3, 1, 2), {n, area});
  const auto q = HierarchyPosterior(out.prior_mu, out.prior_lv, out.head_mu, out.head_lv, nvae);
  out.lower_mu = q.mu;
  out.lower_lv = q.log_variance;
  out.lower_z = GaussianReparameterize(out.lower_mu, out.lower_lv, slice_axis(epsilon, 1, 0, area));
  const auto logits =
      hierarchy_detail::Decoder(parameters, out.lower_z, out.top_feature, height, width, channels);
  out.probability = sigmoid(logits);
  out.nll         = BernoulliPixelNll(logits, image);
  out.kl_lower    = ConditionalGaussianKl(out.lower_mu, out.lower_lv, out.prior_mu, out.prior_lv);
  out.kl_top      = DiagonalGaussianKl(out.top_mu, out.top_lv);
  out.loss        = mean(out.nll + (out.kl_lower + out.kl_top), {0});
  out.regularizer = scalar(0., parameters.dtype());
  if (nvae) {
    std::vector<Tensor> norms;
    for (const std::string name : {"prior_mu/W", "prior_lv/W"})
      norms.push_back(sqrt(sum(square(parameters.at(name))) + 1e-12));
    out.regularizer = 1e-4 * (norms[0] + norms[1]);
  }
  out.train_loss = out.loss + out.regularizer;
  return out;
}
} // namespace datamunge::nn::models
