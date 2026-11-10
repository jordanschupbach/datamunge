#pragma once
/// @file big_gan.hpp
/// @brief Compact hierarchical conditional GAN with explicit spectral caches and captured BN
/// updates.
#include "datamunge/nn/captured_batch_norm.hpp"
#include "datamunge/nn/dense_gan.hpp"
#include "datamunge/nn/gan_priors.hpp"
#include "datamunge/nn/image_ops.hpp"
#include "datamunge/nn/spectral_cache.hpp"

namespace datamunge::nn::models {
/// @brief Internal fixed residual geometry, normal-initialized spectral maps and conditional affine
/// BN.
namespace big_detail {
/// @brief Require a finite matching floating grayscale NHWC batch with fixed square extent.
/// @param image Finite signed CPU[N,side,side,1] with positive N and strides allowed.
/// @param side Required positive spatial extent,8 or28 in the public maps.
/// @param dtype Required Float32/64 dtype.
/// @throws std::invalid_argument For wrong geometry/dtype, nonfinite values or activation-product
/// overflow.
/// @throws std::logic_error For non-CPU storage.
inline void Image(const Tensor& image, std::int64_t side, DType dtype) {
  gan_detail::Finite(image);
  if (image.dim() != 4 || image.shape() != Shape({image.size(0), side, side, 1}) ||
      image.dtype() != dtype ||
      image.size(0) >
          std::numeric_limits<std::int64_t>::max() / std::max<std::int64_t>(side * side, 256))
    throw std::invalid_argument("BigGAN requires matching positive NHWC grayscale geometry");
}
/// @brief Validate the three distinct same-dtype generator stores before any leaves.
/// @param parameters Mutable floating weight owner.
/// @param spectral Distinct matching cached-column owner.
/// @param moving Third distinct matching BN-moment owner.
/// @throws std::invalid_argument For any alias or dtype mismatch.
inline void Stores(ParameterStore& parameters, ParameterStore& spectral, ParameterStore& moving) {
  if (&parameters == &spectral || &parameters == &moving || &spectral == &moving ||
      parameters.dtype() != spectral.dtype() || parameters.dtype() != moving.dtype())
    throw std::invalid_argument("BigGAN generator needs three distinct matching stores");
}
/// @brief Initialize a normal(.02) raw kernel and source-rounded columns, then normalize with live
/// sigma.
/// @param parameters Floating weight owner.
/// @param spectral Distinct matching spectral-column owner, unchanged after initialization.
/// @param name Layer prefix.
/// @param shape Positive fixed kernel shape; final axis is output width.
/// @return Owning live normalized kernel with stopped columns and its original shape.
/// @throws std::invalid_argument For incompatible leaves or nonfinite computation.
/// @throws std::logic_error For non-CPU storage.
inline Tensor Kernel(
    ParameterStore&    parameters,
    ParameterStore&    spectral,
    const std::string& name,
    const Shape&       shape) {
  auto       w   = parameters.get(name + "/W", shape, init::normal(.02));
  const auto out = shape.back(), rows = w.numel() / out;
  auto       u = spectral.get(name + "/u", {out, 1}, init::constant(1.f / std::sqrt(float(out))));
  auto       v = spectral.get(name + "/v", {rows, 1}, init::constant(1.f / std::sqrt(float(rows))));
  return cached_spectral_kernel(w, u, v).weight;
}
/// @brief Apply a fixed normal-initialized spectral dense map with optional zero bias.
/// @param p Floating weight/bias owner.
/// @param s Matching spectral state owner.
/// @param name Layer prefix.
/// @param x Already validated matrix[...,in].
/// @param in Fixed positive input width.
/// @param out Fixed positive output width.
/// @param bias Create/add name/b when true(default).
/// @return Owning live dense output[...,out].
/// @throws std::invalid_argument For incompatible leaves or nonfinite computation.
inline Tensor Dense(
    ParameterStore&    p,
    ParameterStore&    s,
    const std::string& name,
    const Tensor&      x,
    std::int64_t       in,
    std::int64_t       out,
    bool               bias = true) {
  auto y = matmul(x, Kernel(p, s, name, {in, out}));
  if (bias) {
    auto b = p.get(name + "/b", {out}, init::zeros());
    gan_detail::Finite(b);
    y = bias_add(y, b);
  }
  gan_detail::Finite(y);
  return y;
}
/// @brief Apply stride-one same-padded spectral NHWC cross-correlation with optional zero bias.
/// @param p Floating kernel/bias owner.
/// @param s Matching spectral state owner.
/// @param name Layer prefix.
/// @param x Already validated NHWC input.
/// @param in Fixed positive input channel count.
/// @param out Fixed positive output channel count.
/// @param kernel Fixed odd kernel extent,1 or3.
/// @param bias Create/add name/b when true(default).
/// @return Owning live same-spatial-size NHWC output.
/// @throws std::invalid_argument For incompatible leaves or nonfinite computation.
inline Tensor Conv(
    ParameterStore&    p,
    ParameterStore&    s,
    const std::string& name,
    const Tensor&      x,
    std::int64_t       in,
    std::int64_t       out,
    std::int64_t       kernel,
    bool               bias = true) {
  auto y = conv2d(x, Kernel(p, s, name, {kernel, kernel, in, out}));
  if (bias) {
    auto b = p.get(name + "/b", {out}, init::zeros());
    gan_detail::Finite(b);
    y = bias_add(y, b);
  }
  gan_detail::Finite(y);
  return y;
}
/// @brief Apply captured unparameterized BN followed by per-row conditional gain+1 and bias.
/// @param p Floating conditional-affine owner.
/// @param s Matching cached spectral state.
/// @param moving Matching BN moving statistics, unchanged after initialization.
/// @param name BN prefix, bn1 or bn2.
/// @param x Already validated NHWC four-channel values.
/// @param condition Matching matrix[N,8]: raw class embedding followed by one four-coordinate
/// chunk.
/// @param batch Use batch moments when true, saved moments otherwise.
/// @param capture Receives the owning normalized-value/moment record from this forward.
/// @return Owning live conditionally scaled/shifted NHWC values.
/// @throws std::invalid_argument For incompatible leaves/state or nonfinite computation.
inline Tensor ConditionalNorm(
    ParameterStore&    p,
    ParameterStore&    s,
    ParameterStore&    moving,
    const std::string& name,
    const Tensor&      x,
    const Tensor&      condition,
    bool               batch,
    BatchNormCapture&  capture) {
  capture   = capture_batch_norm(moving, name, x, batch);
  auto gain = Dense(p, s, name + "/gain", condition, 8, 4) + 1.;
  auto bias = Dense(p, s, name + "/bias", condition, 8, 4);
  auto y    = capture.value * reshape(gain, {-1, 1, 1, 4}) + reshape(bias, {-1, 1, 1, 4});
  gan_detail::Finite(y);
  return y;
}
} // namespace big_detail

/// @brief Resize finite grayscale MNIST geometry using Double half-pixel bilinear arithmetic.
///
/// input[N,28,28,1] ->[N,8,8,1], N>0. Reuses resize_bilinear with align_corners=false,
/// half_pixel_centers=true, computing in Float64 before one cast to input dtype.
/// Float32 source pixels thereby preserve all four-term Double interpolation/final
/// Float32 rounding. No clipping, range normalization, data loading or fitting.
/// Finite signed Float32/64 CPU inputs and strides retain applicable input paths;
/// all geometry/products validate first. Inputs unchanged, outputs own lifetime.
/// Work/storage O(N*784) plus validation; safe without concurrent input writes.
/// @param images Finite floating CPU[N,28,28,1], positive N.
/// @return Owning live same-dtype NHWC[N,8,8,1] resampled values.
/// @throws std::invalid_argument For invalid shape/dtype/products/values or nonfinite output.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const auto small=dm::nn::models::BigGanResize8(dm::nn::ones({2,28,28,1}));
/// // small is[2,8,8,1], all ones; values are not clipped or rescaled.
/// @endcode
/// @par References
/// - TensorFlow ResizeBilinear half-pixel geometry,
/// https://www.tensorflow.org/api_docs/python/tf/raw_ops/ResizeBilinear .
///   The source preprocessing uses Double interpolation before Float32 output.
/// @see resize_bilinear(), BigGan()
inline Tensor BigGanResize8(const Tensor& images) {
  if (!images.defined())
    throw std::invalid_argument("BigGAN resize needs a defined image batch");
  big_detail::Image(images, 28, images.dtype());
  auto result =
      cast(resize_bilinear(cast(images, DType::Float64), {8, 8}, false, true), images.dtype());
  gan_detail::Finite(result);
  return result;
}

/// @brief Own six captured channel moment vectors from the selected BigGAN batch-statistics
/// generator.
///
/// Each field is live[C4] when captured from a batch forward; all are undefined
/// for an inference-only generator call. Handles retain storage/applicable graphs
/// after parent/stores disappear. They are values from the forward before an
/// optimizer write, not deferred recomputation; no mutation/optimizer is retained.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore g(6803),sn(0),moving(0);
/// auto
/// out=n::models::BigGanGenerate(g,sn,moving,n::zeros({2,12}),n::zeros({2},n::DType::Int32),true);
/// // out.moments.bn1_mean and all five other fields are defined four-channel vectors.
/// @endcode
/// @par References
/// - Brock et al. (2018/2019), https://arxiv.org/abs/1809.11096 .
/// @see BigGanGenerate(), BigGanCommitMoments(), BatchNormCapture
struct BigGanMoments {
  Tensor bn1_mean;              ///< Owning live first conditional-BN batch channel mean[4].
  Tensor bn1_unbiased_variance; ///< Owning live first conditional-BN source-corrected variance[4].
  Tensor bn2_mean;              ///< Owning live second conditional-BN batch channel mean[4].
  Tensor bn2_unbiased_variance; ///< Owning live second conditional-BN source-corrected variance[4].
  Tensor output_mean;           ///< Owning live output-BN batch channel mean[4].
  Tensor output_unbiased_variance; ///< Owning live output-BN source-corrected variance[4].
};

/// @brief Own sigmoid residual-generator pixels and optional captured BN moments.
///
/// image[N,8,8,1] and the moment record retain storage/applicable first/higher
/// graphs after parent/stores disappear. The record owns values, no optimizer,
/// moving/spectral store reference or fitting callback.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore g(6803),sn(0),moving(0);
/// auto out=n::models::BigGanGenerate(g,sn,moving,n::zeros({2,12}),n::zeros({2},n::DType::Int32));
/// // out.image is[2,8,8,1]; inference leaves its moment fields undefined.
/// @endcode
/// @par References
/// - Brock et al. (2018/2019), https://arxiv.org/abs/1809.11096 .
/// @see BigGanGenerate(), BigGanMoments
struct BigGanGenerated {
  Tensor image; ///< Owning live sigmoid pixels[N,8,8,1].
  BigGanMoments
      moments; ///< Owning batch moments, undefined tensor fields for inference-only calls.
};

/// @brief Generate class-conditional8x8 pixels from three explicit hierarchical latent chunks.
///
/// z[N,12] splits into z0,z1,z2 of width4. A raw normal(.1) class_embedding[10,4]
/// is shared by both conditional BNs. Spectral bias-free stem4->64 reshapes z0
/// to4x4x4. Conditional BN1 uses[embedding,z1], ReLU, nearest upsample to8x8,
/// spectral3x3 conv4->4, conditional BN2 using[embedding,z2], ReLU and spectral
/// 3x3 conv4->4. Shortcut is spectral bias-free1x1 on the stem, then nearest-up.
/// Sum -> captured output BN -> learned gamma/beta -> ReLU -> biased spectral
/// 3x3 conv4->1 -> sigmoid. Each conditional gain is spectral dense8->4 plus1;
/// each bias is a separate spectral dense8->4. Every /W kernel starts normal(.02),
/// with cached source-rounded constant u/v and a live signed sigma floor. The
/// raw G class embedding and output affine vectors are not spectrally normalized.
///
/// Creates class_embedding, stem/W, bn1 gain W,b then bias W,b, conv1/W, bn2
/// gain W,b then bias W,b, conv2/W, shortcut/W, output_bn gamma,beta, rgb W,b:
/// 789 values/17 leaves. G spectral state has241 values/18 leaves; BN saved state
/// has24 values/six leaves in bn1,bn2,output_bn mean/variance order. Batch mode
/// captures moments without writes; inference uses detached saved statistics.
/// No noise draw beyond missing leaf initialization, spectral advance, BN commit,
/// optimizer or fitting is hidden. Three distinct matching Float32/64 CPU stores,
/// finite signed noise[N,12], Int32/64 labels[N] in[0,10), positive N, strides and
/// representable products required. Applicable first/higher noise/weight paths
/// stay live; cached columns/saved BN paths stop. Inputs validate before leaves;
/// incompatible/nonfinite leaves may fail later. Inputs unchanged, outputs own
/// lifetime. Work/storage O(N) for this fixed small architecture, plus validation;
/// synchronize stores and exclude both state stores from Adam.
/// @param generator Mutable floating class-conditional generator owner.
/// @param spectral Distinct matching G spectral-column owner, unchanged after initialization.
/// @param moving Third distinct matching BN saved-moment owner, unchanged after initialization.
/// @param noise Finite signed matching floating CPU[N,12], N>0.
/// @param labels Discrete Int32/64 CPU[N] classes in[0,10), strides allowed.
/// @param batch_statistics Normalize/capture batch moments when true; saved statistics when
/// false(default).
/// @return Owning BigGanGenerated with image and optional live captured moments.
/// @throws std::invalid_argument For invalid inputs/stores/products, incompatible leaves or
/// nonfinite computation.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore g(6803),sn(0),moving(0);
/// auto z=n::zeros({2,12}),c=n::zeros({2},n::DType::Int32);
/// n::models::BigGanGenerate(g,sn,moving,z,c);n::advance_cached_spectral_state(g,sn,16);
/// auto out=n::models::BigGanGenerate(g,sn,moving,z,c,true);
/// // Use captured moments after the selected G optimizer write; scoring made no state write.
/// @endcode
/// @par References
/// - Brock, Donahue and Simonyan (2018/2019), https://arxiv.org/abs/1809.11096 .
///   Hierarchical chunks, shared embedding and conditional normalization are retained
///   in this compact dense/residual teaching variant, not a paper-scale benchmark.
/// @see BigGanDiscriminate(), BigGanCommitMoments(), capture_batch_norm(), BigGan()
inline BigGanGenerated BigGanGenerate(
    ParameterStore& generator,
    ParameterStore& spectral,
    ParameterStore& moving,
    const Tensor&   noise,
    const Tensor&   labels,
    bool            batch_statistics = false) {
  big_detail::Stores(generator, spectral, moving);
  gan_detail::Matrix(noise, 12, generator.dtype());
  if (noise.size(0) > std::numeric_limits<std::int64_t>::max() / 256)
    throw std::invalid_argument("BigGAN generator activation products overflow");
  gan_detail::Labels(labels, noise.size(0), 10);
  auto embedding = generator.get("class_embedding", {10, 4}, init::normal(.1));
  gan_detail::Finite(embedding);
  auto shared = matmul(cast(one_hot(cast(labels, DType::Int64), 10), noise.dtype()), embedding);
  auto z0 = slice_axis(noise, 1, 0, 4), z1 = slice_axis(noise, 1, 4, 8),
       z2 = slice_axis(noise, 1, 8, 12);
  auto x = reshape(big_detail::Dense(generator, spectral, "stem", z0, 4, 64, false), {-1, 4, 4, 4});
  auto condition1 = concat({shared, z1}, 1), condition2 = concat({shared, z2}, 1);
  BatchNormCapture first, second;
  auto             h = relu(
      big_detail::ConditionalNorm(
          generator, spectral, moving, "bn1", x, condition1, batch_statistics, first));
  h           = resize_nearest(h, {8, 8});
  auto hidden = big_detail::Conv(generator, spectral, "conv1", h, 4, 4, 3, false);
  hidden      = relu(
      big_detail::ConditionalNorm(
          generator, spectral, moving, "bn2", hidden, condition2, batch_statistics, second));
  hidden = big_detail::Conv(generator, spectral, "conv2", hidden, 4, 4, 3, false);
  auto shortcut =
      resize_nearest(big_detail::Conv(generator, spectral, "shortcut", x, 4, 4, 1, false), {8, 8});
  auto last  = capture_batch_norm(moving, "output_bn", hidden + shortcut, batch_statistics);
  auto gamma = generator.get("output_bn/gamma", {4}, init::ones()),
       beta  = generator.get("output_bn/beta", {4}, init::zeros());
  gan_detail::Finite(gamma);
  gan_detail::Finite(beta);
  auto output = relu(last.value * gamma + beta);
  auto image  = sigmoid(big_detail::Conv(generator, spectral, "rgb", output, 4, 1, 3));
  gan_detail::Finite(image);
  return {
      image,
      {first.mean,
       first.unbiased_variance,
       second.mean,
       second.unbiased_variance,
       last.mean,
       last.unbiased_variance}};
}

/// @brief Own raw discriminator scores, pooled features and their source/projection decomposition.
///
/// score/source/projection[N,1], features[N,4] retain storage and applicable
/// first/higher input/weight paths after parent/stores disappear. No sigmoid,
/// class posterior, optimizer or cached-store reference is retained.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore d(6823),sn(0);
/// auto out=n::models::BigGanDiscriminate(d,sn,n::zeros({2,8,8,1}),n::zeros({2},n::DType::Int32));
/// // score=source+projection; changing only labels changes the projection term.
/// @endcode
/// @par References
/// - Miyato and Koyama (2018), cGANs with Projection Discriminator,
///   https://arxiv.org/abs/1802.05637 .
/// @see BigGanDiscriminate(), BigGanOutput
struct BigGanDiscriminated {
  Tensor score;      ///< Owning live source+projection raw score[N,1].
  Tensor features;   ///< Owning live sum-pooled residual features[N,4].
  Tensor source;     ///< Owning live biased spectral linear source score[N,1].
  Tensor projection; ///< Owning live inner product of class embedding with features[N,1].
};

/// @brief Score image/class pairs using a residual down block and spectral class projection.
///
/// 8x8x1 -> biased spectral3x3 conv1->4, ReLU, biased spectral3x3 conv4->4,
/// 2x2 average pool; shortcut is biased spectral1x1 conv1->4 then average pool.
/// Sum, ReLU, spatial sum-pool yields features[N,4]. Score is spectral biased
/// dense4->1 plus sum(features*one_hot(labels)*spectral projection/W[10,4]).
/// All /W kernels start normal(.02), with the same stopped-column/live-sigma
/// guarded protocol as G. Creation order conv1 W,b; conv2 W,b; shortcut W,b;
/// score W,b; projection/W:241 parameters/nine leaves,77 cache values/ten leaves.
/// No G call, BN, sigmoid, optimizer, spectral advance or fit is hidden.
/// Finite signed matching Float32/64 CPU images[N,8,8,1], positive N, Int32/64
/// labels[N] in[0,10), strides and applicable first/higher continuous paths.
/// Distinct matching stores required. Input checks precede leaves; incompatible
/// leaves/nonfinite computation can fail later. Inputs unchanged, outputs own
/// lifetime. O(N) fixed-architecture work/storage plus validation; synchronize stores.
/// @param discriminator Mutable floating residual/projection weight owner.
/// @param spectral Distinct matching D cached-column owner, unchanged after initialization.
/// @param image Finite signed matching floating CPU[N,8,8,1], N>0.
/// @param labels Int32/64 CPU[N] requested classes in[0,10), strides allowed.
/// @return Owning BigGanDiscriminated, with raw score, features and two score terms.
/// @throws std::invalid_argument For invalid inputs/stores/products, incompatible leaves or
/// nonfinite computation.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore d(6823),sn(0);auto x=n::zeros({2,8,8,1}),c=n::zeros({2},n::DType::Int32);
/// n::models::BigGanDiscriminate(d,sn,x,c);n::advance_cached_spectral_state(d,sn,16);
/// auto out=n::models::BigGanDiscriminate(d,sn,x,c);
/// // out.features is[2,4]; no class prediction is implied by a projection score.
/// @endcode
/// @par References
/// - Miyato and Koyama (2018), https://arxiv.org/abs/1802.05637 .
/// - Brock et al. (2018/2019), https://arxiv.org/abs/1809.11096 . The compact
///   same-padded residual block and initialization follow this repository's source.
/// @see BigGanGenerate(), BigGan(), cached_spectral_kernel()
inline BigGanDiscriminated BigGanDiscriminate(
    ParameterStore& discriminator,
    ParameterStore& spectral,
    const Tensor&   image,
    const Tensor&   labels) {
  if (&discriminator == &spectral || discriminator.dtype() != spectral.dtype())
    throw std::invalid_argument("BigGAN D needs distinct matching stores");
  big_detail::Image(image, 8, discriminator.dtype());
  gan_detail::Labels(labels, image.size(0), 10);
  auto h = big_detail::Conv(discriminator, spectral, "conv1", image, 1, 4, 3);
  h      = big_detail::Conv(discriminator, spectral, "conv2", relu(h), 4, 4, 3);
  h      = avg_pool2d(h, 2, 2);
  auto shortcut =
      avg_pool2d(big_detail::Conv(discriminator, spectral, "shortcut", image, 1, 4, 1), 2, 2);
  auto features   = sum(relu(h + shortcut), {1, 2});
  auto source     = big_detail::Dense(discriminator, spectral, "score", features, 4, 1);
  auto table      = big_detail::Kernel(discriminator, spectral, "projection", {10, 4});
  auto embedding  = matmul(cast(one_hot(cast(labels, DType::Int64), 10), image.dtype()), table);
  auto projection = sum(features * embedding, {1}, true);
  auto score      = source + projection;
  gan_detail::Finite(score);
  return {score, features, source, projection};
}

/// @brief Penalize off-diagonal raw kernel Gram entries with connected zero partials for other
/// leaves.
///
/// For each existing prefix/W kernel, flatten all leading axes into rows M;
/// sum coefficient*sum(square((M^T*M)*(1-I))). Default coefficient=double(.5e-4f).
/// Diagonal entries are excluded; raw kernels, not their spectral normalized
/// versions, are penalized. Every other leaf contributes0*sum(value), preserving
/// defined zero partials even for embeddings/affine/bias leaves. Accumulation
/// follows ParameterStore creation order with the source Float32 mask and grouping.
/// Finite Float32/64 CPU leaves, nonempty store, rank>=2 /W kernels, positive
/// representable products and finite nonnegative coefficient required. Strides
/// and first/higher weight paths remain live. No leaves, optimizer, fitting or
/// state mutation. Owning scalar output retains lifetime. Work O(sum(rows*out^2)),
/// storage O(max(out^2)), plus validation; synchronize concurrent store writes.
/// @param parameters Floating raw parameter owner, read without mutation.
/// @param coefficient Nonnegative finite per-Gram-squared coefficient, default double(.5e-4f).
/// @return Owning live scalar modified orthogonal penalty, with connected zero non-kernel partials.
/// @throws std::invalid_argument For empty/invalid store, kernel geometry, coefficient or nonfinite
/// computation.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore
/// p(1);p.get("layer/W",{2,2},n::init::ones());p.get("bias",{2},n::init::zeros()); auto
/// penalty=n::models::ModifiedOrthogonalPenalty(p,.5);
/// // Gram off-diagonals are2: penalty==4; the bias derivative is defined zero.
/// @endcode
/// @par References
/// - Brock et al. (2018/2019), https://arxiv.org/abs/1809.11096 , modified
///   orthogonal regularization excluding diagonal Gram entries.
/// @see BigGan(), BigGanGenerate()
inline Tensor ModifiedOrthogonalPenalty(
    ParameterStore& parameters,
    double          coefficient = static_cast<double>(.5e-4f)) {
  if (parameters.size() == 0 || !std::isfinite(coefficient) || coefficient < 0)
    throw std::invalid_argument("Invalid modified orthogonal store or coefficient");
  for (const auto& key : parameters.names()) {
    auto value = parameters.at(key);
    gan_detail::Finite(value);
    if (key.size() >= 2 && key.substr(key.size() - 2) == "/W") {
      if (value.dim() < 2 ||
          value.shape().back() > std::numeric_limits<std::int64_t>::max() / value.shape().back())
        throw std::invalid_argument("Invalid modified orthogonal kernel geometry");
    }
  }
  Tensor total;
  for (const auto& key : parameters.names()) {
    auto   value = parameters.at(key);
    Tensor term;
    if (key.size() < 2 || key.substr(key.size() - 2) != "/W")
      term = 0. * sum(value);
    else {
      const auto         out = value.shape().back(), rows = value.numel() / out;
      auto               matrix = reshape(value, {rows, out});
      auto               gram   = matmul(matrix, matrix, true);
      std::vector<float> mask(static_cast<std::size_t>(out * out), 1.f);
      for (std::int64_t i = 0; i < out; ++i)
        mask[static_cast<std::size_t>(i * out + i)] = 0.f;
      term = coefficient * sum(square(gram * from_vector(mask, {out, out})));
    }
    total = total.defined() ? total + term : term;
  }
  gan_detail::Finite(total);
  return total;
}

/// @brief Commit all three captured BigGAN BN moment pairs only after every candidate validates.
///
/// Reads existing bn1,bn2,output_bn moving_mean/moving_var[4]. Computes detached
/// candidates with batch_norm_moving_values() and source retention double(.9f),
/// then copies all six results. No initialization, RNG, spectral advance,
/// gradient, optimizer or recomputation from updated weights. Use moments captured
/// in the SAME G forward that supplied the selected gradient; call after the G
/// Adam write. D phases do not call this function. Invalid/missing state or any
/// invalid/nonfinite candidate fails before writes. ParameterStore flags remain
/// unchanged; existing tensor handles observe copied values. State is excluded
/// from optimizers. O(1) work/storage for24 values plus validation. Synchronize
/// shared store mutation; this is not a concurrent transaction.
/// @param moving Mutable floating existing three-layer BN moment owner; only its six named vectors
/// are written.
/// @param captured Owning six-vector batch capture, matching dtype/shape and nonnegative variances.
/// @throws std::invalid_argument For invalid captured/saved moments or nonfinite candidates.
/// @throws std::out_of_range For any missing saved moment; no writes occur.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore g(6803),sn(0),moving(0);
/// auto
/// capture=n::models::BigGanGenerate(g,sn,moving,n::zeros({2,12}),n::zeros({2},n::DType::Int32),true);
/// // After an explicit G optimizer write, commit this pre-write forward's captures:
/// n::models::BigGanCommitMoments(moving,capture.moments);
/// @endcode
/// @par References
/// - Ioffe and Szegedy (2015), https://arxiv.org/abs/1502.03167 .
/// - Brock et al. (2018/2019), https://arxiv.org/abs/1809.11096 . The explicit
///   post-Adam update from captured unbiased moments follows this teaching schedule.
/// @see BigGanMoments, BigGanGenerate(), batch_norm_moving_values()
inline void BigGanCommitMoments(ParameterStore& moving, const BigGanMoments& captured) {
  const std::vector<std::string> names{"bn1", "bn2", "output_bn"};
  const std::vector<Tensor>      means{captured.bn1_mean, captured.bn2_mean, captured.output_mean},
      variances{
          captured.bn1_unbiased_variance,
          captured.bn2_unbiased_variance,
          captured.output_unbiased_variance};
  std::vector<BatchNormMovingValues> next;
  for (std::size_t k = 0; k < 3; ++k) {
    if (!means[k].defined() || means[k].shape() != Shape({4}) || means[k].dtype() != moving.dtype())
      throw std::invalid_argument("BigGAN moment captures must match four channels and dtype");
    next.push_back(batch_norm_moving_values(
        moving.at(names[k] + "/moving_mean"),
        moving.at(names[k] + "/moving_var"),
        means[k],
        variances[k]));
  }
  for (std::size_t k = 0; k < 3; ++k) {
    moving.set(names[k] + "/moving_mean", next[k].mean);
    moving.set(names[k] + "/moving_var", next[k].variance);
  }
}

/// @brief Own both generator paths, source diagnostics, explicit losses and batch moment captures.
///
/// All tensors and nested moments retain storage/applicable first/higher graphs
/// after parent/stores disappear. The selected fake/fake_score convenience fields
/// affect display only; D/G objectives ALWAYS use fake_training and its D score.
/// No optimizer, write counter, RNG, spectral/BN store reference or mode is retained.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore g(6803),d(6823),gs(0),ds(0),moving(0);
/// auto
/// out=n::models::BigGan(g,d,gs,ds,moving,n::zeros({2,8,8,1}),n::zeros({2,12}),n::zeros({2},n::DType::Int32));
/// // out.fake is inference display; out.g_loss still uses the batch-statistics fake.
/// @endcode
/// @par References
/// - Brock et al. (2018/2019), https://arxiv.org/abs/1809.11096 .
/// @see BigGan(), BigGanGenerated, BigGanDiscriminated
struct BigGanOutput {
  Tensor fake; ///< Owning live selected display pixels[N,8,8,1], inference by default.
  Tensor
      fake_training; ///< Owning live batch-statistics G pixels[N,8,8,1], used in both game losses.
  Tensor fake_inference;       ///< Owning live saved-statistics G pixels[N,8,8,1].
  Tensor fake_score_training;  ///< Owning live D score of training-path fake[N,1].
  Tensor fake_score_inference; ///< Owning live D score of saved-statistics fake[N,1].
  Tensor real_score;           ///< Owning live real image/class D scores[N,1].
  Tensor fake_score; ///< Owning live selected display score[N,1], matching fake convenience field.
  Tensor d_loss;     ///< Scalar mean(relu(1-real))+mean(relu(1+fake_score_training)).
  Tensor g_source;   ///< Scalar -mean(fake_score_training), without a generator hinge.
  Tensor g_orthogonal;  ///< Scalar modified raw G kernel penalty with source coefficient
                        ///< double(.5e-4f).
  Tensor        g_loss; ///< Scalar g_source+g_orthogonal, always using batch-statistics generation.
  BigGanMoments moments; ///< Owning live six captured moment vectors from fake_training's forward.
};

/// @brief Evaluate both compact hierarchical GAN paths and explicit hinge-plus-orthogonal
/// objectives.
///
/// Calls G inference then G batch capture, D real then training-fake then inference-fake,
/// HingeGanObjectives on real/training-fake, and ModifiedOrthogonalPenalty on raw G.
/// display_batch_statistics selects only fake/fake_score convenience endpoints;
/// all losses and moments ALWAYS come from the batch-statistics generator path,
/// even when the displayed image uses saved BN statistics. This preserves the
/// source evaluation convention; use the explicit path names when reporting.
/// Defaults create G789/17 leaves,D241/9,G-SN241/18,D-SN77/10,BN24/6:
/// 1030 optimized values,342 state values,60 tensors in five distinct stores.
///
/// Matching Float32/64 CPU stores, finite signed image[N,8,8,1]/noise[N,12],
/// positive matching N and Int32/64 labels[N] in[0,10) required. Strides and
/// applicable first/higher continuous paths remain live; class labels are discrete,
/// spectral columns/saved BN stop. No noise draw beyond missing leaves, spectral
/// advance, moment commit, optimizer, fitting, freeze or checkpoint is hidden.
/// Caller initializes, explicitly warms each spectral store16 times, then each
/// batch advances D spectral state, writes D, advances G spectral state, recomputes
/// through updated D, captures G moments, writes G and commits those captured
/// moments. D phases freeze every G state; G phases freeze every D state.
/// Input checks precede leaves; incompatible/nonfinite leaves may fail later.
/// Inputs unchanged, outputs own lifetime. O(N) fixed-architecture work/storage,
/// plus validation and raw-kernel Gram matrices. Synchronize shared stores;
/// exclude all three state stores from Adam. Approximate SN is not a norm certificate.
/// @param generator Mutable floating G weight owner.
/// @param discriminator Distinct matching D weight owner.
/// @param generator_spectral Third distinct matching G spectral-column owner.
/// @param discriminator_spectral Fourth distinct matching D spectral-column owner.
/// @param moving Fifth distinct matching G BN saved-moment owner.
/// @param image Finite signed matching floating CPU[N,8,8,1], N>0.
/// @param noise Finite signed matching floating CPU[N,12], same N as image.
/// @param labels Int32/64 CPU[N] classes in[0,10), supplied to both networks.
/// @param display_batch_statistics Select batch fake convenience fields when true; inference
/// display when false(default).
/// @return Owning BigGanOutput with both paths, explicit losses and live captured moments.
/// @throws std::invalid_argument For invalid inputs/stores/products/labels, incompatible leaves or
/// nonfinite computation.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore g(6803),d(6823),gs(0),ds(0),moving(0);
/// auto x=n::zeros({2,8,8,1}),z=n::zeros({2,12}),c=n::zeros({2},n::DType::Int32);
/// n::models::BigGan(g,d,gs,ds,moving,x,z,c);n::advance_cached_spectral_state(g,gs,16);n::advance_cached_spectral_state(d,ds,16);
/// auto out=n::models::BigGan(g,d,gs,ds,moving,x,z,c,true);
/// auto derivative=n::grad(out.g_loss,g.parameters());
/// // After the selected G optimizer write, commit out.moments; no write was hidden here.
/// @endcode
/// @par References
/// - Brock, Donahue and Simonyan (2018/2019), https://arxiv.org/abs/1809.11096 .
///   This compact8x8 residual network and captured-update ordering follow the
///   migrated teaching example rather than the paper's scale or benchmark quality.
/// @see BigGanGenerate(), BigGanDiscriminate(), BigGanCommitMoments(), ModifiedOrthogonalPenalty()
inline BigGanOutput BigGan(
    ParameterStore& generator,
    ParameterStore& discriminator,
    ParameterStore& generator_spectral,
    ParameterStore& discriminator_spectral,
    ParameterStore& moving,
    const Tensor&   image,
    const Tensor&   noise,
    const Tensor&   labels,
    bool            display_batch_statistics = false) {
  ParameterStore* stores[]{
      &generator, &discriminator, &generator_spectral, &discriminator_spectral, &moving};
  for (int i = 0; i < 5; ++i)
    for (int j = 0; j < i; ++j)
      if (stores[i] == stores[j] || stores[i]->dtype() != stores[j]->dtype())
        throw std::invalid_argument("BigGAN needs five distinct matching stores");
  big_detail::Image(image, 8, generator.dtype());
  gan_detail::Matrix(noise, 12, generator.dtype());
  if (image.size(0) != noise.size(0))
    throw std::invalid_argument("BigGAN image/noise rows must match");
  gan_detail::Labels(labels, noise.size(0), 10);
  auto inference = BigGanGenerate(generator, generator_spectral, moving, noise, labels, false);
  auto training  = BigGanGenerate(generator, generator_spectral, moving, noise, labels, true);
  auto real      = BigGanDiscriminate(discriminator, discriminator_spectral, image, labels);
  auto fake_training =
      BigGanDiscriminate(discriminator, discriminator_spectral, training.image, labels);
  auto fake_inference =
      BigGanDiscriminate(discriminator, discriminator_spectral, inference.image, labels);
  auto loss    = HingeGanObjectives(real.score, fake_training.score);
  auto penalty = ModifiedOrthogonalPenalty(generator);
  return {
      display_batch_statistics ? training.image : inference.image,
      training.image,
      inference.image,
      fake_training.score,
      fake_inference.score,
      real.score,
      display_batch_statistics ? fake_training.score : fake_inference.score,
      loss.discriminator,
      loss.generator,
      penalty,
      loss.generator + penalty,
      training.moments};
}
} // namespace datamunge::nn::models
