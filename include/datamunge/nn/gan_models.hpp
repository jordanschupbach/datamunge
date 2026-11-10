#pragma once
/// @file gan_models.hpp
/// @brief Explicit conditional GAN generation, stable logistic objectives and coarse pixel
/// features.
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn::models {
/// @brief Internal finite CPU, architecture-product and discrete-condition validation.
namespace gan_detail {
/// @brief Require nonempty finite floating CPU values, allowing scalar results.
/// @param value Defined Float32/64 CPU tensor with positive axes, or one scalar.
/// @throws std::invalid_argument For invalid dtype/axes or nonfinite values.
/// @throws std::logic_error For non-CPU storage.
inline void Finite(const Tensor& value) {
  if (!value.defined() || value.numel() < 1 ||
      (value.dtype() != DType::Float32 && value.dtype() != DType::Float64))
    throw std::invalid_argument("GAN values must be nonempty floating tensors");
  for (auto size : value.shape())
    if (size <= 0)
      throw std::invalid_argument("GAN axes must be positive");
  for (double x : value.contiguous().to_vector<double>())
    if (!std::isfinite(x))
      throw std::invalid_argument("GAN values must be finite");
}
/// @brief Validate explicit finite floating matrix geometry and computation dtype.
/// @param value Finite CPU[N,width] matrix, N>0.
/// @param width Positive required feature width.
/// @param type Required Float32/64 computation dtype.
/// @throws std::invalid_argument For invalid geometry/dtype/values.
/// @throws std::logic_error For non-CPU storage.
inline void Matrix(const Tensor& value, std::int64_t width, DType type) {
  if (width <= 0 || !value.defined() || value.dim() != 2 || value.size(0) <= 0 ||
      value.size(1) != width || value.dtype() != type)
    throw std::invalid_argument("GAN requires matching nonempty floating matrices");
  Finite(value);
}
/// @brief Validate positive dense architecture widths and their representable parameter products.
/// @param features Output/image coordinate count.
/// @param classes Number of discrete classes.
/// @param hidden Tanh hidden width.
/// @param noise Positive noise width, default 1 for discriminator-only checks.
/// @throws std::invalid_argument For nonpositive widths or int64 sum/product overflow.
inline void
Widths(std::int64_t features, std::int64_t classes, std::int64_t hidden, std::int64_t noise = 1) {
  const auto limit = std::numeric_limits<std::int64_t>::max();
  if (features <= 0 || classes <= 0 || hidden <= 0 || noise <= 0 || noise > limit - classes ||
      features > limit / hidden || classes > limit / hidden || noise + classes > limit / hidden)
    throw std::invalid_argument(
        "GAN widths and parameter products must be positive and representable");
}
/// @brief Validate representable sample counts for every dense intermediate.
/// @param rows Positive sample count N.
/// @param features Positive image/output width F, already checked by Widths().
/// @param classes Positive class count C, already checked by Widths().
/// @param hidden Positive hidden width H, already checked by Widths().
/// @param noise Positive noise width Z, already checked by Widths().
/// @throws std::invalid_argument For nonpositive rows or overflowing N*max(F,C+Z,H).
inline void Rows(
    std::int64_t rows,
    std::int64_t features,
    std::int64_t classes,
    std::int64_t hidden,
    std::int64_t noise) {
  if (rows <= 0 || rows > std::numeric_limits<std::int64_t>::max() /
                              std::max({features, classes + noise, hidden}))
    throw std::invalid_argument("GAN activation products must be positive and representable");
}
/// @brief Validate one nonnegative integer class per row before one-hot allocation or leaf
/// creation.
/// @param labels Int32/64 CPU[N] classes in[0,classes), strides allowed.
/// @param rows Required positive row count N.
/// @param classes Positive class count, with representable N*classes.
/// @throws std::invalid_argument For invalid geometry/dtype/range or product overflow.
/// @throws std::logic_error For non-CPU storage.
inline void Labels(const Tensor& labels, std::int64_t rows, std::int64_t classes) {
  if (rows <= 0 || classes <= 0 || rows > std::numeric_limits<std::int64_t>::max() / classes ||
      !labels.defined() || labels.shape() != Shape({rows}) ||
      (labels.dtype() != DType::Int32 && labels.dtype() != DType::Int64))
    throw std::invalid_argument("GAN requires one valid integer class per row");
  for (auto c : labels.contiguous().to_vector<std::int64_t>())
    if (c < 0 || c >= classes)
      throw std::invalid_argument("GAN class outside range");
}
} // namespace gan_detail

/// @brief Own live logistic discriminator and non-saturating generator scalar losses.
///
/// Plain Tensor handles own their graph/storage lifetimes; no optimizer, labels,
/// parameter store or training state is retained. Caller chooses update groups.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const auto out=dm::nn::models::LogisticGanObjectives(dm::nn::zeros({2,1}),dm::nn::zeros({3,1}));
/// // discriminator is 2*log(2), generator is log(2), with separate pool means.
/// @endcode
/// @see LogisticGanObjectives()
struct LogisticGanLoss {
  Tensor discriminator; ///< Owning scalar mean(softplus(-real))+mean(softplus(fake)).
  Tensor generator;     ///< Owning scalar mean(softplus(-fake)), non-saturating objective.
};

/// @brief Compute source-compatible stable logistic GAN losses from explicit raw logits.
///
/// Requires finite same-dtype Float32/64 CPU real[N,1] and fake[M,1], N,M>0;
/// unequal pool sizes and strides are allowed. Returns D=mean softplus(-real)+
/// mean softplus(fake), G=mean softplus(-fake), preserving operation grouping.
/// No sigmoid/log clipping or softplus(x)-x substitution loses small real-logit
/// derivatives. All applicable first/second paths remain live; D has both input
/// paths, G only fake. No fitting, store mutation or RNG. Owning scalar outputs;
/// inputs unchanged, synchronize external mutation. O(N+M) work/storage with
/// host validation; computed nonfinite losses are rejected.
/// @param real Finite matching floating CPU real logits[N,1], N>0.
/// @param fake Finite matching floating CPU fake logits[M,1], M>0.
/// @return Owning live discriminator/generator scalar objectives in input dtype.
/// @throws std::invalid_argument For invalid geometry/dtype/values or nonfinite computation.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const auto
/// loss=dm::nn::models::LogisticGanObjectives(dm::nn::full({1,1},80.),dm::nn::full({1,1},-80.));
/// // G remains approximately80, with a strong negative fake-logit gradient.
/// @endcode
/// @par References
/// - Goodfellow et al. (2014), Generative Adversarial Networks, https://arxiv.org/abs/1406.2661 .
///   Uses the non-saturating generator variant and explicit stable softplus grouping.
/// @see LogisticGanLoss, AuxiliaryClassifierGan()
inline LogisticGanLoss LogisticGanObjectives(const Tensor& real, const Tensor& fake) {
  if (!real.defined())
    throw std::invalid_argument("GAN real logits must be defined");
  gan_detail::Matrix(real, 1, real.dtype());
  gan_detail::Matrix(fake, 1, real.dtype());
  LogisticGanLoss out{mean(softplus(-real)) + mean(softplus(fake)), mean(softplus(-fake))};
  gan_detail::Finite(out.discriminator);
  gan_detail::Finite(out.generator);
  return out;
}

/// @brief Generate class-conditional sigmoid coordinates from explicit noise and integer labels.
///
/// Concatenates noise[N,Z] and one-hot classes[N,C], then dense hidden tanh H
/// and dense output sigmoid F. Creates/reuses hidden/W,b then output/W,b in
/// the caller's store, using the core dense Glorot/zero initialization. Default
/// Z=16 input with C=10,H=32,F=784 creates 4 leaves/26736 values. Z is inferred
/// from noise, not fixed. Finite signed Float32/64 CPU matrices must match store
/// dtype; labels are discrete Int32/64. All noise/weight paths and strides remain
/// live. No image, discriminator, fitting, optimizer or RNG draw beyond first
/// parameter creation is hidden. Output owns lifetime; inputs unchanged. Invalid
/// inputs fail before leaves; incompatible existing leaves or computed overflow
/// may fail later. Work O(N*((Z+C)*H+H*F)); activations O(N*(Z+C+H+F)).
/// @param generator Caller-owned floating parameter store, mutated only for missing leaves.
/// @param noise Finite signed CPU noise[N,Z], N,Z>0, matching store dtype.
/// @param labels Int32/64 CPU[N] class IDs in[0,classes).
/// @param features Positive output width F, default 784.
/// @param classes Positive class count C, default 10.
/// @param hidden Positive tanh hidden width H, default 32.
/// @return Owning live sigmoid probabilities[N,F], with continuous paths retained.
/// @throws std::invalid_argument For invalid widths/products, inputs, labels, stored leaves or
/// computed nonfinite values.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// dm::nn::ParameterStore g(5209);
/// const auto
/// image=dm::nn::models::ConditionalGanGenerate(g,dm::nn::zeros({2,16}),dm::nn::zeros({2},dm::nn::DType::Int32));
/// // image is [2,784]; g has 4 leaves/26736 values. No discriminator is created.
/// @endcode
/// @par References
/// - Odena, Olah and Shlens (2017), Conditional Image Synthesis With Auxiliary
///   Classifier GANs, https://arxiv.org/abs/1610.09585 . This is the compact source
///   MLP, not the paper's high-resolution architecture or benchmark.
/// @see AuxiliaryGanDiscriminate(), AuxiliaryClassifierGan()
inline Tensor ConditionalGanGenerate(
    ParameterStore& generator,
    const Tensor&   noise,
    const Tensor&   labels,
    std::int64_t    features = 784,
    std::int64_t    classes  = 10,
    std::int64_t    hidden   = 32) {
  if (!noise.defined() || noise.dim() != 2)
    throw std::invalid_argument("GAN noise must be a matrix");
  gan_detail::Widths(features, classes, hidden, noise.size(1));
  gan_detail::Matrix(noise, noise.size(1), generator.dtype());
  gan_detail::Rows(noise.size(0), features, classes, hidden, noise.size(1));
  gan_detail::Labels(labels, noise.size(0), classes);
  const auto condition = cast(one_hot(cast(labels, DType::Int64), classes), noise.dtype());
  auto       out       = dense(
      generator,
      "output",
      dense(generator, "hidden", concat({noise, condition}, 1), hidden, Activation::Tanh),
      features,
      Activation::Sigmoid);
  gan_detail::Finite(out);
  return out;
}

/// @brief Produce explicit 4x4 coarse pixel features for the original GAN game diagnostic.
///
/// Finite signed Float32/64 CPU samples[N,...784], N>0 and rank>=2, flatten
/// each row to28x28 grayscale. Applies7x7 average pooling with stride7, flattens
/// to[N,16], then divides by4, preserving the original Float32 grouping. Strides
/// and applicable first/second input paths survive; output owns lifetime. No
/// labels, fitted normalization, clipping, MMD computation or quality metric is
/// implicit. Use existing ImqMmd() separately; its signed value is not a quality
/// score. O(N*784) work/storage with host finite validation; input unchanged.
/// @param pixels Finite floating CPU samples with first axis N and trailing product784.
/// @return Owning live matching-dtype CPU[N,16] coarse features.
/// @throws std::invalid_argument For invalid geometry/dtype/values or nonfinite output.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const auto coarse=dm::nn::models::GanCoarsePixels(dm::nn::ones({2,784}));
/// // coarse is [2,16], all values approximately .25. No kernel or model is fitted.
/// @endcode
/// @see ImqMmd(), ImqKernel(), AuxiliaryClassifierGan()
inline Tensor GanCoarsePixels(const Tensor& pixels) {
  if (!pixels.defined() || pixels.dim() < 2 || pixels.size(0) <= 0 ||
      pixels.numel() / pixels.size(0) != 784)
    throw std::invalid_argument("GAN coarse features need 28x28 grayscale samples");
  gan_detail::Finite(pixels);
  auto out =
      reshape(
          avg_pool2d(reshape(pixels, {pixels.size(0), 28, 28, 1}), 7, 7), {pixels.size(0), 16}) /
      4.;
  gan_detail::Finite(out);
  return out;
}
} // namespace datamunge::nn::models
