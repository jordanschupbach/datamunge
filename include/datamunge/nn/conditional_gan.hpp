#pragma once
/// @file conditional_gan.hpp
/// @brief Class conditions on both compact GAN networks, with explicit live logistic objectives.
#include "datamunge/nn/dense_gan.hpp"

namespace datamunge::nn::models {
/// @brief Score image/class pairs with one-hot conditioning on the discriminator input.
///
/// Concatenates image[N,F] then one-hot labels[N,C] and delegates the canonical
/// MlpGanDiscriminate() tanh/linear map. Creates hidden/W[F+C,H],b[H],
/// output/W[H,1],b[1] in source order. F784,C10,H32 creates four leaves/25473
/// values. Both image and discrete class condition enter D; there is no auxiliary
/// classifier, target CE, hidden G/E call, sigmoid, fit or optimizer. Finite signed
/// Float32/64 CPU images match store dtype, with positive axes and Int32/64
/// labels in[0,C). Strides and applicable higher continuous image/weight paths
/// stay live; labels are discrete. All input checks precede leaves; incompatible
/// leaves/nonfinite output may fail later. Inputs unchanged; output owns lifetime.
/// O(N*H*(F+C)) work, O(N*(F+C+H)) activations plus host validation; synchronize
/// callers sharing a store. Raw scores are unbounded source logits.
/// @param discriminator Mutable matching floating conditional source store.
/// @param image Finite signed CPU[N,features], N>0, matching store dtype.
/// @param labels Int32/64 CPU[N] requested classes in[0,classes).
/// @param features Positive input width F, default784.
/// @param classes Positive class count C, default10.
/// @param hidden Positive hidden tanh width H, default32.
/// @return Owning live conditional source logits[N,1], without a class prediction.
/// @throws std::invalid_argument For invalid inputs/labels/dimensions/products, incompatible leaves
/// or nonfinite computation.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore d(5113);
/// const auto
/// score=n::models::ConditionalGanDiscriminate(d,n::zeros({2,784}),n::zeros({2},n::DType::Int32));
/// // score is[2,1]; changing only labels can change real-image source predictions.
/// @endcode
/// @par References
/// - Mirza and Osindero (2014), Conditional Generative Adversarial Nets,
///   https://arxiv.org/abs/1411.1784 . Both networks are conditioned here;
///   this compact dense source architecture is not a reproduced paper benchmark.
/// @see ConditionalGanGenerate(), ConditionalGan(), AuxiliaryGanDiscriminate()
inline Tensor ConditionalGanDiscriminate(
    ParameterStore& discriminator,
    const Tensor&   image,
    const Tensor&   labels,
    std::int64_t    features = 784,
    std::int64_t    classes  = 10,
    std::int64_t    hidden   = 32) {
  gan_detail::Widths(features, classes, hidden, features);
  gan_detail::Matrix(image, features, discriminator.dtype());
  gan_detail::Rows(image.size(0), features, classes, hidden, features);
  dense_gan_detail::Geometry(image.size(0), features + classes, 1, hidden);
  gan_detail::Labels(labels, image.size(0), classes);
  auto condition = cast(one_hot(cast(labels, DType::Int64), classes), image.dtype());
  return MlpGanDiscriminate(
      discriminator, concat({image, condition}, 1), features + classes, hidden);
}

/// @brief Own seven live conditional GAN endpoints with separate logistic objectives.
///
/// Fields retain storage and applicable gradients after this record/stores are
/// destroyed. Source probabilities are explicit sigmoid reporting fields; D's
/// losses consume raw logits. No class predictions, optimizer, RNG, training mode,
/// checkpoint, fitting counter or quadratic MMD diagnostic is retained.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore g(5101),d(5113);
/// const auto
/// out=n::models::ConditionalGan(g,d,n::zeros({2,784}),n::zeros({2,16}),n::zeros({2},n::DType::Int32));
/// // fake is[2,784]; real/fake logits and source probabilities are[2,1].
/// @endcode
/// @see ConditionalGan(), ConditionalGanDiscriminate(), DenseGanOutput
struct ConditionalGanOutput {
  Tensor fake;               ///< Owning live class/noise-conditioned sigmoid coordinates[N,F].
  Tensor real_logit;         ///< Owning live real image/class source logits[N,1].
  Tensor fake_logit;         ///< Owning live generated image/class source logits[N,1].
  Tensor real_probability;   ///< Owning live sigmoid real source probabilities[N,1].
  Tensor fake_probability;   ///< Owning live sigmoid generated source probabilities[N,1].
  Tensor discriminator_loss; ///< Scalar mean softplus(-real_logit)+mean softplus(fake_logit).
  Tensor generator_loss;     ///< Scalar non-saturating mean softplus(-fake_logit).
};

/// @brief Evaluate a class-conditioned generator and discriminator with explicit logistic losses.
///
/// ConditionalGanGenerate() concatenates noise[N,Z] with one-hot classes[N,C],
/// then hidden tanh H/output sigmoid F. ConditionalGanDiscriminate() concatenates
/// image F with the same class condition and predicts one raw source logit. Both
/// real and generated D branches consume labels; there is no auxiliary class
/// head or categorical CE. Default Z16,C10,H32,F784 creates G26736+D25473=52209
/// values in eight leaves, each store's hidden then output W,b, in G,D order.
/// Source objectives are LogisticGanObjectives(), preserving stable softplus(-x)
/// grouping and non-saturating G. All mathematical paths remain live: select
/// only D leaves, write D, recompute through the updated D, then select G leaves.
///
/// Requires distinct stores and finite signed CPU image[N,F]/noise[N,Z] sharing
/// Float32/64 dtype; the single Int32/64[N] class vector conditions both pools,
/// so row counts must match. Positive widths/products, strides and applicable
/// higher continuous paths are supported; labels are discrete. No noise draw
/// except initial leaf creation, optimizer, fitting, implicit freeze, evaluation
/// guard, checkpoint or O(N^2) MMD is hidden. Source sigmoid probabilities are
/// reporting fields, not a quality or requested-class accuracy score. All input
/// checks precede leaf creation; incompatible leaves/nonfinite output may fail
/// later. Inputs unchanged, outputs own lifetime. Work O(N*H*(Z+C+F)); activation
/// storage O(N*(Z+C+F+H)), plus host validation. Synchronize shared stores.
/// @param generator Mutable independent floating conditional generator store.
/// @param discriminator Mutable independent floating conditional D store, matching G dtype.
/// @param image Finite signed CPU[N,features] real coordinates, N>0, matching dtype.
/// @param noise Finite signed CPU[N,Z] explicit codes, Z>0, matching dtype.
/// @param labels Int32/64 CPU[N] classes in[0,classes), supplied to both G and D.
/// @param features Positive image/output width F, default784.
/// @param classes Positive discrete class count C, default10.
/// @param hidden Positive hidden tanh width H, default32.
/// @return Owning seven-field ConditionalGanOutput with two explicit scalar objectives.
/// @throws std::invalid_argument For invalid inputs/options/products, aliased or mismatched stores,
/// incompatible leaves or nonfinite computation.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore g(5101),d(5113);
/// const auto
/// out=n::models::ConditionalGan(g,d,n::zeros({2,784}),n::zeros({2,16}),n::zeros({2},n::DType::Int32));
/// const auto dd=n::grad(out.discriminator_loss,d.parameters());
/// // Caller writes only D, then recomputes before selecting G's gradient.
/// @endcode
/// @par References
/// - Mirza and Osindero (2014), Conditional Generative Adversarial Nets,
///   https://arxiv.org/abs/1411.1784 . This compact dense teaching model uses
///   non-saturating logistic G loss and its own explicit source schedule.
/// @see ConditionalGanGenerate(), ConditionalGanDiscriminate(), LogisticGanObjectives(), DenseGan()
inline ConditionalGanOutput ConditionalGan(
    ParameterStore& generator,
    ParameterStore& discriminator,
    const Tensor&   image,
    const Tensor&   noise,
    const Tensor&   labels,
    std::int64_t    features = 784,
    std::int64_t    classes  = 10,
    std::int64_t    hidden   = 32) {
  if (&generator == &discriminator || generator.dtype() != discriminator.dtype())
    throw std::invalid_argument("Conditional GAN needs distinct matching-dtype stores");
  if (!noise.defined() || noise.dim() != 2)
    throw std::invalid_argument("Conditional GAN noise must be a matrix");
  gan_detail::Widths(features, classes, hidden, noise.size(1));
  gan_detail::Matrix(image, features, generator.dtype());
  gan_detail::Matrix(noise, noise.size(1), generator.dtype());
  if (image.size(0) != noise.size(0))
    throw std::invalid_argument("Conditional GAN image/noise rows must match");
  // Validate the concatenated D architecture before any G leaf can be created.
  gan_detail::Widths(features, classes, hidden, features);
  dense_gan_detail::Geometry(image.size(0), features + classes, 1, hidden);
  gan_detail::Rows(image.size(0), features, classes, hidden, noise.size(1));
  gan_detail::Labels(labels, image.size(0), classes);
  auto fake = ConditionalGanGenerate(generator, noise, labels, features, classes, hidden);
  auto real = ConditionalGanDiscriminate(discriminator, image, labels, features, classes, hidden),
       generated =
           ConditionalGanDiscriminate(discriminator, fake, labels, features, classes, hidden);
  auto loss = LogisticGanObjectives(real, generated);
  return {
      fake, real, generated, sigmoid(real), sigmoid(generated), loss.discriminator, loss.generator};
}
} // namespace datamunge::nn::models
