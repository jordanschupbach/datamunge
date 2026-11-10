#pragma once
/// @file auxiliary_classifier_gan.hpp
/// @brief Image-only auxiliary discriminator and explicit class-conditional GAN objectives.
#include "datamunge/nn/gan_models.hpp"

namespace datamunge::nn::models {
/// @brief Own an image-only discriminator's source logits and auxiliary predictions.
///
/// Three Tensor handles retain storage and available gradients after this record
/// or its parameter store is destroyed. No target class, training state, fitting
/// or optimizer is retained. Positive source logits mean real; class probabilities
/// sum to one per image. Synchronize external mutation of shared tensor storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// dm::nn::ParameterStore d(5219);
/// const auto out=dm::nn::models::AuxiliaryGanDiscriminate(d,dm::nn::zeros({2,784}));
/// // source is [2,1], class_logits and class_probability are [2,10].
/// @endcode
/// @see AuxiliaryGanDiscriminate(), AuxiliaryClassifierGanOutput
struct AuxiliaryDiscriminator {
  Tensor source;            ///< Owning live linear real/fake logits [N,1].
  Tensor class_logits;      ///< Owning live linear auxiliary logits [N,classes].
  Tensor class_probability; ///< Owning live softmax probabilities [N,classes].
};

/// @brief Predict real/fake source and auxiliary classes from images alone.
///
/// Dense hidden tanh H feeds class logits C and a raw source logit. Creates
/// hidden/W,b, class/W,b, source/W,b in that order: the class head precedes the
/// source head in the initialization stream. Defaults create six leaves/25483
/// values. Images are finite signed Float32/64 CPU [N,F], matching store dtype;
/// strides and all continuous first/higher paths are retained. No class condition,
/// target, noise, sigmoid source probability, loss, RNG draw beyond missing-leaf
/// creation, fitting or update is hidden. Inputs remain unchanged; outputs own
/// lifetime. Invalid inputs fail before leaves; incompatible existing leaves or
/// nonfinite computation may fail later. Work O(N*H*(F+C+1)), activation storage
/// O(N*(H+C+1)); synchronize callers sharing a mutable store.
/// @param discriminator Caller-owned floating store, mutated only for missing leaves.
/// @param image Finite signed CPU matrix [N,features], N>0, matching store dtype.
/// @param features Positive image width F, default 784.
/// @param classes Positive auxiliary class count C, default 10.
/// @param hidden Positive tanh hidden width H, default 32.
/// @return Owning source logits, class logits and softmax class probabilities.
/// @throws std::invalid_argument For invalid widths/products, inputs, stored leaves or computed
/// nonfinite values.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore d(5219,n::DType::Float64);
/// const auto
/// out=n::models::AuxiliaryGanDiscriminate(d,n::full({2,3},-.2,n::DType::Float64),3,2,4);
/// // No labels enter D; out.class_probability has two columns.
/// @endcode
/// @par References
/// - Odena, Olah and Shlens (2017), Conditional Image Synthesis with Auxiliary
///   Classifier GANs, https://arxiv.org/abs/1610.09585 . This compact dense
///   teaching discriminator differs from the paper's convolutional architecture.
/// @see ConditionalGanGenerate(), AuxiliaryClassLoss(), AuxiliaryClassifierGan()
inline AuxiliaryDiscriminator AuxiliaryGanDiscriminate(
    ParameterStore& discriminator,
    const Tensor&   image,
    std::int64_t    features = 784,
    std::int64_t    classes  = 10,
    std::int64_t    hidden   = 32) {
  gan_detail::Widths(features, classes, hidden);
  gan_detail::Matrix(image, features, discriminator.dtype());
  gan_detail::Rows(image.size(0), features, classes, hidden, 1);
  auto                   h      = dense(discriminator, "hidden", image, hidden, Activation::Tanh);
  auto                   logits = dense(discriminator, "class", h, classes);
  auto                   source = dense(discriminator, "source", h, 1);
  AuxiliaryDiscriminator out{source, logits, softmax(logits)};
  gan_detail::Finite(out.source);
  gan_detail::Finite(out.class_logits);
  gan_detail::Finite(out.class_probability);
  return out;
}

/// @brief Average source-grouped categorical cross entropy against integer class targets.
///
/// Computes mean(-sum(one_hot(labels,C)*log_softmax(logits),axis=1)), preserving
/// the teaching source's operation grouping. Finite Float32/64 CPU [N,C] logits
/// and Int32/64 CPU [N] labels in [0,C) are required, N,C>0; strides are allowed.
/// No clipping, label smoothing, fitting, RNG or optimizer. First/higher logit
/// gradients remain live; labels are discrete. Inputs are unchanged, output owns
/// lifetime; synchronize external mutation. Work/storage O(N*C), including host
/// validation. Extreme finite logits whose arithmetic overflows are rejected.
/// @param logits Finite floating CPU class logits [N,C], N,C>0.
/// @param labels Integer CPU class targets [N], in [0,C).
/// @return Owning live scalar mean categorical cross entropy, in logits dtype.
/// @throws std::invalid_argument For invalid geometry, dtype, targets, product overflow or
/// nonfinite computation.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const auto
/// loss=dm::nn::models::AuxiliaryClassLoss(dm::nn::zeros({2,10}),dm::nn::zeros({2},dm::nn::DType::Int32));
/// // Uniform predictions give log(10), independent of the two target IDs.
/// @endcode
/// @par References
/// - Odena, Olah and Shlens (2017), https://arxiv.org/abs/1610.09585 ,
///   auxiliary class log likelihood, expressed here as a minimized mean loss.
/// @see AuxiliaryGanDiscriminate(), AuxiliaryClassifierGan()
inline Tensor AuxiliaryClassLoss(const Tensor& logits, const Tensor& labels) {
  if (!logits.defined() || logits.dim() != 2)
    throw std::invalid_argument("Auxiliary logits must be a matrix");
  gan_detail::Matrix(logits, logits.size(1), logits.dtype());
  gan_detail::Labels(labels, logits.size(0), logits.size(1));
  auto out = mean(-sum(
      cast(one_hot(cast(labels, DType::Int64), logits.size(1)), logits.dtype()) *
          log_softmax(logits),
      {1}));
  gan_detail::Finite(out);
  return out;
}

/// @brief Own fifteen live class-conditional GAN endpoints and source/class objectives.
///
/// Every Tensor owns storage and available autograd history independently of this
/// record and the caller's stores. No optimizer, random generator, mode switch,
/// fitting count, checkpoint selection or quadratic diagnostic is stored. Choose
/// update groups explicitly; no implicit detachment freezes either model.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore g(5209),d(5219);
/// const auto
/// out=n::models::AuxiliaryClassifierGan(g,d,n::zeros({2,784}),n::zeros({2,16}),n::zeros({2},n::DType::Int32));
/// // fake is [2,784]; the two training losses are scalar tensors.
/// @endcode
/// @see AuxiliaryClassifierGan(), AuxiliaryDiscriminator
struct AuxiliaryClassifierGanOutput {
  Tensor fake;                      ///< Live sigmoid generated coordinates [N,features].
  Tensor real_logit;                ///< Image-only real source logits [N,1].
  Tensor fake_logit;                ///< Generated-image source logits [N,1].
  Tensor real_probability;          ///< Real-image sigmoid source probabilities [N,1].
  Tensor fake_probability;          ///< Generated-image sigmoid source probabilities [N,1].
  Tensor discriminator_loss;        ///< Scalar source D loss plus (real CE + fake CE).
  Tensor generator_loss;            ///< Scalar non-saturating source G loss plus fake CE.
  Tensor real_class_logits;         ///< Image-only real auxiliary logits [N,classes].
  Tensor fake_class_logits;         ///< Generated-image auxiliary logits [N,classes].
  Tensor real_class_probability;    ///< Image-only real softmax predictions [N,classes].
  Tensor fake_class_probability;    ///< Generated-image softmax predictions [N,classes].
  Tensor real_class_loss;           ///< Scalar mean real-image CE against supplied targets.
  Tensor fake_class_loss;           ///< Scalar mean generated-image CE against requested classes.
  Tensor source_discriminator_loss; ///< Scalar mean softplus(-real) + mean softplus(fake).
  Tensor source_generator_loss;     ///< Scalar non-saturating mean softplus(-fake).
};

/// @brief Evaluate a conditional generator and image-only auxiliary discriminator with explicit
/// losses.
///
/// ConditionalGanGenerate() concatenates noise [N,Z] with one-hot labels [N,C],
/// then uses hidden tanh H and output sigmoid F. AuxiliaryGanDiscriminate() sees
/// only real or generated coordinates and predicts source plus class. Source
/// objectives come from LogisticGanObjectives(); D minimizes source_D+(CE_real+
/// CE_fake), G minimizes source_G+CE_fake, with unit class weights. The supplied
/// labels condition G and serve as CE targets; they never enter D's forward.
/// Defaults with Z=16 create G four leaves/26736 and D six leaves/25483 values,
/// in hidden/output and hidden/class/source order (each W then b), respectively.
///
/// All mathematical paths remain live. Select only D parameters for D loss,
/// then recompute through the updated D and select only G parameters for G loss.
/// There is no hidden optimizer, alternating write, fit, noise draw except initial
/// leaf creation, checkpoint, evaluation guard or O(N^2) MMD. Use GanCoarsePixels()
/// and ImqMmd() explicitly for the separate source game diagnostic. Finite signed
/// Float32/64 CPU image/noise matrices and two distinct matching-dtype stores are
/// required; integer labels are discrete. Strides and higher continuous gradients
/// are retained. All input/width checks precede any leaf creation; incompatible
/// existing leaves or nonfinite computed states may fail later. Inputs unchanged,
/// output fields own lifetime; synchronize shared stores. Work O(N*H*(Z+C+F))
/// plus the two O(N*H*(F+C)) discriminator branches; activation storage
/// O(N*(Z+C+H+F)). No claim of convergence or image quality is implied.
/// @param generator Mutable independent floating generator store.
/// @param discriminator Mutable independent floating discriminator store, matching dtype.
/// @param image Finite signed CPU real matrix [N,features], matching dtype.
/// @param noise Finite signed CPU explicit noise [N,Z], N,Z>0, matching dtype.
/// @param labels Int32/64 CPU [N] requested classes and real/fake CE targets, in [0,classes).
/// @param features Positive image/output width F, default 784.
/// @param classes Positive class count C, default 10.
/// @param hidden Positive tanh hidden width H, default 32.
/// @return Owning fifteen-field record with live endpoints and separate scalar objectives.
/// @throws std::invalid_argument For invalid inputs/geometry/products, aliased or mismatched
/// stores, incompatible leaves or nonfinite computation.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore g(5209),d(5219);
/// const auto
/// out=n::models::AuxiliaryClassifierGan(g,d,n::zeros({2,784}),n::zeros({2,16}),n::zeros({2},n::DType::Int32));
/// const auto dg=n::grad(out.discriminator_loss,d.parameters());
/// // Caller applies dg only to D, then recomputes before differentiating G loss.
/// @endcode
/// @par References
/// - Odena, Olah and Shlens (2017), Conditional Image Synthesis with Auxiliary
///   Classifier GANs, https://arxiv.org/abs/1610.09585 . This compact MLP teaching
///   variant uses the non-saturating source G loss, not the paper's architecture.
/// - Goodfellow et al. (2014), Generative Adversarial Networks,
///   https://arxiv.org/abs/1406.2661 .
/// @see ConditionalGanGenerate(), AuxiliaryGanDiscriminate(), AuxiliaryClassLoss(),
/// LogisticGanObjectives()
inline AuxiliaryClassifierGanOutput AuxiliaryClassifierGan(
    ParameterStore& generator,
    ParameterStore& discriminator,
    const Tensor&   image,
    const Tensor&   noise,
    const Tensor&   labels,
    std::int64_t    features = 784,
    std::int64_t    classes  = 10,
    std::int64_t    hidden   = 32) {
  if (&generator == &discriminator || generator.dtype() != discriminator.dtype())
    throw std::invalid_argument("Auxiliary GAN needs distinct matching-dtype stores");
  if (!noise.defined() || noise.dim() != 2)
    throw std::invalid_argument("Auxiliary GAN noise must be a matrix");
  gan_detail::Widths(features, classes, hidden, noise.size(1));
  gan_detail::Matrix(image, features, generator.dtype());
  gan_detail::Matrix(noise, noise.size(1), generator.dtype());
  if (image.size(0) != noise.size(0))
    throw std::invalid_argument("Auxiliary GAN image/noise row counts differ");
  gan_detail::Rows(image.size(0), features, classes, hidden, noise.size(1));
  gan_detail::Labels(labels, image.size(0), classes);
  auto fake      = ConditionalGanGenerate(generator, noise, labels, features, classes, hidden);
  auto real      = AuxiliaryGanDiscriminate(discriminator, image, features, classes, hidden);
  auto generated = AuxiliaryGanDiscriminate(discriminator, fake, features, classes, hidden);
  auto source    = LogisticGanObjectives(real.source, generated.source);
  auto real_ce   = AuxiliaryClassLoss(real.class_logits, labels),
       fake_ce   = AuxiliaryClassLoss(generated.class_logits, labels);
  AuxiliaryClassifierGanOutput out{
      fake,
      real.source,
      generated.source,
      sigmoid(real.source),
      sigmoid(generated.source),
      source.discriminator + (real_ce + fake_ce),
      source.generator + fake_ce,
      real.class_logits,
      generated.class_logits,
      real.class_probability,
      generated.class_probability,
      real_ce,
      fake_ce,
      source.discriminator,
      source.generator};
  gan_detail::Finite(out.discriminator_loss);
  gan_detail::Finite(out.generator_loss);
  return out;
}
} // namespace datamunge::nn::models
