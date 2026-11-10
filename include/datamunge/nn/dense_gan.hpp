#pragma once
/// @file dense_gan.hpp
/// @brief Explicit compact dense GAN maps and raw-score logistic, least-squares and hinge losses.
#include "datamunge/nn/adversarial_autoencoder.hpp"
#include "datamunge/nn/gan_models.hpp"

namespace datamunge::nn::models {
/// @brief Internal positive dense-GAN geometry and raw-score pair validation.
namespace dense_gan_detail {
/// @brief Require positive representable architecture and activation products.
/// @param rows Positive real or generated pool size.
/// @param features Positive data width F.
/// @param latent Positive input/noise width Z.
/// @param hidden Positive hidden width H.
/// @throws std::invalid_argument For nonpositive dimensions or int64 sum/product overflow.
inline void
Geometry(std::int64_t rows, std::int64_t features, std::int64_t latent, std::int64_t hidden) {
  const auto limit = std::numeric_limits<std::int64_t>::max();
  if (rows <= 0 || features <= 0 || latent <= 0 || hidden <= 0 || features > limit - latent ||
      features + latent > limit / hidden || rows > limit / std::max(features + latent, hidden))
    throw std::invalid_argument("Dense GAN dimensions/products must be positive and representable");
}
/// @brief Validate finite matching floating source-column pools, allowing unequal row counts.
/// @param real Nonempty finite Float32/64 CPU[N,1] scores.
/// @param fake Nonempty finite matching CPU[M,1] scores.
/// @throws std::invalid_argument For invalid dtype, geometry or values.
/// @throws std::logic_error For non-CPU storage.
inline void Scores(const Tensor& real, const Tensor& fake) {
  if (!real.defined())
    throw std::invalid_argument("Dense GAN real scores must be defined");
  gan_detail::Matrix(real, 1, real.dtype());
  gan_detail::Matrix(fake, 1, real.dtype());
}
} // namespace dense_gan_detail

/// @brief Choose the raw-source objective of a dense GAN without selecting an optimizer.
///
/// This changes only the two losses, not architecture, parameter names, input
/// normalization or fitting. Values outside the three named enumerators are
/// rejected by DenseGan() before creating leaves. No probability calibration is
/// implied by a raw score; explicitly apply sigmoid only for logistic reporting.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const auto objective=dm::nn::models::DenseGanObjective::LeastSquares;
/// // This selects a=0,b=c=1 squared losses, without changing any weights.
/// @endcode
/// @see DenseGan(), LeastSquaresGanObjectives(), HingeGanObjectives()
enum class DenseGanObjective {
  Logistic,     ///< Stable logistic D and non-saturating fake-as-real G objectives.
  LeastSquares, ///< Squared raw-score targets a=0,b=c=1, with half-weighted means.
  Hinge         ///< D margin hinge with zero boundary subgradient; G is negative mean raw score.
};

/// @brief Own live discriminator/generator scalar objectives without fitting state.
///
/// Tensor handles retain storage and applicable graph history after this record
/// is destroyed. The formula is selected by the producing function; no store,
/// optimizer, labels or RNG is retained. Synchronize external tensor mutation.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const auto
/// loss=dm::nn::models::LeastSquaresGanObjectives(dm::nn::ones({2,1}),dm::nn::zeros({3,1}));
/// // loss.discriminator==0; loss.generator==.5, each pool averaged separately.
/// @endcode
/// @see LeastSquaresGanObjectives(), HingeGanObjectives(), DenseGanOutput
struct DenseGanLoss {
  Tensor discriminator; ///< Owning live scalar source-separation objective.
  Tensor generator;     ///< Owning live scalar generated-source objective.
};

/// @brief Compute source-grouped least-squares GAN losses with a=0 and b=c=1.
///
/// D=.5*(mean((real-1)^2)+mean(fake^2)); G=.5*mean((fake-1)^2).
/// The half factor follows the sum of pool means in D, preserving source
/// Float32 operation grouping. Raw scores are unbounded, not probabilities;
/// no sigmoid, clipping, cross entropy, smoothing or fitted threshold is applied.
/// Finite same-dtype Float32/64 CPU[N,1]/[M,1] scores allow unequal positive
/// pool sizes and strides. All applicable first/higher continuous paths remain
/// live; G depends only on fake. Inputs unchanged, scalar outputs own lifetime.
/// O(N+M) work/storage plus host validation; no global state or mutation. Finite
/// inputs whose square/reduction overflows are rejected rather than repaired.
/// @param real Finite floating CPU real-score column[N,1], N>0.
/// @param fake Finite matching floating CPU fake-score column[M,1], M>0.
/// @return Owning DenseGanLoss in input dtype, with the two explicit scalar objectives.
/// @throws std::invalid_argument For invalid pools/dtypes/values or nonfinite computed losses.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const auto
/// loss=dm::nn::models::LeastSquaresGanObjectives(dm::nn::zeros({2,1}),dm::nn::zeros({3,1}));
/// // D=.5 and G=.5, without a sigmoid on either score pool.
/// @endcode
/// @par References
/// - Mao et al. (2017), Least Squares Generative Adversarial Networks,
///   https://arxiv.org/abs/1611.04076 , equation9 (0/1 coding), not the
///   a=-1,b=1,c=0 parameterization used for the Pearson-divergence derivation.
/// @see HingeGanObjectives(), LogisticGanObjectives(), DenseGan()
inline DenseGanLoss LeastSquaresGanObjectives(const Tensor& real, const Tensor& fake) {
  dense_gan_detail::Scores(real, fake);
  DenseGanLoss out{
      .5 * (mean(square(real - 1.)) + mean(square(fake))), .5 * mean(square(fake - 1.))};
  gan_detail::Finite(out.discriminator);
  gan_detail::Finite(out.generator);
  return out;
}

/// @brief Compute positive D margin hinges and an unhinged negative-mean G objective.
///
/// D=mean(relu(1-real))+mean(relu(1+fake)); G=-mean(fake). Scores are raw,
/// unbounded and need not lie in[-1,1]; G loss may be negative. D's real-score
/// derivative is -1/N for real<1 and0 otherwise; fake derivative is1/M for
/// fake>-1 and0 otherwise. Exact margins real=1/fake=-1 use the native ReLU
/// zero subgradient. G derivative remains -1/M at every score, including the
/// margins: there is no hinge or sigmoid on G. First/higher paths are retained
/// away from D's kinks; a classical second derivative at the kink is undefined.
/// Finite matching Float32/64 CPU[N,1]/[M,1] pools may differ in positive size
/// and have strides. No clipping, spectral normalization, fit, optimizer or
/// threshold is hidden. Inputs unchanged, outputs own lifetime. O(N+M)
/// work/storage plus host validation; synchronize external mutation. Computed
/// nonfinite losses are rejected without repair.
/// @param real Finite floating CPU real scores[N,1], N>0.
/// @param fake Finite matching floating CPU fake scores[M,1], M>0.
/// @return Owning live DenseGanLoss with separate scalar D/G losses in input dtype.
/// @throws std::invalid_argument For invalid pools/dtypes/values or nonfinite computation.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const auto loss=dm::nn::models::HingeGanObjectives(dm::nn::ones({2,1}),dm::nn::full({3,1},-1.));
/// // D=0 at both margins, G=1; G's fake-score derivative is still -1/3.
/// @endcode
/// @par References
/// - Miyato et al. (2018), Spectral Normalization for Generative Adversarial
///   Networks, https://arxiv.org/abs/1802.05957 , equations16-17, with D's
///   maximized negative hinge expressed as a minimized positive loss. This
///   function supplies the loss only, without the paper's spectral normalization.
/// @see LeastSquaresGanObjectives(), LogisticGanObjectives(), DenseGan()
inline DenseGanLoss HingeGanObjectives(const Tensor& real, const Tensor& fake) {
  dense_gan_detail::Scores(real, fake);
  DenseGanLoss out{mean(relu(1. - real)) + mean(relu(1. + fake)), -mean(fake)};
  gan_detail::Finite(out.discriminator);
  gan_detail::Finite(out.generator);
  return out;
}

/// @brief Generate sigmoid data coordinates from explicit noise through the source dense MLP.
///
/// Reuses canonical AaeDecode() tanh/linear computation and applies sigmoid.
/// Noise width Z is inferred. Creates hidden/W[Z,H],b[H],output/W[H,F],b[F]
/// in that order with dense Glorot/zero initialization; Z16,H32,F784 creates
/// four leaves/26416 values. Finite signed Float32/64 CPU[M,Z] matches store
/// dtype; positive axes and strides preserve first/higher continuous paths.
/// No data, class, D, optimizer, fitting or input RNG draw is hidden. Validation
/// precedes leaves; incompatible leaves/nonfinite output may fail later. Input
/// unchanged; output owns lifetime. O(M*H*(Z+F)) work, O(M*(H+F)) activations
/// plus host validation; synchronize shared stores.
/// @param generator Mutable matching floating generator store.
/// @param noise Finite signed CPU[M,Z] explicit codes, M,Z>0, matching dtype.
/// @param features Positive output width F, default784.
/// @param hidden Positive hidden tanh width H, default32.
/// @return Owning live sigmoid coordinates[M,F], not random pixel draws.
/// @throws std::invalid_argument For invalid inputs/widths/products, incompatible leaves or
/// nonfinite output.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// dm::nn::ParameterStore g(5407);
/// const auto image=dm::nn::models::MlpGanGenerate(g,dm::nn::zeros({2,16}));
/// // image is[2,784]; G has four leaves/26416 values and requires no class label.
/// @endcode
/// @par References
/// - The migrated compact dense teaching architecture; AaeDecode() supplies its
///   shared tanh/linear implementation, not a paper's convolutional benchmark.
/// @see MlpGanDiscriminate(), DenseGan(), ConditionalGanGenerate()
inline Tensor MlpGanGenerate(
    ParameterStore& generator,
    const Tensor&   noise,
    std::int64_t    features = 784,
    std::int64_t    hidden   = 32) {
  if (!noise.defined() || noise.dim() != 2)
    throw std::invalid_argument("Dense GAN noise must be a matrix");
  dense_gan_detail::Geometry(noise.size(0), features, noise.size(1), hidden);
  gan_detail::Matrix(noise, noise.size(1), generator.dtype());
  auto out = sigmoid(AaeDecode(generator, noise, features, noise.size(1), hidden));
  gan_detail::Finite(out);
  return out;
}

/// @brief Predict an unbounded source score from images alone with the source tanh/linear MLP.
///
/// Reuses AaeDecode() with output width1. Creates hidden/W[F,H],b[H],
/// output/W[H,1],b[1] in that order; F784,H32 gives four leaves/25153 values.
/// Finite signed Float32/64 CPU[N,F] matches store dtype; positive axes and
/// strides retain input/parameter first/higher paths. No class, G, sigmoid,
/// fit, optimizer or target is hidden. Input checks precede leaves; incompatible
/// leaves/nonfinite output may fail later. Input unchanged; owning[N,1] output.
/// O(N*H*F) work, O(N*H) activations plus host validation; synchronize stores.
/// @param discriminator Mutable matching floating source-score store.
/// @param image Finite signed CPU[N,features], N>0, matching store dtype.
/// @param features Positive input width F, default784.
/// @param hidden Positive hidden tanh width H, default32.
/// @return Owning live raw scores[N,1], without a probability interpretation.
/// @throws std::invalid_argument For invalid inputs/widths/products, incompatible leaves or
/// nonfinite computation.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// dm::nn::ParameterStore d(5413);
/// const auto score=dm::nn::models::MlpGanDiscriminate(d,dm::nn::zeros({2,784}));
/// // score is[2,1]; no sigmoid, class condition or generated input is supplied.
/// @endcode
/// @par References
/// - The migrated compact teaching MLP; canonical AaeDecode() supplies its raw
///   tanh/linear computation. Scores are not calibrated class probabilities.
/// @see MlpGanGenerate(), DenseGan(), ConditionalGanDiscriminate()
inline Tensor MlpGanDiscriminate(
    ParameterStore& discriminator,
    const Tensor&   image,
    std::int64_t    features = 784,
    std::int64_t    hidden   = 32) {
  if (!image.defined() || image.dim() != 2)
    throw std::invalid_argument("Dense GAN images must be a matrix");
  dense_gan_detail::Geometry(image.size(0), features, 1, hidden);
  gan_detail::Matrix(image, features, discriminator.dtype());
  auto out = AaeDecode(discriminator, image, 1, features, hidden);
  gan_detail::Finite(out);
  return out;
}

/// @brief Own five live dense-GAN endpoints without an optimizer or experiment state.
///
/// Fields retain storage and applicable autograd history after this record or
/// its stores are destroyed. Raw scores have no implicit sigmoid or calibration.
/// No RNG, fitting counter, checkpoint or quadratic diagnostic is retained.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore g(5407),d(5413);
/// const auto
/// out=n::models::DenseGan(g,d,n::zeros({2,784}),n::zeros({3,16}),n::models::DenseGanObjective::LeastSquares);
/// // fake is[3,784]; real_score has2 rows and fake_score has3.
/// @endcode
/// @see DenseGan(), DenseGanLoss
struct DenseGanOutput {
  Tensor fake;               ///< Live sigmoid generated coordinates[M,F].
  Tensor real_score;         ///< Live unbounded real source scores[N,1].
  Tensor fake_score;         ///< Live unbounded generated source scores[M,1].
  Tensor discriminator_loss; ///< Live scalar selected source D objective.
  Tensor generator_loss;     ///< Live scalar selected source G objective.
};

/// @brief Evaluate explicit unconditional dense-GAN data/noise with a chosen raw-source objective.
///
/// MlpGanGenerate() maps noise[M,Z] to sigmoid coordinates F; the same
/// MlpGanDiscriminate() scores real[N,F] and fake[M,F]. Separate G/D stores
/// create hidden/output W,b in source order, using dense Glorot/zero initialization.
/// Defaults Z16,H32,F784 give G26416+D25153=51569 values in eight leaves.
/// Positive N/M may differ; each loss averages its own pool. Labels are absent.
///
/// Logistic selects LogisticGanObjectives(); LeastSquares selects a0,b=c1;
/// Hinge selects positive D margins and unhinged -mean(fake). Raw scores are
/// returned for every objective; any logistic reporting sigmoid is explicit.
/// All mathematical paths stay live, with hinge derivatives piecewise-defined
/// and zero ReLU subgradient at its margins. Caller selects only D leaves,
/// writes D, recomputes through the updated D and selects only G leaves. No
/// optimizer, fitting, noise draw except initial leaf creation, implicit freeze,
/// checkpoint or O(N^2) diagnostic is hidden. Use GanCoarsePixels()/ImqMmd()
/// explicitly for the source game diagnostic, without a quality guarantee.
///
/// Distinct stores and finite signed CPU inputs share Float32/64 dtype; strides
/// and applicable higher paths are retained. All input/options checks precede
/// leaves; incompatible leaves/nonfinite computed output may fail later. Inputs
/// unchanged, outputs own lifetime. Work O((N+M)*H*F+M*H*Z), activation storage
/// O((N+M)*(F+H)+M*Z), plus host validation. Synchronize shared stores.
/// @param generator Mutable independent floating generator store G.
/// @param discriminator Mutable independent floating score store D, matching G dtype.
/// @param image Finite signed CPU[N,features] real coordinates, N>0, matching dtype.
/// @param noise Finite signed CPU[M,Z] explicit codes, M,Z>0, matching dtype.
/// @param objective Required valid DenseGanObjective; no default objective is selected.
/// @param features Positive image/output width F, default784.
/// @param hidden Positive hidden tanh width H, default32.
/// @return Owning DenseGanOutput with raw scores and two selected scalar losses.
/// @throws std::invalid_argument For invalid inputs/geometry/objective, aliased or mismatched
/// stores, incompatible leaves or nonfinite computation.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore g(5701),d(5711);
/// const auto
/// out=n::models::DenseGan(g,d,n::zeros({2,784}),n::zeros({2,16}),n::models::DenseGanObjective::Hinge);
/// const auto dd=n::grad(out.discriminator_loss,d.parameters());
/// // Apply dd only to D, then recompute before differentiating G's loss.
/// @endcode
/// @par References
/// - Mao et al. (2017), https://arxiv.org/abs/1611.04076 , equation9.
/// - Miyato et al. (2018), https://arxiv.org/abs/1802.05957 , equations16-17.
///   These compact teaching models preserve source losses, not paper architectures,
///   spectral normalization, benchmark results or convergence claims.
/// @see LeastSquaresGanObjectives(), HingeGanObjectives(), LogisticGanObjectives(),
/// ConditionalGan()
inline DenseGanOutput DenseGan(
    ParameterStore&   generator,
    ParameterStore&   discriminator,
    const Tensor&     image,
    const Tensor&     noise,
    DenseGanObjective objective,
    std::int64_t      features = 784,
    std::int64_t      hidden   = 32) {
  if (objective != DenseGanObjective::Logistic && objective != DenseGanObjective::LeastSquares &&
      objective != DenseGanObjective::Hinge)
    throw std::invalid_argument("Invalid dense GAN objective");
  if (&generator == &discriminator || generator.dtype() != discriminator.dtype())
    throw std::invalid_argument("Dense GAN needs distinct matching-dtype stores");
  if (!noise.defined() || noise.dim() != 2)
    throw std::invalid_argument("Dense GAN noise must be a matrix");
  dense_gan_detail::Geometry(noise.size(0), features, noise.size(1), hidden);
  gan_detail::Matrix(image, features, generator.dtype());
  gan_detail::Matrix(noise, noise.size(1), generator.dtype());
  dense_gan_detail::Geometry(image.size(0), features, noise.size(1), hidden);
  auto fake      = MlpGanGenerate(generator, noise, features, hidden),
       real      = MlpGanDiscriminate(discriminator, image, features, hidden),
       generated = MlpGanDiscriminate(discriminator, fake, features, hidden);
  DenseGanLoss loss;
  if (objective == DenseGanObjective::LeastSquares)
    loss = LeastSquaresGanObjectives(real, generated);
  else if (objective == DenseGanObjective::Hinge)
    loss = HingeGanObjectives(real, generated);
  else {
    auto logistic = LogisticGanObjectives(real, generated);
    loss          = {logistic.discriminator, logistic.generator};
  }
  return {fake, real, generated, loss.discriminator, loss.generator};
}
} // namespace datamunge::nn::models
