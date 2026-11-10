#pragma once
/// @file bidirectional_gan.hpp
/// @brief Explicit tanh/linear maps, joint pair scores and bidirectional adversarial objectives.
#include "datamunge/nn/adversarial_autoencoder.hpp"
#include "datamunge/nn/gan_models.hpp"

namespace datamunge::nn::models {
/// @brief Internal positive BiGAN coordinate, concatenation and activation-product checks.
namespace bigan_detail {
/// @brief Require representable positive dense weights and paired-coordinate activations.
/// @param rows Positive real or prior pool size.
/// @param features Positive data width F.
/// @param latent Positive code width Z.
/// @param hidden Positive tanh width H.
/// @throws std::invalid_argument For nonpositive dimensions or int64 sum/product overflow.
inline void
Geometry(std::int64_t rows, std::int64_t features, std::int64_t latent, std::int64_t hidden) {
  const auto limit = std::numeric_limits<std::int64_t>::max();
  if (rows <= 0 || features <= 0 || latent <= 0 || hidden <= 0 || features > limit - latent ||
      features + latent > limit / hidden || rows > limit / std::max(features + latent, hidden))
    throw std::invalid_argument("BiGAN dimensions and products must be positive and representable");
}
} // namespace bigan_detail

/// @brief Map signed coordinates through the source hidden-tanh/output-linear network.
///
/// Delegates the canonical AaeDecode() tanh/linear computation without a sigmoid,
/// interpreting its raw outputs as coordinates. Input width I is inferred.
/// Creates/reuses hidden/W[I,H],hidden/b[H],output/W[H,O],output/b[O] in that
/// order, with ordinary dense Glorot/zero initialization. I=O=2,H=16 gives four
/// leaves/82 values. Finite signed Float32/64 CPU[N,I] input matches store dtype;
/// positive axes and strides preserve all first/higher continuous paths. No
/// encoder, discriminator, fit, optimizer, random input draw or target is hidden.
/// Outputs are unbounded coordinates, not pixel probabilities or posterior samples.
/// Input checks precede creation; incompatible leaves/nonfinite computed output
/// may fail later. Input unchanged, output owns lifetime. O(N*H*(I+O)) work,
/// O(N*(H+O)) activations, plus host validation; synchronize shared mutable stores.
/// @param parameters Mutable matching floating map store; only missing leaves are initialized.
/// @param input Finite signed floating CPU[N,I], N,I>0, matching store dtype.
/// @param output Positive output-coordinate width O, default 2.
/// @param hidden Positive hidden tanh width H, default 16.
/// @return Owning live unbounded CPU[N,output] coordinates in input dtype.
/// @throws std::invalid_argument For invalid input/widths/products, incompatible leaves or
/// nonfinite computation.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// dm::nn::ParameterStore g(6007);
/// const auto points=dm::nn::models::BiganMap(g,dm::nn::zeros({3,2}));
/// // points is [3,2]; no encoder or image enters this prior-only generation.
/// @endcode
/// @par References
/// - Donahue, Krähenbühl and Darrell (2016/2017), Adversarial Feature Learning,
///   https://arxiv.org/abs/1605.09782 . This is the compact teaching MLP,
///   not the paper's image-model architecture or benchmark.
/// @see AaeDecode(), BiganJointScore(), BidirectionalGan()
inline Tensor BiganMap(
    ParameterStore& parameters,
    const Tensor&   input,
    std::int64_t    output = 2,
    std::int64_t    hidden = 16) {
  if (!input.defined() || input.dim() != 2)
    throw std::invalid_argument("BiGAN map requires matrix coordinates");
  bigan_detail::Geometry(input.size(0), output, input.size(1), hidden);
  gan_detail::Matrix(input, input.size(1), parameters.dtype());
  auto out = AaeDecode(parameters, input, output, input.size(1), hidden);
  gan_detail::Finite(out);
  return out;
}

/// @brief Score supplied data/code pairs jointly, without pair labels or hidden encoding.
///
/// Concatenates image[N,F],code[N,Z] in that order and delegates the canonical
/// tanh/linear AaeDecode() with one raw output. Creates hidden/W[F+Z,H],b[H],
/// output/W[H,1],b[1] in source order; F=Z=2,H=16 gives 97 values/four leaves.
/// Finite signed Float32/64 CPU pairs match row counts and store dtype; strides
/// and all first/higher paths through both pair parts remain live. Positive means
/// the (data,encoded-data) source, not a class probability. No E/G forward, target,
/// sigmoid, loss, fit or update is hidden. Checks precede creation; incompatible
/// leaves or nonfinite output may fail later. Inputs unchanged; output owns
/// lifetime. O(N*H*(F+Z)) work, O(N*(F+Z+H)) activations and host validation;
/// synchronize shared stores.
/// @param discriminator Mutable floating joint-discriminator store.
/// @param image Finite signed CPU[N,F] data coordinates, N,F>0, matching dtype.
/// @param code Finite signed CPU[N,Z] code coordinates, same N and dtype, Z>0.
/// @param hidden Positive tanh hidden width H, default 16.
/// @return Owning live raw source logits[N,1], depending on both supplied pair parts.
/// @throws std::invalid_argument For invalid pairs/widths/products, incompatible leaves or
/// nonfinite computation.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// dm::nn::ParameterStore d(6029);
/// const auto score=dm::nn::models::BiganJointScore(d,dm::nn::ones({3,2}),dm::nn::zeros({3,2}));
/// // score is [3,1]; both pair coordinates are explicit, with no encoder call.
/// @endcode
/// @par References
/// - Donahue et al. (2016/2017), https://arxiv.org/abs/1605.09782 .
/// - Dumoulin et al. (2016/2017), Adversarially Learned Inference,
///   https://arxiv.org/abs/1606.00704 . Both discriminate joint data/code sources.
/// @see BiganMap(), LogisticGanObjectives(), BidirectionalGan()
inline Tensor BiganJointScore(
    ParameterStore& discriminator,
    const Tensor&   image,
    const Tensor&   code,
    std::int64_t    hidden = 16) {
  if (!image.defined() || image.dim() != 2 || !code.defined() || code.dim() != 2 ||
      image.size(0) != code.size(0))
    throw std::invalid_argument("BiGAN joint score requires aligned matrix pairs");
  bigan_detail::Geometry(image.size(0), image.size(1), code.size(1), hidden);
  gan_detail::Matrix(image, image.size(1), discriminator.dtype());
  gan_detail::Matrix(code, code.size(1), discriminator.dtype());
  auto out =
      AaeDecode(discriminator, concat({image, code}, 1), 1, image.size(1) + code.size(1), hidden);
  gan_detail::Finite(out);
  return out;
}

/// @brief Own eight live bidirectional GAN endpoints, including two unfitted cycle diagnostics.
///
/// Tensor fields retain storage and applicable autograd history after this record
/// and parameter stores are destroyed. No optimizer, random generator, mode switch,
/// fit counter or checkpoint state is stored. Reconstruction/cycle fields are
/// computed for inspection; their residuals do not enter either fitting objective.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore g(6007),e(6011),d(6029);
/// const auto out=n::models::BidirectionalGan(g,e,d,n::ones({2,2}),n::zeros({3,2}));
/// // reconstruction has 2 data rows; latent_cycle has 3 prior rows.
/// @endcode
/// @see BidirectionalGan(), BiganMap(), BiganJointScore()
struct BiganOutput {
  Tensor fake;               ///< Live generated coordinates [M,F], from prior only.
  Tensor encoded;            ///< Live encoded coordinates [N,Z], from data only.
  Tensor real_score;         ///< Live joint logits [N,1] for (image,encoded).
  Tensor fake_score;         ///< Live joint logits [M,1] for (fake,prior).
  Tensor discriminator_loss; ///< Scalar mean softplus(-real_score) + mean softplus(fake_score).
  Tensor eg_loss;            ///< Scalar mean softplus(real_score) + mean softplus(-fake_score).
  Tensor reconstruction;     ///< Live G(E(image))[N,F], excluded from fitting objectives.
  Tensor latent_cycle;       ///< Live E(G(prior))[M,Z], excluded from fitting objectives.
};

/// @brief Evaluate explicit data/prior joint sources and separate bidirectional adversarial losses.
///
/// G maps prior[M,Z] through hidden tanh H to linear coordinates F; E maps data
/// [N,F] through the same architecture to Z. D scores (image,E(image)) and
/// (G(prior),prior), concatenating data before code. Stores are separate; each
/// creates hidden/W,b then output/W,b, in G,E,D order. F=Z=2,H=16 gives
/// G82+E82+D97=261 values in 12 leaves. Widths F/Z are inferred from inputs.
/// The reusable tanh/linear computation is AaeDecode(), with no sigmoid.
///
/// D minimizes mean softplus(-real_score)+mean softplus(fake_score). E/G minimize
/// mean softplus(real_score)+mean softplus(-fake_score), the source's explicit
/// non-saturating reversed-source objective. Each pool is averaged separately;
/// N and M may differ. Reconstruction G(E(image)) and latent cycle E(G(prior))
/// stay live diagnostics, with no reconstruction/cycle term in either fit loss.
/// There is no inference-likelihood, learned inverse or convergence guarantee.
///
/// All mathematical paths remain live. Caller selects D leaves, writes D,
/// recomputes for E, writes E, then recomputes for G. With D fixed, E's write
/// leaves G's loss partial unchanged because the two joint branches separate.
/// No optimizer, fit loop, noise draw except missing-leaf initialization, clipping,
/// target label, implicit freeze, evaluation guard or checkpoint is hidden.
/// All three distinct stores and finite signed CPU inputs share Float32/64 dtype.
/// Strides and higher continuous paths are preserved. All input/geometry checks
/// precede creation; incompatible leaves or nonfinite computed states may fail
/// later. Inputs unchanged; owning outputs retain lifetime. Work
/// O((N+M)*H*(F+Z)), including the two cycle diagnostics; activation storage
/// O((N+M)*(F+Z+H)). Synchronize callers sharing mutable stores.
/// @param generator Mutable independent floating generator store G.
/// @param encoder Mutable independent floating encoder store E, matching G dtype.
/// @param discriminator Mutable independent floating joint store D, matching G dtype.
/// @param image Finite signed CPU data[N,F], N,F>0, matching store dtype.
/// @param prior Finite signed CPU prior[M,Z], M,Z>0, matching store dtype.
/// @param hidden Positive tanh width H, default 16.
/// @return Owning eight-field BiganOutput, with separate D and E/G scalar objectives.
/// @throws std::invalid_argument For invalid inputs/dimensions/products, aliased or mismatched
/// stores, incompatible leaves or nonfinite computation.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore g(6007),e(6011),d(6029);
/// const auto out=n::models::BidirectionalGan(g,e,d,n::ones({2,2}),n::zeros({2,2}));
/// const auto dd=n::grad(out.discriminator_loss,d.parameters());
/// // Caller writes dd only to D, then recomputes the next selected-group loss.
/// @endcode
/// @par References
/// - Donahue, Krähenbühl and Darrell (2016/2017), Adversarial Feature Learning,
///   https://arxiv.org/abs/1605.09782 .
/// - Dumoulin et al. (2016/2017), Adversarially Learned Inference,
///   https://arxiv.org/abs/1606.00704 . This compact deterministic MLP uses
///   the teaching source's reversed-source non-saturating objective and schedule,
///   not either paper's image benchmark or idealized optimum assumptions.
/// @see BiganMap(), BiganJointScore(), LogisticGanObjectives(), GaussianRingPoints()
inline BiganOutput BidirectionalGan(
    ParameterStore& generator,
    ParameterStore& encoder,
    ParameterStore& discriminator,
    const Tensor&   image,
    const Tensor&   prior,
    std::int64_t    hidden = 16) {
  if (&generator == &encoder || &generator == &discriminator || &encoder == &discriminator ||
      generator.dtype() != encoder.dtype() || generator.dtype() != discriminator.dtype())
    throw std::invalid_argument("BiGAN needs three distinct matching-dtype stores");
  if (!image.defined() || image.dim() != 2 || !prior.defined() || prior.dim() != 2)
    throw std::invalid_argument("BiGAN needs matrix data and prior");
  bigan_detail::Geometry(image.size(0), image.size(1), prior.size(1), hidden);
  bigan_detail::Geometry(prior.size(0), image.size(1), prior.size(1), hidden);
  gan_detail::Matrix(image, image.size(1), generator.dtype());
  gan_detail::Matrix(prior, prior.size(1), generator.dtype());
  auto fake          = BiganMap(generator, prior, image.size(1), hidden),
       encoded       = BiganMap(encoder, image, prior.size(1), hidden);
  auto real          = BiganJointScore(discriminator, image, encoded, hidden),
       generated     = BiganJointScore(discriminator, fake, prior, hidden);
  auto        source = LogisticGanObjectives(real, generated);
  BiganOutput out{
      fake,
      encoded,
      real,
      generated,
      source.discriminator,
      mean(softplus(real)) + source.generator,
      BiganMap(generator, encoded, image.size(1), hidden),
      BiganMap(encoder, fake, prior.size(1), hidden)};
  gan_detail::Finite(out.eg_loss);
  return out;
}
} // namespace datamunge::nn::models
