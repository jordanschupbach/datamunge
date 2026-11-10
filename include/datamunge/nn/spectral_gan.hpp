#pragma once
/// @file spectral_gan.hpp
/// @brief Compact hinge GAN with explicit cached spectral columns and caller-owned update phases.
#include "datamunge/nn/cached_spectral_dense.hpp"
#include "datamunge/nn/dense_gan.hpp"

namespace datamunge::nn::models {
/// @brief Score coordinates using two cached spectral kernels with tanh then a raw scalar output.
///
/// image[N,F] -> hidden tanh H -> raw source score[N,1]. Each W is divided by
/// max(sum(detach(v)*(W*detach(u))),double(1e-6f)); biases are not normalized.
/// Creates hidden/W[F,H],b[H],output/W[H,1],b[1] in that order. Each layer's
/// state creates prefix/u[out,1],prefix/v[in,1]. W's uniform limit is computed
/// as sqrt(6.f/float(in+out)); initial columns are 1.f/sqrt(float(length)),
/// preserving Float32-rounded constants for both computation dtypes. Default
/// F784/H32 creates four trainable leaves/25153 values and four cache leaves/849
/// values. Cache flags follow ParameterStore but their math is always detached;
/// exclude the state store from optimization and checkpoint it explicitly.
///
/// Does not advance existing state, warm up, normalize u initially, draw noise
/// beyond missing leaf initialization, fit, or choose an optimizer. Call
/// advance_cached_spectral_state() explicitly; repeated inference leaves existing
/// state bit exact. Finite signed Float32/64 CPU images match both distinct
/// stores, positive widths/products, strides supported. All applicable image/W
/// first/higher paths including sigma stay live. All input/geometry checks precede
/// leaves; incompatible/nonfinite existing leaves or computation may fail later
/// after earlier missing leaves were created. Inputs unchanged; outputs own lifetime.
/// Work O(N*F*H), activations O(N*(F+H)) plus kernel validation. Synchronize stores.
/// Finite power iteration does not certify exact unit kernel or network norms.
/// @param discriminator Mutable floating spectral dense weight/bias owner.
/// @param state Distinct matching cache owner, read unchanged after initialization.
/// @param image Finite signed matching floating CPU[N,features], N>0.
/// @param features Positive image/input width F, default784.
/// @param hidden Positive tanh width H, default32.
/// @return Owning live raw source scores[N,1], without a sigmoid.
/// @throws std::invalid_argument For invalid inputs/geometry, aliased/mismatched stores,
/// incompatible leaves or nonfinite computation.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore d(5807),s(0);
/// n::models::SpectralGanDiscriminate(d,s,n::zeros({2,784}));
/// n::advance_cached_spectral_state(d,s,16); // explicit warm-up, outside inference.
/// auto score=n::models::SpectralGanDiscriminate(d,s,n::zeros({2,784}));
/// // Existing u/v are unchanged by scoring; sigma's W path remains live.
/// @endcode
/// @par References
/// - Miyato, Kataoka, Koyama and Yoshida (2018), Spectral Normalization for
///   Generative Adversarial Networks, https://arxiv.org/abs/1802.05957 .
///   This compact tanh MLP and explicit guarded cached-column protocol are teaching choices.
/// @see SpectralGan(), cached_spectral_kernel(), advance_cached_spectral_state()
inline Tensor SpectralGanDiscriminate(
    ParameterStore& discriminator,
    ParameterStore& state,
    const Tensor&   image,
    std::int64_t    features = 784,
    std::int64_t    hidden   = 32) {
  if (&discriminator == &state || discriminator.dtype() != state.dtype())
    throw std::invalid_argument("Spectral discriminator needs distinct matching-dtype stores");
  dense_gan_detail::Geometry(1, features, 1, hidden);
  gan_detail::Matrix(image, features, discriminator.dtype());
  dense_gan_detail::Geometry(image.size(0), features, 1, hidden);
  auto h = cached_spectral_dense(discriminator, state, "hidden", image, hidden, Activation::Tanh);
  return cached_spectral_dense(discriminator, state, "output", h, 1, Activation::None);
}

/// @brief Evaluate a compact sigmoid generator and cached spectral discriminator with hinge
/// objectives.
///
/// G is MlpGanGenerate(): noise[M,Z] -> tanh H -> sigmoid F. D is
/// SpectralGanDiscriminate(): real[N,F]/fake[M,F] -> spectral tanh H -> spectral
/// raw scalar. HingeGanObjectives() gives D=mean(relu(1-real))+mean(relu(1+fake))
/// with zero margin subgradient and G=-mean(fake), with no G hinge or sigmoid.
/// Default Z16/H32/F784 creates G26416/4 leaves, D25153/4 leaves, cache849/4 leaves
/// in G,D order, hidden then output W,b and state u,v per layer. Signed floating
/// images/noise match three distinct Float32/64 CPU stores. Unequal positive N/M,
/// positive widths/products, strides and applicable first/higher continuous paths
/// are supported. Fixed cache columns are stopped; the W-dependent sigma is live.
///
/// No cache advance, warm-up, fitting, optimizer, implicit freeze, checkpoint,
/// MMD diagnostic or noise draw beyond missing leaf initialization is hidden.
/// Caller initializes, explicitly warms state, then for each batch advances state,
/// writes only D, recomputes through updated D with the SAME cache, and writes only G.
/// Cache requires_grad flags follow ParameterStore; exclude this store from Adam.
/// All input/geometry checks precede leaves; incompatible/nonfinite existing leaves
/// may fail later after earlier missing leaves were created. Inputs unchanged,
/// outputs own lifetime. Work O((N+M)*F*H+M*Z*H); activations O((N+M)*(F+H)+M*Z)
/// plus validation. Synchronize shared stores. An estimated norm is not a certificate.
/// @param generator Mutable floating generator owner.
/// @param discriminator Distinct matching floating spectral dense weight/bias owner.
/// @param state Third distinct matching cache owner; unchanged after initialization.
/// @param image Finite signed matching floating CPU[N,features], N>0.
/// @param noise Finite signed matching floating CPU[M,Z], M,Z>0; independent of N.
/// @param features Positive image/output width F, default784.
/// @param hidden Positive hidden tanh width H, default32 for both networks.
/// @return Owning DenseGanOutput: fake, real/fake scores and scalar D/G objectives.
/// @throws std::invalid_argument For invalid inputs/geometry, aliased/mismatched stores,
/// incompatible leaves or nonfinite computation.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore g(5801),d(5807),s(0);
/// auto x=n::zeros({2,784}),z=n::zeros({3,16});
/// n::models::SpectralGan(g,d,s,x,z); // initialize only.
/// n::advance_cached_spectral_state(d,s,16);
/// n::advance_cached_spectral_state(d,s); // before the explicit D phase.
/// auto out=n::models::SpectralGan(g,d,s,x,z);
/// auto derivative=n::grad(out.discriminator_loss,d.parameters());
/// // Write only D, then recompute the G objective with unchanged s.
/// @endcode
/// @par References
/// - Miyato et al. (2018), https://arxiv.org/abs/1802.05957 , spectral normalization
///   and raw-score hinge adversarial objectives. This small MNIST MLP is not a paper benchmark.
/// @see SpectralGanDiscriminate(), MlpGanGenerate(), HingeGanObjectives(),
/// advance_cached_spectral_state()
inline DenseGanOutput SpectralGan(
    ParameterStore& generator,
    ParameterStore& discriminator,
    ParameterStore& state,
    const Tensor&   image,
    const Tensor&   noise,
    std::int64_t    features = 784,
    std::int64_t    hidden   = 32) {
  if (&generator == &discriminator || &generator == &state || &discriminator == &state ||
      generator.dtype() != discriminator.dtype() || generator.dtype() != state.dtype())
    throw std::invalid_argument("Spectral GAN needs three distinct matching-dtype stores");
  if (!noise.defined() || noise.dim() != 2)
    throw std::invalid_argument("Spectral GAN noise must be a matrix");
  dense_gan_detail::Geometry(1, features, noise.size(1), hidden);
  gan_detail::Matrix(image, features, generator.dtype());
  gan_detail::Matrix(noise, noise.size(1), generator.dtype());
  dense_gan_detail::Geometry(image.size(0), features, noise.size(1), hidden);
  dense_gan_detail::Geometry(noise.size(0), features, noise.size(1), hidden);
  auto fake      = MlpGanGenerate(generator, noise, features, hidden);
  auto real      = SpectralGanDiscriminate(discriminator, state, image, features, hidden);
  auto generated = SpectralGanDiscriminate(discriminator, state, fake, features, hidden);
  auto loss      = HingeGanObjectives(real, generated);
  return {fake, real, generated, loss.discriminator, loss.generator};
}
} // namespace datamunge::nn::models
