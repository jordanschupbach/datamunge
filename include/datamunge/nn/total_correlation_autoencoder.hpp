#pragma once
/// @file total_correlation_autoencoder.hpp
/// @brief Explicit empirical-mixture and discriminator total-correlation VAE components.
#include <algorithm>
#include <numeric>
#include <random>

#include "datamunge/nn/gaussian_autoencoder.hpp"

namespace datamunge::nn::models {
/// @brief Shared validation for explicit TC-augmented Gaussian forwards.
namespace tc_detail {
/// @brief Check finite nonnegative extra TC weight in the computation dtype.
/// @param weight Extra weight (FactorVAE gamma or beta_TC-1).
/// @param type Float32/64 computation dtype, already validated by tensor inputs.
/// @throws std::invalid_argument For negative/nonfinite or unrepresentable positive weights.
inline void Weight(double weight, DType type) {
  if (!std::isfinite(weight) || weight < 0 ||
      (type == DType::Float32 &&
       (!std::isfinite(float(weight)) || (weight > 0 && float(weight) <= 0))))
    throw std::invalid_argument("TC weight must be finite, nonnegative and representable");
}
/// @brief Validate all Gaussian-model inputs before any learned leaves are created.
/// @param p Floating store matching both inputs.
/// @param image Finite nonempty unit-range pixel rows.
/// @param epsilon Explicit finite Gaussian noise, matching image batch/dtype.
/// @param weight Finite nonnegative extra TC weight.
/// @param features Positive image width.
/// @param latent Positive latent width.
/// @param hidden Positive ReLU width.
/// @throws std::invalid_argument For invalid geometry, data, dtype, batch or weight.
inline void Inputs(
    const ParameterStore& p,
    const Tensor&         image,
    const Tensor&         epsilon,
    double                weight,
    std::int64_t          features,
    std::int64_t          latent,
    std::int64_t          hidden) {
  aae_detail::Widths(features, latent, hidden);
  aae_detail::Matrix(image, features, p.dtype(), true);
  aae_detail::Matrix(epsilon, latent, p.dtype(), false);
  Weight(weight, p.dtype());
  if (image.size(0) != epsilon.size(0))
    throw std::invalid_argument("TC VAE batches must match");
}
} // namespace tc_detail

/// @brief Own exact finite-mixture log densities and signed pointwise total-correlation ratios.
///
/// All fields own lifetime and retain sample/reference-head gradients. A negative
/// pointwise ratio or finite sample mean is preserved, not clipped to zero.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// const auto
/// out=n::models::EmpiricalTotalCorrelation(n::zeros({2,3}),n::zeros({1,3}),n::zeros({1,3}));
/// // One diagonal Gaussian factorizes: out.per_case is [0,0], up to roundoff.
/// @endcode
/// @see EmpiricalTotalCorrelation()
struct EmpiricalTcOutput {
  Tensor log_joint;   ///< log((1/R)*sum_r product_d q_r(z_d)), [B], nats.
  Tensor log_product; ///< sum_d log((1/R)*sum_r q_r(z_d)), [B], nats.
  Tensor per_case;    ///< Signed log_joint-log_product for each supplied sample, [B], nats.
  Tensor mean;        ///< Arithmetic mean of per_case over B supplied samples, scalar nats.
};

/// @brief Evaluate the joint/product-marginal log-density ratio of an equally weighted Gaussian
/// mixture.
///
/// Uses ALL R supplied diagonal-Gaussian reference components, with stable
/// logsumexp over component log densities. Joint: logsumexp_r(sum_d log q_rd)-log R;
/// product: sum_d(logsumexp_r(log q_rd)-log R). This is exact finite empirical
/// density evaluation, not the paper's minibatch-weighted estimator. Expectations
/// remain Monte Carlo when z is sampled; pointwise/finite-mean estimates can be
/// negative. A single diagonal component or one latent coordinate factorizes.
/// References must be fitting-only when used for training/evaluation selection;
/// the function cannot infer corpus membership. Both reference heads stay LIVE,
/// including reference encoder parameter paths; no detach/RNG/clipping is hidden.
///
/// Inputs are finite nonempty Float32/64 matrices with matching dtype/latent
/// width; reference means/log variances share [R,L], samples may have B!=R.
/// Strides and all three continuous gradients are retained; inputs unchanged,
/// output owns lifetime. Float32 log(2*pi) and log R are rounded to Float32 before
/// graph arithmetic, matching the source; Float64 uses full double constants.
/// Work/storage O(B*R*L). Exp/square overflow or all-minus-infinity logsumexp
/// states are not repaired. No shared mutable state.
/// @param z Finite floating samples [B,L], B,L>0.
/// @param mu Finite matching-dtype reference means [R,L], R>0.
/// @param log_variance Finite matching-shape/dtype reference log variances [R,L].
/// @return Owning four-field empirical log-density ratio and mean.
/// @throws std::invalid_argument For invalid input shape, dtype or values.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// auto z=n::tensor({.3,-.2,1.,.4},{2,2}).requires_grad_();
/// auto mu=n::tensor({0.,0.,1.,-.3,-.5,.8},{3,2}).requires_grad_();
/// auto lv=n::zeros({3,2}).requires_grad_();
/// const auto out=n::models::EmpiricalTotalCorrelation(z,mu,lv);
/// const auto gradients=n::grad(out.mean,{z,mu,lv}); // all three paths, including references.
/// @endcode
/// @par References
/// - Chen, Li, Grosse and Duvenaud (2018), Isolating Sources of Disentanglement in
///   Variational Autoencoders, https://arxiv.org/abs/1802.04942 . This variant uses
///   a complete equally weighted finite reference mixture, not its minibatch estimator.
/// @see BetaTcVariationalAutoencoder(), EmpiricalTcOutput, logsumexp()
inline EmpiricalTcOutput
EmpiricalTotalCorrelation(const Tensor& z, const Tensor& mu, const Tensor& log_variance) {
  gaussian_detail::Match(mu, log_variance);
  aae_detail::Matrix(z, mu.size(1), mu.dtype(), false);
  double log2pi = std::log(2 * 3.14159265358979323846), log_refs = std::log(double(mu.size(0)));
  if (z.dtype() == DType::Float32) {
    log2pi   = double(float(log2pi));
    log_refs = double(float(log_refs));
  }
  const auto delta = expand_dims(z, 1) - expand_dims(mu, 0), lv0 = expand_dims(log_variance, 0);
  const auto components = -0.5 * ((square(delta) * exp(neg(lv0)) + lv0) + log2pi);
  EmpiricalTcOutput out;
  out.log_joint   = logsumexp(sum(components, {2}), {1}) - log_refs;
  out.log_product = sum(logsumexp(components, {1}) - log_refs, {1});
  out.per_case    = out.log_joint - out.log_product;
  out.mean        = mean(out.per_case, {0});
  return out;
}

/// @brief Evaluate a tanh/linear density-ratio discriminator, with positive logit meaning joint.
///
/// Uses hidden/W[latent,hidden], hidden/b[hidden], ratio/W[hidden,1],ratio/b[1],
/// ordinary Glorot-uniform weights and zero biases: four default leaves/97 scalars.
/// Negative logit means product of marginals. No probability, labels, detach,
/// RNG or training is hidden; all code/store derivatives remain. Finite signed/zero
/// strided Float32/64 rows match store dtype. Validation precedes leaf creation;
/// incompatible leaves can fail later, computed nonfinite states are not sanitized.
/// Output owns lifetime; inputs unchanged. Work O(B*latent*hidden); synchronize stores.
/// @param discriminator Mutable floating store, normally separate from the VAE.
/// @param z Finite nonempty code rows [B,latent], matching store dtype.
/// @param latent Positive code width, default4.
/// @param hidden Positive tanh width, default16.
/// @return Owning differentiable logits [B], not probabilities.
/// @throws std::invalid_argument For invalid width/input/dtype or incompatible leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// dm::nn::ParameterStore d(2309);
/// const auto logits=dm::nn::models::FactorDiscriminator(d,dm::nn::zeros({3,4}));
/// // [3] logits; four leaves/97 scalars with hidden and ratio prefixes.
/// @endcode
/// @par References
/// - Kim and Mnih (2018), Disentangling by Factorising,
///   https://proceedings.mlr.press/v80/kim18b.html . Small tanh teaching discriminator.
/// @see FactorDiscriminatorLoss(), FactorVariationalAutoencoder()
inline Tensor FactorDiscriminator(
    ParameterStore& discriminator,
    const Tensor&   z,
    std::int64_t    latent = 4,
    std::int64_t    hidden = 16) {
  aae_detail::Widths(1, latent, hidden);
  aae_detail::Matrix(z, latent, discriminator.dtype(), false);
  const auto h = dense(discriminator, "hidden", z, hidden, Activation::Tanh);
  return reshape(dense(discriminator, "ratio", h, 1), {z.size(0)});
}

/// @brief Average balanced joint-versus-product softplus cross-entropy for FactorVAE's
/// discriminator.
///
/// Returns0.5*(mean softplus(-D(joint))+mean softplus(D(product))). Each batch
/// is averaged separately, so row counts can differ. Joint has label1/product0;
/// the factor0.5 matches the source's equal-prior binary classification objective.
/// Inputs/store retain all mathematical gradients; the caller MUST detach posterior
/// samples and request/update only D when implementing a discriminator-only phase.
/// No optimizer, detachment, sampling or permuting is hidden. Matching finite
/// Float32/64 matrices and both positive widths are checked before leaf creation.
/// Owning scalar; inputs unchanged. Same numerical/store synchronization limits
/// and per-row work as FactorDiscriminator().
/// @param discriminator Mutable floating discriminator store.
/// @param joint Finite nonempty joint rows [B,latent], matching dtype.
/// @param product Finite nonempty product rows [P,latent], P may differ from B.
/// @param latent Positive code width, default4.
/// @param hidden Positive tanh width, default16.
/// @return Owning differentiable scalar cross-entropy, in nats.
/// @throws std::invalid_argument For invalid input/options or incompatible leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore d(2309);
/// const auto loss=n::models::FactorDiscriminatorLoss(d,n::zeros({2,4}),n::ones({3,4}));
/// const auto gradients=n::grad(loss,d.parameters()); // caller decides when to update D.
/// @endcode
/// @par References
/// - Kim and Mnih (2018), https://proceedings.mlr.press/v80/kim18b.html .
/// @see FactorDiscriminator(), PermuteLatentColumns()
inline Tensor FactorDiscriminatorLoss(
    ParameterStore& discriminator,
    const Tensor&   joint,
    const Tensor&   product,
    std::int64_t    latent = 4,
    std::int64_t    hidden = 16) {
  aae_detail::Widths(1, latent, hidden);
  aae_detail::Matrix(joint, latent, discriminator.dtype(), false);
  aae_detail::Matrix(product, latent, discriminator.dtype(), false);
  const auto j = FactorDiscriminator(discriminator, joint, latent, hidden),
             p = FactorDiscriminator(discriminator, product, latent, hidden);
  return 0.5 * (mean(softplus(neg(j)), {0}) + mean(softplus(p), {0}));
}

/// @brief Independently shuffle each latent column with a local seeded MT19937 stream.
///
/// Resets row indices for each column and calls std::shuffle on one local stream,
/// preserving each empirical marginal exactly while breaking row alignment. Output
/// is a DETACHED owning same-shape/dtype Float32/64 matrix; inputs unchanged. There
/// is no global RNG/state, optimizer or gradient through the discrete permutation.
/// Reproducible for matching seed, data and C++ standard-library shuffle implementation;
/// the standard does not promise identical shuffle sequences across libraries.
/// Finite nonempty strided rows are accepted. Work/storage O(B*L). Finite shuffled
/// batches are a product-marginal surrogate, not independent population draws.
/// @param joint Finite nonempty [B,L] Float32/64 joint samples.
/// @param seed Explicit uint32 seed for the local shuffle stream.
/// @return Detached owning shuffled [B,L], preserving all column value bytes.
/// @throws std::invalid_argument For invalid shape, dtype or nonfinite values.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// const auto x=n::tensor({1.,10.,2.,20.,3.,30.},{3,2}).requires_grad_();
/// const auto product=n::models::PermuteLatentColumns(x,2311);
/// // Each column's multiset is unchanged; product.requires_grad() is false.
/// @endcode
/// @par References
/// - Kim and Mnih (2018), https://proceedings.mlr.press/v80/kim18b.html , empirical
///   product-of-marginals construction via independent within-batch permutations.
/// @see FactorDiscriminatorLoss()
inline Tensor PermuteLatentColumns(const Tensor& joint, std::uint32_t seed) {
  if (!joint.defined() || joint.dim() != 2)
    throw std::invalid_argument("latent permutation requires matrices");
  aae_detail::Matrix(joint, joint.size(1), joint.dtype(), false);
  const auto                n = joint.size(0), width = joint.size(1);
  const auto                values = joint.contiguous().to_vector<double>();
  std::vector<double>       product(values.size());
  std::vector<std::int64_t> rows(n);
  std::mt19937              random(seed);
  for (std::int64_t d = 0; d < width; ++d) {
    std::iota(rows.begin(), rows.end(), 0);
    std::shuffle(rows.begin(), rows.end(), random);
    for (std::int64_t i = 0; i < n; ++i)
      product[i * width + d] = values[rows[i] * width + d];
  }
  return cast(from_vector(product, joint.shape()), joint.dtype());
}

/// @brief Own a base Gaussian VAE forward, scalar TC surrogate and augmented scalar loss.
///
/// net.loss/net.unweighted_loss remain base NLL+KL; outer loss adds extra_weight*tc.
/// tc is either mean discriminator logit or the exact finite-mixture sample mean.
/// All continuous paths are retained; no optimizer, RNG or fitted state is stored.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(2237),d(2309);
/// const auto out=n::models::FactorVariationalAutoencoder(p,d,n::zeros({2,784}),n::zeros({2,4}));
/// // out.net owns Gaussian endpoints; out.loss=out.net.unweighted_loss+3*out.tc.
/// @endcode
/// @see FactorVariationalAutoencoder(), BetaTcVariationalAutoencoder()
struct TcVaeOutput {
  GaussianVaeOutput net; ///< Owning base Gaussian/Bernoulli endpoints, with base losses.
  Tensor            tc;  ///< Scalar signed TC surrogate, nats/sample, with all continuous paths.
  Tensor loss;           ///< Scalar base loss+extra_weight*tc; not the common validation criterion.
};

/// @brief Evaluate FactorVAE's Gaussian model and discriminator-logit TC surrogate with explicit
/// noise.
///
/// Composes BetaVariationalAutoencoder(beta1) with mean FactorDiscriminator(z).
/// loss=mean(summed-pixel NLL+summed-latent analytic KL)+extra_weight*tc. Default
/// gamma3 gives51416 VAE scalars/ten leaves plus97 discriminator scalars/four leaves.
/// Positive logit means joint; this surrogate may be negative and does not certify
/// true TC or disentanglement. No detach is hidden: VAE gradients pass through D
/// to z while the caller requests/updates VAE leaves only. D's separate phase must
/// detach fresh posterior samples, construct product rows and update D separately.
/// Distinct same-dtype stores, finite unit-range image and explicit noise, positive
/// widths and representable nonnegative weight are checked before any creation.
/// Incompatible leaves/nonfinite computed states may fail later. Owning fields;
/// inputs unchanged. Same continuous-gradient/overflow/soft-target semantics as
/// BetaVariationalAutoencoder(); no RNG, fitting, annealing or checkpoint selection.
/// Work is the VAE plus O(B*latent*disc_hidden); shared stores need synchronization.
/// @param parameters Mutable floating VAE store, distinct from discriminator.
/// @param discriminator Mutable matching-dtype store, distinct from parameters.
/// @param image Finite nonempty unit-range [B,features] pixels.
/// @param epsilon Explicit finite same-dtype/batch [B,latent] Gaussian noise.
/// @param extra_weight Finite nonnegative gamma, default3;0 is an ablation.
/// @param features Positive image width, default784.
/// @param latent Positive code width, default4.
/// @param hidden Positive VAE ReLU width, default32.
/// @param disc_hidden Positive discriminator tanh width, default16.
/// @return Owning TcVaeOutput; base net losses are not overwritten.
/// @throws std::invalid_argument For invalid inputs/options, aliased/mismatched stores or
/// incompatible leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(2237),d(2309);
/// const auto out=n::models::FactorVariationalAutoencoder(p,d,n::full({2,784},.2),n::zeros({2,4}));
/// const auto vae_gradients=n::grad(out.loss,p.parameters());
/// // Fixed D still propagates gradients to z.
/// @endcode
/// @par References
/// - Kim and Mnih (2018), Disentangling by Factorising,
///   https://proceedings.mlr.press/v80/kim18b.html . This uses the migrated small
///   ReLU/Bernoulli VAE and tanh discriminator, not the paper's benchmark network.
/// @see TcVaeOutput, FactorDiscriminatorLoss(), BetaVariationalAutoencoder()
inline TcVaeOutput FactorVariationalAutoencoder(
    ParameterStore& parameters,
    ParameterStore& discriminator,
    const Tensor&   image,
    const Tensor&   epsilon,
    double          extra_weight = 3,
    std::int64_t    features     = 784,
    std::int64_t    latent       = 4,
    std::int64_t    hidden       = 32,
    std::int64_t    disc_hidden  = 16) {
  tc_detail::Inputs(parameters, image, epsilon, extra_weight, features, latent, hidden);
  aae_detail::Widths(1, latent, disc_hidden);
  if (&parameters == &discriminator || parameters.dtype() != discriminator.dtype())
    throw std::invalid_argument("FactorVAE requires distinct matching-dtype parameter stores");
  TcVaeOutput out;
  out.net  = BetaVariationalAutoencoder(parameters, image, epsilon, 1, features, latent, hidden);
  out.tc   = mean(FactorDiscriminator(discriminator, out.net.z, latent, disc_hidden), {0});
  out.loss = out.net.unweighted_loss + extra_weight * out.tc;
  return out;
}

/// @brief Evaluate betaTCVAE with all supplied fitting posteriors as live finite-mixture
/// references.
///
/// Base model is BetaVariationalAutoencoder(beta1). Reuses its encoder/mu/
/// log_variance leaves on reference_images, preserving ALL reference encoder
/// gradients, then evaluates EmpiricalTotalCorrelation on sampled z. No reference
/// decoder or random draw is computed. loss=base NLL+analytic KL+extra_weight*tc;
/// default extra3 corresponds to beta_TC4, leaving other decomposition weights1.
/// This complete equally weighted finite mixture is an educational adaptation,
/// not the original minibatch-weighted density estimator. It is exact as a finite
/// density; supplied z only estimates its expected TC, so negatives are retained.
///
/// reference_images MUST contain fitting rows only, never validation/test; this
/// is caller-owned provenance. Matching finite unit-range matrices, explicit noise,
/// positive widths and representable nonnegative weight are validated before
/// creation. Inputs unchanged; all fields own lifetime, with sample/reference
/// input/encoder paths live. Incompatible leaves/nonfinite arithmetic may fail
/// later. No implicit detach, sampling, training or selection. Work/storage
/// includes O(B*R*latent), with no hidden reference chunking. Synchronize stores.
/// @param parameters Mutable floating VAE store shared by sample/reference encoders.
/// @param image Finite nonempty [B,features] unit-range pixels.
/// @param epsilon Explicit finite matching-dtype/batch [B,latent] Gaussian noise.
/// @param reference_images Finite nonempty [R,features] fitting-only pixel rows, computation dtype.
/// @param extra_weight Finite nonnegative beta_TC-1, default3.
/// @param features Positive image width, default784.
/// @param latent Positive code width, default4.
/// @param hidden Positive encoder/decoder ReLU width, default32.
/// @return Owning TcVaeOutput with base losses and a signed exact-mixture TC sample mean.
/// @throws std::invalid_argument For invalid inputs/options/reference rows or incompatible leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(2237);
/// const auto
/// out=n::models::BetaTcVariationalAutoencoder(p,n::full({2,784},.2),n::zeros({2,4}),n::full({3,784},.3));
/// const auto gradients=n::grad(out.loss,p.parameters()); // Includes fitting-reference head paths.
/// @endcode
/// @par References
/// - Chen, Li, Grosse and Duvenaud (2018), Isolating Sources of Disentanglement in
///   Variational Autoencoders, https://arxiv.org/abs/1802.04942 . Complete finite
///   reference mixture replaces its minibatch estimator; no MIG claim is made.
/// @see EmpiricalTotalCorrelation(), TcVaeOutput, BetaVariationalAutoencoder()
inline TcVaeOutput BetaTcVariationalAutoencoder(
    ParameterStore& parameters,
    const Tensor&   image,
    const Tensor&   epsilon,
    const Tensor&   reference_images,
    double          extra_weight = 3,
    std::int64_t    features     = 784,
    std::int64_t    latent       = 4,
    std::int64_t    hidden       = 32) {
  tc_detail::Inputs(parameters, image, epsilon, extra_weight, features, latent, hidden);
  aae_detail::Matrix(reference_images, features, parameters.dtype(), true);
  TcVaeOutput out;
  out.net = BetaVariationalAutoencoder(parameters, image, epsilon, 1, features, latent, hidden);
  const auto h  = dense(parameters, "encoder", reference_images, hidden, Activation::Relu);
  const auto mu = dense(parameters, "mu", h, latent),
             lv = dense(parameters, "log_variance", h, latent);
  out.tc        = EmpiricalTotalCorrelation(out.net.z, mu, lv).mean;
  out.loss      = out.net.unweighted_loss + extra_weight * out.tc;
  return out;
}
} // namespace datamunge::nn::models
