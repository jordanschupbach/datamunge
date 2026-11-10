#pragma once
/// @file flow_variational_autoencoder.hpp
/// @brief Conditional affine posterior couplings and explicit-noise Gaussian flow VAEs.
#include <limits>

#include "datamunge/nn/gaussian_autoencoder.hpp"
#include "datamunge/nn/io.hpp"

namespace datamunge::nn::models {
/// @brief Own the transformed code and signed log-Jacobian of one conditional affine coupling.
///
/// All fields own lifetime and retain code, context and parameter derivatives.
/// The context is held fixed when interpreting the code-to-code Jacobian.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(3049);
/// const auto out=n::models::ConditionalAffinePosteriorFlow(
///     p,"flow0",n::zeros({2,4}),n::ones({2,2}));
/// // out.value is [2,4]; out.log_determinant is [2].
/// @endcode
/// @see ConditionalAffinePosteriorFlow()
struct AffinePosteriorFlowOutput {
  Tensor value;           ///< Transformed [B,L] code, retaining the original coordinate ordering.
  Tensor log_determinant; ///< Signed forward/inverse log absolute determinant [B], nats.
  Tensor scale;           ///< Bounded forward log scales [B,L/2], also returned for inverse calls.
  Tensor shift;           ///< Forward affine shifts [B,L/2], also returned for inverse calls.
};

/// @brief Apply or invert a context-conditioned affine coupling, preserving the unchanged half.
///
/// Split code into equal halves; swap=false conditions on the first and transforms
/// the second, swap=true conditions on the second and transforms the first.
/// Concatenate the unchanged half with supplied context[C], then use
/// name/conditioner: (L/2+C)->hidden ReLU and name/heads: hidden->L linear.
/// Heads give scale=0.5*tanh(raw_scale) and shift, each L/2 wide. Forward computes
/// b*exp(scale)+shift and sum(scale); inverse computes (b-shift)*exp(-scale)
/// and -sum(scale). Coordinate ordering is preserved, not swapped on output.
/// The Jacobian is triangular in split coordinates with context fixed; each
/// row's absolute logdet is bounded by L/4. Computed exp overflow is not repaired.
///
/// Finite nonempty matching Float32/64 matrices preserve strides and all continuous
/// gradients. Positive even latent width and positive context/hidden widths are
/// checked before leaf creation. Weights use ordinary Glorot uniform, biases zero.
/// No RNG draws beyond lazy parameter initialization, detach, fit or optimizer is
/// hidden. Inputs unchanged; returned handles own lifetime. Synchronize shared
/// stores. Work O(B*hidden*(L+C)); incompatible stored leaves can fail later.
/// @param parameters Mutable matching-dtype floating store, with shared forward/inverse leaves.
/// @param name Nonempty parameter prefix; reuse exactly for inversion, e.g. flow0.
/// @param code Finite code [B,latent], B>0, with store dtype.
/// @param context Finite same-batch/dtype conditioning rows [B,C], C>0.
/// @param swap Condition on the second half when true; default false.
/// @param inverse Invert the same conditioned mapping when true; default false.
/// @param latent Positive even code width, default4.
/// @param hidden Positive conditioner ReLU width, default16.
/// @return Owning transformed code, signed logdet, forward scale and shift.
/// @throws std::invalid_argument For invalid inputs/options, width overflow or incompatible leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(3049);
/// const auto z=n::ones({2,4}),c=n::zeros({2,2});
/// const auto f=n::models::ConditionalAffinePosteriorFlow(p,"f",z,c);
/// const auto back=n::models::ConditionalAffinePosteriorFlow(p,"f",f.value,c,false,true);
/// // back.value recovers z up to roundoff; log determinants cancel.
/// @endcode
/// @par References
/// - Dinh, Sohl-Dickstein and Bengio (2016/2017), Density estimation using Real NVP,
///   https://arxiv.org/abs/1605.08803 . Uses affine coupling with an added external
///   context and bounded log scales, not an image-space Real NVP generative model.
/// @see AffinePosteriorFlowOutput, FlowVariationalAutoencoder()
inline AffinePosteriorFlowOutput ConditionalAffinePosteriorFlow(
    ParameterStore&    parameters,
    const std::string& name,
    const Tensor&      code,
    const Tensor&      context,
    bool               swap    = false,
    bool               inverse = false,
    std::int64_t       latent  = 4,
    std::int64_t       hidden  = 16) {
  aae_detail::Widths(1, latent, hidden);
  aae_detail::Matrix(code, latent, parameters.dtype(), false);
  if (name.empty() || latent % 2 || !context.defined() || context.dim() != 2)
    throw std::invalid_argument(
        "conditional affine flow needs an even code width, context and prefix");
  aae_detail::Matrix(context, context.size(1), parameters.dtype(), false);
  const auto half = latent / 2;
  if (context.size(0) != code.size(0) ||
      context.size(1) > std::numeric_limits<std::int64_t>::max() - half)
    throw std::invalid_argument("conditional affine flow batch mismatch or width overflow");
  const auto first = slice_axis(code, 1, 0, half), second = slice_axis(code, 1, half, latent);
  const auto a = swap ? second : first, b = swap ? first : second;
  const auto h =
      dense(parameters, name + "/conditioner", concat({a, context}, 1), hidden, Activation::Relu);
  const auto                heads = dense(parameters, name + "/heads", h, latent);
  AffinePosteriorFlowOutput out;
  out.scale = 0.5 * tanh(slice_axis(heads, 1, 0, half));
  out.shift = slice_axis(heads, 1, half, latent);
  const auto transformed =
      inverse ? (b - out.shift) * exp(-out.scale) : b * exp(out.scale) + out.shift;
  out.value           = swap ? concat({transformed, a}, 1) : concat({a, transformed}, 1);
  const auto total    = sum(out.scale, {1});
  out.log_determinant = inverse ? -total : total;
  return out;
}

/// @brief Own all eighteen Gaussian, conditional-flow, density and reconstruction endpoints.
///
/// Every Tensor owns its lifetime and preserves continuous gradients. Per-row
/// density/reconstruction values are in nats; loss alone averages the batch.
/// Inverse endpoints are diagnostics and are not part of the training loss.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(3001);
/// const auto out=n::models::FlowVariationalAutoencoder(
///     p,n::zeros({2,784}),n::zeros({2,4}));
/// // out.z0 equals mu; out.z is flowed mu, not generally a posterior mean.
/// @endcode
/// @see FlowVariationalAutoencoder(), Named()
struct FlowVaeOutput {
  Tensor context;      ///< Encoder ReLU context [B,hidden], shared by both conditioners.
  Tensor mu;           ///< Base Gaussian mean [B,L], independent of epsilon.
  Tensor log_variance; ///< Base Gaussian log variance [B,L], independent of epsilon.
  Tensor z0;           ///< Base sample mu+exp(log_variance/2)*epsilon [B,L].
  Tensor z1;           ///< First coupling output [B,L].
  Tensor z;            ///< Second coupling output [B,L], the sole decoder input.
  Tensor logdet0;      ///< First forward log absolute determinant [B].
  Tensor logdet1;      ///< Second forward log absolute determinant [B].
  Tensor logdet;       ///< Sum of the two forward log determinants [B].
  Tensor logq0;        ///< Base diagonal-Gaussian log density evaluated at z0 [B].
  Tensor logq;         ///< Flowed posterior log density logq0-logdet [B].
  Tensor logp;         ///< Standard-normal prior log density evaluated at final z [B].
  Tensor logratio;     ///< Signed logq-logp per supplied sample [B], not analytic KL.
  Tensor probability;  ///< Sigmoid Bernoulli probabilities [B,features], not sampled pixels.
  Tensor nll;          ///< Summed pixel Bernoulli NLL (binary) or cross-entropy (soft) [B].
  Tensor loss;         ///< Scalar mean(nll+logratio), sampled negative ELBO.
  Tensor inverse_z0;   ///< Second inverse then first inverse recover z0 [B,L] up to roundoff.
  Tensor
      inverse_logdet; ///< Sum of inverse log determinants [B], cancelling forward up to roundoff.

  /// @brief Copy the eighteen endpoint handles into their stable named archive order.
  ///
  /// Arrays share reference-counted tensor storage/graphs with this object and
  /// outlive it. No values, gradient state or parameter leaves are changed.
  /// @return Owning NamedTensors from context through inverse_logdet, in field declaration order.
  /// @par Example
  /// @code
  /// #include <dm/dm.hpp>
  /// namespace n=dm::nn;
  /// n::ParameterStore p(3001);
  /// const auto arrays=n::models::FlowVariationalAutoencoder(
  ///     p,n::zeros({1,784}),n::zeros({1,4})).Named();
  /// // arrays has eighteen entries; the temporary result can be destroyed safely.
  /// @endcode
  /// @see FlowVaeOutput, save_npz()
  datamunge::nn::NamedTensors Named() const {
    return {
        {"context", context},
        {"mu", mu},
        {"log_variance", log_variance},
        {"z0", z0},
        {"z1", z1},
        {"z", z},
        {"logdet0", logdet0},
        {"logdet1", logdet1},
        {"logdet", logdet},
        {"logq0", logq0},
        {"logq", logq},
        {"logp", logp},
        {"logratio", logratio},
        {"probability", probability},
        {"nll", nll},
        {"loss", loss},
        {"inverse_z0", inverse_z0},
        {"inverse_logdet", inverse_logdet}};
  }
};

/// @brief Evaluate a Gaussian VAE with two alternating conditional affine posterior couplings.
///
/// Encoder features->hidden ReLU feeds two latent-wide linear Gaussian heads.
/// Explicit epsilon samples z0; two ConditionalAffinePosteriorFlow calls use that
/// same live image context, first with swap=false, then true. Only final z enters
/// GaussianVaeDecode; no deterministic context bypass exists. Forward/inverse
/// share names/parameters; weights are ordinary Glorot uniform, biases zero.
/// Default architecture has eighteen leaves/52672 scalars, including1256 flow
/// scalars; creation order is encoder,mu,log_variance,flow0,flow1,decoder,output.
///
/// logq=logq0-SUM(forward logdet); logp is standard normal at final z. Train
/// mean(summed pixel NLL+logq-logp). This sampled ratio can be negative and is
/// not clipped or replaced by base analytic Gaussian KL. Float32 uses the original
/// float-rounded log(2*pi), Float64 uses full precision. Epsilon-zero decoding
/// yields decode(f(mu)); nonlinear f(mu) is not generally the flowed mean.
/// Prior sampling uses GaussianVaeDecode directly on standard-normal FINAL codes,
/// without image, context or posterior flow. This is a flexible inference family,
/// not an image-space exact-likelihood flow or a learned flow prior.
///
/// Validates finite unit-range nonempty image and matching finite explicit noise,
/// positive widths/even latent and addition overflow before creating leaves.
/// Strides and all image/noise/context/parameter gradients stay live. No fitting,
/// RNG draws beyond lazy initialization, detach or checkpoint selection is hidden.
/// Computed nonfinite values are not sanitized; incompatible leaves may fail later.
/// Inputs unchanged; all outputs own lifetime. Shared stores require synchronization.
/// Work includes O(B*hidden*(features+latent)+B*flow_hidden*(latent+hidden));
/// inverse diagnostics evaluate the shared conditioners again without extra leaves.
/// @param parameters Mutable matching floating store; normally seed3001 for this experiment.
/// @param image Finite Float32/64 pixel rows [B,features], B>0, in [0,1].
/// @param epsilon Finite same-dtype/batch explicit standard-normal rows [B,latent].
/// @param features Positive pixel width, default784.
/// @param latent Positive even code width, default4.
/// @param hidden Positive encoder-context/decoder width, default32.
/// @param flow_hidden Positive coupling-conditioner width, default16.
/// @return Owning eighteen-field flow VAE output with sampled scalar negative ELBO.
/// @throws std::invalid_argument For invalid inputs/options, overflow or incompatible leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(3001);
/// const auto out=n::models::FlowVariationalAutoencoder(
///     p,n::full({2,784},.2),n::ones({2,4}));
/// const auto gradients=n::grad(out.loss,p.parameters());
/// // Caller owns the optimizer and updates; all eighteen leaves receive derivatives.
/// @endcode
/// @par References
/// - Rezende and Mohamed (2015), Variational Inference with Normalizing Flows,
///   https://proceedings.mlr.press/v37/rezende15.html . Uses posterior change of variables.
/// - Dinh, Sohl-Dickstein and Bengio (2016/2017), https://arxiv.org/abs/1605.08803 .
///   Uses two small context-conditioned affine couplings, not the paper's benchmark network.
/// @see FlowVaeOutput, ConditionalAffinePosteriorFlow(), GaussianVaeDecode()
inline FlowVaeOutput FlowVariationalAutoencoder(
    ParameterStore& parameters,
    const Tensor&   image,
    const Tensor&   epsilon,
    std::int64_t    features    = 784,
    std::int64_t    latent      = 4,
    std::int64_t    hidden      = 32,
    std::int64_t    flow_hidden = 16) {
  aae_detail::Widths(features, latent, hidden);
  aae_detail::Matrix(image, features, parameters.dtype(), true);
  aae_detail::Matrix(epsilon, latent, parameters.dtype(), false);
  if (latent % 2 || flow_hidden <= 0 || image.size(0) != epsilon.size(0) ||
      hidden > std::numeric_limits<std::int64_t>::max() - latent / 2)
    throw std::invalid_argument("flow VAE needs matching batches, even codes and positive widths");
  FlowVaeOutput out;
  out.context      = dense(parameters, "encoder", image, hidden, Activation::Relu);
  out.mu           = dense(parameters, "mu", out.context, latent);
  out.log_variance = dense(parameters, "log_variance", out.context, latent);
  out.z0           = GaussianReparameterize(out.mu, out.log_variance, epsilon);
  const auto first = ConditionalAffinePosteriorFlow(
                 parameters, "flow0", out.z0, out.context, false, false, latent, flow_hidden),
             second = ConditionalAffinePosteriorFlow(
                 parameters, "flow1", first.value, out.context, true, false, latent, flow_hidden);
  out.z1        = first.value;
  out.z         = second.value;
  out.logdet0   = first.log_determinant;
  out.logdet1   = second.log_determinant;
  out.logdet    = out.logdet0 + out.logdet1;
  double log2pi = std::log(2 * 3.14159265358979323846);
  if (image.dtype() == DType::Float32)
    log2pi = double(float(log2pi));
  out.logq0 =
      -0.5 * sum(square(out.z0 - out.mu) * exp(-out.log_variance) + out.log_variance + log2pi, {1});
  out.logq          = out.logq0 - out.logdet;
  out.logp          = -0.5 * sum(square(out.z) + log2pi, {1});
  out.logratio      = out.logq - out.logp;
  const auto logits = GaussianVaeDecode(parameters, out.z, features, latent, hidden);
  out.probability   = sigmoid(logits);
  out.nll           = BernoulliPixelNll(logits, image);
  out.loss          = mean(out.nll + out.logratio, {0});
  const auto back1  = ConditionalAffinePosteriorFlow(
                 parameters, "flow1", out.z, out.context, true, true, latent, flow_hidden),
             back0 = ConditionalAffinePosteriorFlow(
                 parameters, "flow0", back1.value, out.context, false, true, latent, flow_hidden);
  out.inverse_z0     = back0.value;
  out.inverse_logdet = back1.log_determinant + back0.log_determinant;
  return out;
}
} // namespace datamunge::nn::models
