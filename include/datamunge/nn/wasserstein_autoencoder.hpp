#pragma once
/// @file wasserstein_autoencoder.hpp
/// @brief Deterministic WAE-MMD inference, pixel-summed squared cost and explicit signed objective.
#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/kernel_discrepancy.hpp"

namespace datamunge::nn::models {
/// @brief Validation for deterministic WAE controls and architecture products.
namespace wae_detail {
/// @brief Validate nonnegative MMD strength in the computation dtype.
/// @param weight Finite nonnegative discrepancy coefficient.
/// @param type Float32 or Float64 computation dtype.
/// @return Coefficient narrowed to Float32 when applicable, then promoted to double.
/// @throws std::invalid_argument For invalid controls/type, overflow or positive underflow.
inline double Weight(double weight, DType type) {
  if (!std::isfinite(weight) || weight < 0 || (type != DType::Float32 && type != DType::Float64))
    throw std::invalid_argument("WAE needs a finite nonnegative MMD weight and floating dtype");
  if (type == DType::Float32) {
    const float value = float(weight);
    if (!std::isfinite(value) || (weight > 0 && value == 0))
      throw std::invalid_argument("WAE weight must be representable in the computation dtype");
    return value;
  }
  return weight;
}
/// @brief Check positive widths, batch count and representable dense products.
/// @param batch Positive image/code row count.
/// @param features Positive pixel width.
/// @param latent Positive code width.
/// @param hidden Positive tanh width.
/// @throws std::invalid_argument For nonpositive dimensions or int64 product overflow.
inline void
Geometry(std::int64_t batch, std::int64_t features, std::int64_t latent, std::int64_t hidden) {
  aae_detail::Widths(features, latent, hidden);
  const auto limit = std::numeric_limits<std::int64_t>::max();
  if (batch <= 0 || features > limit / hidden || latent > limit / hidden ||
      batch > limit / std::max(features, std::max(latent, hidden)))
    throw std::invalid_argument("WAE architecture products overflow");
}
} // namespace wae_detail

/// @brief Sum squared unit-range pixel differences independently for each image.
///
/// Returns sum_j(image_ij-probability_ij)^2, with no pixel averaging or clipping.
/// Both arguments are finite matching nonempty Float32/64[B,F] matrices in[0,1].
/// This is a squared reconstruction cost, not a Gaussian/Bernoulli log density.
/// Strides, first/second derivatives through both inputs and owning lifetime
/// are preserved. Inputs unchanged; no state, RNG or fitting. Work/storage O(B*F).
/// @param image Finite unit-range targets[B,F], B,F positive.
/// @param probability Finite same-shape/dtype decoder values in[0,1].
/// @return Owning per-image squared pixel sums[B], in the input dtype.
/// @throws std::invalid_argument For invalid shapes, dtype or values.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// const auto cost=n::models::WaeSquaredPixelCost(n::zeros({2,4}),n::full({2,4},.5));
/// // cost is[1,1]: four squared errors of .25 per image.
/// @endcode
/// @par References
/// - Tolstikhin, Bousquet, Gelly and Schoelkopf (2018), Wasserstein Auto-Encoders,
///   https://arxiv.org/abs/1711.01558 . Uses the pixel-summed squared cost.
/// @see WaeObjective(), WassersteinAutoencoder()
inline Tensor WaeSquaredPixelCost(const Tensor& image, const Tensor& probability) {
  if (!image.defined() || image.dim() != 2)
    throw std::invalid_argument("WAE squared cost needs pixel matrices");
  aae_detail::Matrix(image, image.size(1), image.dtype(), true);
  aae_detail::Matrix(probability, image.size(1), image.dtype(), true);
  if (image.shape() != probability.shape())
    throw std::invalid_argument("WAE squared cost shapes must match");
  return sum(square(image - probability), {1});
}

/// @brief Average per-image squared cost and add explicitly weighted signed MMD squared.
///
/// Computes mean(squared_cost)+weight*mmd. Cost is a finite nonnegative nonempty
/// Float32/64[B] vector; mmd is one finite matching-dtype value and may be negative.
/// No square root or nonnegative clipping is applied. Zero weight retains a zero
/// derivative to mmd. Float32 narrows weight before multiplication; positive
/// underflow/overflow is rejected. This penalized finite-sample objective is not
/// an exact Wasserstein distance, image likelihood or negative ELBO. Inputs
/// unchanged, owning scalar and live first/second derivatives. Work O(B); no state.
/// @param squared_cost Finite nonnegative per-image pixel sums[B], B>0.
/// @param mmd Finite signed scalar/one-element aggregate discrepancy, matching dtype.
/// @param weight Finite nonnegative coefficient, default10.
/// @return Owning scalar mean reconstruction plus weighted signed MMD.
/// @throws std::invalid_argument For invalid tensors or unrepresentable controls.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// const auto loss=n::models::WaeObjective(n::tensor({2.,4.}),n::scalar(-.1));
/// // mean([2,4])+10*(-.1) = 2; negative discrepancy is retained.
/// @endcode
/// @par References
/// - Tolstikhin et al. (2018), Wasserstein Auto-Encoders,
///   https://arxiv.org/abs/1711.01558 . Uses the WAE-MMD penalized objective.
/// @see WaeSquaredPixelCost(), ImqMmd(), WassersteinAutoencoder()
inline Tensor WaeObjective(const Tensor& squared_cost, const Tensor& mmd, double weight = 10) {
  if (!squared_cost.defined() || squared_cost.dim() != 1 || squared_cost.size(0) <= 0 ||
      !mmd.defined() || mmd.numel() != 1 || mmd.dtype() != squared_cost.dtype())
    throw std::invalid_argument(
        "WAE objective needs floating cost rows and one matching MMD value");
  const auto coefficient = wae_detail::Weight(weight, squared_cost.dtype());
  for (double value : squared_cost.contiguous().to_vector<double>())
    if (!std::isfinite(value) || value < 0)
      throw std::invalid_argument("WAE squared costs must be finite and nonnegative");
  if (!std::isfinite(mmd.item<double>()))
    throw std::invalid_argument("WAE discrepancy must be finite");
  return mean(squared_cost, {0}) + coefficient * reshape(mmd, {});
}

/// @brief Decode explicit deterministic WAE codes through a tanh/linear network.
///
/// Creates/reuses decoder/W[L,H],decoder/b[H],output/W[H,F],output/b[F], in that
/// order, with ordinary Glorot-uniform weights and zero biases. Four default
/// leaves contain13,408 scalars and are shared with WassersteinAutoencoder().
/// Only supplied codes enter; no image, encoder, prior draws, MMD, sigmoid or fit
/// is hidden. Apply sigmoid() for unit-range decoder values. Strided finite
/// signed/zero Float32/64 codes retain input/parameter derivatives and owning
/// lifetime. Validation precedes leaves; incompatible existing leaves can fail
/// later. Computed overflow is not repaired. Inputs unchanged; synchronize shared
/// stores. Work O(B*H*(L+F)), activation storage O(B*(H+F)).
/// @param parameters Mutable matching floating store, containing/creating decoder leaves.
/// @param code Finite nonempty code matrix[B,latent].
/// @param features Positive output width, default784.
/// @param latent Positive code width, default4.
/// @param hidden Positive tanh width, default16.
/// @return Owning differentiable linear logits[B,features].
/// @throws std::invalid_argument For invalid tensors/widths/products or incompatible leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(3701);
/// const auto images=n::sigmoid(n::models::WaeDecode(p,n::zeros({20,4})));
/// // Twenty code-only decoder values[20,784]; no pixel sampling is performed.
/// @endcode
/// @par References
/// - Tolstikhin et al. (2018), https://arxiv.org/abs/1711.01558 . This is the
///   migrated small tanh teaching architecture, not the paper's benchmark network.
/// @see WassersteinAutoencoder(), WaeOutput
inline Tensor WaeDecode(
    ParameterStore& parameters,
    const Tensor&   code,
    std::int64_t    features = 784,
    std::int64_t    latent   = 4,
    std::int64_t    hidden   = 16) {
  if (!code.defined() || code.dim() != 2)
    throw std::invalid_argument("WAE decoder needs matrix codes");
  wae_detail::Geometry(code.size(0), features, latent, hidden);
  aae_detail::Matrix(code, latent, parameters.dtype(), false);
  return dense(
      parameters, "output", dense(parameters, "decoder", code, hidden, Activation::Tanh), features);
}

/// @brief Own six deterministic WAE endpoints and one nested seven-field MMD record.
///
/// Every tensor owns lifetime and retains autograd history, including nested
/// kernel matrices and signed scalar terms. No RNG, optimizer or fitting state
/// is stored. probability names unit-range decoder values, not sampled pixels.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(3701);
/// const auto out=n::models::WassersteinAutoencoder(p,n::full({2,784},.2),n::zeros({3,4}));
/// // code[2,4], squared_cost[2], mmd.xy[2,3], scalar loss; all own lifetime.
/// @endcode
/// @see WassersteinAutoencoder(), MmdOutput
struct WaeOutput {
  Tensor    code;                ///< Owning deterministic linear code[B,L].
  Tensor    logits;              ///< Owning code-only decoder logits[B,F].
  Tensor    probability;         ///< Owning sigmoid decoder values[B,F], not sampled pixels.
  Tensor    squared_cost;        ///< Owning per-image sum of squared pixel differences[B].
  Tensor    reconstruction_loss; ///< Owning image-mean squared cost scalar.
  Tensor    loss;                ///< Owning mean squared cost plus weighted signed MMD scalar.
  MmdOutput mmd;                 ///< Owning within/cross kernels and unbiased signed discrepancy.
};

/// @brief Evaluate a deterministic tanh WAE with explicit independent-prior MMD regularization.
///
/// Encodes image through encoder tanh then linear code, decodes through WaeDecode
/// and sigmoid, and returns pixel-summed squared costs plus signed ImqMmd(code,prior).
/// Defaults F784/L4/H16 create eight leaves/26,036 scalars in source order:
/// encoder/W,b; code/W,b; decoder/W,b; output/W,b. Ordinary Glorot-uniform weights
/// and zero biases; source seed3701. No per-image Gaussian posterior, KL,
/// Bernoulli training loss, discriminator, sampling, optimizer or fitting loop.
///
/// Image and prior counts B,P>=2 can differ; even weight0 computes full MMD and
/// requires these counts. Prior values do not affect codes or reconstructions.
/// Independent standard-normal sampling and fitting-only provenance are caller
/// responsibilities; no array-level check establishes statistical independence.
/// IMQ averages scales1,4,16 without dimension scaling. Within estimates exclude
/// diagonals, cross includes all pairs; a negative finite-sample estimate is
/// retained. This teaching variant is not an exact transport-distance solver.
///
/// Finite matching Float32/64 strided inputs preserve image, prior and parameter
/// first/second derivatives and all owning result lifetimes. Widths, controls,
/// matrices and pairwise/architecture products are checked before leaves.
/// Incompatible existing leaves or nonfinite computed values can fail later;
/// no rollback or overflow repair. Inputs unchanged; synchronize shared stores.
/// Work includes O(B*H*(F+L)+(B^2+P^2+B*P)*L), materializing pairwise differences.
/// @param parameters Mutable matching floating store; original seed3701.
/// @param image Finite unit-range pixels[B,features], B>=2.
/// @param prior Explicit finite independent standard-normal samples[P,latent], P>=2.
/// @param weight Finite nonnegative signed-MMD coefficient, default10.
/// @param features Positive pixel width, default784.
/// @param latent Positive code width, default4.
/// @param hidden Positive tanh width, default16.
/// @return Owning six-tensor plus nested MMD output, without fitting or random prior draws.
/// @throws std::invalid_argument For invalid inputs/options/products or incompatible leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(3701);
/// const auto out=n::models::WassersteinAutoencoder(p,n::full({2,784},.2),n::zeros({3,4}));
/// const auto derivatives=n::grad(out.loss,p.parameters());
/// // Caller applies its own Adam step to all eight leaves.
/// @endcode
/// @par References
/// - Tolstikhin, Bousquet, Gelly and Schoelkopf (2018), Wasserstein Auto-Encoders,
///   ICLR2018, https://arxiv.org/abs/1711.01558 . Uses a compact deterministic
///   WAE-MMD teaching model, fixed three-scale averaged IMQ kernels and weight10.
/// @see WaeOutput, WaeDecode(), WaeSquaredPixelCost(), WaeObjective(), ImqMmd()
inline WaeOutput WassersteinAutoencoder(
    ParameterStore& parameters,
    const Tensor&   image,
    const Tensor&   prior,
    double          weight   = 10,
    std::int64_t    features = 784,
    std::int64_t    latent   = 4,
    std::int64_t    hidden   = 16) {
  wae_detail::Weight(weight, parameters.dtype());
  if (!image.defined() || image.dim() != 2 || !prior.defined() || prior.dim() != 2)
    throw std::invalid_argument("WAE needs image and independent-prior matrices");
  wae_detail::Geometry(image.size(0), features, latent, hidden);
  aae_detail::Matrix(image, features, parameters.dtype(), true);
  aae_detail::Matrix(prior, latent, parameters.dtype(), false);
  const auto b = image.size(0), p = prior.size(0);
  if (b < 2 || p < 2)
    throw std::invalid_argument("unbiased WAE MMD needs at least two samples in each set");
  mmd_detail::Products(b, b, latent);
  mmd_detail::Products(p, p, latent);
  mmd_detail::Products(b, p, latent);
  WaeOutput out;
  out.code = dense(
      parameters, "code", dense(parameters, "encoder", image, hidden, Activation::Tanh), latent);
  out.logits              = WaeDecode(parameters, out.code, features, latent, hidden);
  out.probability         = sigmoid(out.logits);
  out.squared_cost        = WaeSquaredPixelCost(image, out.probability);
  out.reconstruction_loss = mean(out.squared_cost, {0});
  out.mmd                 = ImqMmd(out.code, prior);
  out.loss                = WaeObjective(out.squared_cost, out.mmd.value, weight);
  return out;
}
} // namespace datamunge::nn::models
