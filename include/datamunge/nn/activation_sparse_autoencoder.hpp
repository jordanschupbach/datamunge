#pragma once
/// @file activation_sparse_autoencoder.hpp
/// @brief Activation-L1 and clamped mean-unit Bernoulli-KL objectives for sigmoid autoencoders.

#include <cmath>
#include <stdexcept>

#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn::models {
/// @brief Validation and dtype-consistent scalar constants for activation sparsity.
namespace activation_sparse_detail {
/// @brief Own scalar weights, target/complement and finite clipping endpoints in the computation
/// dtype.
struct Options {
  double l1;    ///< Nonnegative activation-L1 weight.
  double kl;    ///< Nonnegative mean-unit KL weight.
  double rho;   ///< Target probability.
  double other; ///< Separately rounded complement.
  double low;   ///< Strictly positive lower clipping endpoint.
  double high;  ///< Strictly below-one upper clipping endpoint.
};
/// @brief Validate and round scalar options once to the activation dtype.
/// @param l1 Nonnegative finite activation-L1 weight.
/// @param kl Nonnegative finite mean-unit KL weight.
/// @param rho Target probability strictly between zero and one.
/// @param epsilon Lower clipping bound strictly between zero and one half.
/// @param dtype Float32 or Float64 computation dtype.
/// @return Owned validated constants, rounded to dtype including both complements.
/// @throws std::invalid_argument For invalid values or endpoints collapsed by dtype rounding.
inline Options MakeOptions(double l1, double kl, double rho, double epsilon, DType dtype) {
  if ((dtype != DType::Float32 && dtype != DType::Float64) || !std::isfinite(l1) || l1 < 0 ||
      !std::isfinite(kl) || kl < 0 || !(rho > 0 && rho < 1) || !(epsilon > 0 && epsilon < .5))
    throw std::invalid_argument(
        "activation sparsity requires finite nonnegative weights, 0<rho<1 and 0<epsilon<.5");
  /// @brief Round a scalar to the tensor dtype before complements are formed.
  /// @param x Scalar to round.
  /// @return Value represented in the selected computation dtype.
  const auto rounded = [dtype](double x) { return dtype == DType::Float32 ? double(float(x)) : x; };
  Options    o{rounded(l1), rounded(kl), rounded(rho), 0, rounded(epsilon), 0};
  o.other = rounded(1. - o.rho);
  o.high  = rounded(1. - o.low);
  if (!std::isfinite(o.l1) || !std::isfinite(o.kl) || !(o.rho > 0 && o.rho < 1) ||
      !(o.other > 0 && o.other < 1) || !(o.low > 0 && o.low < .5) || !(o.high > .5 && o.high < 1))
    throw std::invalid_argument(
        "activation sparsity constants must remain finite/interior after dtype rounding");
  return o;
}
/// @brief Reject malformed/nonfinite activation or image matrices without changing their gradients.
/// @param x Nonempty floating matrix to validate.
/// @throws std::invalid_argument For undefined, nonmatrix, empty, nonfloating or nonfinite values.
inline void Matrix(const Tensor& x) {
  if (!x.defined() || x.dim() != 2 || x.size(0) <= 0 || x.size(1) <= 0 ||
      (x.dtype() != DType::Float32 && x.dtype() != DType::Float64))
    throw std::invalid_argument("activation sparsity requires nonempty Float32/64 [B,F]");
  for (double v : x.contiguous().to_vector<double>())
    if (!std::isfinite(v))
      throw std::invalid_argument("activation sparsity inputs must be finite");
}
} // namespace activation_sparse_detail

/// @brief Own activation penalties and raw/clipped per-unit batch means.
///
/// Every field owns its tensor lifetime and retains continuous autograd paths.
/// This value contains no learned parameters, optimizer or fit. A nested value
/// copied from SparseAeOutput remains valid after the parent result is released.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const auto penalty=dm::nn::models::ActivationSparsity(dm::nn::full({2,3},.05),.1,.2);
/// // unit_mean is [3]; l1, kl and total are scalar tensors.
/// @endcode
/// @see ActivationSparsity(), SparseAeOutput
struct SparsityPenalty {
  Tensor l1;           ///< Mean absolute activation over B*F entries, not a weight penalty.
  Tensor kl;           ///< Mean Bernoulli target-to-clipped-unit-mean divergence over F units.
  Tensor unit_mean;    ///< Unclipped mean activation of each unit over B observations, [F].
  Tensor clipped_mean; ///< Unit means clipped to interior dtype-rounded endpoints, [F].
  Tensor total;        ///< l1_weight*l1 + kl_weight*kl, scalar.
};

/// @brief Compute activation-L1 and clamped mean-unit Bernoulli-KL penalties.
///
/// For finite nonempty code[B,F], l1=mean(abs(code)), q=mean(code,axis0),
/// c=clip(q,epsilon,1-epsilon) and kl=mean_j[rho*log(rho/c_j)+
/// (1-rho)*log((1-rho)/(1-c_j))]. Unlike summing unit divergences, the mean
/// divides by F. Constants, complements and endpoints round to the input dtype;
/// Float32 reproduces the source's float constants, while Float64 keeps double
/// precision. Rounding can give tiny signed KL residuals near the minimum.
/// Signed codes are accepted: KL then concerns clipped means, not raw probabilities.
/// Values are not mutated; strides and all code gradients are retained. abs has
/// zero derivative at zero. clip passes gradients at either endpoint and zeros
/// them outside; no smoothness is promised at zero or clipping boundaries.
/// Splitting a batch generally changes KL and its derivatives. Both penalties
/// are evaluated even when their weight is zero. Work/storage O(B*F), no shared
/// mutable state. This does not impose exact zeros or penalize model weights.
/// @param code Finite Float32/64 matrix [B,F], B,F>0; conventionally sigmoid activations.
/// @param l1_weight Finite nonnegative activation-L1 coefficient.
/// @param kl_weight Finite nonnegative mean-unit KL coefficient.
/// @param rho Target mean probability in (0,1), default .05.
/// @param epsilon Interior clipping bound in (0,.5), default 1e-5.
/// @return Owning scalar components and per-unit means with code gradients retained.
/// @throws std::invalid_argument For invalid code/options or constants collapsed by dtype rounding.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// const auto h=n::tensor({.1,.3,.3,.5},{2,2},n::DType::Float64).requires_grad_();
/// const auto penalty=n::models::ActivationSparsity(h,.1,.2,.2);
/// // unit means are [.2,.4]; l1 is .3. The caller adds penalty.total to its loss.
/// @endcode
/// @par References
/// - A. Ng, Sparse autoencoder, CS294A lecture notes,
///   https://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf .
///   This implementation uses mean unit divergence, clipping and activation L1,
///   with the source experiment's defaults rather than the notes' full objective.
/// @see SparseAutoencoder(), SparsityPenalty, clip()
inline SparsityPenalty ActivationSparsity(
    const Tensor& code,
    double        l1_weight,
    double        kl_weight,
    double        rho     = .05,
    double        epsilon = 1e-5) {
  activation_sparse_detail::Matrix(code);
  const auto o =
      activation_sparse_detail::MakeOptions(l1_weight, kl_weight, rho, epsilon, code.dtype());
  SparsityPenalty out;
  out.l1              = mean(abs(code));
  out.unit_mean       = mean(code, {0});
  out.clipped_mean    = clip(out.unit_mean, o.low, o.high);
  const auto positive = o.rho * log(o.rho / out.clipped_mean);
  const auto negative = o.other * log(o.other / (1. - out.clipped_mean));
  out.kl              = mean(positive + negative, {0});
  out.total           = o.l1 * out.l1 + o.kl * out.kl;
  return out;
}

/// @brief Own sigmoid encoder/decoder values and their complete activation-regularized objective.
///
/// All handles, including nested penalty fields, own lifetimes and retain
/// gradients. The store owns learned leaves; no optimizer or training state is
/// hidden in this result.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(1601);
/// const auto net=n::models::SparseAutoencoder(p,n::zeros({2,784}),.05,.05);
/// // 64 sigmoid activities, 784 sigmoid predictions and MSE+L1+KL loss.
/// @endcode
/// @see SparseAutoencoder(), SparsityPenalty
struct SparseAeOutput {
  Tensor          code;           ///< Sigmoid activities [B,latent], without hard selection.
  Tensor          reconstruction; ///< Independent sigmoid decoder output [B,features].
  Tensor          mse;            ///< Mean squared reconstruction error over all B*features pixels.
  Tensor          loss;           ///< mse + penalty.total, scalar.
  SparsityPenalty penalty;        ///< Owning activation-only components and unit means.
};

/// @brief Evaluate an untied sigmoid autoencoder with activation-L1 and mean-unit KL objectives.
///
/// Creates/reuses encoder/W[features,latent], encoder/b[latent],
/// decoder/W[latent,features] and decoder/b[features], with Glorot-uniform
/// weights and zero biases. Defaults give 101200 scalars in four leaves.
/// Accepts finite nonempty matching Float32/64 images[B,features], without
/// clipping or range normalization; code and reconstruction use sigmoid.
/// Both weights may be zero for a matched plain control. Batch-dependent KL
/// changes loss, not the observation-wise encoder/decoder prediction. Complete
/// input/parameter gradients remain, including direct target derivatives when
/// image requires gradients. The caller owns Adam, batch order, epochs and
/// validation selection. Invalid inputs/options fail before leaf creation;
/// incompatible leaves or nonfinite computed activities can fail after creation. No input
/// mutation; shared stores need synchronization. Dense work O(B*features*latent),
/// activation storage O(B*(features+latent)).
/// @param parameters Mutable matching Float32/64 parameter store.
/// @param image Finite [B,features] image rows, B>0.
/// @param l1_weight Finite nonnegative activation-L1 coefficient.
/// @param kl_weight Finite nonnegative mean-unit KL coefficient.
/// @param features Positive input/output width, default784.
/// @param latent Positive code width, default64; no strict bottleneck is required.
/// @param rho Target mean in (0,1), default .05.
/// @param epsilon Interior clipping bound in (0,.5), default 1e-5.
/// @return Owning code, reconstruction, MSE, both penalties and total loss.
/// @throws std::invalid_argument For invalid options/input or incompatible stored leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(7,n::DType::Float64);
/// const auto net=n::models::SparseAutoencoder(p,n::full({2,3},.2,n::DType::Float64),.1,.2,3,2);
/// // Seventeen scalars; the caller explicitly optimizes net.loss.
/// @endcode
/// @par References
/// - A. Ng, Sparse autoencoder, https://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf .
///   This variant averages pixel MSE and unit KL and adds activation L1, not weight decay.
/// @see ActivationSparsity(), SparseAeOutput, dense()
inline SparseAeOutput SparseAutoencoder(
    ParameterStore& parameters,
    const Tensor&   image,
    double          l1_weight,
    double          kl_weight,
    std::int64_t    features = 784,
    std::int64_t    latent   = 64,
    double          rho      = .05,
    double          epsilon  = 1e-5) {
  activation_sparse_detail::Matrix(image);
  if (features <= 0 || latent <= 0 || image.size(1) != features ||
      image.dtype() != parameters.dtype())
    throw std::invalid_argument(
        "sparse autoencoder requires matching [B,features] and positive widths");
  activation_sparse_detail::MakeOptions(l1_weight, kl_weight, rho, epsilon, image.dtype());
  SparseAeOutput out;
  out.code           = dense(parameters, "encoder", image, latent, Activation::Sigmoid);
  out.reconstruction = dense(parameters, "decoder", out.code, features, Activation::Sigmoid);
  out.mse            = mean(square(out.reconstruction - image));
  out.penalty        = ActivationSparsity(out.code, l1_weight, kl_weight, rho, epsilon);
  out.loss           = out.mse + out.penalty.total;
  return out;
}
} // namespace datamunge::nn::models
