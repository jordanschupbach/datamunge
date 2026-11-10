#pragma once
/// @file tied_autoencoder.hpp
/// @brief Live transposed-weight ReLU/sigmoid autoencoders and their independent-weight control.

#include <cmath>
#include <stdexcept>

#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn::models {
/// @brief Validation shared by tied-weight forward and standalone decoding.
namespace tied_ae_detail {
/// @brief Validate strict undercomplete geometry, floating rows and compatible tied-store names.
/// @param p Matching Float32/64 parameter store.
/// @param x Finite nonempty matrix with the specified column count.
/// @param features Original input/output width, greater than one.
/// @param latent Positive code width strictly below features.
/// @param columns Required matrix column count.
/// @param tied Reject a pre-existing independent decoder/W when true.
/// @throws std::invalid_argument For invalid widths, input, dtype or a conflicting decoder leaf.
inline void Validate(
    const ParameterStore& p,
    const Tensor&         x,
    std::int64_t          features,
    std::int64_t          latent,
    std::int64_t          columns,
    bool                  tied) {
  if (features <= 1 || latent <= 0 || latent >= features)
    throw std::invalid_argument("tied autoencoder requires 0<latent<features and features>1");
  if (!x.defined() || x.dim() != 2 || x.size(0) <= 0 || x.size(1) != columns ||
      (x.dtype() != DType::Float32 && x.dtype() != DType::Float64) || x.dtype() != p.dtype())
    throw std::invalid_argument(
        "tied autoencoder requires matching nonempty Float32/64 [B,features/code]");
  for (double value : x.contiguous().to_vector<double>())
    if (!std::isfinite(value))
      throw std::invalid_argument("tied autoencoder input must be finite");
  if (tied && p.contains("decoder/W"))
    throw std::invalid_argument("tied autoencoder store must not contain independent decoder/W");
}
} // namespace tied_ae_detail

/// @brief Own the live encoder/decoder weights, ReLU code and sigmoid reconstruction.
///
/// Tensor handles outlive this result and retain autograd history. In tied mode
/// decoder_weight is a transposed view of encoder_weight, not a parameter copy;
/// the reconstruction gradient sums both uses into the same encoder leaf.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// dm::nn::ParameterStore p(1853);
/// const auto net=dm::nn::models::TiedAutoencoder(p,dm::nn::zeros({2,3}),3,2);
/// // code is [2,2], reconstruction [2,3]; p owns three leaves/11 scalars.
/// @endcode
/// @see TiedAutoencoder(), TiedAutoencoderDecode()
struct TiedAutoencoderOutput {
  Tensor encoder_weight; ///< Shared encoder leaf [features,latent].
  Tensor decoder_weight; ///< Live transpose [latent,features], or independent control leaf.
  Tensor code; ///< ReLU activities [B,latent], with zero derivative at nonpositive preactivations.
  Tensor reconstruction; ///< Raw sigmoid pixels [B,features], without copied inputs or targets.
};

/// @brief Evaluate a ReLU encoder and sigmoid decoder with a live shared-weight transpose.
///
/// Computes H=relu(XW+encoder/b), R=sigmoid(HD+decoder/b). By default D=transpose(W)
/// is recomputed on every call and remains differentiable: encoder and decoder
/// contributions sum into encoder/W. Creates/reuses encoder/W[features,latent],
/// encoder/b[latent], decoder/b[features] in that order, with Glorot-uniform W
/// and zero biases. For tied=false, inserts independent decoder/W[latent,features]
/// before decoder/b, with the same Glorot limit; no automatic initial matching.
/// Default 784–32–784 has 25,904 scalars (50,992 independent control).
/// Inputs are finite nonempty Float32/64 matrices matching the store; strided
/// rows and signed/unbounded values are accepted without clipping. Input is not
/// mutated. All returned handles own lifetimes and retain input/parameter gradients.
/// There is no loss, optimizer or fitting policy. Width/input/conflicting decoder
/// checks precede creation; incompatible existing shapes can fail later. Nonfinite
/// stored/computed values propagate. Shared stores need external synchronization.
/// Work O(B*features*latent), activation storage O(B*(features+latent)).
/// @param parameters Mutable matching floating store with fixed encoder/decoder prefixes.
/// @param image Finite nonempty [B,features] input rows; flatten images explicitly.
/// @param features Input/output width, default 784, greater than one.
/// @param latent Code width, default 32, strictly between zero and features.
/// @param tied True by default; false selects the independent-decoder control.
/// @return Owning weights, code and reconstruction, retaining both shared gradient paths.
/// @throws std::invalid_argument For invalid widths/input/dtype, tied decoder/W conflict or
/// incompatible leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(1853);
/// const auto x=n::full({2,3},.2);
/// const auto net=n::models::TiedAutoencoder(p,x,3,2);
/// const auto gradients=n::grad(n::mse(net.reconstruction,x),p.parameters());
/// // Three gradients; encoder/W receives encoder and decoder contributions.
/// @endcode
/// @par References
/// - Theano Deep Learning Tutorials, Denoising Autoencoders, Autoencoders section,
///   https://deeplearningtutorials.readthedocs.io/en/latest/dA.html . Follows its live
///   W transpose construction; this source experiment uses ReLU codes, sigmoid pixels,
///   ordinary Glorot initialization and mean squared reconstruction without corruption.
/// @see TiedAutoencoderOutput, TiedAutoencoderDecode(), transpose(), grad()
inline TiedAutoencoderOutput TiedAutoencoder(
    ParameterStore& parameters,
    const Tensor&   image,
    std::int64_t    features = 784,
    std::int64_t    latent   = 32,
    bool            tied     = true) {
  tied_ae_detail::Validate(parameters, image, features, latent, features, tied);
  const double          limit = std::sqrt(6. / (double(features) + double(latent)));
  TiedAutoencoderOutput out;
  out.encoder_weight = parameters.get("encoder/W", {features, latent}, init::uniform(limit));
  const auto eb      = parameters.get("encoder/b", {latent}, init::zeros());
  out.code           = relu(bias_add(matmul(image, out.encoder_weight), eb));
  out.decoder_weight = tied ? transpose(out.encoder_weight, {1, 0})
                            : parameters.get("decoder/W", {latent, features}, init::uniform(limit));
  const auto db      = parameters.get("decoder/b", {features}, init::zeros());
  out.reconstruction = sigmoid(bias_add(matmul(out.code, out.decoder_weight), db));
  return out;
}

/// @brief Decode explicit codes using the live encoder transpose or independent control matrix.
///
/// Consumes only finite Float32/64 code[B,latent], B>0, matching the store.
/// In tied mode creates/reuses encoder/W[features,latent] and decoder/b[features];
/// it never creates encoder/b or decoder/W. Otherwise creates/reuses decoder/W
/// [latent,features] and decoder/b. Weights use ordinary Glorot uniform, bias zero.
/// A fresh tied decoder contains features*latent+features scalars, without unused
/// encoder bias. Restored encoder/W is transposed afresh; no image or target enters.
/// Output owns lifetime, retaining code/weight/bias gradients. Signed and zero codes
/// are allowed without ReLU or clipping. Same validation, error timing, mutation,
/// nonfinite-parameter and synchronization rules as TiedAutoencoder(); work
/// O(B*features*latent), activation storage O(B*features).
/// @param parameters Matching floating store with the selected matrix and decoder/b names.
/// @param code Finite nonempty [B,latent] explicit codes.
/// @param features Output width, default 784, greater than one.
/// @param latent Code width, default 32, strictly between zero and features.
/// @param tied True uses encoder/W transpose; false uses independent decoder/W.
/// @return Owning raw sigmoid reconstruction [B,features] retaining all decoder gradients.
/// @throws std::invalid_argument For invalid widths/input/dtype, tied decoder/W conflict or
/// incompatible leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(1853);
/// const auto net=n::models::TiedAutoencoder(p,n::zeros({2,3}),3,2);
/// const auto decoded=n::models::TiedAutoencoderDecode(p,net.code,3,2);
/// // decoded equals net.reconstruction; no image or target is supplied.
/// @endcode
/// @par References
/// - Theano Deep Learning Tutorials, Denoising Autoencoders,
///   https://deeplearningtutorials.readthedocs.io/en/latest/dA.html . Uses the same
///   live-transpose convention as TiedAutoencoder(), with source-specific sigmoid/MSE behavior.
/// @see TiedAutoencoder(), sigmoid()
inline Tensor TiedAutoencoderDecode(
    ParameterStore& parameters,
    const Tensor&   code,
    std::int64_t    features = 784,
    std::int64_t    latent   = 32,
    bool            tied     = true) {
  tied_ae_detail::Validate(parameters, code, features, latent, latent, tied);
  const double limit = std::sqrt(6. / (double(features) + double(latent)));
  const auto   weight =
      tied
            ? transpose(parameters.get("encoder/W", {features, latent}, init::uniform(limit)), {1, 0})
            : parameters.get("decoder/W", {latent, features}, init::uniform(limit));
  const auto bias = parameters.get("decoder/b", {features}, init::zeros());
  return sigmoid(bias_add(matmul(code, weight), bias));
}
} // namespace datamunge::nn::models
