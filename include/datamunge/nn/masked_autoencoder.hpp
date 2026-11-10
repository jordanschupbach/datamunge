#pragma once
/// @file masked_autoencoder.hpp
/// @brief Visible-only MNIST patch autoencoder components, deterministic masks and masked
/// reconstruction loss.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "datamunge/nn/attention.hpp"
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn::models {
/// @brief Shape, discrete-selector and unsigned-hash implementation details.
namespace mae_detail {
/// @brief Validate a finite floating tensor without changing its values or gradient history.
/// @param value Tensor to inspect.
/// @throws std::invalid_argument For undefined, nonfloating or nonfinite values.
inline void Floating(const Tensor& value) {
  if (!value.defined() || (value.dtype() != DType::Float32 && value.dtype() != DType::Float64))
    throw std::invalid_argument("MAE requires defined Float32/64 tensors");
  for (double x : value.contiguous().to_vector<double>())
    if (!std::isfinite(x))
      throw std::invalid_argument("MAE input values must be finite");
}
/// @brief Check four distinct one-hot visible patch rows for every observation.
/// @param selector Detached floating [batch,4,16] selector, with the supplied dtype.
/// @param batch Required nonempty batch size.
/// @param dtype Required floating dtype.
/// @throws std::invalid_argument For wrong shape/dtype, gradient-enabled, soft, duplicate or
/// missing rows.
inline void Selection(const Tensor& selector, std::int64_t batch, DType dtype) {
  if (!selector.defined() || selector.shape() != Shape{batch, 4, 16} || selector.dtype() != dtype ||
      selector.requires_grad() || batch <= 0)
    throw std::invalid_argument("MAE selector must be detached matching [B,4,16]");
  const auto values = selector.contiguous().to_vector<double>();
  for (std::int64_t b = 0; b < batch; ++b) {
    std::uint32_t seen = 0;
    for (int row = 0; row < 4; ++row) {
      int chosen = -1;
      for (int patch = 0; patch < 16; ++patch) {
        const double v = values[b * 64 + row * 16 + patch];
        if (v != 0 && v != 1)
          throw std::invalid_argument("MAE selector rows must be binary one-hot");
        if (v == 1) {
          if (chosen != -1)
            throw std::invalid_argument("MAE selector rows must be one-hot");
          chosen = patch;
        }
      }
      if (chosen < 0 || (seen & (1u << chosen)))
        throw std::invalid_argument("MAE selector must choose four distinct patches");
      seen |= 1u << chosen;
    }
  }
}
/// @brief Preserve the source example's fixed wrapping 32-bit avalanche transform.
/// @param x Raw hash input.
/// @return Mixed bits, using unsigned modulo-2^32 multiplication and logical shifts.
inline std::uint32_t Mix(std::uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  return x ^ (x >> 16);
}
} // namespace mae_detail

/// @brief Map a row-major 7-by-7 patch element to its flat 28-by-28 image index.
/// @param patch Patch index in [0,15], ordered by grid row then column.
/// @param within Pixel index in [0,48], ordered by patch row then column.
/// @return Zero-based flat image index in [0,783].
/// @throws std::invalid_argument For an out-of-range index.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const auto index=dm::nn::models::MaePixel(5,0); // 203
/// @endcode
/// @see MaePatches(), MaeImage()
inline std::int64_t MaePixel(std::int64_t patch, std::int64_t within) {
  if (patch < 0 || patch >= 16 || within < 0 || within >= 49)
    throw std::invalid_argument("MAE patch/pixel index out of range");
  return ((patch / 4) * 7 + within / 7) * 28 + (patch % 4) * 7 + within % 7;
}

/// @brief Extract sixteen nonoverlapping row-major 7-by-7 grayscale patches.
///
/// Requires finite Float32/64 NHWC images [B,28,28,1], B>0. No clipping or
/// normalization is applied. Strides and input gradients are retained through
/// reshape/transpose; values are never mutated. Work/storage are O(B*784).
/// @param image Finite floating nonempty grayscale image batch.
/// @return Differentiable [B,16,49] patch tensor in row/column patch order.
/// @throws std::invalid_argument For wrong shape/dtype or nonfinite pixels.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const auto patches=dm::nn::models::MaePatches(dm::nn::ones({2,28,28,1}));
/// // patches is [2,16,49], with every element one.
/// @endcode
/// @see MaeImage(), MaePixel(), MaskedAutoencoder()
inline Tensor MaePatches(const Tensor& image) {
  mae_detail::Floating(image);
  if (image.dim() != 4 || image.size(0) <= 0 || image.shape() != Shape{image.size(0), 28, 28, 1})
    throw std::invalid_argument("MAE images must be nonempty [B,28,28,1]");
  const auto b = image.size(0);
  return reshape(transpose(reshape(image, {b, 4, 7, 4, 7}), {0, 1, 3, 2, 4}), {b, 16, 49});
}

/// @brief Invert row-major patch extraction with differentiable tensor permutations.
///
/// Supports strided Float32/64 tensors, including empty batches. No pixel range
/// or finiteness check is imposed; values/gradients are permuted without mutation.
/// @param patches Floating [B,16,49] patch values.
/// @return Differentiable flat images [B,784], preserving dtype and pixel bits.
/// @throws std::invalid_argument For undefined/nonfloating/wrong-shaped inputs.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// const auto image=n::ones({1,28,28,1});
/// const auto flat=n::models::MaeImage(n::models::MaePatches(image));
/// // flat is [1,784], an exact inverse of patch extraction.
/// @endcode
/// @see MaePatches(), MaePixel()
inline Tensor MaeImage(const Tensor& patches) {
  if (!patches.defined() || patches.dim() != 3 || patches.size(1) != 16 || patches.size(2) != 49 ||
      (patches.dtype() != DType::Float32 && patches.dtype() != DType::Float64))
    throw std::invalid_argument("MAE patch images require floating [B,16,49]");
  const auto b = patches.size(0);
  return reshape(transpose(reshape(patches, {b, 4, 4, 7, 7}), {0, 1, 3, 2, 4}), {b, 784});
}

/// @brief Generate fixed grouped sine/cosine positions for the four-by-four patch grid.
///
/// Columns are row-sines, row-cosines, column-sines, column-cosines, each width/4
/// long. Frequency k is 10000^(-4*k/width); coordinates are grid indices 0..3.
/// This grouped layout differs from interleaved sinusoidal_encoding_2d(). Values
/// are computed in Float64 then rounded to dtype; no shared state or gradients.
/// @param width Positive encoding width divisible by four.
/// @param dtype Float32 by default, or Float64.
/// @return Detached [16,width] position table.
/// @throws std::invalid_argument For invalid width or nonfloating dtype.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const auto positions=dm::nn::models::MaePosition(4);
/// // First row is [0,1,0,1].
/// @endcode
/// @par References
/// - He et al. (2021/2022), Masked Autoencoders Are Scalable Vision Learners,
///   https://arxiv.org/abs/2111.06377 . This is the source example's small fixed-grid variant.
/// @see MaeEncode(), MaeDecode()
inline Tensor MaePosition(std::int64_t width, DType dtype = DType::Float32) {
  if (width <= 0 || width % 4 || (dtype != DType::Float32 && dtype != DType::Float64))
    throw std::invalid_argument(
        "MAE position width must be a positive multiple of four with floating dtype");
  std::vector<double> values(static_cast<std::size_t>(shape_numel({16, width})));
  for (int patch = 0; patch < 16; ++patch)
    for (int axis = 0; axis < 2; ++axis)
      for (std::int64_t k = 0; k < width / 4; ++k) {
        const double angle =
            (axis ? patch % 4 : patch / 4) * std::pow(10000., -2. * k / (width / 2));
        values[patch * width + axis * width / 2 + k]             = std::sin(angle);
        values[patch * width + axis * width / 2 + width / 4 + k] = std::cos(angle);
      }
  return cast(from_vector(values, {16, width}), dtype);
}

/// @brief Select four distinct visible patches by stable per-record unsigned hashes.
///
/// Ranks the source's exact wrapping 32-bit mix of ID, epoch, seed and patch index;
/// hash ties prefer lower patch indices. Visible rows follow hash order, not
/// spatial order. This is deterministic selection, not a uniform-randomness claim.
/// No language/global RNG state is read or changed. Empty ID vectors are allowed.
/// @param source Int32/64 vector of source IDs in [0,4294967295].
/// @param epoch Whole epoch index in [0,4294967295], default zero for evaluation.
/// @param seed Explicit unsigned seed, default 1907; changing it changes the masks.
/// @param dtype Float32 by default, or Float64, matching the later image/store.
/// @return Detached binary [B,4,16] selector with four distinct one-hot rows per record.
/// @throws std::invalid_argument For invalid ID shape/dtype/range, epoch or output dtype.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// const auto selection=n::models::MaeSelection(n::zeros({2},n::DType::Int32));
/// // Two identical source IDs yield identical [4,16] selectors.
/// @endcode
/// @par References
/// - He et al. (2021/2022), https://arxiv.org/abs/2111.06377 . The source's
///   hash-derived 75-percent masks preserve its experiment, rather than the paper's random sampler.
/// @see MaskedAutoencoder(), MaeLoss()
inline Tensor MaeSelection(
    const Tensor& source,
    std::int64_t  epoch = 0,
    std::uint32_t seed  = 1907,
    DType         dtype = DType::Float32) {
  if (!source.defined() || source.dim() != 1 ||
      (source.dtype() != DType::Int32 && source.dtype() != DType::Int64) || epoch < 0 ||
      epoch > std::numeric_limits<std::uint32_t>::max() ||
      (dtype != DType::Float32 && dtype != DType::Float64))
    throw std::invalid_argument(
        "MAE source IDs must be integral vectors with a uint32 epoch and floating output dtype");
  const auto          ids = source.contiguous().to_vector<std::int64_t>();
  std::vector<double> values(static_cast<std::size_t>(shape_numel({source.size(0), 4, 16})), 0);
  for (std::size_t b = 0; b < ids.size(); ++b) {
    if (ids[b] < 0 || ids[b] > std::numeric_limits<std::uint32_t>::max())
      throw std::invalid_argument("MAE source IDs must fit uint32");
    std::vector<std::pair<std::uint32_t, int>> keys;
    for (int i = 0; i < 16; ++i)
      keys.emplace_back(
          mae_detail::Mix(
              seed ^ (std::uint32_t(ids[b]) * 0x9e3779b9u) ^ (std::uint32_t(epoch) * 0x85ebca6bu) ^
              (std::uint32_t(i) * 0xc2b2ae35u)),
          i);
    std::sort(keys.begin(), keys.end());
    for (int k = 0; k < 4; ++k)
      values[b * 64 + k * 16 + keys[k].second] = 1;
  }
  return cast(from_vector(values, {source.size(0), 4, 16}), dtype);
}

/// @brief Own a transformer block's residual output and per-head attention probabilities.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// dm::nn::ParameterStore p(17);
/// const auto block=dm::nn::models::MaeBlock(p,"example",dm::nn::ones({1,4,8}),8);
/// // output [1,4,8], attention [1,2,4,4].
/// @endcode
/// @see MaeBlock()
struct MaeBlockOutput {
  Tensor output;    ///< Differentiable residual block output [B,T,width].
  Tensor attention; ///< Differentiable softmax weights [B,2,T,T].
};

/// @brief Apply a two-head pre-normalized residual transformer block with tanh-GELU MLP.
///
/// LayerNorm epsilon is 1e-5. Bias-free Q/K/V/output attention projections use
/// scaled dot-product softmax, followed by a second pre-norm residual MLP with
/// widths width→2*width→width. Dense MLPs have biases; no dropout or causal mask.
/// Leaves use standard layer initializers and names name/ln1, attention, ln2,
/// ff1 and ff2. Inputs are not mutated; shared stores require synchronization.
/// @param parameters Mutable Float32/64 store matching x.
/// @param name Nonempty parameter prefix.
/// @param x Finite floating [B,T,width], with positive B,T.
/// @param width Positive even feature width matching x's last dimension.
/// @return Owning block output and per-head attention handles with gradients retained.
/// @throws std::invalid_argument For invalid geometry/dtype/prefix or incompatible stored leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(17);
/// const auto block=n::models::MaeBlock(p,"block",n::ones({2,3,4}),4);
/// // output [2,3,4]; attention [2,2,3,3].
/// @endcode
/// @par References
/// - He et al. (2021/2022), https://arxiv.org/abs/2111.06377 . Width, depth and
///   MLP ratio here retain the tiny source example, not a full ViT benchmark.
/// @see MaeEncode(), MaeDecode(), multi_head_attention()
inline MaeBlockOutput
MaeBlock(ParameterStore& parameters, const std::string& name, const Tensor& x, std::int64_t width) {
  mae_detail::Floating(x);
  if (name.empty() || x.dim() != 3 || x.size(0) <= 0 || x.size(1) <= 0 || width <= 0 || width % 2 ||
      x.size(2) != width || parameters.dtype() != x.dtype())
    throw std::invalid_argument(
        "MAE block requires matching floating [B,T,width], positive even width and a prefix");
  const auto       normalized = layer_norm(parameters, name + "/ln1", x);
  MultiHeadOptions options;
  options.need_weights = true;
  const auto attended =
      multi_head_attention(parameters, name + "/attention", normalized, normalized, 2, options);
  const auto residual = x + attended.output;
  const auto ff       = dense(
      parameters,
      name + "/ff1",
      layer_norm(parameters, name + "/ln2", residual),
      2 * width,
      Activation::GeluTanh);
  return {residual + dense(parameters, name + "/ff2", ff, width), attended.weights};
}

/// @brief Own patch, code and decoder endpoints without optimizer state.
///
/// Full forwards populate every field. MaeEncode populates patches, visible,
/// mask, encoder_input, code and encoder_attention; MaeDecode populates code,
/// mask, projected, restored, decoder_input, prediction and decoder_attention.
/// Other partial-result fields are undefined. All handles own their tensor
/// lifetimes and retain continuous autograd paths; masks/positions are discrete data.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(1877);
/// const auto s=n::models::MaeSelection(n::zeros({1},n::DType::Int32));
/// const auto net=n::models::MaskedAutoencoder(p,n::zeros({1,28,28,1}),s);
/// // code [1,4,32], prediction [1,16,49]; all endpoint handles are owning.
/// @endcode
/// @see MaeEncode(), MaeDecode(), MaskedAutoencoder()
struct MaskedAeOutput {
  Tensor patches;    ///< All original patches [B,16,49], retained for caller-owned scoring only.
  Tensor visible;    ///< Selected image content [B,4,49].
  Tensor mask;       ///< Hidden indicator [B,16], exactly twelve ones per full-model row.
  Tensor code;       ///< Visible-only normalized encoder tokens [B,4,32].
  Tensor prediction; ///< Raw sigmoid patch predictions [B,16,49], with no visible-pixel copy.
  Tensor encoder_attention; ///< Per-head encoder weights [B,2,4,4].
  Tensor decoder_attention; ///< Per-head decoder weights [B,2,16,16].
  Tensor encoder_input;     ///< Visible embeddings plus selected fixed positions [B,4,32].
  Tensor projected;         ///< Visible codes projected into decoder width [B,4,16].
  Tensor restored;          ///< Projected codes scattered to their original locations [B,16,16].
  Tensor decoder_input;     ///< Restored codes, hidden mask tokens and positions [B,16,16].
};

/// @brief Encode only four visible patches into normalized width-32 tokens.
///
/// Gather precedes the learned 49→32 patch embedding, so finite hidden pixel
/// changes do not enter the encoder. Selected fixed positions, one MaeBlock and
/// final LayerNorm form the code. No target is supplied or scored. Invalid
/// images/selectors/dtypes fail before parameter creation; later backend or
/// incompatible-leaf failures can leave earlier parameters created. No fitting,
/// clipping or input mutation; shared stores require external synchronization.
/// @param parameters Mutable floating store matching image and selector.
/// @param image Finite Float32/64 [B,28,28,1], B>0.
/// @param selector Matching detached [B,4,16], four distinct one-hot visible rows.
/// @return Owning encoder fields documented in MaskedAeOutput; decoder fields undefined.
/// @throws std::invalid_argument For invalid inputs or incompatible stored leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(1877);
/// const auto s=n::models::MaeSelection(n::zeros({1},n::DType::Int32));
/// const auto encoded=n::models::MaeEncode(p,n::zeros({1,28,28,1}),s);
/// // encoded.visible [1,4,49]; encoded.code [1,4,32].
/// @endcode
/// @par References
/// - He et al. (2021/2022), https://arxiv.org/abs/2111.06377 .
/// @see MaeDecode(), MaskedAutoencoder(), MaeLoss()
inline MaskedAeOutput
MaeEncode(ParameterStore& parameters, const Tensor& image, const Tensor& selector) {
  MaskedAeOutput out;
  out.patches = MaePatches(image);
  if (parameters.dtype() != image.dtype())
    throw std::invalid_argument("MAE store/image dtype mismatch");
  mae_detail::Selection(selector, image.size(0), image.dtype());
  out.visible           = matmul(selector, out.patches);
  out.mask              = 1. - sum(selector, {1});
  const auto embedded   = dense(parameters, "patch", out.visible, 32);
  const auto position   = matmul(selector, expand_dims(MaePosition(32, image.dtype()), 0));
  out.encoder_input     = embedded + position;
  const auto encoder    = MaeBlock(parameters, "encoder", out.encoder_input, 32);
  out.code              = layer_norm(parameters, "encoder/norm", encoder.output);
  out.encoder_attention = encoder.attention;
  return out;
}

/// @brief Decode visible codes and patch locations using learned mask tokens, without an image.
///
/// Projects 32→16, scatters the four visible tokens with selector transpose,
/// inserts one shared learned 16-value mask token at hidden locations, adds fixed
/// positions, and runs one width-16 block, final LayerNorm and 16→49 sigmoid head.
/// Targets and hidden image pixels are absent from the signature. No visible
/// pixels are copied into predictions; a composite display is the caller's choice.
/// Invalid input fails before decoder parameter creation. Retains code gradients;
/// inputs are not mutated and shared stores need external synchronization.
/// @param parameters Mutable floating store matching code/selector.
/// @param code Finite Float32/64 [B,4,32], B>0.
/// @param selector Matching detached [B,4,16], four distinct one-hot rows.
/// @return Owning decoder fields documented in MaskedAeOutput; encoder-only fields undefined.
/// @throws std::invalid_argument For invalid inputs or incompatible stored leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(1877);
/// const auto s=n::models::MaeSelection(n::zeros({1},n::DType::Int32));
/// const auto decoded=n::models::MaeDecode(p,n::zeros({1,4,32}),s);
/// // decoded.prediction [1,16,49]; no original image or target was supplied.
/// @endcode
/// @par References
/// - He et al. (2021/2022), https://arxiv.org/abs/2111.06377 .
/// @see MaeEncode(), MaskedAutoencoder()
inline MaskedAeOutput
MaeDecode(ParameterStore& parameters, const Tensor& code, const Tensor& selector) {
  mae_detail::Floating(code);
  if (code.dim() != 3 || code.size(0) <= 0 || code.size(1) != 4 || code.size(2) != 32 ||
      parameters.dtype() != code.dtype())
    throw std::invalid_argument("MAE decoder requires matching floating [B,4,32]");
  mae_detail::Selection(selector, code.size(0), code.dtype());
  MaskedAeOutput out;
  out.code         = code;
  out.mask         = 1. - sum(selector, {1});
  out.projected    = dense(parameters, "decoder_projection", code, 16);
  out.restored     = matmul(selector, out.projected, true);
  const auto token = parameters.get("mask_token", {16}, init::zeros());
  out.decoder_input =
      out.restored + expand_dims(out.mask, 2) * token + MaePosition(16, code.dtype());
  const auto decoder    = MaeBlock(parameters, "decoder", out.decoder_input, 16);
  out.decoder_attention = decoder.attention;
  out.prediction        = dense(
      parameters,
      "prediction",
      layer_norm(parameters, "decoder/norm", decoder.output),
      49,
      Activation::Sigmoid);
  return out;
}

/// @brief Reconstruct sixteen patches with the original tiny visible-only transformer MAE.
///
/// Composes MaeEncode and MaeDecode without fitting or scoring. The fixed
/// width-32 encoder sees four 7-by-7 patches; the width-16 decoder sees sixteen
/// locations. No CLS token or dropout; 35 parameter tensors contain 13649 scalars.
/// Names preserve patch, encoder, decoder_projection, mask_token, decoder and
/// prediction prefixes. Standard layer initializers use independent weights;
/// LayerNorm has unit gamma/zero beta and the mask token starts at zero.
/// Targets are retained in patches only for caller-owned loss calculation.
/// All discrete selector checks precede parameter creation; numerical/later
/// backend failures may occur after partial creation. Inputs are not mutated;
/// shared stores require synchronization. Supports strided Float32/64 tensors.
/// @param parameters Mutable floating store matching image/selector.
/// @param image Finite nonempty grayscale [B,28,28,1].
/// @param selector Detached matching [B,4,16] with four distinct one-hot rows.
/// @return All owning encoder/decoder endpoints with continuous gradients retained.
/// @throws std::invalid_argument For invalid inputs or incompatible stored leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(1877);
/// const auto s=n::models::MaeSelection(n::zeros({2},n::DType::Int32));
/// const auto net=n::models::MaskedAutoencoder(p,n::zeros({2,28,28,1}),s);
/// const auto loss=n::models::MaeLoss(net.prediction,net.patches,net.mask);
/// // p.count() is 13649. The caller owns Adam and the minibatch loop.
/// @endcode
/// @par References
/// - He, Chen, Xie, Li, Dollar and Girshick (2021/2022), Masked Autoencoders Are
///   Scalable Vision Learners, https://arxiv.org/abs/2111.06377 . This MNIST model
///   changes depth, widths, patches, sampler and sigmoid/raw-pixel loss; it is not
///   an ImageNet or transfer-learning reproduction.
/// @see MaeEncode(), MaeDecode(), MaeLoss()
inline MaskedAeOutput
MaskedAutoencoder(ParameterStore& parameters, const Tensor& image, const Tensor& selector) {
  auto       out        = MaeEncode(parameters, image, selector);
  const auto decoded    = MaeDecode(parameters, out.code, selector);
  out.prediction        = decoded.prediction;
  out.decoder_attention = decoded.decoder_attention;
  out.projected         = decoded.projected;
  out.restored          = decoded.restored;
  out.decoder_input     = decoded.decoder_input;
  return out;
}

/// @brief Average squared reconstruction error over explicitly hidden patch pixels only.
///
/// Computes sum((prediction-patches)^2*mask[...,None])/(D*sum(mask)), for
/// [B,P,D] patch arrays and binary [B,P] hidden indicators. No stop-gradient is
/// applied to predictions or targets; mask must be detached. Values are not
/// range-checked: nonfinite arithmetic can propagate even through zero weights.
/// Shapes must be nonempty and at least one patch hidden. Strides/continuous
/// gradients are retained, with no mutation or shared state. Work/storage O(BPD).
/// @param prediction Float32/64 predicted patch pixels [B,P,D].
/// @param patches Matching clean target patch pixels.
/// @param mask Matching detached binary hidden indicators [B,P], one meaning hidden.
/// @return Differentiable scalar masked pixel MSE, normalized by hidden pixel count.
/// @throws std::invalid_argument For invalid shapes/dtypes/mask values or no hidden patches.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// const auto loss=n::models::MaeLoss(n::zeros({1,2,2}),n::ones({1,2,2}),
///                                   n::tensor({0.,1.},{1,2}));
/// // loss is 1: only the second patch's two pixels are scored.
/// @endcode
/// @par References
/// - He et al. (2021/2022), https://arxiv.org/abs/2111.06377 . This variant uses
///   raw pixel targets without per-patch target normalization.
/// @see MaskedAutoencoder(), MaePatches()
inline Tensor MaeLoss(const Tensor& prediction, const Tensor& patches, const Tensor& mask) {
  if (!prediction.defined() || !patches.defined() || !mask.defined() || prediction.dim() != 3 ||
      prediction.size(0) <= 0 || prediction.size(1) <= 0 || prediction.size(2) <= 0 ||
      prediction.shape() != patches.shape() ||
      mask.shape() != Shape{prediction.size(0), prediction.size(1)} ||
      prediction.dtype() != patches.dtype() || prediction.dtype() != mask.dtype() ||
      mask.requires_grad() ||
      (prediction.dtype() != DType::Float32 && prediction.dtype() != DType::Float64))
    throw std::invalid_argument(
        "MAE loss requires matching nonempty floating [B,P,D] and detached binary [B,P]");
  bool hidden = false;
  for (double v : mask.contiguous().to_vector<double>()) {
    if (v != 0 && v != 1)
      throw std::invalid_argument("MAE loss mask must be binary");
    hidden = hidden || v == 1;
  }
  if (!hidden)
    throw std::invalid_argument("MAE loss requires at least one hidden patch");
  return sum(square(prediction - patches) * expand_dims(mask, 2)) /
         (double(prediction.size(2)) * sum(mask));
}

} // namespace datamunge::nn::models
