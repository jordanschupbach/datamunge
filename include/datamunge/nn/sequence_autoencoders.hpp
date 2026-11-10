#pragma once
/// @file sequence_autoencoders.hpp
/// @brief Final-state-only tanh sequence autoencoders with independent repeated-code decoders.

#include <cmath>
#include <stdexcept>
#include <vector>

#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn::models {
/// @brief Validation details for finite floating sequence autoencoders.
namespace sequence_ae_detail {
/// @brief Validate defined finite Float32/64 values without changing gradient history.
/// @param x Tensor to inspect, including arbitrary strides.
/// @throws std::invalid_argument For undefined, integer or nonfinite inputs.
inline void Floating(const Tensor& x) {
  if (!x.defined() || (x.dtype() != DType::Float32 && x.dtype() != DType::Float64))
    throw std::invalid_argument("sequence autoencoder requires Float32/64 tensors");
  for (double v : x.contiguous().to_vector<double>())
    if (!std::isfinite(v))
      throw std::invalid_argument("sequence autoencoder inputs must be finite");
}
} // namespace sequence_ae_detail

/// @brief View grayscale images or flat rows as an ordered fixed-length row sequence.
///
/// Accepts finite Float32/64 [B,T,F,1], [B,T,F] or [B,T*F], with positive
/// B,T,F. Values, dtype and row order are preserved; no resize, clipping or
/// normalization. Reshape retains gradients and materializes strides when
/// needed. No mutation or shared state. Work/storage O(B*T*F) in the copy case.
/// @param images Grayscale batch, sequence batch or flat image rows.
/// @param steps Positive sequence length T, default28.
/// @param features Positive row width F, default28.
/// @return Owning differentiable tensor handle with shape [B,T,F].
/// @throws std::invalid_argument For invalid shapes, dimensions, dtype or nonfinite values.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const auto sequence=dm::nn::models::ImageRowSequence(dm::nn::ones({2,28,28,1}));
/// // sequence is [2,28,28], with all values one.
/// @endcode
/// @see RecurrentEncode(), RecurrentAutoencoder()
inline Tensor
ImageRowSequence(const Tensor& images, std::int64_t steps = 28, std::int64_t features = 28) {
  sequence_ae_detail::Floating(images);
  if (steps <= 0 || features <= 0 || images.dim() < 2 || images.size(0) <= 0)
    throw std::invalid_argument("image row sequence requires positive B,T,F");
  const auto batch = images.size(0);
  const bool flat =
      images.dim() == 2 && images.size(1) / steps == features && images.size(1) % steps == 0;
  if (!flat && images.shape() != Shape{batch, steps, features} &&
      images.shape() != Shape{batch, steps, features, 1})
    throw std::invalid_argument("image row sequence requires [B,T,F,1], [B,T,F] or [B,T*F]");
  return reshape(images, {batch, steps, features});
}

/// @brief Own code, reconstruction and complete state histories for a sequence autoencoder.
///
/// RecurrentEncode fills code and encoder_states only. RecurrentDecode fills
/// code, reconstruction and decoder_states only. Other fields in partial
/// results are undefined. RecurrentAutoencoder fills every field. Handles own
/// tensor lifetimes and retain continuous autograd paths; no optimizer state.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(1871);
/// const auto net=n::models::RecurrentAutoencoder(p,n::zeros({2,28,28}));
/// // code [2,32]; reconstruction [2,28,28]; both histories [2,28,32].
/// @endcode
/// @see RecurrentEncode(), RecurrentDecode(), RecurrentAutoencoder()
struct RecurrentAeOutput {
  Tensor code;           ///< Final encoder state [B,U], the decoder's only sample-specific input.
  Tensor reconstruction; ///< Independent shared sigmoid row projections [B,T,F].
  Tensor encoder_states; ///< All post-update encoder states [B,T,U], excluding zero initial state.
  Tensor decoder_states; ///< All post-update decoder states [B,T,U], excluding zero initial state.
};

/// @brief Encode an ordered sequence into only its final tanh recurrent state.
///
/// Starts at zero and applies h_t=tanh(x_t*Wx+h_(t-1)*Wh+b), sharing
/// encoder/Wx[F,U], encoder/Wh[U,U] and encoder/b[U] across all T steps.
/// Requires finite nonempty Float32/64 [B,T,F], matching store dtype, T>1,
/// F>0 and 0<U<T*F (strict scalar-count bottleneck). Every time step retains
/// gradients; no truncation, detach, dropout or state carried between calls.
/// Work O(B*T*(F*U+U*U)); stores O(B*T*U) recurrent activations. The store
/// may create/reuse leaves with standard rnn_step initializers and needs
/// synchronization when shared. Inputs are not mutated.
/// @param parameters Mutable matching floating parameter store.
/// @param input Finite floating [B,T,F] sequence.
/// @param steps T, default28, matching the input's middle dimension.
/// @param features F, default28, matching the input's last dimension.
/// @param units U, default32, strictly below T*F.
/// @return Owning code and complete encoder history; decoder fields undefined.
/// @throws std::invalid_argument For invalid inputs/dimensions/dtype or incompatible stored leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(5);
/// const auto encoded=n::models::RecurrentEncode(p,n::ones({2,3,1}),3,1,1);
/// // Three steps and one feature; code [2,1], encoder_states [2,3,1].
/// @endcode
/// @par References
/// - J. L. Elman (1990), Finding Structure in Time. Cognitive Science14,179–211.
///   https://doi.org/10.1207/s15516709cog1402_1 . The cell is a simple tanh recurrence.
/// - P. J. Werbos (1990), Backpropagation through time: what it does and how to do it.
///   https://doi.org/10.1109/5.58337 . All source-example time steps remain differentiable.
/// @see RecurrentDecode(), RecurrentAutoencoder(), rnn_step()
inline RecurrentAeOutput RecurrentEncode(
    ParameterStore& parameters,
    const Tensor&   input,
    std::int64_t    steps    = 28,
    std::int64_t    features = 28,
    std::int64_t    units    = 32) {
  sequence_ae_detail::Floating(input);
  if (steps <= 1 || features <= 0 || units <= 0 || units / steps >= features || input.dim() != 3 ||
      input.size(0) <= 0 || input.shape() != Shape{input.size(0), steps, features} ||
      parameters.dtype() != input.dtype())
    throw std::invalid_argument("recurrent encoder requires matching [B,T,F], T>1 and 0<U<T*F");
  Tensor              h = zeros({input.size(0), units}, input.dtype());
  std::vector<Tensor> history;
  for (std::int64_t t = 0; t < steps; ++t) {
    h = rnn_step(parameters, "encoder", select(input, 1, t), h);
    history.push_back(h);
  }
  RecurrentAeOutput out;
  out.code           = h;
  out.encoder_states = stack(history, 1);
  return out;
}

/// @brief Decode a repeated continuous code with a zero-initialized tanh recurrence.
///
/// Every decoder step receives the same code z, not a target/previous output:
/// d_t=tanh(z*Wx+d_(t-1)*Wh+b), y_t=sigmoid(d_t*Wout+bout).
/// Shares decoder/Wx[U,U], decoder/Wh[U,U], decoder/b[U], output/W[U,F]
/// and output/b[F] across T steps. Code is finite nonempty Float32/64 [B,U]
/// matching the store, and T,F are positive. No teacher forcing, image input,
/// attention or encoder-state bypass. All code/state gradients remain, with
/// a fresh zero initial state every call. Shapes and dtypes are checked before
/// creating leaves; incompatible existing leaves can also throw. No input
/// mutation. Work O(B*T*(U*U+U*F)); stores O(B*T*(U+F)) activations. Shared
/// stores need synchronization; the function performs no optimization.
/// @param parameters Mutable floating store matching code dtype.
/// @param code Finite [B,U] code, U>0, with gradients preserved.
/// @param steps Positive output length T, default28.
/// @param features Positive output row width F, default28.
/// @return Owning code, reconstruction and decoder history; encoder history undefined.
/// @throws std::invalid_argument For invalid geometry/dtype/nonfinite code or incompatible stored
/// leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(5);
/// const auto decoded=n::models::RecurrentDecode(p,n::zeros({2,1}),3,1);
/// // reconstruction [2,3,1]; no image or target is an argument.
/// @endcode
/// @par References
/// - J. L. Elman (1990), https://doi.org/10.1207/s15516709cog1402_1 .
///   This example repeats a code through an independent tanh decoder.
/// @see RecurrentEncode(), RecurrentAutoencoder(), rnn_step(), dense()
inline RecurrentAeOutput RecurrentDecode(
    ParameterStore& parameters,
    const Tensor&   code,
    std::int64_t    steps    = 28,
    std::int64_t    features = 28) {
  sequence_ae_detail::Floating(code);
  if (code.dim() != 2 || code.size(0) <= 0 || code.size(1) <= 0 || steps <= 0 || features <= 0 ||
      parameters.dtype() != code.dtype())
    throw std::invalid_argument(
        "recurrent decoder requires matching finite [B,U] and positive T,F");
  RecurrentAeOutput out;
  out.code              = code;
  Tensor              h = zeros_like(code);
  std::vector<Tensor> history, rows;
  for (std::int64_t t = 0; t < steps; ++t) {
    h = rnn_step(parameters, "decoder", code, h);
    history.push_back(h);
    rows.push_back(dense(parameters, "output", h, features, Activation::Sigmoid));
  }
  out.reconstruction = stack(rows, 1);
  out.decoder_states = stack(history, 1);
  return out;
}

/// @brief Compose a final-state-only tanh encoder and independent repeated-code decoder.
///
/// RecurrentEncode validates the strict scalar bottleneck and input geometry;
/// RecurrentDecode receives only its final code. All encoder/decoder steps
/// retain gradients. At defaults T=28,F=28,U=32, eight leaves contain
/// 2*F*U+3*U*U+2*U+F=4956 learned scalars. Parameter creation order and
/// initialization retain the original source. No fitting, batching, scoring
/// or checkpoint selection is hidden here. Inputs are not mutated; shared
/// stores need synchronization. Work O(B*T*(F*U+U*U)) and history O(B*T*U),
/// in addition to reconstructed output storage O(B*T*F).
/// @param parameters Mutable matching Float32/64 store.
/// @param input Finite nonempty [B,T,F] ordered sequence.
/// @param steps T>1, default28.
/// @param features F>0, default28.
/// @param units 0<U<T*F, default32.
/// @return Owning final code, reconstructed sequence and both complete histories.
/// @throws std::invalid_argument For invalid dimensions/dtype/nonfinite input or incompatible
/// leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(1871);
/// const auto x=n::full({2,28,28},.2);
/// const auto net=n::models::RecurrentAutoencoder(p,x);
/// const auto loss=n::mse(net.reconstruction,x);
/// // The caller chooses Adam and explicitly runs every fitting update.
/// @endcode
/// @par References
/// - J. L. Elman (1990), https://doi.org/10.1207/s15516709cog1402_1 .
/// - P. J. Werbos (1990), https://doi.org/10.1109/5.58337 .
/// @see RecurrentEncode(), RecurrentDecode(), ImageRowSequence()
inline RecurrentAeOutput RecurrentAutoencoder(
    ParameterStore& parameters,
    const Tensor&   input,
    std::int64_t    steps    = 28,
    std::int64_t    features = 28,
    std::int64_t    units    = 32) {
  const auto encoded     = RecurrentEncode(parameters, input, steps, features, units);
  auto       decoded     = RecurrentDecode(parameters, encoded.code, steps, features);
  decoded.encoder_states = encoded.encoder_states;
  return decoded;
}
} // namespace datamunge::nn::models
