#pragma once
/// @file layered_autoencoders.hpp
/// @brief Configurable symmetric sigmoid autoencoders and standalone decoders, without fitting
/// orchestration.

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn::models {
/// @brief Configuration and input validation for symmetric sigmoid stacks.
namespace layered_ae_detail {
/// @brief Validate positive mirrored widths that strictly decrease into the central code.
/// @param widths Odd integer width list including input and output, with at least three entries.
/// @throws std::invalid_argument For short/even, nonpositive, asymmetric or nondecreasing encoder
/// widths.
inline void Widths(const Shape& widths) {
  if (widths.size() < 3 || widths.size() % 2 == 0)
    throw std::invalid_argument(
        "sigmoid autoencoder requires an odd mirrored width list of length >=3");
  for (std::size_t i = 0; i < widths.size(); ++i) {
    if (widths[i] <= 0 || widths[i] != widths[widths.size() - 1 - i])
      throw std::invalid_argument("sigmoid autoencoder widths must be positive and mirrored");
    if (i < widths.size() / 2 && widths[i] <= widths[i + 1])
      throw std::invalid_argument("sigmoid autoencoder encoder widths must strictly decrease");
  }
}
/// @brief Validate finite floating matrix rows and a matching parameter-store dtype.
/// @param parameters Store whose dtype must match input.
/// @param input Nonempty finite Float32/64 matrix.
/// @param features Required last dimension.
/// @throws std::invalid_argument For invalid input geometry, dtype or nonfinite values.
inline void Matrix(const ParameterStore& parameters, const Tensor& input, std::int64_t features) {
  if (!input.defined() || input.dim() != 2 || input.size(0) <= 0 || input.size(1) != features ||
      (input.dtype() != DType::Float32 && input.dtype() != DType::Float64) ||
      input.dtype() != parameters.dtype())
    throw std::invalid_argument(
        "sigmoid autoencoder requires matching nonempty Float32/64 [B,features]");
  for (double x : input.contiguous().to_vector<double>())
    if (!std::isfinite(x))
      throw std::invalid_argument("sigmoid autoencoder inputs must be finite");
}
} // namespace layered_ae_detail

/// @brief Evaluate every layer of an untied symmetric sigmoid autoencoder.
///
/// widths=[d,h1,...,c,...,h1,d] has odd length >=3, positive mirrored entries
/// and strictly decreasing encoder widths. For layer i=0,...,L-2, creates/reuses
/// layer<i>/W[widths[i],widths[i+1]] and layer<i>/b[widths[i+1]], with independent
/// Glorot-uniform matrices and zero biases. Every layer uses sigmoid. Input is
/// finite nonempty Float32/64 [B,d], matching the store; values are neither clipped
/// nor normalized. Returns L-1 layer outputs; code is at index L/2-1, final
/// reconstruction at index L-2. Handles own lifetimes and retain input/parameter
/// gradients, including through every decoder layer. No objective, optimizer,
/// greedy fitting, transfer or data caching is hidden. Invalid inputs/configuration
/// fail before leaf creation; incompatible existing leaves can fail later, and
/// nonfinite stored/computed values are not sanitized. Input is not mutated;
/// shared stores require synchronization. Work O(B*sum_i widths[i]*widths[i+1]),
/// activation storage O(B*sum_i widths[i+1]).
/// @param parameters Mutable matching Float32/64 store, using fixed layer<i> prefixes.
/// @param image Finite nonempty [B,widths.front()] rows, including cached representations.
/// @param widths Complete odd mirrored integer geometry, e.g. {784,64,16,64,784}.
/// @return Owning layer outputs in forward order, excluding the input and any loss.
/// @throws std::invalid_argument For invalid geometry/input/dtype or incompatible stored leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(1553);
/// const auto layers=n::models::SigmoidAutoencoder(p,n::zeros({2,4}),{4,3,2,3,4});
/// const auto loss=n::mean(n::square(layers.back()-n::zeros({2,4})));
/// // Four outputs; layers[1] is code [2,2]. Eight leaves contain 48 scalars.
/// @endcode
/// @par References
/// - Bengio, Lamblin, Popovici and Larochelle (NIPS2006), Greedy Layer-Wise Training of Deep
/// Networks,
///   https://papers.nips.cc/paper_files/paper/2006/hash/5da713a690c067105aeb2fae32403405-Abstract.html
///   . The source experiment uses all-sigmoid MSE autoencoders, explicit greedy stages and decoder
///   transfer; this function only supplies their forward components.
/// @see SigmoidAutoencoderDecode(), dense(), ParameterStore
inline std::vector<Tensor>
SigmoidAutoencoder(ParameterStore& parameters, const Tensor& image, const Shape& widths) {
  layered_ae_detail::Widths(widths);
  layered_ae_detail::Matrix(parameters, image, widths.front());
  std::vector<Tensor> outputs;
  auto                value = image;
  for (std::size_t i = 1; i < widths.size(); ++i) {
    value =
        dense(parameters, "layer" + std::to_string(i - 1), value, widths[i], Activation::Sigmoid);
    outputs.push_back(value);
  }
  return outputs;
}

/// @brief Decode an explicit code through the decoder half of a configured sigmoid autoencoder.
///
/// Validates the same complete mirrored/strictly decreasing geometry as
/// SigmoidAutoencoder(), then consumes only finite Float32/64 code[B,widths[L/2]],
/// B>0, matching the store. Uses original layer indices L/2,...,L-2, rather
/// than renaming decoder layers from zero. Creates/reuses only their independent
/// Glorot-uniform weights and zero biases; a fresh decoder-only store therefore
/// need not contain encoder leaves. All layers use sigmoid. Output owns its
/// lifetime and retains code/decoder gradients. There is no image/target input,
/// copied-pixel path, fitting or state mutation beyond parameter creation.
/// Zero or signed finite codes are allowed; they are not clipped. Configuration
/// and input validation precede creation, but incompatible leaves can fail later;
/// nonfinite stored/computed values propagate. Shared stores need synchronization.
/// Work/storage are the decoder subset of SigmoidAutoencoder().
/// @param parameters Mutable matching Float32/64 store, with original layer<i> names.
/// @param code Finite nonempty [B,widths[widths.size()/2]] code.
/// @param widths Complete odd mirrored geometry, including the unused encoder widths.
/// @return Owning differentiable [B,widths.back()] raw sigmoid reconstruction.
/// @throws std::invalid_argument For invalid widths/code/dtype or incompatible stored leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(1553);
/// const auto layers=n::models::SigmoidAutoencoder(p,n::zeros({2,4}),{4,3,2,3,4});
/// const auto decoded=n::models::SigmoidAutoencoderDecode(p,layers[1],{4,3,2,3,4});
/// // decoded equals layers.back(); only the explicit two-value codes enter decoding.
/// @endcode
/// @par References
/// - Bengio et al. (NIPS2006),
///   https://papers.nips.cc/paper_files/paper/2006/hash/5da713a690c067105aeb2fae32403405-Abstract.html
///   . This independent sigmoid decoder retains the small source experiment's geometry.
/// @see SigmoidAutoencoder(), dense()
inline Tensor
SigmoidAutoencoderDecode(ParameterStore& parameters, const Tensor& code, const Shape& widths) {
  layered_ae_detail::Widths(widths);
  layered_ae_detail::Matrix(parameters, code, widths[widths.size() / 2]);
  auto value = code;
  for (std::size_t i = widths.size() / 2 + 1; i < widths.size(); ++i)
    value =
        dense(parameters, "layer" + std::to_string(i - 1), value, widths[i], Activation::Sigmoid);
  return value;
}
} // namespace datamunge::nn::models
