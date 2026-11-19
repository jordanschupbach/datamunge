#pragma once

/// @file deepspeech_ctc.hpp
/// @brief Compact convolutional bidirectional GRU acoustic model.

#include <stdexcept>
#include <vector>

#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Compute DeepSpeech-style frame logits for CTC.
/// Normalizes input log-mel values as (x+6)*0.2, subsamples time by two
/// with a width-five convolution, then concatenates forward and reverse
/// GRU states before a shared output projection.
/// @param model Store whose parameters are initialized on first use.
/// @param features Floating log-mel tensor [batch, frames, bands] matching the store dtype.
/// @param units Convolution channels and recurrent units per direction, default 24.
/// @param classes Output classes including the CTC blank, default 11.
/// @return Time-major logits [ceil(frames/2), batch, classes].
/// @throws std::invalid_argument For invalid sizes, dtype or incompatible parameters.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(42);
/// auto logits = dm::nn::deepspeech_ctc_logits(model, dm::nn::zeros({1, 28, 24}));
/// @endcode
/// @par References
/// - D. Amodei et al. (2016), Deep Speech 2. https://proceedings.mlr.press/v48/amodei16.html
/// @see ctc_loss(), ctc_greedy_decode()
inline Tensor deepspeech_ctc_logits(ParameterStore& model, const Tensor& features,
                                  int units = 24, int classes = 11) {
  if (!features.defined() || features.dim() != 3 || features.size(0) < 1 ||
      features.size(1) < 1 || features.size(2) < 1 || features.dtype() != model.dtype() ||
      units < 1 || classes < 2) {
    throw std::invalid_argument("Invalid acoustic model input");
  }
  const auto conv = conv1d(model, "conv", (features + 6) * .2, units, 5,
                          {.stride = 2, .activation = Activation::Relu});
  auto forward = zeros({features.size(0), units}, features.dtype());
  auto backward = zeros_like(forward);
  const auto frames = conv.size(1);
  std::vector<Tensor> left(frames), right(frames);
  for (std::int64_t t = 0; t < frames; ++t) {
    const auto reverse = frames - 1 - t;
    forward = gru_step(model, "forward", select(conv, 1, t), forward);
    backward = gru_step(model, "backward", select(conv, 1, reverse), backward);
    left[t] = forward;
    right[reverse] = backward;
  }
  return dense(model, "head", concat({stack(left), stack(right)}, 2), classes);
}

}  // namespace datamunge::nn
