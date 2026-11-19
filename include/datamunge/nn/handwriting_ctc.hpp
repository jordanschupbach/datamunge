/// @file
/// @brief A masked bidirectional recognizer for composite MNIST digit strings.
#pragma once

#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/ops.hpp>

namespace datamunge::nn {

/// @brief Encode 28-by-96 digit strips as time-major CTC logits.
///
/// Four adjacent columns make one frame. A tanh projection feeds two
/// 16-unit LSTMs scanning in opposite directions. Each recurrence
/// preserves its state on invalid frames. Class 10 is the CTC blank.
///
/// @param model Owns trainable projection, recurrent and readout tensors.
/// @param images Floating [N,28,96] strips in the store dtype, N positive.
/// @param lengths Integral [N] valid frame counts in [1,24].
/// @return Differentiable logits [24,N,11]; use lengths to mask CTC and decoding.
/// @throws std::invalid_argument If image geometry, dtype or lengths are incompatible.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(7);
/// const auto logits = dm::nn::handwriting_ctc_logits(model, dm::nn::zeros({1,28,96}),
///     dm::nn::full({1}, 12., dm::nn::DType::Int64));
/// @endcode
/// @see ctc_loss()
inline Tensor handwriting_ctc_logits(ParameterStore &model, const Tensor &images, const Tensor &lengths) {
  if (images.dim() != 3 || images.shape()[0] < 1 || images.shape()[1] != 28 ||
      images.shape()[2] != 96 || images.dtype() != model.dtype() ||
      lengths.shape() != Shape{images.shape()[0]} || !is_integral(lengths.dtype()) ||
      min(lengths).item() < 1 || max(lengths).item() > 24)
    throw std::invalid_argument("handwriting_ctc_logits: invalid strips or length");
  const auto count = images.shape()[0];
  const auto x = dense(model, "input", images.transpose({0,2,1}).reshape({count,24,112}),
                       16, Activation::Tanh);
  auto scan = [&](const std::string &name, bool reverse) {
    auto state = lstm_zero_state(count,16,images.dtype());
    std::vector<Tensor> steps(24);
    for (int i = 0; i < 24; ++i) {
      const int t = reverse ? 23-i : i;
      const auto next = lstm_step(model,name,select(x,1,t),state);
      const auto keep = greater(lengths,t).reshape({count,1});
      state.h = where(keep,next.h,state.h);
      state.c = where(keep,next.c,state.c);
      steps[t] = state.h;
    }
    return stack(steps,0);
  };
  return dense(model,"readout",concat({scan("forward",false),scan("backward",true)},2),11);
}

}  // namespace datamunge::nn
