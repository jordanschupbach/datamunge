// Tiny recurrent handwriting features for unaligned CTC supervision (port of
// tflow's common/nn/ctc_handwriting.h): each of the 8 frames (49 pixels) goes
// through tanh(dense "input" 49 -> 8), an 8-unit LSTM "lstm" whose state is
// frozen past each sequence's length, and a dense "head" 8 -> 11 (digits 0-9
// and the CTC blank 10). Logits are time-major [8, B, 11]; 1043 parameters.
#pragma once

#include <vector>

#include <datamunge/nn/nn.hpp>

namespace ctc_handwriting {

namespace nn = datamunge::nn;
using nn::Tensor;

constexpr int kFrames = 8, kPixels = 49, kClasses = 11, kBlank = 10;

/// input [B, 8, 49], lengths int32 [B] -> logits [8, B, 11].
inline Tensor CtcHandwriting(nn::ParameterStore& p, const Tensor& input, const Tensor& lengths) {
  const auto   B        = input.shape()[0];
  const Tensor features = nn::reshape(nn::tanh(nn::dense(p, "input", nn::reshape(input, {-1, kPixels}), 8)), {B, kFrames, 8});
  const auto   steps    = nn::time_steps(features);
  nn::LstmState s       = nn::lstm_zero_state(B, 8, input.dtype());
  std::vector<Tensor> logits;
  for (int t = 0; t < kFrames; ++t) {
    const auto   next  = nn::lstm_step(p, "lstm", steps[t], s);
    const Tensor valid = nn::expand_dims(nn::greater(lengths, double(t)), 1);
    s                  = {nn::where(valid, next.h, s.h), nn::where(valid, next.c, s.c)};
    logits.push_back(nn::dense(p, "head", s.h, kClasses));
  }
  return nn::transpose(nn::stack_time(logits), {1, 0, 2});
}

}  // namespace ctc_handwriting
