// Note-event LSTM shared by the teacher-forced likelihood and stateful
// generation (port of tflow's common/nn/midi_lstm.h): 13 input tokens (12
// pitch classes + start token 12) one-hot embedded by "embedding/W" [13, 8]
// ~ N(0, .1^2), one 8-unit LSTM "lstm", a 12-way dense "head"; 756
// parameters in 6 tensors.
#pragma once

#include <vector>

#include <datamunge/nn/nn.hpp>

namespace midi {

namespace nn = datamunge::nn;
using nn::Tensor;

constexpr int kInputs = 13, kNotes = 12, kStart = 12, kUnits = 8, kSteps = 16;

struct Step {
  nn::LstmState state;
  Tensor        logits;
};

/// One step from a one-hot note [B, 13] and the previous state.
inline Step MidiLstmStep(nn::ParameterStore& p, const Tensor& onehot, const nn::LstmState& previous) {
  const Tensor w    = p.get("embedding/W", {kInputs, kUnits}, nn::init::normal(.1));
  const auto   next = nn::lstm_step(p, "lstm", nn::matmul(onehot, w), previous);
  return {next, nn::dense(p, "head", next.h, kNotes)};
}

struct Path {
  Tensor logits, h, c;  // [B, 16, 12], [B, 17, 8] (incl. the zero initial state)
};

/// Teacher-forced pass over one-hot notes [B, 16, 13] from a zero state.
inline Path MidiLstm(nn::ParameterStore& p, const Tensor& x) {
  const auto          n = x.shape()[0];
  nn::LstmState       s{nn::zeros({n, kUnits}, x.dtype()), nn::zeros({n, kUnits}, x.dtype())};
  std::vector<Tensor> hs{s.h}, cs{s.c}, logits;
  for (const auto& step : nn::time_steps(x)) {
    const auto next = MidiLstmStep(p, step, s);
    s               = next.state;
    hs.push_back(s.h);
    cs.push_back(s.c);
    logits.push_back(next.logits);
  }
  return {nn::stack_time(logits), nn::stack_time(hs), nn::stack_time(cs)};
}

}  // namespace midi
