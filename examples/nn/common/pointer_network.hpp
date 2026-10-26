// Position-indexed pointer network (port of tflow's common/nn/pointer_network.h
// and the model of common/ml/pointer_network.h): 2-D points [B, 5, 2] are
// embedded by tanh(dense "input"), a length-masked 8-unit encoder LSTM gives
// five memories, a learned STOP memory is appended as position 5, and an
// 8-unit decoder LSTM is fed the memory selected by the previous pointer
// (a learned BOS vector at the first step, pointer id -1). Additive scores
// over valid positions + STOP are the logits; no vocabulary head, repeats are
// allowed. 1272 parameters in 14 tensors.
#pragma once

#include <vector>

#include <datamunge/nn/nn.hpp>

#include "common/seq2seq.hpp"

namespace pointer_network {

namespace nn = datamunge::nn;
using nn::Tensor;
using seq2seq::kSteps;
using seq2seq::kUnits;

constexpr int kPoints = 5, kStop = 5, kChoices = 6, kBos = -1;

/// Additive pointer scores [B, 6] over the memory [B, 6, 8]; invalid input
/// positions (>= length, except STOP) are -1e9.
inline Tensor PointerScores(nn::ParameterStore& p, const Tensor& memory, const Tensor& lengths, const Tensor& query) {
  const Tensor we     = p.get("pointer/Wmemory", {kUnits, kUnits}, nn::init::normal(.1));
  const Tensor wq     = p.get("pointer/Wquery", {kUnits, kUnits}, nn::init::normal(.1));
  const Tensor bias   = p.get("pointer/b", {kUnits}, nn::init::zeros());
  const Tensor v      = p.get("pointer/v", {kUnits, 1}, nn::init::normal(.1));
  const Tensor keys   = nn::reshape(nn::matmul(nn::reshape(memory, {-1, kUnits}), we), {-1, kChoices, kUnits});
  const Tensor q      = nn::expand_dims(nn::matmul(query, wq), 1);
  const Tensor hidden = nn::tanh(nn::add(nn::add(keys, q), bias));
  const Tensor scores = nn::reshape(nn::matmul(nn::reshape(hidden, {-1, kUnits}), v), {-1, kChoices});
  const Tensor positions = nn::reshape(nn::arange(0, kChoices, 1, nn::DType::Int32), {1, -1});
  const Tensor valid     = nn::logical_or(nn::less(positions, nn::reshape(lengths, {-1, 1})), nn::equal(positions, double(kStop)));
  return nn::where(valid, scores, -1e9);
}

struct Step {
  nn::LstmState state;
  Tensor        logits, selected;
};

/// pointer: one-hot previous pointer [B, 6] (all-zero for BOS); bos: bool [B].
inline Step PointerStep(nn::ParameterStore& p, const Tensor& pointer, const Tensor& bos, const nn::LstmState& previous,
                        const Tensor& memory, const Tensor& lengths) {
  Tensor       selected = nn::sum(nn::mul(memory, nn::expand_dims(pointer, 2)), {1});
  const Tensor b        = p.get("decoder/BOS", {kUnits}, nn::init::normal(.1));
  selected              = nn::add(selected, nn::mul(nn::expand_dims(nn::cast(bos, selected.dtype()), 1), b));
  const auto next       = nn::lstm_step(p, "decoder", selected, previous);
  return {next, PointerScores(p, memory, lengths, next.h), selected};
}

struct Path {
  Tensor encoder_h, encoder_c, decoder_h, decoder_c, logits, bridge_h, bridge_c, memory, selected;
};

/// Teacher-forced pass: points [B, 5, 2], teacher_weights one-hot [B, 6, 6], teacher_ids int32 [B, 6].
inline Path PointerNetwork(nn::ParameterStore& p, const Tensor& points, const Tensor& teacher_weights, const Tensor& teacher_ids,
                           const Tensor& lengths, const Tensor& out_lengths, const Tensor& ih, const Tensor& ic) {
  p.get("input/W", {2, kUnits}, nn::init::normal(.1));
  p.get("input/b", {kUnits}, nn::init::zeros());
  const Tensor embedding = nn::reshape(nn::dense(p, "input", nn::reshape(points, {-1, 2}), kUnits, nn::Activation::Tanh),
                                       {-1, kPoints, kUnits});
  const auto   xs = nn::time_steps(embedding), teachers = nn::time_steps(teacher_weights), ids = nn::time_steps(teacher_ids);
  nn::LstmState       s{ih, ic};
  std::vector<Tensor> eh{s.h}, ec{s.c}, dh, dc, logits, memory, selected;
  for (int t = 0; t < kPoints; ++t) {
    const auto   next  = nn::lstm_step(p, "encoder", xs[t], s);
    const Tensor valid = seq2seq::ValidAt(lengths, t);
    s                  = {nn::where(valid, next.h, s.h), nn::where(valid, next.c, s.c)};
    eh.push_back(s.h);
    ec.push_back(s.c);
    memory.push_back(nn::where(valid, s.h, 0.0));
  }
  const Tensor stop = p.get("pointer/STOP", {kUnits}, nn::init::normal(.1));
  memory.push_back(nn::add(nn::zeros_like(s.h), stop));
  Path out;
  out.memory   = nn::stack_time(memory);
  out.bridge_h = s.h;
  out.bridge_c = s.c;
  dh.push_back(s.h);
  dc.push_back(s.c);
  for (int t = 0; t < kSteps; ++t) {
    const Step   next  = PointerStep(p, teachers[t], nn::less(ids[t], 0.0), s, out.memory, lengths);
    const Tensor valid = seq2seq::ValidAt(out_lengths, t);
    s                  = {nn::where(valid, next.state.h, s.h), nn::where(valid, next.state.c, s.c)};
    logits.push_back(nn::where(valid, next.logits, 0.0));
    selected.push_back(nn::where(valid, next.selected, 0.0));
    dh.push_back(s.h);
    dc.push_back(s.c);
  }
  out.encoder_h = nn::stack_time(eh);
  out.encoder_c = nn::stack_time(ec);
  out.decoder_h = nn::stack_time(dh);
  out.decoder_c = nn::stack_time(dc);
  out.logits    = nn::stack_time(logits);
  out.selected  = nn::stack_time(selected);
  return out;
}

}  // namespace pointer_network
