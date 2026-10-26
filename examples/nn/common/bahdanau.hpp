// Additive (Bahdanau) encoder attention queried by the PREVIOUS decoder hidden
// state, with the context fed into the decoder LSTM (port of tflow's
// common/nn/bahdanau.h and the model of common/ml/bahdanau.h). 14-token
// vocabulary, 8-D shared embedding, 8-unit encoder/decoder LSTMs; 1726
// parameters in 13 tensors (embedding/W, encoder/*, attention/{Wenc, Wquery,
// b, v}, decoder/* with 16 inputs, head/*).
//
//   score_t = v . tanh(memory_t Wenc + h_prev Wquery + b),  masked softmax over valid t
//
// nn::additive_attention computes the same score without the bias b and with
// Glorot-initialised "<name>/Wq", "/Wk", "/v"; tflow's parameter names,
// N(0, .1^2) initialisation and empty-source normalisation are kept here.
#pragma once

#include <vector>

#include <datamunge/nn/nn.hpp>

#include "common/seq2seq.hpp"

namespace bahdanau {

namespace nn = datamunge::nn;
using nn::Tensor;
using seq2seq::kSteps;
using seq2seq::kUnits;

constexpr int kVocab = 14;

struct Attended {
  Tensor weights, context;
};

inline Attended BahdanauContext(nn::ParameterStore& p, const Tensor& memory, const Tensor& lengths, const Tensor& query) {
  const Tensor we = p.get("attention/Wenc", {kUnits, kUnits}, nn::init::normal(.1));
  const Tensor wq = p.get("attention/Wquery", {kUnits, kUnits}, nn::init::normal(.1));
  const Tensor b  = p.get("attention/b", {kUnits}, nn::init::zeros());
  const Tensor v  = p.get("attention/v", {kUnits, 1}, nn::init::normal(.1));
  const Tensor keys   = nn::reshape(nn::matmul(nn::reshape(memory, {-1, kUnits}), we), {-1, kSteps, kUnits});
  const Tensor q      = nn::expand_dims(nn::matmul(query, wq), 1);
  const Tensor hidden = nn::tanh(nn::add(nn::add(keys, q), b));
  const Tensor scores = nn::reshape(nn::matmul(nn::reshape(hidden, {-1, kUnits}), v), {-1, kSteps});
  Attended     out;
  out.weights = seq2seq::MaskedAttention(scores, lengths);
  out.context = seq2seq::Context(memory, out.weights);
  return out;
}

struct Step {
  nn::LstmState state;
  Tensor        logits, weights, context;
};

inline Step BahdanauStep(nn::ParameterStore& p, const Tensor& embedded, const nn::LstmState& previous, const Tensor& memory,
                         const Tensor& lengths) {
  const Attended att   = BahdanauContext(p, memory, lengths, previous.h);
  const auto     state = nn::lstm_step(p, "decoder", nn::concat({embedded, att.context}, 1), previous);
  return {state, seq2seq::Head(p, state.h, kVocab), att.weights, att.context};
}

struct Path {
  Tensor encoder_h, encoder_c, decoder_h, decoder_c, logits, bridge_h, bridge_c, attention, contexts, memory;
};

/// Teacher-forced pass; x, d are one-hot [B, 6, 14].
inline Path Bahdanau(nn::ParameterStore& p, const Tensor& x, const Tensor& d, const Tensor& lengths, const Tensor& out_lengths,
                     const Tensor& ih, const Tensor& ic) {
  const Tensor xe  = seq2seq::Embedding(p, x, kVocab);
  const Tensor de  = seq2seq::Embedding(p, d, kVocab);
  const auto   enc = seq2seq::Encode(p, xe, lengths, ih, ic);
  Path         out;
  out.memory   = enc.memory;
  out.bridge_h = enc.h.back();
  out.bridge_c = enc.c.back();
  nn::LstmState       s{out.bridge_h, out.bridge_c};
  std::vector<Tensor> dh{s.h}, dc{s.c}, logits, attention, contexts;
  const auto          ds = nn::time_steps(de);
  for (int t = 0; t < kSteps; ++t) {
    const Step   next = BahdanauStep(p, ds[t], s, enc.memory, lengths);
    const Tensor mask = seq2seq::ValidAt(out_lengths, t);
    s                 = {nn::where(mask, next.state.h, s.h), nn::where(mask, next.state.c, s.c)};
    logits.push_back(nn::where(mask, next.logits, 0.0));
    attention.push_back(nn::where(mask, next.weights, 0.0));
    contexts.push_back(nn::where(mask, next.context, 0.0));
    dh.push_back(s.h);
    dc.push_back(s.c);
  }
  out.encoder_h = nn::stack_time(enc.h);
  out.encoder_c = nn::stack_time(enc.c);
  out.decoder_h = nn::stack_time(dh);
  out.decoder_c = nn::stack_time(dc);
  out.logits    = nn::stack_time(logits);
  out.attention = nn::stack_time(attention);
  out.contexts  = nn::stack_time(contexts);
  return out;
}

}  // namespace bahdanau
