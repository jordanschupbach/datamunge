// Global Luong "general" attention from the CURRENT decoder hidden state;
// context and state form an attentional output, no input feeding (port of
// tflow's common/nn/luong.h and the model of common/ml/luong.h). The same
// model as common/translation.hpp with the 14-token synthetic vocabulary:
// 1526 parameters in 12 tensors (embedding/W, encoder/*, decoder/*,
// attention/W, combine/{W,b}, head/{W,b}).
//
//   score_t = (h W) . memory_t,  a = masked softmax,  out = tanh([context, h] Wc + bc)
#pragma once

#include <vector>

#include <datamunge/nn/nn.hpp>

#include "common/seq2seq.hpp"
#include "common/translation.hpp"

namespace luong {

namespace nn = datamunge::nn;
using nn::Tensor;
using seq2seq::kSteps;
using seq2seq::kUnits;

constexpr int kVocab = 14;

struct Step {
  nn::LstmState state;
  Tensor        logits, weights, context;
};

inline Step LuongStep(nn::ParameterStore& p, const Tensor& embedded, const nn::LstmState& previous, const Tensor& memory,
                      const Tensor& lengths) {
  const auto   state = nn::lstm_step(p, "decoder", embedded, previous);
  const auto   att   = translation::TranslationContext(p, memory, lengths, state.h);
  const Tensor wc    = p.get("combine/W", {2 * kUnits, kUnits}, nn::init::normal(.1));
  const Tensor bc    = p.get("combine/b", {kUnits}, nn::init::zeros());
  const Tensor attentional = nn::tanh(nn::bias_add(nn::matmul(nn::concat({att.context, state.h}, 1), wc), bc));
  return {state, seq2seq::Head(p, attentional, kVocab), att.weights, att.context};
}

using Path = translation::Path;

/// Teacher-forced pass; x, d are one-hot [B, 6, 14].
inline Path Luong(nn::ParameterStore& p, const Tensor& x, const Tensor& d, const Tensor& lengths, const Tensor& out_lengths,
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
    const Step   next = LuongStep(p, ds[t], s, enc.memory, lengths);
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

}  // namespace luong
