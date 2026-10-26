// Hybrid vocabulary generation and source-token copying with an extended
// dictionary (port of tflow's common/nn/pointer_generator.h and the model of
// common/ml/pointer_generator.h). Base vocabulary 14 (ids >= 14 are source
// OOVs embedded as id 13), extended output dictionary 20; Bahdanau attention
// (common/bahdanau.hpp) queried by the previous decoder state, decoder LSTM
// fed [embedding, context], a learned generation gate from [context, h,
// embedding]; 1751 parameters in 15 tensors.
//
//   p = g softmax(head(h)) (padded to 20) + (1 - g) sum_t a_t onehot20(source_t),
//   g = 1 for an empty source;  logits = log(p + 1e-9)
#pragma once

#include <algorithm>
#include <vector>

#include <datamunge/nn/nn.hpp>

#include "common/bahdanau.hpp"
#include "common/seq2seq.hpp"

namespace pointer_generator {

namespace nn = datamunge::nn;
using nn::Tensor;
using seq2seq::kSteps;
using seq2seq::kUnits;

constexpr int kBase = 14, kExtended = 20;

struct Step {
  nn::LstmState state;
  Tensor        logits, weights, context;
};

/// One step; `source` is the one-hot extended source [B, 6, 20].
inline Step PointerGeneratorStep(nn::ParameterStore& p, const Tensor& embedded, const nn::LstmState& previous,
                                 const Tensor& memory, const Tensor& lengths, const Tensor& source) {
  const auto   att      = bahdanau::BahdanauContext(p, memory, lengths, previous.h);
  const auto   state    = nn::lstm_step(p, "decoder", nn::concat({embedded, att.context}, 1), previous);
  const Tensor features = nn::concat({att.context, state.h, embedded}, 1);
  const Tensor gw       = p.get("gate/W", {3 * kUnits, 1}, nn::init::normal(.1));
  const Tensor gb       = p.get("gate/b", {1}, nn::init::zeros());
  const Tensor learned  = nn::sigmoid(nn::add(nn::matmul(features, gw), gb));
  // With no copyable source, use the normalized generator alone.
  const Tensor gate  = nn::where(nn::expand_dims(nn::greater(lengths, 0.0), 1), learned, 1.0);
  const Tensor vocab = nn::pad(nn::softmax(seq2seq::Head(p, state.h, kBase), -1), {{0, 0}, {0, kExtended - kBase}});
  const Tensor copy  = nn::sum(nn::mul(source, nn::expand_dims(att.weights, 2)), {1});
  const Tensor mixed = nn::add(nn::mul(gate, vocab), nn::mul(nn::sub(nn::scalar(1.0, gate.dtype()), gate), copy));
  // Tiny explicit smoothing keeps the log finite at absent extended IDs.
  return {state, nn::log(nn::add(mixed, nn::scalar(1e-9, mixed.dtype()))), att.weights, att.context};
}

struct Path {
  Tensor encoder_h, encoder_c, decoder_h, decoder_c, logits, bridge_h, bridge_c, attention, contexts, memory;
};

/// Teacher-forced pass; x, d are base one-hot [B, 6, 14], source the extended one-hot [B, 6, 20].
inline Path PointerGenerator(nn::ParameterStore& p, const Tensor& x, const Tensor& d, const Tensor& lengths,
                             const Tensor& out_lengths, const Tensor& ih, const Tensor& ic, const Tensor& source) {
  const Tensor xe  = seq2seq::Embedding(p, x, kBase);
  const Tensor de  = seq2seq::Embedding(p, d, kBase);
  const auto   enc = seq2seq::Encode(p, xe, lengths, ih, ic);
  Path         out;
  out.memory   = enc.memory;
  out.bridge_h = enc.h.back();
  out.bridge_c = enc.c.back();
  nn::LstmState       s{out.bridge_h, out.bridge_c};
  std::vector<Tensor> dh{s.h}, dc{s.c}, logits, attention, contexts;
  const auto          ds = nn::time_steps(de);
  for (int t = 0; t < kSteps; ++t) {
    const Step   next = PointerGeneratorStep(p, ds[t], s, enc.memory, lengths, source);
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

/// Base-vocabulary ids: min(ids, 13).
inline Tensor Base(const Tensor& ids) {
  auto v = ids.to_vector<std::int32_t>();
  for (auto& x : v) x = std::min(x, kBase - 1);
  return nn::from_vector(v, ids.shape());
}

}  // namespace pointer_generator
