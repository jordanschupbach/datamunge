// Shared pieces of tflow's small encoder-decoder LSTMs (common/ml/seq2seq.h and
// common/nn/seq2seq.h, used by bahdanau, luong, translation, beam_search ...):
// 6-token source/target batches, a shared one-hot token embedding, a linear
// vocabulary head, the length-masked encoder that also returns the attention
// memory, the masked teacher-forced token cross-entropy and label-free greedy
// decoding (BOS = 1, EOS = 2).
//
// A batch is a replay::Map with int32 "input", "teacher", "target" [B, 6],
// "input_lengths", "output_lengths" and "source_ids" [B].
#pragma once

#include <functional>
#include <string>
#include <vector>

#include <datamunge/nn/nn.hpp>

#include "common/tf_replay.hpp"

namespace seq2seq {

namespace nn = datamunge::nn;
using nn::Tensor;
using replay::Map;

constexpr int kSteps = 6, kUnits = 8, kBos = 1, kEos = 2;

inline const std::vector<std::string>& BatchKeys() {
  static const std::vector<std::string> keys{"input", "target", "teacher", "input_lengths", "output_lengths", "source_ids"};
  return keys;
}

/// The fit / validation / test pools of a seq2seq data.npz.
inline std::vector<Map> LoadPools(const std::filesystem::path& file) {
  const auto       all = nn::load_npz(file.string());
  std::vector<Map> pools;
  for (const std::string split : {"fit", "validation", "test"}) pools.push_back(replay::Split(all, split, BatchKeys()));
  return pools;
}

/// one-hot [..., V] x "embedding/W" [V, 8] ~ N(0, .1^2) -> [..., 8].
inline Tensor Embedding(nn::ParameterStore& p, const Tensor& encoded, std::int64_t vocab) {
  const Tensor W     = p.get("embedding/W", {vocab, kUnits}, nn::init::normal(.1));
  nn::Shape    shape = encoded.shape();
  shape.back()       = kUnits;
  return nn::reshape(nn::matmul(nn::reshape(encoded, {-1, vocab}), W), shape);
}

/// "head/W" [in, V] ~ N(0, .1^2), "head/b" = 0 (tflow pre-adds them before Dense).
inline Tensor Head(nn::ParameterStore& p, const Tensor& h, std::int64_t vocab) {
  const Tensor W = p.get("head/W", {h.shape().back(), vocab}, nn::init::normal(.1));
  const Tensor b = p.get("head/b", {vocab}, nn::init::zeros());
  return nn::bias_add(nn::matmul(h, W), b);
}

/// [B, 1] bool: lengths > t.
inline Tensor ValidAt(const Tensor& lengths, int t) { return nn::expand_dims(nn::greater(lengths, double(t)), 1); }
/// [B, 6] bool: position < lengths.
inline Tensor ValidPositions(const Tensor& lengths) {
  return nn::less(nn::reshape(nn::arange(0, kSteps, 1, nn::DType::Int32), {1, -1}), nn::reshape(lengths, {-1, 1}));
}

struct Encoded {
  std::vector<Tensor> h, c;  // 7 states each: the initial one, then one per source position
  Tensor              memory;  // [B, 6, 8]: the encoder state at valid positions, zero elsewhere
};

/// Length-masked LSTM "encoder" over the embedded source xe [B, 6, 8].
inline Encoded Encode(nn::ParameterStore& p, const Tensor& xe, const Tensor& lengths, const Tensor& ih, const Tensor& ic) {
  Encoded       out;
  nn::LstmState s{ih, ic};
  out.h.push_back(s.h);
  out.c.push_back(s.c);
  std::vector<Tensor> memory;
  const auto          xs = nn::time_steps(xe);
  for (int t = 0; t < kSteps; ++t) {
    const auto   next = nn::lstm_step(p, "encoder", xs[t], s);
    const Tensor mask = ValidAt(lengths, t);
    s                 = {nn::where(mask, next.h, s.h), nn::where(mask, next.c, s.c)};
    out.h.push_back(s.h);
    out.c.push_back(s.c);
    memory.push_back(nn::where(mask, s.h, 0.0));
  }
  out.memory = nn::stack_time(memory);
  return out;
}

/// Masked mean token cross-entropy of logits [B, 6, V] (tflow's seq2seq loss).
inline Tensor TokenCE(const Tensor& logits, const Tensor& target, const Tensor& out_lengths) {
  const auto   V    = logits.shape().back();
  const Tensor mask = nn::cast(ValidPositions(out_lengths), logits.dtype());
  const Tensor ce   = nn::neg(nn::sum(nn::mul(nn::log_softmax(logits, -1), nn::one_hot(target, V)), {2}));
  return nn::div(nn::sum(nn::mul(mask, ce)), nn::maximum(nn::sum(mask), nn::scalar(1.0, logits.dtype())));
}

/// One decoder step from owned state: logits [B, V], next h, c [B, 8] and
/// attention weights [B, 6] (undefined for attention-free decoders).
struct DecoderStep {
  Tensor logits, h, c, attention;
};
using StepFn = std::function<DecoderStep(const Tensor& token, const Tensor& h, const Tensor& c)>;

/// Greedy decoding from the final encoder state (h0, c0): BOS first, then each
/// row's own previous prediction, frozen after EOS. Returns int32 [B, 6]
/// (0 after EOS); `trace` (when given) receives the attention [B, 6, 6].
inline Tensor Greedy(const StepFn& step, Tensor h, Tensor c, Tensor* trace = nullptr) {
  nn::NoGradGuard           guard;
  const auto                n = h.shape()[0];
  std::vector<std::int32_t> tokens(n, kBos), result(n * kSteps, 0);
  std::vector<float>        attention(n * kSteps * kSteps, 0.f);
  std::vector<char>         done(n, 0);
  for (int t = 0; t < kSteps; ++t) {
    const auto out = step(nn::from_vector(tokens), h, c);
    const auto V   = out.logits.shape().back();
    const auto lg  = out.logits.to_vector<float>();
    auto       hv = h.to_vector<float>(), cv = c.to_vector<float>();
    const auto nh = out.h.to_vector<float>(), nc = out.c.to_vector<float>();
    const auto aw = out.attention.defined() ? out.attention.to_vector<float>() : std::vector<float>();
    for (std::int64_t j = 0; j < n; ++j) {
      if (done[j]) continue;
      if (!aw.empty())
        for (int k = 0; k < kSteps; ++k) attention[(j * kSteps + t) * kSteps + k] = aw[j * kSteps + k];
      int best = 0;
      for (int k = 1; k < V; ++k)
        if (lg[j * V + k] > lg[j * V + best]) best = k;
      result[j * kSteps + t] = best;
      tokens[j]              = best;
      done[j]                = best == kEos;
      for (int k = 0; k < kUnits; ++k) hv[j * kUnits + k] = nh[j * kUnits + k], cv[j * kUnits + k] = nc[j * kUnits + k];
    }
    h = nn::from_vector(hv, {n, kUnits});
    c = nn::from_vector(cv, {n, kUnits});
  }
  if (trace) *trace = nn::from_vector(attention, {n, kSteps, kSteps});
  return nn::from_vector(result, {n, kSteps});
}

/// Masked attention normalisation shared by tflow's attentive decoders:
/// softmax over valid source positions (invalid scores -1e9, then zeroed),
/// divided by max(sum, 1) so that an empty source gives all-zero weights.
inline Tensor MaskedAttention(const Tensor& scores, const Tensor& lengths) {
  const Tensor valid = ValidPositions(lengths);
  const Tensor raw   = nn::mul(nn::softmax(nn::where(valid, scores, -1e9), -1), nn::cast(valid, scores.dtype()));
  return nn::div(raw, nn::maximum(nn::sum(raw, {1}, true), nn::scalar(1.0, scores.dtype())));
}

/// sum_t weights[b, t] memory[b, t, :] -> [B, 8].
inline Tensor Context(const Tensor& memory, const Tensor& weights) {
  return nn::sum(nn::mul(memory, nn::expand_dims(weights, 2)), {1});
}

}  // namespace seq2seq

// ---- added by job 4.2b: the plain (attention-free) model of tflow's
// common/nn/seq2seq.h and greedy decoding with configurable BOS/EOS/fill. ----
namespace seq2seq {

/// Greedy decoding like Greedy() with the start token `bos`, the stop token
/// `eos` and `fill` written after the stop (tflow's pointer network: -1, 5, -1).
inline Tensor GreedyWith(const StepFn& step, Tensor h, Tensor c, Tensor* trace, int bos, int eos, int fill) {
  nn::NoGradGuard           guard;
  const auto                n = h.shape()[0];
  std::vector<std::int32_t> tokens(n, bos), result(n * kSteps, fill);
  std::vector<float>        attention(n * kSteps * kSteps, 0.f);
  std::vector<char>         done(n, 0);
  for (int t = 0; t < kSteps; ++t) {
    const auto out = step(nn::from_vector(tokens), h, c);
    const auto V   = out.logits.shape().back();
    const auto lg  = out.logits.to_vector<float>();
    auto       hv = h.to_vector<float>(), cv = c.to_vector<float>();
    const auto nh = out.h.to_vector<float>(), nc = out.c.to_vector<float>();
    const auto aw = out.attention.defined() ? out.attention.to_vector<float>() : std::vector<float>();
    for (std::int64_t j = 0; j < n; ++j) {
      if (done[j]) continue;
      if (!aw.empty())
        for (int k = 0; k < kSteps; ++k) attention[(j * kSteps + t) * kSteps + k] = aw[j * kSteps + k];
      int best = 0;
      for (int k = 1; k < V; ++k)
        if (lg[j * V + k] > lg[j * V + best]) best = k;
      result[j * kSteps + t] = best;
      tokens[j]              = best;
      done[j]                = best == eos;
      for (int k = 0; k < kUnits; ++k) hv[j * kUnits + k] = nh[j * kUnits + k], cv[j * kUnits + k] = nc[j * kUnits + k];
    }
    h = nn::from_vector(hv, {n, kUnits});
    c = nn::from_vector(cv, {n, kUnits});
  }
  if (trace) *trace = nn::from_vector(attention, {n, kSteps, kSteps});
  return nn::from_vector(result, {n, kSteps});
}

/// tflow's plain encoder-decoder (nn::Seq2seq): the decoder LSTM starts from
/// the final encoder state and sees only the embedded teacher tokens; no
/// attention. 14-token vocabulary: 1326 parameters in 9 tensors.
struct PlainPath {
  Tensor encoder_h, encoder_c, decoder_h, decoder_c, logits, bridge_h, bridge_c;
};

/// One plain decoder step: embedded token [B, 8] -> state and logits [B, V].
inline DecoderStep PlainStep(nn::ParameterStore& p, const Tensor& embedded, const nn::LstmState& previous, std::int64_t vocab) {
  const auto s = nn::lstm_step(p, "decoder", embedded, previous);
  return {Head(p, s.h, vocab), s.h, s.c, Tensor()};
}

/// Teacher-forced pass; x, d are one-hot [B, 6, V].
inline PlainPath Plain(nn::ParameterStore& p, const Tensor& x, const Tensor& d, const Tensor& lengths, const Tensor& out_lengths,
                       const Tensor& ih, const Tensor& ic, std::int64_t vocab) {
  const Tensor xe  = Embedding(p, x, vocab);
  const Tensor de  = Embedding(p, d, vocab);
  const auto   enc = Encode(p, xe, lengths, ih, ic);
  PlainPath    out;
  out.bridge_h = enc.h.back();
  out.bridge_c = enc.c.back();
  nn::LstmState       s{out.bridge_h, out.bridge_c};
  std::vector<Tensor> dh{s.h}, dc{s.c}, logits;
  const auto          ds = nn::time_steps(de);
  for (int t = 0; t < kSteps; ++t) {
    const auto   next = PlainStep(p, ds[t], s, vocab);
    const Tensor mask = ValidAt(out_lengths, t);
    s                 = {nn::where(mask, next.h, s.h), nn::where(mask, next.c, s.c)};
    logits.push_back(nn::where(mask, next.logits, 0.0));
    dh.push_back(s.h);
    dc.push_back(s.c);
  }
  out.encoder_h = nn::stack_time(enc.h);
  out.encoder_c = nn::stack_time(enc.c);
  out.decoder_h = nn::stack_time(dh);
  out.decoder_c = nn::stack_time(dc);
  out.logits    = nn::stack_time(logits);
  return out;
}

}  // namespace seq2seq
