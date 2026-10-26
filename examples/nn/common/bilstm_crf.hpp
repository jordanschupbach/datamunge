// End-to-end BiLSTM-CRF conditional sequence likelihood (port of tflow's
// common/ml/bilstm_crf.h and common/nn/bilstm_tagging.h): a 512-word
// embedding (8-D), two independent 8-unit LSTMs (forward and backward, state
// frozen past each sentence's length), a linear NER head on the 16-D
// context and a linear-chain CRF with transition[from, to], start and end
// scores. Sentences are 24 tokens, padded.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <datamunge/nn/nn.hpp>
#include <datamunge/nn/sequence.hpp>

namespace bilstm_crf {

namespace nn = datamunge::nn;
using nn::Tensor;

constexpr std::int64_t kSteps = 24, kVocab = 512, kDim = 8, kUnits = 8, kTags = 9;

struct TaggingBatch {
  Tensor input, lengths, pos, ner, source_ids;  // int32: [B, 24], [B], [B, 24], [B, 24], [B]
};

inline TaggingBatch TaggingRows(const TaggingBatch& x, const std::vector<std::int64_t>& rows) {
  return {nn::take_rows(x.input, rows), nn::take_rows(x.lengths, rows), nn::take_rows(x.pos, rows),
          nn::take_rows(x.ner, rows), nn::take_rows(x.source_ids, rows)};
}

inline Tensor Named(const nn::NamedTensors& arrays, const std::string& key) {
  for (const auto& [name, t] : arrays)
    if (name == key) return t;
  throw std::runtime_error("missing array " + key);
}

/// [B, 1] bool: lengths > t.
inline Tensor ValidAt(const Tensor& lengths, std::int64_t t) {
  return nn::reshape(nn::greater(lengths, static_cast<double>(t)), {-1, 1});
}

struct Path {
  Tensor embeddings, forward_states, backward_aligned, features, unaries;
};

/// Label-free forward pass: words and lengths only.
inline Path Encode(nn::ParameterStore& p, const Tensor& input, const Tensor& lengths) {
  Path         out;
  const Tensor table = p.get("embedding/W", {kVocab, kDim}, nn::init::normal(.1));
  out.embeddings     = nn::gather(table, input);  // == one_hot(input) @ W
  const auto   xs    = nn::time_steps(out.embeddings);
  const auto   B     = input.shape()[0];
  nn::LstmState s    = nn::lstm_zero_state(B, kUnits);
  std::vector<Tensor> forward{s.h}, aligned(kSteps);
  for (std::int64_t t = 0; t < kSteps; ++t) {
    const auto next  = nn::lstm_step(p, "forward", xs[t], s, 1.0);
    const auto valid = ValidAt(lengths, t);
    s                = {nn::where(valid, next.h, s.h), nn::where(valid, next.c, s.c)};
    forward.push_back(s.h);
  }
  s = nn::lstm_zero_state(B, kUnits);
  for (std::int64_t t = kSteps - 1; t >= 0; --t) {
    const auto next  = nn::lstm_step(p, "backward", xs[t], s, 1.0);
    const auto valid = ValidAt(lengths, t);
    s                = {nn::where(valid, next.h, s.h), nn::where(valid, next.c, s.c)};
    aligned[t]       = nn::where(valid, s.h, 0.0);
  }
  std::vector<Tensor> features;
  for (std::int64_t t = 0; t < kSteps; ++t)
    features.push_back(nn::where(ValidAt(lengths, t), nn::concat({forward[t + 1], aligned[t]}, 1), 0.0));
  out.forward_states   = nn::stack(forward, 1);
  out.backward_aligned = nn::stack(aligned, 1);
  out.features         = nn::stack(features, 1);
  nn::DenseOptions head;
  head.init        = nn::init::normal(.1);
  const Tensor raw = nn::dense(p, "ner", out.features, kTags, head);
  const Tensor valid3 =
      nn::expand_dims(nn::less(nn::reshape(nn::arange(0, kSteps, 1, nn::DType::Int32), {1, -1}),
                               nn::reshape(lengths, {-1, 1})), 2);
  out.unaries = nn::where(valid3, raw, 0.0);
  return out;
}

struct Crf {
  Tensor transition, start, end;
};
inline Crf CrfParams(nn::ParameterStore& p) {
  return {p.get("crf/transition", {kTags, kTags}, nn::init::normal(.05)), p.get("crf/start", {kTags}, nn::init::zeros()),
          p.get("crf/end", {kTags}, nn::init::zeros())};
}

struct Objective {
  Path   path;
  Tensor log_partition, gold_score, nll, loss;
};

/// Mean conditional sequence NLL of the gold NER tags.
inline Objective Evaluate(nn::ParameterStore& p, const TaggingBatch& x) {
  Objective o;
  o.path        = Encode(p, x.input, x.lengths);
  const Crf crf = CrfParams(p);
  o.log_partition = nn::crf_log_norm(o.path.unaries, x.lengths, crf.transition, crf.start, crf.end);
  o.gold_score    = nn::crf_sequence_score(o.path.unaries, x.ner, x.lengths, crf.transition, crf.start, crf.end);
  o.nll           = nn::sub(o.log_partition, o.gold_score);
  o.loss          = nn::mean(o.nll);
  return o;
}

/// Viterbi decoding from words and lengths only (-1 in padded positions).
inline nn::CrfDecoded Decode(nn::ParameterStore& p, const TaggingBatch& x) {
  nn::NoGradGuard guard;
  const Path      path = Encode(p, x.input, x.lengths);
  const Crf       crf  = CrfParams(p);
  return nn::crf_decode(path.unaries, x.lengths, crf.transition, crf.start, crf.end, -1);
}

/// Creates every parameter (in tflow's order) without running data through.
inline void Build(nn::ParameterStore& p) {
  TaggingBatch dummy{nn::zeros({1, kSteps}, nn::DType::Int32), nn::ones({1}, nn::DType::Int32), {}, {}, {}};
  nn::NoGradGuard guard;
  Encode(p, dummy.input, dummy.lengths);
  CrfParams(p);
}

}  // namespace bilstm_crf
