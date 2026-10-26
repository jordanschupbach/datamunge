// Offline bidirectional LSTM token features with original-time alignment and
// independent POS and NER heads (port of tflow's common/nn/bilstm_tagging.h and
// the model of common/ml/bilstm_tagging.h). Sentences are 24 word ids from a
// 512-word vocabulary, fed as their one-hot encoding [B, 24, 512] (tflow's
// `encoded`, so that word partials exist). 6102 parameters in 11 tensors:
// embedding/W [512, 8] ~ N(0, .1^2); forward/{Wx,Wh,b} and backward/{Wx,Wh,b}
// (8 units, forget bias 1, states frozen past each sentence's length);
// pos/W [16, 45], ner/W [16, 9] ~ N(0, .1^2) with zero biases.
#pragma once

#include <string>
#include <vector>

#include <datamunge/nn/nn.hpp>

namespace bilstm_tagging {

namespace nn = datamunge::nn;
using nn::Tensor;

constexpr int kSteps = 24, kVocab = 512, kUnits = 8, kPos = 45, kNer = 9;

/// The four LSTM gates (i, f, g, o) after their nonlinearities, [B, 32].
inline Tensor LstmGates(nn::ParameterStore& p, const std::string& name, const Tensor& x, const Tensor& h) {
  const Tensor pre =
      nn::bias_add(nn::add(nn::matmul(x, p.at(name + "/Wx")), nn::matmul(h, p.at(name + "/Wh"))), p.at(name + "/b"));
  const auto parts = nn::split(pre, 4, 1);
  return nn::concat({nn::sigmoid(parts[0]), nn::sigmoid(parts[1]), nn::tanh(parts[2]), nn::sigmoid(parts[3])}, 1);
}

struct Features {
  Tensor embeddings, forward_states, forward_cells, backward_processing, backward_cells, backward_aligned, features,
      forward_gates, backward_gates;
};

/// [B, 1] bool: lengths > t.
inline Tensor ValidAt(const Tensor& lengths, int t) { return nn::expand_dims(nn::greater(lengths, double(t)), 1); }

/// initial = {forward h, forward c, backward h, backward c}, each [B, 8].
inline Features BiLstmFeatures(nn::ParameterStore& p, const Tensor& encoded, const Tensor& lengths,
                               const std::vector<Tensor>& initial) {
  const auto   B  = encoded.shape()[0];
  const Tensor ew = p.get("embedding/W", {kVocab, 8}, nn::init::normal(.1));
  Features     out;
  out.embeddings = nn::reshape(nn::matmul(nn::reshape(encoded, {-1, kVocab}), ew), {B, kSteps, 8});
  const auto xs  = nn::time_steps(out.embeddings);
  nn::LstmState s{initial[0], initial[1]};
  std::vector<Tensor> forward{s.h}, fcells{s.c}, fgates, backward{initial[2]}, bcells{initial[3]}, aligned(kSteps),
      bgates(kSteps);
  for (int t = 0; t < kSteps; ++t) {
    const auto   next  = nn::lstm_step(p, "forward", xs[t], s, 1.0);
    const Tensor gate  = LstmGates(p, "forward", xs[t], s.h);
    const Tensor valid = ValidAt(lengths, t);
    s                  = {nn::where(valid, next.h, s.h), nn::where(valid, next.c, s.c)};
    forward.push_back(s.h);
    fcells.push_back(s.c);
    fgates.push_back(nn::where(valid, gate, 0.0));
  }
  s = {initial[2], initial[3]};
  for (int t = kSteps - 1; t >= 0; --t) {
    const auto   next  = nn::lstm_step(p, "backward", xs[t], s, 1.0);
    const Tensor gate  = LstmGates(p, "backward", xs[t], s.h);
    const Tensor valid = ValidAt(lengths, t);
    s                  = {nn::where(valid, next.h, s.h), nn::where(valid, next.c, s.c)};
    backward.push_back(s.h);
    bcells.push_back(s.c);
    aligned[t] = nn::where(valid, s.h, 0.0);
    bgates[t]  = nn::where(valid, gate, 0.0);
  }
  std::vector<Tensor> features;
  for (int t = 0; t < kSteps; ++t)
    features.push_back(nn::where(ValidAt(lengths, t), nn::concat({forward[t + 1], aligned[t]}, 1), 0.0));
  out.forward_states      = nn::stack_time(forward);
  out.forward_cells       = nn::stack_time(fcells);
  out.backward_processing = nn::stack_time(backward);
  out.backward_cells      = nn::stack_time(bcells);
  out.backward_aligned    = nn::stack_time(aligned);
  out.features            = nn::stack_time(features);
  out.forward_gates       = nn::stack_time(fgates);
  out.backward_gates      = nn::stack_time(bgates);
  return out;
}

struct Path {
  Features context;
  Tensor   pos_logits, ner_logits;
};

inline Path BiLstmTagging(nn::ParameterStore& p, const Tensor& encoded, const Tensor& lengths,
                          const std::vector<Tensor>& initial) {
  Path out;
  out.context = BiLstmFeatures(p, encoded, lengths, initial);
  auto head   = [&](const std::string& name, std::int64_t k) {
    const Tensor W = p.get(name + "/W", {16, k}, nn::init::normal(.1));
    const Tensor b = p.get(name + "/b", {k}, nn::init::zeros());
    return nn::reshape(nn::bias_add(nn::matmul(nn::reshape(out.context.features, {-1, 16}), W), b), {-1, kSteps, k});
  };
  const Tensor pos   = head("pos", kPos);
  const Tensor ner   = head("ner", kNer);
  const Tensor valid = nn::expand_dims(nn::less(nn::reshape(nn::arange(0, kSteps, 1, nn::DType::Int32), {1, -1}),
                                                nn::reshape(lengths, {-1, 1})),
                                       2);
  out.pos_logits = nn::where(valid, pos, 0.0);
  out.ner_logits = nn::where(valid, ner, 0.0);
  return out;
}

}  // namespace bilstm_tagging
