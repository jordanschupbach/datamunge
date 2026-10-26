// Learned word embeddings and a length-masked eight-unit LSTM over short IMDB
// review prefixes (port of tflow's common/nn/imdb_lstm.h and the model of
// common/ml/imdb_lstm.h). Reviews are 48 word ids from a 256-word vocabulary;
// the input is their one-hot encoding [B, 48, 256] so that gradients with
// respect to it exist (tflow's `encoded`). 2610 parameters in 6 tensors:
// embedding/W [256, 8] ~ N(0, .1^2), cell/{Wx,Wh,b} (forget bias 1),
// head/W [8, 2] ~ N(0, .1^2), head/b.
#pragma once

#include <string>
#include <vector>

#include <datamunge/nn/nn.hpp>

namespace imdb {

namespace nn = datamunge::nn;
using nn::Tensor;

constexpr int kSteps = 48, kVocab = 256, kUnits = 8;

struct Path {
  Tensor logits, states, cell_states, final_state, final_cell, gates, embeddings;
};

/// The four LSTM gates (i, f, g, o) after their nonlinearities, [B, 32].
inline Tensor Gates(nn::ParameterStore& p, const Tensor& x, const Tensor& h) {
  const Tensor pre   = nn::bias_add(nn::add(nn::matmul(x, p.at("cell/Wx")), nn::matmul(h, p.at("cell/Wh"))), p.at("cell/b"));
  const auto   parts = nn::split(pre, 4, 1);
  return nn::concat({nn::sigmoid(parts[0]), nn::sigmoid(parts[1]), nn::tanh(parts[2]), nn::sigmoid(parts[3])}, 1);
}

inline Path ImdbLstm(nn::ParameterStore& p, const Tensor& encoded, const Tensor& lengths, const Tensor& initial,
                     const Tensor& initial_cell) {
  const auto   B  = encoded.shape()[0];
  const Tensor ew = p.get("embedding/W", {kVocab, 8}, nn::init::normal(.1));
  Path         out;
  out.embeddings  = nn::reshape(nn::matmul(nn::reshape(encoded, {-1, kVocab}), ew), {B, kSteps, 8});
  const auto xs   = nn::time_steps(out.embeddings);
  nn::LstmState s{initial, initial_cell};
  std::vector<Tensor> states{s.h}, cells{s.c}, gates;
  for (int t = 0; t < kSteps; ++t) {
    const auto   proposed = nn::lstm_step(p, "cell", xs[t], s, 1.0);
    const Tensor gate     = Gates(p, xs[t], s.h);
    const Tensor valid    = nn::expand_dims(nn::greater(lengths, double(t)), 1);
    s                     = {nn::where(valid, proposed.h, s.h), nn::where(valid, proposed.c, s.c)};
    states.push_back(s.h);
    cells.push_back(s.c);
    gates.push_back(nn::where(valid, gate, 0.0));
  }
  const Tensor hw = p.get("head/W", {kUnits, 2}, nn::init::normal(.1));
  const Tensor hb = p.get("head/b", {2}, nn::init::zeros());
  out.logits      = nn::bias_add(nn::matmul(s.h, hw), hb);
  out.states      = nn::stack_time(states);
  out.cell_states = nn::stack_time(cells);
  out.final_state = s.h;
  out.final_cell  = s.c;
  out.gates       = nn::stack_time(gates);
  return out;
}

}  // namespace imdb
