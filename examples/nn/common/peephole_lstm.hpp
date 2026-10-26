// Diagonal cell-to-gate peepholes: old cell for input/forget, NEW cell for
// output. Port of tflow's common/nn/peephole_lstm.h.
#pragma once

#include <cmath>

#include "common/recurrent_model.hpp"

namespace rnn_cells {

struct GatedStep {
  nn::LstmState state;
  Tensor        gates;  // concat(i, f, g, o) or, for the GRU, concat(z, r, candidate)
  Tensor        normalized_gate_affine, normalized_cell;  // layer-norm LSTM only
};

/// The LSTM gate bias: zeros except the forget block, which starts at `forget_bias`.
inline nn::Initializer LstmBias(std::int64_t units, double forget_bias = 1.0) {
  return [units, forget_bias](const nn::Shape&, nn::DType d, nn::Generator&) {
    return nn::concat({nn::zeros({units}, d), nn::full({units}, forget_bias, d), nn::zeros({2 * units}, d)}, 0);
  };
}

inline GatedStep PeepholeLstmStep(nn::ParameterStore& p, const std::string& name, const Tensor& x, std::int64_t incoming,
                                  std::int64_t units, const nn::LstmState& state) {
  const Tensor wx = p.get(name + "/Wx", {incoming, 4 * units}, nn::init::uniform(std::sqrt(6.f / float(incoming + 4 * units))));
  const Tensor wh = p.get(name + "/Wh", {units, 4 * units}, nn::init::uniform(std::sqrt(6.f / float(5 * units))));
  const Tensor b  = p.get(name + "/b", {4 * units}, LstmBias(units));
  const Tensor pi = p.get(name + "/peephole_i", {units}, nn::init::normal(.05f));
  const Tensor pf = p.get(name + "/peephole_f", {units}, nn::init::normal(.05f));
  const Tensor po = p.get(name + "/peephole_o", {units}, nn::init::normal(.05f));
  const auto   parts = nn::split(nn::bias_add(nn::add(nn::matmul(x, wx), nn::matmul(state.h, wh)), b), 4, 1);
  const Tensor i = nn::sigmoid(nn::add(parts[0], nn::mul(state.c, pi))), f = nn::sigmoid(nn::add(parts[1], nn::mul(state.c, pf)));
  const Tensor g = nn::tanh(parts[2]);
  const Tensor cell = nn::add(nn::mul(f, state.c), nn::mul(i, g));
  const Tensor o = nn::sigmoid(nn::add(parts[3], nn::mul(cell, po)));
  return {{nn::mul(o, nn::tanh(cell)), cell}, nn::concat({i, f, g, o}, 1), {}, {}};
}

}  // namespace rnn_cells
