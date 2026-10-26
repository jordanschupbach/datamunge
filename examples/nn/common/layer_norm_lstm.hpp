// Per-example, per-gate layer normalization; the raw cell is retained and the
// normalized cell is used ONLY for the hidden readout. Port of tflow's
// common/nn/layer_norm_lstm.h.
#pragma once

#include <cmath>

#include "common/peephole_lstm.hpp"

namespace rnn_cells {

/// (x - mean) * rsqrt(var + 1e-5) along `axis` (no gain or bias).
inline Tensor LstmLayerNormalize(const Tensor& x, std::int64_t axis) {
  const Tensor centered = nn::sub(x, nn::mean(x, {axis}, true));
  const Tensor variance = nn::mean(nn::square(centered), {axis}, true);
  return nn::mul(centered, nn::rsqrt(nn::add(variance, double(1e-5f))));
}

inline GatedStep LayerNormLstmStep(nn::ParameterStore& p, const std::string& name, const Tensor& x, std::int64_t incoming,
                                   std::int64_t units, const nn::LstmState& state) {
  const Tensor wx = p.get(name + "/Wx", {incoming, 4 * units}, nn::init::uniform(std::sqrt(6.f / float(incoming + 4 * units))));
  const Tensor wh = p.get(name + "/Wh", {units, 4 * units}, nn::init::uniform(std::sqrt(6.f / float(5 * units))));
  const Tensor bias       = p.get(name + "/b", {4 * units}, LstmBias(units));
  const Tensor gate_gamma = p.get(name + "/gate_gamma", {4 * units}, nn::init::ones());
  const Tensor cell_gamma = p.get(name + "/cell_gamma", {units}, nn::init::ones());
  const Tensor cell_beta  = p.get(name + "/cell_beta", {units}, nn::init::zeros());
  const Tensor affine     = nn::add(nn::matmul(x, wx), nn::matmul(state.h, wh));
  const Tensor normalized = nn::reshape(LstmLayerNormalize(nn::reshape(affine, {-1, 4, units}), 2), {-1, 4 * units});
  const auto   parts      = nn::split(nn::add(nn::mul(normalized, gate_gamma), bias), 4, 1);
  const Tensor i = nn::sigmoid(parts[0]), f = nn::sigmoid(parts[1]), g = nn::tanh(parts[2]), o = nn::sigmoid(parts[3]);
  const Tensor cell            = nn::add(nn::mul(f, state.c), nn::mul(i, g));
  const Tensor normalized_cell = nn::add(nn::mul(LstmLayerNormalize(cell, 1), cell_gamma), cell_beta);
  return {{nn::mul(o, nn::tanh(normalized_cell)), cell}, nn::concat({i, f, g, o}, 1), normalized, normalized_cell};
}

}  // namespace rnn_cells
