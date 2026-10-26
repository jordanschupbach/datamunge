// Explicit masked sequence unroll around the handwritten LSTM, reset-before
// GRU, peephole LSTM and layer-norm LSTM cells. Port of tflow's
// common/nn/gated_sequence.h (path) and common/ml/gated_sequence.h (model:
// 12 hidden units, 12 steps, 12->2 head, Adam .01, Params seed 27003).
#pragma once

#include <cmath>
#include <memory>

#include "common/layer_norm_lstm.hpp"
#include "common/peephole_lstm.hpp"
#include "common/recurrent_model.hpp"
#include "common/rnn.hpp"

namespace rnn_cells {

enum class GatedCell { kLstm, kGru, kPeephole, kLayerNorm };

/// Plain LSTM step (nn::lstm_step's parameters and arithmetic) that also returns its gates.
inline GatedStep LstmGatedStep(nn::ParameterStore& p, const std::string& name, const Tensor& x, std::int64_t in,
                               std::int64_t units, const nn::LstmState& state, double forget_bias = 1.0) {
  const Tensor wx = p.get(name + "/Wx", {in, 4 * units}, nn::init::uniform(std::sqrt(6.0 / double(in + 4 * units))));
  const Tensor wh = p.get(name + "/Wh", {units, 4 * units}, nn::init::uniform(std::sqrt(6.0 / double(5 * units))));
  const Tensor b  = p.get(name + "/b", {4 * units}, LstmBias(units, forget_bias));
  const auto   parts = nn::split(nn::bias_add(nn::add(nn::matmul(x, wx), nn::matmul(state.h, wh)), b), 4, 1);
  const Tensor i = nn::sigmoid(parts[0]), f = nn::sigmoid(parts[1]), g = nn::tanh(parts[2]), o = nn::sigmoid(parts[3]);
  const Tensor c = nn::add(nn::mul(f, state.c), nn::mul(i, g));
  return {{nn::mul(o, nn::tanh(c)), c}, nn::concat({i, f, g, o}, 1), {}, {}};
}

/// Reset-before GRU step (nn::gru_step's parameters and arithmetic) that also
/// returns concat(z, r, candidate).
inline GatedStep GruGatedStep(nn::ParameterStore& p, const std::string& name, const Tensor& x, std::int64_t in,
                              std::int64_t u, const Tensor& h, double update_bias = 0.0) {
  const Tensor Wx = p.get(name + "/Wx_zr", {in, 2 * u}, nn::init::uniform(std::sqrt(6.0 / double(in + 2 * u))));
  const Tensor Wh = p.get(name + "/Wh_zr", {u, 2 * u}, nn::init::uniform(std::sqrt(6.0 / double(3 * u))));
  const Tensor b  = p.get(name + "/b_zr", {2 * u}, [u, update_bias](const nn::Shape&, nn::DType d, nn::Generator&) {
    return nn::concat({nn::full({u}, update_bias, d), nn::zeros({u}, d)}, 0);
  });
  const Tensor Wxn = p.get(name + "/Wx_n", {in, u}, nn::init::uniform(std::sqrt(6.0 / double(in + u))));
  const Tensor Whn = p.get(name + "/Wh_n", {u, u}, nn::init::uniform(std::sqrt(6.0 / double(2 * u))));
  const Tensor bn  = p.get(name + "/b_n", {u}, nn::init::zeros());
  const auto   zr  = nn::split(nn::sigmoid(nn::bias_add(nn::add(nn::matmul(x, Wx), nn::matmul(h, Wh)), b)), 2, 1);
  const Tensor n   = nn::tanh(nn::bias_add(nn::add(nn::matmul(x, Wxn), nn::matmul(nn::mul(zr[1], h), Whn)), bn));
  return {{nn::add(nn::mul(nn::sub(1.0, zr[0]), n), nn::mul(zr[0], h)), Tensor()}, nn::concat({zr[0], zr[1], n}, 1), {}, {}};
}

inline Outputs GatedSequence(nn::ParameterStore& p, const Tensor& input, const Tensor& lengths,
                             const std::vector<Tensor>& initial, GatedCell kind) {
  const auto          xs = nn::time_steps(input);
  Tensor              h = initial[0], c = kind != GatedCell::kGru ? initial[1] : nn::mul(initial[0], 0.0);
  std::vector<Tensor> states{h}, cells{c}, gates, normed_gates, normed_cells;
  for (int t = 0; t < 12; ++t) {
    const Tensor mask  = StepMask(lengths, t);
    const Tensor valid = nn::greater(mask, 0.0);
    GatedStep    next;
    switch (kind) {
      case GatedCell::kLstm: next = LstmGatedStep(p, "cell", xs[t], 2, 12, {h, c}, 1.0); break;
      case GatedCell::kPeephole: next = PeepholeLstmStep(p, "cell", xs[t], 2, 12, {h, c}); break;
      case GatedCell::kLayerNorm:
        next = LayerNormLstmStep(p, "cell", xs[t], 2, 12, {h, c});
        normed_gates.push_back(MaskedDiagnostics(valid, next.normalized_gate_affine));
        normed_cells.push_back(MaskedDiagnostics(valid, next.normalized_cell));
        break;
      case GatedCell::kGru:
        next          = GruGatedStep(p, "cell", xs[t], 2, 12, h, 0.0);
        next.state.c  = c;
        break;
    }
    h = Blend(mask, next.state.h, h);
    c = Blend(mask, next.state.c, c);
    states.push_back(h);
    cells.push_back(c);
    gates.push_back(MaskedDiagnostics(valid, next.gates));
  }
  Outputs out{{"logits", Head(p, "head", h, 12, 2)},
              {"states", nn::stack_time(states)},
              {"cell_states", nn::stack_time(cells)},
              {"final_state", h},
              {"final_cell", c},
              {"gates", nn::stack_time(gates)}};
  if (!normed_gates.empty()) {
    out["normalized_gate_affine"] = nn::stack_time(normed_gates);
    out["normalized_cell"]        = nn::stack_time(normed_cells);
  }
  return out;
}

inline std::unique_ptr<RecurrentModel> GatedSequenceModel(GatedCell kind, bool training = true) {
  std::vector<std::string> fields{"logits", "probability", "states", "final_state", "cell_states", "final_cell", "gates"};
  if (kind == GatedCell::kLayerNorm) fields.insert(fields.end(), {"normalized_gate_affine", "normalized_cell"});
  auto m = std::make_unique<RecurrentModel>(
      27003,
      [kind](nn::ParameterStore& p, const Tensor& x, const Tensor& l, const std::vector<Tensor>& s) {
        return GatedSequence(p, x, l, s, kind);
      },
      nn::Shape{12, 2}, std::vector<nn::Shape>{{12}, {12}},
      std::vector<std::string>{"initial_gradient", kind != GatedCell::kGru ? "initial_cell_gradient" : ""}, fields);
  m->Build(training);
  return m;
}

}  // namespace rnn_cells
