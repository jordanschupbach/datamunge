// Original input-only SRU gates, channelwise memory, tanh readout and projected
// highway skip. Port of tflow's common/nn/sru.h (path) and common/ml/sru.h
// (model: 146 parameters in 5 tensors, Adam .01, seed 34003).
//
//   [z | f~ | o~] = x W (input only, batched across time),  skip = x Wskip
//   f = sigmoid(f~ + bf), o = sigmoid(o~ + bo)
//   c' = f c + (1 - f) z,  h' = o tanh(c') + (1 - o) skip
#pragma once

#include <cmath>
#include <memory>

#include "common/recurrent_model.hpp"
#include "common/rnn.hpp"

namespace rnn_cells {

inline Outputs Sru(nn::ParameterStore& p, const Tensor& input, const Tensor& lengths, const std::vector<Tensor>& initial) {
  const Tensor weights      = p.get("project/W", {2, 36}, nn::init::uniform(std::sqrt(6.f / 38)));
  const Tensor bias         = p.get("project/b", {24}, nn::init::zeros());
  const Tensor skip_weights = p.get("skip/W", {2, 12}, nn::init::uniform(std::sqrt(6.f / 14)));
  // Input-only transforms are batched across time; the recurrence is channelwise.
  const Tensor flat     = nn::reshape(input, {-1, 2});
  const auto   features = nn::time_steps(nn::reshape(nn::matmul(flat, weights), {-1, 12, 36}));
  const auto   skip     = nn::time_steps(nn::reshape(nn::matmul(flat, skip_weights), {-1, 12, 12}));
  const auto   gate_bias = nn::split(bias, 2, 0);
  Tensor       h = initial[0], c = initial[1];
  std::vector<Tensor> states{h}, cells{c}, gates;
  for (int t = 0; t < 12; ++t) {
    const auto   parts  = nn::split(features[t], 3, 1);
    const Tensor z      = parts[0];
    const Tensor f      = nn::sigmoid(nn::add(parts[1], gate_bias[0])), o = nn::sigmoid(nn::add(parts[2], gate_bias[1]));
    const Tensor next_c = nn::add(nn::mul(f, c), nn::mul(nn::sub(1.0, f), z));
    const Tensor next_h = nn::add(nn::mul(o, nn::tanh(next_c)), nn::mul(nn::sub(1.0, o), skip[t]));
    const Tensor mask   = StepMask(lengths, t);
    h = Blend(mask, next_h, h);
    c = Blend(mask, next_c, c);
    gates.push_back(MaskedDiagnostics(StepValid(lengths, t), nn::concat({z, f, o, skip[t]}, 1)));
    states.push_back(h);
    cells.push_back(c);
  }
  return {{"logits", Head(p, "head", h, 12, 2)}, {"states", nn::stack_time(states)}, {"cell_states", nn::stack_time(cells)},
          {"final_state", h},                     {"final_cell", c},                   {"gates", nn::stack_time(gates)}};
}

inline std::unique_ptr<RecurrentModel> SruModel(bool training = true) {
  auto m = std::make_unique<RecurrentModel>(
      34003, Sru, nn::Shape{12, 2}, std::vector<nn::Shape>{{12}, {12}}, std::vector<std::string>{"initial_gradient", ""},
      std::vector<std::string>{"logits", "probability", "states", "final_state", "cell_states", "final_cell", "gates"});
  // Sum of every first-step gate diagnostic, differentiated w.r.t. the input.
  m->evaluate_probes = {{"first_gate_input_gradient",
                         [](const Outputs& o) { return nn::sum(nn::slice(o.at("gates"), {0, 0, 0}, {-1, 1, -1})); }, 0}};
  m->Build(training);
  return m;
}

}  // namespace rnn_cells
