// Width-two causal QRNN filters followed by parameter-free channelwise fo
// pooling. Port of tflow's common/nn/qrnn.h (path) and common/ml/qrnn.h (model:
// 206 parameters in 5 tensors, Adam .01, seed 33003).
//
//   [z~ | f~ | o~] = x_{t-1} Wprior + x_t Wcurrent + b   (all steps before pooling; x_{-1} = 0)
//   z = tanh(z~), f = sigmoid(f~), o = sigmoid(o~)
//   c' = f c + (1 - f) z,  h' = o c'   (no tanh readout, no hidden-to-hidden matrix)
#pragma once

#include <cmath>
#include <memory>

#include "common/recurrent_model.hpp"
#include "common/rnn.hpp"

namespace rnn_cells {

inline Outputs Qrnn(nn::ParameterStore& p, const Tensor& input, const Tensor& lengths, const std::vector<Tensor>& initial) {
  const Tensor prior   = p.get("filter/prior", {2, 36}, nn::init::uniform(std::sqrt(6.f / 40)));
  const Tensor current = p.get("filter/current", {2, 36}, nn::init::uniform(std::sqrt(6.f / 40)));
  const Tensor bias    = p.get("filter/b", {36}, nn::init::zeros());
  // All temporal features are computed together BEFORE the sequential pooling.
  const Tensor shifted = nn::slice(nn::pad(input, {{0, 0}, {1, 0}, {0, 0}}), {0, 0, 0}, {-1, 12, 2});
  const Tensor affine  = nn::reshape(
      nn::add(nn::add(nn::matmul(nn::reshape(shifted, {-1, 2}), prior), nn::matmul(nn::reshape(input, {-1, 2}), current)), bias),
      {-1, 12, 36});
  const auto features = nn::time_steps(affine);
  Tensor     h = initial[0], c = initial[1];
  std::vector<Tensor> states{h}, cells{c}, gates;
  for (int t = 0; t < 12; ++t) {
    const auto   parts  = nn::split(features[t], 3, 1);
    const Tensor z      = nn::tanh(parts[0]);
    const Tensor f      = nn::sigmoid(parts[1]), o = nn::sigmoid(parts[2]);
    const Tensor next_c = nn::add(nn::mul(f, c), nn::mul(nn::sub(1.0, f), z));
    const Tensor next_h = nn::mul(o, next_c);
    const Tensor mask   = StepMask(lengths, t);
    h = Blend(mask, next_h, h);
    c = Blend(mask, next_c, c);
    gates.push_back(MaskedDiagnostics(StepValid(lengths, t), nn::concat({z, f, o}, 1)));
    states.push_back(h);
    cells.push_back(c);
  }
  return {{"logits", Head(p, "head", h, 12, 2)}, {"states", nn::stack_time(states)}, {"cell_states", nn::stack_time(cells)},
          {"final_state", h},                     {"final_cell", c},                   {"gates", nn::stack_time(gates)}};
}

inline std::unique_ptr<RecurrentModel> QrnnModel(bool training = true) {
  auto m = std::make_unique<RecurrentModel>(
      33003, Qrnn, nn::Shape{12, 2}, std::vector<nn::Shape>{{12}, {12}}, std::vector<std::string>{"initial_gradient", ""},
      std::vector<std::string>{"logits", "probability", "states", "final_state", "cell_states", "final_cell", "gates"});
  // Sum of every first-step gate diagnostic, differentiated w.r.t. the input.
  m->evaluate_probes = {{"first_gate_input_gradient",
                         [](const Outputs& o) { return nn::sum(nn::slice(o.at("gates"), {0, 0, 0}, {-1, 1, -1})); }, 0}};
  m->Build(training);
  return m;
}

}  // namespace rnn_cells
