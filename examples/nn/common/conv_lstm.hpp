// Spatial ConvLSTM: 3x3 input/recurrent kernels, two-channel hidden/cell maps
// and a 1x1 pixel head. Port of tflow's common/nn/conv_lstm.h (path) and
// common/ml/conv_lstm.h (model: 227 parameters, Adam .01, Params seed 37003,
// positive-weighted (x15) sigmoid cross-entropy on the next frame).
#pragma once

#include <cmath>
#include <memory>

#include "common/recurrent_model.hpp"
#include "common/rnn.hpp"

namespace rnn_cells {

/// input [B, T, 4, 4, 1], initial {h, c} [B, 4, 4, 2]; gates i, f, z, o along channels.
inline Outputs ConvLstm(nn::ParameterStore& p, const Tensor& input, const Tensor& lengths, const std::vector<Tensor>& initial) {
  const Tensor wx = p.get("cell/Wx", {3, 3, 1, 8}, nn::init::uniform(std::sqrt(6.f / 81)));
  const Tensor wh = p.get("cell/Wh", {3, 3, 2, 8}, nn::init::uniform(std::sqrt(6.f / 90)));
  const Tensor b  = p.get("cell/b", {8}, [](const nn::Shape&, nn::DType d, nn::Generator&) {
    return nn::concat({nn::zeros({2}, d), nn::full({2}, 1.0, d), nn::zeros({4}, d)}, 0);
  });
  const auto          xs = nn::time_steps(input);
  Tensor              h = initial[0], c = initial[1];
  std::vector<Tensor> states{h}, cells{c}, gates;
  for (int t = 0; t < static_cast<int>(xs.size()); ++t) {
    const Tensor pre   = nn::bias_add(nn::add(nn::conv2d(xs[t], wx), nn::conv2d(h, wh)), b);
    const auto   split = nn::split(pre, 4, 3);
    const Tensor i = nn::sigmoid(split[0]), f = nn::sigmoid(split[1]), o = nn::sigmoid(split[3]), z = nn::tanh(split[2]);
    const Tensor next_c = nn::add(nn::mul(f, c), nn::mul(i, z));
    const Tensor next_h = nn::mul(o, nn::tanh(next_c));
    const Tensor valid  = nn::reshape(nn::greater(lengths, double(t)), {-1, 1, 1, 1});
    h = nn::where(valid, next_h, h);
    c = nn::where(valid, next_c, c);
    gates.push_back(MaskedDiagnostics(valid, nn::concat({i, f, z, o}, 3)));
    states.push_back(h);
    cells.push_back(c);
  }
  const Tensor head_w = p.get("head/W", {1, 1, 2, 1}, nn::init::uniform(std::sqrt(6.f / 3)));
  const Tensor head_b = p.get("head/b", {1}, nn::init::zeros());
  return {{"logits", nn::bias_add(nn::conv2d(h, head_w), head_b)},
          {"states", nn::stack_time(states)},
          {"cell_states", nn::stack_time(cells)},
          {"final_state", h},
          {"final_cell", c},
          {"gates", nn::stack_time(gates)}};
}

/// Mean of 15 * y * softplus(-logit) + (1 - y) * softplus(logit); chosen = sum(logits * y).
inline std::pair<Tensor, Tensor> WeightedPixelLoss(const Outputs& out, const Tensor& labels) {
  const Tensor& logits   = out.at("logits");
  const Tensor  positive = nn::mul(nn::mul(labels, 15.0), nn::softplus(nn::neg(logits)));
  const Tensor  negative = nn::mul(nn::sub(1.0, labels), nn::softplus(logits));
  return {nn::mean(nn::add(positive, negative)), nn::sum(nn::mul(logits, labels))};
}

inline std::unique_ptr<RecurrentModel> ConvLstmModel(bool training = true) {
  auto m = std::make_unique<RecurrentModel>(
      37003, ConvLstm, nn::Shape{4, 4, 4, 1}, std::vector<nn::Shape>{{4, 4, 2}, {4, 4, 2}},
      std::vector<std::string>{"initial_gradient", "initial_cell_gradient"},
      std::vector<std::string>{"logits", "probability", "states", "final_state", "cell_states", "final_cell", "gates"});
  m->loss_fn     = WeightedPixelLoss;
  m->probability = [](const Tensor& logits) { return nn::sigmoid(logits); };
  // First spatial hidden state (time 1, pixel (1, 1), channel 0): only time-0 inputs in its 3x3 neighbourhood.
  m->evaluate_probes = {{"first_gate_input_gradient",
                         [](const Outputs& o) { return nn::sum(nn::slice(o.at("states"), {0, 1, 1, 1, 0}, {-1, 1, 1, 1, 1})); }, 0}};
  m->Build(training);
  return m;
}

}  // namespace rnn_cells
