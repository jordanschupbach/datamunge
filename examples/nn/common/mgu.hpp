// Minimal gated unit: one shared sigmoid controls the reset-before candidate
// and the update interpolation. Port of tflow's common/nn/mgu.h (path) and
// common/ml/mgu.h (model: 386 parameters, Adam .01, seed 38003).
#pragma once

#include <cmath>
#include <memory>

#include "common/recurrent_model.hpp"
#include "common/rnn.hpp"

namespace rnn_cells {

inline Outputs Mgu(nn::ParameterStore& p, const Tensor& input, const Tensor& lengths, const std::vector<Tensor>& initial) {
  const Tensor wxf = p.get("cell/Wxf", {2, 12}, nn::init::uniform(std::sqrt(6.f / 14)));
  const Tensor whf = p.get("cell/Whf", {12, 12}, nn::init::uniform(std::sqrt(6.f / 24)));
  const Tensor bf  = p.get("cell/bf", {12}, nn::init::zeros());
  const Tensor wxn = p.get("cell/Wxn", {2, 12}, nn::init::uniform(std::sqrt(6.f / 14)));
  const Tensor whn = p.get("cell/Whn", {12, 12}, nn::init::uniform(std::sqrt(6.f / 24)));
  const Tensor bn  = p.get("cell/bn", {12}, nn::init::zeros());
  const auto   xs  = nn::time_steps(input);
  Tensor       h   = initial[0];
  std::vector<Tensor> states{h}, gates;
  for (int t = 0; t < 12; ++t) {
    const Tensor f = nn::sigmoid(nn::bias_add(nn::add(nn::matmul(xs[t], wxf), nn::matmul(h, whf)), bf));
    const Tensor candidate =
        nn::tanh(nn::bias_add(nn::add(nn::matmul(xs[t], wxn), nn::matmul(nn::mul(f, h), whn)), bn));
    const Tensor next  = nn::add(nn::mul(nn::sub(1.0, f), h), nn::mul(f, candidate));
    const Tensor valid = StepValid(lengths, t);
    h = nn::where(valid, next, h);
    gates.push_back(MaskedDiagnostics(valid, nn::concat({f, candidate}, 1)));
    states.push_back(h);
  }
  return {{"logits", Head(p, "head", h, 12, 2)},
          {"states", nn::stack_time(states)},
          {"final_state", h},
          {"gates", nn::stack_time(gates)}};
}

inline std::unique_ptr<RecurrentModel> MguModel(bool training = true) {
  auto m = std::make_unique<RecurrentModel>(38003, Mgu, nn::Shape{12, 2}, std::vector<nn::Shape>{{12}},
                                            std::vector<std::string>{"initial_gradient"},
                                            std::vector<std::string>{"logits", "probability", "states", "final_state", "gates"});
  m->Build(training);
  return m;
}

}  // namespace rnn_cells
