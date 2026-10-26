// Independent ReLU recurrence: one scalar recurrent coefficient per channel.
// Port of tflow's common/nn/indrnn.h (path) and common/ml/indrnn.h (model:
// 74 parameters, Adam .01, seed 35003, u projected to [0, 2^(1/12)] after
// every update).
#pragma once

#include <cmath>
#include <memory>

#include "common/recurrent_model.hpp"
#include "common/rnn.hpp"

namespace rnn_cells {

inline float IndRnnUpper() { return std::pow(2.f, 1.f / 12); }

inline Outputs IndRnn(nn::ParameterStore& p, const Tensor& input, const Tensor& lengths, const std::vector<Tensor>& initial) {
  const Tensor wx = p.get("cell/Wx", {2, 12}, nn::init::uniform(std::sqrt(6.f / 14)));
  const Tensor u = p.get("cell/u", {12}, nn::init::constant(.9f)), b = p.get("cell/b", {12}, nn::init::constant(.05f));
  const auto   features =
      nn::time_steps(nn::reshape(nn::bias_add(nn::matmul(nn::reshape(input, {-1, 2}), wx), b), {-1, 12, 12}));
  Tensor              h = initial[0];
  std::vector<Tensor> states{h};
  for (int t = 0; t < 12; ++t) {
    const Tensor candidate = nn::relu(nn::add(features[t], nn::mul(h, u)));
    h = Blend(StepMask(lengths, t), candidate, h);
    states.push_back(h);
  }
  return {{"logits", Head(p, "head", h, 12, 2)}, {"states", nn::stack_time(states)}, {"final_state", h}};
}

inline std::unique_ptr<RecurrentModel> IndRnnModel(bool training = true) {
  auto m = std::make_unique<RecurrentModel>(35003, IndRnn, nn::Shape{12, 2}, std::vector<nn::Shape>{{12}},
                                            std::vector<std::string>{"initial_gradient"},
                                            std::vector<std::string>{"logits", "probability", "states", "final_state"});
  m->Build(training);
  return m;
}

/// Projects u onto [0, 2^(1/12)] in place (tflow's project_u Assign(ClipByValue)).
inline void ProjectIndRnnCoefficients(nn::ParameterStore& store) {
  store.set("cell/u", nn::clip(store.at("cell/u").detach(), 0.0, double(IndRnnUpper())));
}

/// d final_state[:, c] / d initial state for the 12 channels ("channel0".."channel11").
inline std::vector<Probe> IndRnnChannelProbes() {
  std::vector<Probe> probes;
  for (int c = 0; c < 12; ++c)
    probes.push_back({"channel" + std::to_string(c),
                      [c](const Outputs& o) { return nn::sum(nn::slice(o.at("final_state"), {0, c}, {-1, 1})); }, 1});
  return probes;
}

}  // namespace rnn_cells
