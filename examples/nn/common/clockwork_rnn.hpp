// Clockwork modules with periods 1,2,4,8 and slow-to-fast recurrent
// connectivity. Port of tflow's common/nn/clockwork_rnn.h (path) and
// common/ml/clockwork_rnn.h (model: 206 parameters, Adam .01, seed 32003).
#pragma once

#include <memory>

#include "common/recurrent_model.hpp"
#include "common/rnn.hpp"

namespace rnn_cells {

/// [from, to] = 1 when module(from)'s period >= module(to)'s period (slow feeds fast).
inline Tensor ClockworkConnectivity() {
  std::vector<float> mask(144);
  for (int from = 0; from < 12; ++from)
    for (int to = 0; to < 12; ++to) mask[from * 12 + to] = ((1 << (from / 3)) >= (1 << (to / 3))) ? 1.f : 0.f;
  return nn::from_vector(mask, {12, 12});
}

/// [t, unit] = 1 when the unit's module ticks at step t.
inline Tensor ClockworkSchedule() {
  std::vector<float> mask(144);
  for (int t = 0; t < 12; ++t)
    for (int j = 0; j < 12; ++j) mask[t * 12 + j] = t % (1 << (j / 3)) == 0 ? 1.f : 0.f;
  return nn::from_vector(mask, {12, 12});
}

inline Outputs ClockworkRnn(nn::ParameterStore& p, const Tensor& input, const Tensor& lengths,
                            const std::vector<Tensor>& initial) {
  const auto   steps = nn::time_steps(input);
  const Tensor wx = p.get("cell/Wx", {2, 12}, nn::init::normal(.1)), wh = p.get("cell/Wh", {12, 12}, nn::init::normal(.1)),
               bias = p.get("cell/b", {12}, nn::init::zeros());
  const Tensor masked_wh = nn::mul(wh, ClockworkConnectivity());
  const Tensor schedule  = ClockworkSchedule();
  Tensor       h         = initial[0];
  std::vector<Tensor> states{h};
  for (int t = 0; t < 12; ++t) {
    const Tensor candidate = nn::tanh(nn::add(nn::add(nn::matmul(steps[t], wx), nn::matmul(h, masked_wh)), bias));
    const Tensor clock     = nn::slice(schedule, {t, 0}, {1, 12});
    h = Blend(nn::mul(StepMask(lengths, t), clock), candidate, h);
    states.push_back(h);
  }
  return {{"logits", Head(p, "head", h, 12, 2)}, {"states", nn::stack_time(states)}, {"final_state", h}};
}

inline std::unique_ptr<RecurrentModel> ClockworkRnnModel(bool training = true) {
  auto m = std::make_unique<RecurrentModel>(32003, ClockworkRnn, nn::Shape{12, 2}, std::vector<nn::Shape>{{12}},
                                            std::vector<std::string>{"initial_gradient"},
                                            std::vector<std::string>{"logits", "probability", "states", "final_state"});
  m->Build(training);
  return m;
}

/// d final_state[:, 3m] / d input for the four modules ("module0".."module3").
inline std::vector<Probe> ClockworkModuleProbes() {
  std::vector<Probe> probes;
  for (int m = 0; m < 4; ++m)
    probes.push_back({"module" + std::to_string(m),
                      [m](const Outputs& o) { return nn::sum(nn::slice(o.at("final_state"), {0, m * 3}, {-1, 1})); }, 0});
  return probes;
}

}  // namespace rnn_cells
