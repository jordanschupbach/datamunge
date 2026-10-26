// Two recurrent tanh layers: the upper cell consumes the CURRENT lower state at
// each time, not a delayed one. Port of tflow's common/nn/stacked_rnn.h (path)
// and common/ml/stacked_rnn.h (model: 506 parameters in 8 tensors, Adam .01,
// seed 29003, separate lower/upper initial states and two input-Jacobian probes).
#pragma once

#include <memory>

#include "common/recurrent_model.hpp"
#include "common/rnn.hpp"

namespace rnn_cells {

inline Outputs StackedRnn(nn::ParameterStore& p, const Tensor& input, const Tensor& lengths, const std::vector<Tensor>& initial) {
  const auto          steps = nn::time_steps(input);
  Tensor              lower = initial[0], upper = initial[1];
  std::vector<Tensor> lower_trace{lower}, upper_trace{upper};
  for (int t = 0; t < 12; ++t) {
    const Tensor mask = StepMask(lengths, t);
    lower             = Blend(mask, nn::rnn_step(p, "lower", steps[t], lower), lower);
    upper             = Blend(mask, nn::rnn_step(p, "upper", lower, upper), upper);
    lower_trace.push_back(lower);
    upper_trace.push_back(upper);
  }
  return {{"logits", Head(p, "head", upper, 12, 2)},
          {"states", nn::stack_time(lower_trace)},
          {"upper_states", nn::stack_time(upper_trace)},
          {"final_state", upper},
          {"final_lower", lower}};
}

/// Sum over the batch of state 1 (after the first step), unit 0 of `field`.
inline Tensor FirstStateProbe(const Outputs& out, const std::string& field) {
  return nn::sum(nn::slice(out.at(field), {0, 1, 0}, {-1, 1, 1}));
}

inline std::unique_ptr<RecurrentModel> StackedRnnModel(bool training = true) {
  auto m = std::make_unique<RecurrentModel>(
      29003, StackedRnn, nn::Shape{12, 2}, std::vector<nn::Shape>{{12}, {12}},
      std::vector<std::string>{"initial_gradient", "initial_upper_gradient"},
      std::vector<std::string>{"logits", "probability", "states", "final_state", "upper_states", "final_lower"});
  m->evaluate_probes = {{"lower_probe_gradient", [](const Outputs& o) { return FirstStateProbe(o, "states"); }, 0},
                        {"upper_probe_gradient", [](const Outputs& o) { return FirstStateProbe(o, "upper_states"); }, 0}};
  m->Build(training);
  return m;
}

}  // namespace rnn_cells
