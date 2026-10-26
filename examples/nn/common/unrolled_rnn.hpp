// Explicit twelve-step shared-weight tanh recurrence and valid-length state
// masking. Port of tflow's common/nn/unrolled_rnn.h (206 parameters).
#pragma once

#include <memory>

#include "common/recurrent_model.hpp"
#include "common/rnn.hpp"

namespace rnn_cells {

inline Outputs UnrolledRnn(nn::ParameterStore& p, const Tensor& input, const Tensor& lengths,
                           const std::vector<Tensor>& initial) {
  const auto          steps = nn::time_steps(input);
  Tensor              h     = initial[0];
  std::vector<Tensor> states{h};
  for (int t = 0; t < 12; ++t) {
    h = Blend(StepMask(lengths, t), nn::rnn_step(p, "cell", steps[t], h), h);
    states.push_back(h);
  }
  return {{"logits", Head(p, "head", h, 12, 2)}, {"states", nn::stack_time(states)}, {"final_state", h}};
}

/// tflow's ml::UnrolledRnnModel (vanilla_rnn): 206 parameters, Adam .01, seed 26003.
inline std::unique_ptr<RecurrentModel> UnrolledRnnModel(bool training = true) {
  auto m = std::make_unique<RecurrentModel>(26003, UnrolledRnn, nn::Shape{12, 2}, std::vector<nn::Shape>{{12}},
                                            std::vector<std::string>{"initial_gradient"},
                                            std::vector<std::string>{"logits", "probability", "states", "final_state"});
  m->Build(training);
  return m;
}

}  // namespace rnn_cells
