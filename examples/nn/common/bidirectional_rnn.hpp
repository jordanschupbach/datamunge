// Independent forward/backward tanh cells; reverse traversal masks padding and
// preserves original-time alignment. Port of tflow's common/nn/bidirectional_rnn.h
// (the path) and common/ml/bidirectional_rnn.h (the model: 410 parameters,
// Adam .01, Params seed 28003).
#pragma once

#include <memory>

#include "common/recurrent_model.hpp"
#include "common/rnn.hpp"

namespace rnn_cells {

inline Outputs BidirectionalRnn(nn::ParameterStore& p, const Tensor& input, const Tensor& lengths,
                                const std::vector<Tensor>& initial) {
  const auto          xs = nn::time_steps(input);
  Tensor              h = initial[0], b = initial[1];
  std::vector<Tensor> forward{h}, backward{b}, aligned(12);
  for (int t = 0; t < 12; ++t) h = Blend(StepMask(lengths, t), nn::rnn_step(p, "forward", xs[t], h), h), forward.push_back(h);
  for (int t = 11; t >= 0; --t) {
    b = Blend(StepMask(lengths, t), nn::rnn_step(p, "backward", xs[t], b), b);
    backward.push_back(b);
    aligned[t] = b;
  }
  const Tensor joined = nn::concat({h, b}, 1);
  return {{"logits", Head(p, "head", joined, 24, 2)},   {"states", nn::stack_time(forward)},
          {"final_state", joined},                       {"backward_states", nn::stack_time(backward)},
          {"backward_aligned", nn::stack_time(aligned)}, {"final_backward", b}};
}

inline std::unique_ptr<RecurrentModel> BidirectionalRnnModel(bool training = true) {
  auto m = std::make_unique<RecurrentModel>(
      28003, BidirectionalRnn, nn::Shape{12, 2}, std::vector<nn::Shape>{{12}, {12}},
      std::vector<std::string>{"initial_gradient", "backward_initial_gradient"},
      std::vector<std::string>{"logits", "probability", "states", "final_state", "backward_states", "backward_aligned",
                               "final_backward"});
  // First forward state (time 1, unit 0) and first backward-aligned context (time 0, unit 0).
  m->evaluate_probes = {
      {"forward_probe_gradient", [](const Outputs& o) { return nn::sum(nn::slice(o.at("states"), {0, 1, 0}, {-1, 1, 1})); }, 0},
      {"backward_probe_gradient",
       [](const Outputs& o) { return nn::sum(nn::slice(o.at("backward_aligned"), {0, 0, 0}, {-1, 1, 1})); }, 0}};
  m->Build(training);
  return m;
}

}  // namespace rnn_cells
