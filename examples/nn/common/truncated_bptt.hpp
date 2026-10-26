// Final-window truncated BPTT: carry the complete forward state, stop only the
// backward signal at each sequence's boundary max(length - window, 0). Port of
// tflow's common/nn/truncated_bptt.h (path) and common/ml/truncated_bptt.h
// (model: the 206-parameter tanh RNN, Adam .01, seed 39003; window 0 = full BPTT).
#pragma once

#include <memory>

#include "common/recurrent_model.hpp"
#include "common/rnn.hpp"

namespace rnn_cells {

inline Outputs TruncatedBptt(nn::ParameterStore& p, const Tensor& input, const Tensor& lengths, const std::vector<Tensor>& initial,
                             int window) {
  const auto          steps = nn::time_steps(input);
  Tensor              h = initial[0], boundary = initial[0];
  std::vector<Tensor> states{h};
  const Tensor        shifted = nn::sub(lengths, double(window));
  const Tensor        boundary_index = nn::maximum(shifted, nn::zeros_like(shifted));
  for (int t = 0; t < 12; ++t) {
    if (window > 0) {
      const Tensor selected = nn::expand_dims(nn::equal(boundary_index, double(t)), 1);
      boundary              = nn::where(selected, h, boundary);
      if (t > 0) h = nn::where(selected, nn::stop_gradient(h), h);
    }
    h = Blend(StepMask(lengths, t), nn::rnn_step(p, "cell", steps[t], h), h);
    states.push_back(h);
  }
  return {{"logits", Head(p, "head", h, 12, 2)}, {"states", nn::stack_time(states)}, {"final_state", h}, {"boundary_state", boundary}};
}

inline std::unique_ptr<RecurrentModel> TruncatedBpttModel(bool training = true, int window = 3) {
  auto m = std::make_unique<RecurrentModel>(
      39003,
      [window](nn::ParameterStore& p, const Tensor& x, const Tensor& l, const std::vector<Tensor>& s) { return TruncatedBptt(p, x, l, s, window); },
      nn::Shape{12, 2}, std::vector<nn::Shape>{{12}}, std::vector<std::string>{"initial_gradient"},
      std::vector<std::string>{"logits", "probability", "states", "final_state", "boundary_state"});
  m->Build(training);
  return m;
}

}  // namespace rnn_cells
