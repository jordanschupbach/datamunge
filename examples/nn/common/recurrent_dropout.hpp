// Explicit hidden-to-hidden inverted dropout on the 206-parameter tanh RNN; the
// caller supplies per-time or time-shared masks [B, 12, 12]. Port of tflow's
// common/nn/recurrent_dropout.h (path) and common/ml/recurrent_dropout.h
// (model: Adam .01, seed 30003). The mask is tflow's extra placeholder; here it
// is the model's side input "recurrent_mask" (all ones when absent, i.e.
// inference with dropout disabled).
#pragma once

#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "common/recurrent_model.hpp"
#include "common/rnn.hpp"

namespace rnn_cells {

inline Outputs RecurrentDropout(nn::ParameterStore& p, const Tensor& input, const Tensor& lengths, const std::vector<Tensor>& initial,
                                const Tensor& recurrent_mask) {
  const auto          steps = nn::time_steps(input), masks = nn::time_steps(recurrent_mask);
  Tensor              h = initial[0];
  std::vector<Tensor> states{h};
  for (int t = 0; t < 12; ++t) {
    h = Blend(StepMask(lengths, t), nn::rnn_step(p, "cell", steps[t], nn::mul(h, masks[t])), h);
    states.push_back(h);
  }
  return {{"logits", Head(p, "head", h, 12, 2)}, {"states", nn::stack_time(states)}, {"final_state", h}, {"recurrent_mask", recurrent_mask}};
}

inline std::unique_ptr<RecurrentModel> RecurrentDropoutModel(bool training = true) {
  auto m = std::make_unique<RecurrentModel>(30003, nullptr, nn::Shape{12, 2}, std::vector<nn::Shape>{{12}},
                                            std::vector<std::string>{"initial_gradient"},
                                            std::vector<std::string>{"logits", "probability", "states", "final_state", "recurrent_mask"});
  RecurrentModel* model = m.get();
  m->forward = [model](nn::ParameterStore& p, const Tensor& x, const Tensor& l, const std::vector<Tensor>& s) {
    const auto it = model->side_inputs.find("recurrent_mask");
    return RecurrentDropout(p, x, l, s, it != model->side_inputs.end() ? it->second : nn::ones({x.shape()[0], 12, 12}));
  };
  m->Build(training);
  return m;
}

/// tflow's Mask(n, seed, shared): uniforms (float64 [n, 12, 12]) from std::mt19937;
/// keep (1.25) when u >= .2, else drop (0). `shared` reuses time 0's draws for every step.
inline std::pair<Tensor, Tensor> DropoutMask(int n, int seed, bool shared) {
  std::vector<double> uniform(static_cast<std::size_t>(n) * 144);
  std::vector<float>  mask(uniform.size());
  std::mt19937        rng(static_cast<std::mt19937::result_type>(seed));
  for (auto& u : uniform) u = (double(rng()) + .5) / 4294967296.;
  for (int b = 0; b < n; ++b)
    for (int t = 0; t < 12; ++t)
      for (int j = 0; j < 12; ++j)
        mask[(static_cast<std::size_t>(b) * 12 + t) * 12 + j] = uniform[(static_cast<std::size_t>(b) * 12 + (shared ? 0 : t)) * 12 + j] >= .2 ? 1.25f : 0.f;
  return {nn::from_vector(mask, {n, 12, 12}), nn::from_vector(uniform, {n, 12, 12})};
}

}  // namespace rnn_cells
