// Explicit gradient policies: one global scale preserves direction; coordinate
// clipping generally changes it. Port of tflow's common/nn/gradient_control.h
// (policy 0 none, 1 global norm, 2 coordinate value clipping), kept separate
// from nn::clip_by_global_norm so the float32 arithmetic follows tflow's
// (scale = min(1, limit / max(norm, 1e-30))).
#pragma once

#include <datamunge/nn/nn.hpp>
#include <vector>

namespace rnn_cells {

namespace nn = datamunge::nn;

struct ControlledGradients {
  std::vector<nn::Tensor> values;
  nn::Tensor              raw_norm, applied_norm, scale;
};

inline nn::Tensor GlobalNorm(const std::vector<nn::Tensor>& gradients) {
  nn::Tensor total;
  for (const auto& g : gradients) {
    const nn::Tensor s = nn::sum(nn::square(g));
    total              = total.defined() ? nn::add(total, s) : s;
  }
  return nn::sqrt(total);
}

inline ControlledGradients ControlGradients(const std::vector<nn::Tensor>& gradients, int policy, float norm_limit = .05f,
                                            float value_limit = .01f) {
  const nn::Tensor norm = GlobalNorm(gradients);
  const nn::Tensor scale =
      policy == 1 ? nn::minimum(nn::full({}, 1.0, norm.dtype()),
                                nn::div(nn::full({}, double(norm_limit), norm.dtype()), nn::maximum(norm, nn::full({}, double(1e-30f), norm.dtype()))))
                  : nn::full({}, 1.0, norm.dtype());
  std::vector<nn::Tensor> result;
  for (const auto& g : gradients)
    result.push_back(policy == 2 ? nn::clip(g, -double(value_limit), double(value_limit)) : nn::mul(g, scale));
  return {result, norm, GlobalNorm(result), scale};
}

}  // namespace rnn_cells
