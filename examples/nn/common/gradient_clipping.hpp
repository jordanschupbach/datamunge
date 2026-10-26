// The unrolled tanh RNN trained with plain SGD .1 under an explicit gradient
// policy (none / global norm .05 / coordinate value .01). Port of tflow's
// common/ml/gradient_clipping.h (Params seed 41003, 206 parameters).
#pragma once

#include <memory>

#include "common/gradient_control.hpp"
#include "common/recurrent_model.hpp"
#include "common/unrolled_rnn.hpp"

namespace rnn_cells {

inline std::unique_ptr<RecurrentModel> GradientClippingModel(bool training = true, int policy = 0) {
  auto m = std::make_unique<RecurrentModel>(41003, UnrolledRnn, nn::Shape{12, 2}, std::vector<nn::Shape>{{12}},
                                            std::vector<std::string>{"initial_gradient"},
                                            std::vector<std::string>{"logits", "probability", "states", "final_state"});
  m->gradient_policy = [policy](const std::vector<Tensor>& g) { return ControlGradients(g, policy).values; };
  m->Build(training, .1, nn::OptimizerKind::Sgd);
  return m;
}

/// The controlled `raw` loss gradients by parameter name plus raw_norm,
/// applied_norm and scale (tflow's Applied()).
inline nn::NamedTensors AppliedGradients(const RecurrentModel& model, const std::vector<Tensor>& raw, int policy) {
  const auto       controlled = ControlGradients(raw, policy);
  nn::NamedTensors out;
  for (std::size_t i = 0; i < controlled.values.size(); ++i) out.emplace_back(model.store.names()[i], controlled.values[i].detach());
  out.emplace_back("raw_norm", controlled.raw_norm.detach());
  out.emplace_back("applied_norm", controlled.applied_norm.detach());
  out.emplace_back("scale", controlled.scale.detach());
  return out;
}

}  // namespace rnn_cells
