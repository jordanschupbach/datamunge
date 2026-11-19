#pragma once

/// @file multi_gate_experts.hpp
/// @brief Shared expert banks with independent task gates and regression heads.

#include <stdexcept>
#include <string>
#include <vector>

#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Differentiable output and routing diagnostics from a multitask forward pass.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(42);
/// const auto fit = n::multi_gate_regression(p, "model", n::ones({2,5}), 2, 3);
/// const auto first_gate = fit.gates.front(); // Shape [2,3]
/// @endcode
/// @see multi_gate_regression()
struct MultiGateRegression {
  Tensor prediction; ///< Scalar prediction per row and task [N,T].
  Tensor experts; ///< Shared expert activations [N,E,W], undefined for shared-bottom.
  std::vector<Tensor> gates; ///< One softmax [N,E] per task, empty for shared-bottom.
  std::vector<Tensor> representations; ///< One task input [N,W] for each head.
};

/// @brief Predict multiple continuous tasks from a common input batch.
///
/// Each expert is a tanh dense layer. Each task learns a separate softmax
/// mixture over the expert bank and applies a tanh layer with a scalar
/// linear head. With zero experts, every head sees the same dense shared
/// representation. Keep the architecture fixed when reusing the prefix.
/// @param parameters Store owning the trainable weights.
/// @param name Parameter prefix.
/// @param input Nonempty feature tensor [N,D].
/// @param tasks Positive number of scalar tasks.
/// @param experts Positive number of experts, or zero for shared-bottom.
/// @param width Positive expert, shared representation and head hidden width.
/// @return Live predictions and intermediate tensors.
/// @throws std::invalid_argument Invalid dimensions or architecture.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(42);
/// const auto fit = n::multi_gate_regression(p, "shared", n::ones({2,5}), 2, 0);
/// const auto prediction = fit.prediction; // Shape [2,2]
/// @endcode
/// @par References
/// - Ma et al., Modeling Task Relationships in Multi-task Learning with
///   Multi-gate Mixture-of-Experts, KDD 2018. https://doi.org/10.1145/3219819.3220007
/// @see MultiGateRegression, dense()
inline MultiGateRegression multi_gate_regression(ParameterStore& parameters, const std::string& name,
                                                 const Tensor& input, int tasks, int experts,
                                                 int width = 12) {
  if (!input.defined() || input.dim() != 2 || input.size(0) == 0 || input.size(1) == 0 ||
      tasks <= 0 || experts < 0 || width <= 0)
    throw std::invalid_argument("invalid multi-gate regression geometry");
  MultiGateRegression out;
  Tensor shared;
  if (experts == 0) {
    shared = dense(parameters, name + "/shared", input, width, Activation::Tanh);
  } else {
    std::vector<Tensor> bank;
    for (int expert = 0; expert < experts; ++expert)
      bank.push_back(dense(parameters, name + "/expert_" + std::to_string(expert),
                           input, width, Activation::Tanh));
    out.experts = stack(bank, 1);
  }
  std::vector<Tensor> predictions;
  for (int task = 0; task < tasks; ++task) {
    const auto prefix = name + "/task_" + std::to_string(task);
    auto representation = shared;
    if (experts > 0) {
      auto gate = softmax(dense(parameters, prefix + "/gate", input, experts));
      representation = sum(out.experts * expand_dims(gate, 2), {1});
      out.gates.push_back(gate);
    }
    out.representations.push_back(representation);
    const auto hidden = dense(parameters, prefix + "/hidden", representation, width, Activation::Tanh);
    predictions.push_back(dense(parameters, prefix + "/head", hidden, 1));
  }
  out.prediction = concat(predictions, 1);
  return out;
}

}  // namespace datamunge::nn
