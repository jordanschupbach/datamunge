#pragma once

/// @file tabnet.hpp
/// @brief Sparsemax and sequential feature selection for tabular regression.

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Project each row onto the probability simplex using sparsemax.
///
/// The projection has exact zeros outside its support. Reverse mode treats
/// the support as fixed and subtracts the mean adjoint of its active entries.
/// @param scores Nonempty finite floating matrix [N,D].
/// @return Differentiable sparse probability matrix [N,D].
/// @throws std::invalid_argument Invalid geometry, dtype or nonfinite scores.
/// @par Example
/// @code
/// const auto probability = dm::nn::sparsemax(dm::nn::tensor({{2.,0.}})); // [1,0]
/// @endcode
/// @par References
/// - Martins and Astudillo (2016), From Softmax to Sparsemax.
///   https://proceedings.mlr.press/v48/martins16.html
/// @see tabnet_regression()
inline Tensor sparsemax(const Tensor& scores) {
  if (!scores.defined() || scores.dim() != 2 || scores.size(0) == 0 || scores.size(1) == 0 ||
      (scores.dtype() != DType::Float32 && scores.dtype() != DType::Float64) ||
      !all(is_finite(scores)).item<bool>())
    throw std::invalid_argument("sparsemax requires a finite nonempty floating matrix");
  const auto centered = maximum(scores-max(scores.detach(),{1},true),full_like(scores,-2.));
  const auto x = centered.detach();
  const auto sorted = top_k(x,x.size(1)).first;
  const auto rank = arange(1,x.size(1)+1,1,scores.dtype()).to(x.context());
  const auto cumulative = cumsum(sorted,1);
  const auto supported = greater(1.+rank*sorted,cumulative);
  const auto preliminary = max(where(supported,(cumulative-1.)/rank,-3.),{1},true);
  const auto active = greater(x,preliminary).to(scores.dtype());
  const auto threshold = (sum(active*centered,{1},true)-1.)/sum(active,{1},true);
  return relu(centered-threshold);
}

/// @brief Live TabNet regression output and per-step diagnostics.
/// @par Example
/// @code
/// dm::nn::ParameterStore p(42);
/// const auto out = dm::nn::tabnet_regression(p,"tab",dm::nn::zeros({2,4}));
/// const auto shape = out.masks.shape(); // [2,3,4]
/// @endcode
/// @see tabnet_regression()
struct TabNetRegression {
  Tensor prediction; ///< Scalar response per row [N,1].
  Tensor masks; ///< Sparse feature weights [N,steps,D].
  Tensor attention_inputs; ///< Prior-weighted mask logits [N,steps,D].
  Tensor decisions; ///< Nonnegative decision channels [N,steps,width/2].
};

/// @brief Apply sequential sparse feature selection and gated residual transforms.
///
/// This compact TabNet architecture shares the first gated transform over
/// steps and uses step-specific attention and residual transforms. It sums
/// nonnegative decision channels for regression. Keep geometry fixed when
/// reusing the parameter prefix.
/// @param parameters Store owning the trainable weights.
/// @param name Parameter prefix.
/// @param input Nonempty feature matrix [N,D].
/// @param width Positive even transform width.
/// @param steps Positive decision step count.
/// @param relaxation Finite feature reuse multiplier, at least one.
/// @return Differentiable predictions and per-step diagnostics.
/// @throws std::invalid_argument Invalid architecture or sparsemax input.
/// @par Example
/// @code
/// dm::nn::ParameterStore p(42);
/// auto out=dm::nn::tabnet_regression(p,"tab",dm::nn::ones({3,2}),8,2);
/// auto prediction=out.prediction; // [3,1]
/// @endcode
/// @par References
/// - Arik and Pfister, TabNet: Attentive Interpretable Tabular Learning.
///   https://arxiv.org/abs/1908.07442
/// @see sparsemax(), TabNetRegression
inline TabNetRegression tabnet_regression(ParameterStore& parameters, const std::string& name,
                                          const Tensor& input, int width = 16, int steps = 3,
                                          double relaxation = 1.4) {
  if (!input.defined() || input.dim() != 2 || input.size(0) == 0 || input.size(1) == 0 ||
      width <= 0 || width % 2 != 0 || steps <= 0 || !std::isfinite(relaxation) || relaxation < 1.)
    throw std::invalid_argument("invalid TabNet geometry or feature reuse");
  const auto rows = input.size(0), fields = input.size(1);
  const auto gated = [&](const std::string& key, const Tensor& x) {
    const auto wide = dense(parameters,key,x,2LL*width);
    return slice(wide,{0,0},{rows,width})*sigmoid(slice(wide,{0,width},{rows,width}));
  };
  auto context = gated(name+"/shared",input), prior = ones_like(input);
  std::vector<Tensor> masks, attention, decisions;
  for (int i = 0; i < steps; ++i) {
    const auto prefix = name+"/step_"+std::to_string(i);
    const auto scores = prior*dense(parameters,prefix+"/attention",context,fields);
    const auto mask = sparsemax(scores);
    prior = prior*(relaxation-mask);
    const auto selected = gated(name+"/shared",input*mask);
    const auto transformed = std::sqrt(.5)*(selected+gated(prefix+"/transform",selected));
    decisions.push_back(relu(slice(transformed,{0,0},{rows,width/2})));
    if (i+1 < steps)
      context = dense(parameters,prefix+"/context",slice(transformed,{0,width/2},{rows,width/2}),width);
    masks.push_back(mask);
    attention.push_back(scores);
  }
  auto stack_decisions = stack(decisions,1);
  auto prediction = dense(parameters,name+"/head",sum(stack_decisions,{1}),1);
  return {prediction,stack(masks,1),stack(attention,1),stack_decisions};
}
}  // namespace datamunge::nn
