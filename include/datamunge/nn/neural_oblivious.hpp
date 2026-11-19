#pragma once

/// @file neural_oblivious.hpp
/// @brief Differentiable oblivious trees with sparse feature selection.

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Apply alpha=1.5 entmax along each row of a logit matrix.
///
/// Solves sum(relu(logits/2 - threshold)^2)=1 by bisection and uses
/// the implicit threshold derivative in reverse mode.
/// @param logits Nonempty finite floating matrix [N,D].
/// @return Live sparse probabilities with rows summing to one up to rounding.
/// @throws std::invalid_argument Invalid geometry, dtype, or nonfinite logits.
/// @par Example
/// @code
/// const auto p = dm::nn::entmax15(dm::nn::zeros({1,2})); // [0.5,0.5]
/// @endcode
/// @par References
/// - Peters, Niculae and Martins, Sparse Sequence-to-Sequence Models.
///   https://aclanthology.org/P19-1146/
/// @see neural_oblivious_regression()
inline Tensor entmax15(const Tensor& logits) {
  if (!logits.defined() || logits.dim() != 2 || logits.size(0) == 0 || logits.size(1) == 0 ||
      (logits.dtype() != DType::Float32 && logits.dtype() != DType::Float64) ||
      !all(is_finite(logits)).item<bool>())
    throw std::invalid_argument("entmax requires a finite nonempty floating matrix");
  const auto scaled = .5 * logits;
  const auto centered = scaled - max(scaled.detach(), {1}, true);
  const auto x = centered.detach();
  auto hi = max(x, {1}, true), lo = hi - 1.;
  const int iterations = logits.dtype() == DType::Float32 ? 24 : 53;
  for (int i = 0; i < iterations; ++i) {
    const auto mid = .5 * (lo + hi);
    const auto above = greater(sum(square(relu(x - mid)), {1}, true), 1.);
    lo = where(above, mid, lo);
    hi = where(above, hi, mid);
  }
  const auto threshold = .5 * (lo + hi);
  const auto root = relu(x - threshold);
  const auto correction = sum(root * (centered - x), {1}, true) / sum(root, {1}, true);
  return square(relu(centered - threshold - correction));
}

/// @brief Live outputs and explainable routing from an oblivious tree ensemble.
/// @par Example
/// @code
/// dm::nn::ParameterStore p(42);
/// const auto out = dm::nn::neural_oblivious_regression(p,"trees",dm::nn::zeros({2,3}),2,2);
/// const auto shape = out.routes.shape(); // [2,2,4]
/// @endcode
/// @see neural_oblivious_regression()
struct NeuralObliviousRegression {
  Tensor prediction; ///< Ensemble mean [N,1].
  Tensor feature_weights; ///< Entmax probabilities [trees*depth,D].
  Tensor split_probability; ///< Right-branch probabilities [N,trees*depth].
  Tensor routes; ///< Leaf probabilities [N,trees,2^depth], with depth 0 as the low bit.
};

/// @brief Predict with differentiable oblivious trees and trainable scalar leaves.
///
/// One shared split applies to all nodes at each tree depth. An entmax feature
/// selector and a temperature-scaled entmoid determine the split probability.
/// Routing uses the product of the branch probabilities along each leaf path.
/// Keep the same geometry for repeated calls with the same parameter prefix.
/// @param parameters Store owning selectors, thresholds, log temperatures and leaves.
/// @param name Parameter prefix.
/// @param input Nonempty finite floating feature tensor [N,D].
/// @param trees Positive tree count.
/// @param depth Number of splits per tree, from 1 to 16.
/// @return Differentiable predictions and routing diagnostics.
/// @throws std::invalid_argument Invalid geometry or selector logits.
/// @par Example
/// @code
/// dm::nn::ParameterStore p(42);
/// const auto out = dm::nn::neural_oblivious_regression(p,"trees",dm::nn::zeros({2,3}),2,1);
/// const auto mean_prediction = out.prediction; // Shape [2,1]
/// @endcode
/// @par References
/// - Popov, Morozov and Babenko, Neural Oblivious Decision Ensembles.
///   https://openreview.net/forum?id=r1eiu2VtwH
/// @see entmax15(), NeuralObliviousRegression
inline NeuralObliviousRegression neural_oblivious_regression(
    ParameterStore& parameters, const std::string& name, const Tensor& input,
    std::int64_t trees, int depth) {
  if (!input.defined() || input.dim() != 2 || input.size(0) == 0 || input.size(1) == 0 ||
      trees <= 0 || depth < 1 || depth > 16 || trees > std::numeric_limits<std::int64_t>::max() / depth)
    throw std::invalid_argument("invalid oblivious tree geometry");
  const auto splits = trees * depth;
  const auto selectors = parameters.get(name + "/selectors", {splits,input.size(1)}, init::uniform(1.5));
  const auto thresholds = parameters.get(name + "/thresholds", {splits}, init::zeros());
  const auto log_temperature = parameters.get(name + "/log_temperature", {splits}, init::constant(-1.));
  const auto leaves = parameters.get(name + "/leaves", {trees,1 << depth}, init::uniform(.25));
  const auto weights = entmax15(selectors);
  const auto z = clip((matmul(input, weights, false, true) - thresholds) / exp(log_temperature), -2., 2.);
  const auto probability = .5 + .125 * z * sqrt(8. - square(z));
  const auto n = input.size(0);
  const auto splits_by_tree = probability.reshape({n,trees,depth});
  auto routes = ones_like(slice(splits_by_tree, {0,0,0}, {n,trees,1}));
  for (int d = 0; d < depth; ++d) {
    const auto right = slice(splits_by_tree, {0,0,d}, {n,trees,1});
    routes = concat({routes * (1. - right), routes * right}, 2);
  }
  return {mean(sum(routes * leaves, {2}), {1}, true), weights, probability, routes};
}
}  // namespace datamunge::nn
