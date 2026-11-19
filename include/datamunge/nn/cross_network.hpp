#pragma once

/// @file cross_network.hpp
/// @brief Trainable feature crossings for deep and cross networks.

#include <stdexcept>
#include <string>

#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Apply a residual feature crossing to two batches of the same width.
///
/// Full-matrix DCN-v2 computes state + base * (state W + b), where *
/// is elementwise. The vector DCN option computes state + base * (state w)
/// + b, broadcasting the scalar projection per row. Pass the same original
/// base at every depth. Weights start at normal standard deviation 0.1
/// and biases at zero. Use the same variant and width per parameter name.
/// @param parameters Store owning differentiable weights and biases.
/// @param name Parameter prefix, extended by /W and /b.
/// @param base Original input [N,D] with positive D.
/// @param state Current cross-network state [N,D] with the same tensor type as base.
/// @param full_matrix True for DCN-v2 (default), false for vector DCN.
/// @return Differentiable crossed state [N,D].
/// @throws std::invalid_argument Undefined or incompatible input geometry.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore parameters(42);
/// const auto base = n::ones({2,3});
/// const auto crossed = n::cross_layer(parameters, "cross", base, base);
/// const auto width = crossed.size(1); // 3
/// @endcode
/// @par References
/// - Wang et al., DCN V2: Improved Deep & Cross Network and Practical Lessons
///   for Web-scale Learning to Rank Systems. https://arxiv.org/abs/2008.13535
/// - Wang et al., Deep & Cross Network for Ad Click Predictions.
///   https://arxiv.org/abs/1708.05123
/// @see dense()
inline Tensor cross_layer(ParameterStore& parameters, const std::string& name,
                          const Tensor& base, const Tensor& state, bool full_matrix = true) {
  if (!base.defined() || !state.defined() || base.dim() != 2 || base.shape() != state.shape() ||
      base.size(1) == 0 || base.dtype() != state.dtype())
    throw std::invalid_argument("cross layer requires matching [N,D] inputs");
  const auto width = base.size(1);
  const auto weight = parameters.get(name + "/W", {width, full_matrix ? width : 1}, init::normal(.1));
  const auto bias = parameters.get(name + "/b", {width}, init::zeros());
  const auto projection = matmul(state, weight);
  return full_matrix ? state + base * (projection + bias) : state + base * projection + bias;
}

}  // namespace datamunge::nn
