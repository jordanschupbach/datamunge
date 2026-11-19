#pragma once

/// @file neural_tangent_kernel.hpp
/// @brief Finite and infinite-width kernels for a bias-free two-layer ReLU network.

#include <datamunge/nn/ops.hpp>

#include <numbers>
#include <stdexcept>

namespace datamunge::nn {

/// @brief Computes the parameter-gradient Gram matrix of a scalar two-layer ReLU network.
/// The network sums its weighted hidden activations and divides by sqrt(width).
/// Both input and output weights are parameters. The ReLU derivative at zero is zero.
/// @param inputs Nonempty CPU floating matrix [rows, features].
/// @param hidden CPU hidden weight matrix [features, width] of matching dtype.
/// @param output CPU output weight vector [width] of matching dtype.
/// @return Tensor [rows, rows] in the input dtype.
/// @throws std::invalid_argument Incompatible shapes, dtypes or devices.
/// @par Example
/// @code
/// const auto x = dm::nn::ones({2, 1});
/// const auto k = dm::nn::relu_neural_tangent_kernel(x, dm::nn::ones({1, 1}), dm::nn::ones({1}));
/// std::print("{}\n", k.at({0, 1})); // 2
/// @endcode
/// @par References
/// - Jacot, Gabriel and Hongler (2018), https://arxiv.org/abs/1806.07572.
/// @see relu_neural_tangent_kernel_limit()
inline Tensor relu_neural_tangent_kernel(const Tensor& inputs, const Tensor& hidden,
                                        const Tensor& output) {
  if (!inputs.defined() || !hidden.defined() || !output.defined() ||
      inputs.dim() != 2 || hidden.dim() != 2 || output.dim() != 1 ||
      inputs.size(0) == 0 || inputs.size(1) == 0 || hidden.size(0) != inputs.size(1) ||
      hidden.size(1) == 0 || output.size(0) != hidden.size(1) ||
      !is_floating(inputs.dtype()) || hidden.dtype() != inputs.dtype() || output.dtype() != inputs.dtype() ||
      inputs.device().backend != compute::Backend::CPU || hidden.device().backend != compute::Backend::CPU ||
      output.device().backend != compute::Backend::CPU)
    throw std::invalid_argument("relu_neural_tangent_kernel: incompatible CPU floating inputs and weights");
  const auto affine = matmul(inputs, hidden);
  const auto active = relu(affine);
  const auto gate = cast(greater(affine, 0.), inputs.dtype());
  const auto output_part = matmul(active, active, false, true);
  const auto input_part = matmul(inputs, inputs, false, true) *
                          matmul(gate * square(output), gate, false, true);
  return (output_part + input_part) / double(hidden.size(1));
}

/// @brief Evaluates the two-layer ReLU tangent kernel at infinite width.
/// Uses the arc-cosine expression for independent standard normal hidden and
/// output weights. Zero input rows produce zero kernel entries.
/// @param inputs Nonempty CPU floating matrix [rows, features].
/// @return Tensor [rows, rows] in the input dtype.
/// @throws std::invalid_argument Invalid shape, dtype or device.
/// @par Example
/// @code
/// const auto k = dm::nn::relu_neural_tangent_kernel_limit(dm::nn::ones({2, 1}));
/// std::print("{}\n", k.at({0, 1})); // 1
/// @endcode
/// @par References
/// - Jacot, Gabriel and Hongler (2018), https://arxiv.org/abs/1806.07572.
/// @see relu_neural_tangent_kernel()
inline Tensor relu_neural_tangent_kernel_limit(const Tensor& inputs) {
  if (!inputs.defined() || inputs.dim() != 2 || inputs.size(0) == 0 || inputs.size(1) == 0 ||
      !is_floating(inputs.dtype()) || inputs.device().backend != compute::Backend::CPU)
    throw std::invalid_argument("relu_neural_tangent_kernel_limit: expected a nonempty CPU floating matrix");
  const auto gram = matmul(inputs, inputs, false, true);
  const auto lengths = sqrt(sum(square(inputs), {1}, true));
  const auto norm = matmul(lengths, lengths, false, true);
  const auto cosine = clip(gram / where(greater(norm, 0.), norm, 1.), -1., 1.);
  const auto angle = acos(cosine);
  return (norm * sin(angle) / 2. + gram * (std::numbers::pi - angle)) / std::numbers::pi;
}

} // namespace datamunge::nn
