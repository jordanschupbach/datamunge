#pragma once

/// @file equivariant.hpp
/// @brief Tied convolution kernels for quarter turns and a scalar-message Euclidean graph update.

#include <datamunge/nn/ops.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace datamunge::nn {

/// @brief Expands a scalar spatial kernel into four rotated orientation channels.
/// @param filter CPU floating matrix [side, side], with positive odd side length.
/// @return Kernel [side, side, 1, 4] for unit-stride SAME convolution.
/// @throws std::invalid_argument Invalid shape, dtype or device.
/// @par Example
/// @code
/// const auto kernel = dm::nn::c4_lift_kernel(dm::nn::ones({3, 3}));
/// std::print("{}\n", kernel.size(3)); // 4
/// @endcode
/// @par References
/// Cohen and Welling, Group Equivariant Convolutional Networks (2016).
/// @see c4_group_kernel(), conv2d()
inline Tensor c4_lift_kernel(const Tensor& filter) {
  if (!filter.defined() || filter.dim() != 2 || filter.size(0) != filter.size(1) ||
      filter.size(0) % 2 != 1 || !is_floating(filter.dtype()) ||
      filter.device().backend != compute::Backend::CPU)
    throw std::invalid_argument("c4_lift_kernel: expected an odd square CPU floating matrix");
  std::vector<Tensor> turns{filter};
  for (int turn = 1; turn < 4; ++turn)
    turns.push_back(reverse(transpose(turns.back(), {1, 0}), {0}));
  return expand_dims(stack(turns, 2), 2);
}

/// @brief Expands a relative-orientation kernel into a tied C4 convolution kernel.
/// @param filter CPU floating tensor [side, side, 4] with positive odd side length.
/// @return Kernel [side, side, input orientation, output orientation]. An
/// output orientation g uses the spatial rotation g and relative channel h-g modulo four.
/// @throws std::invalid_argument Invalid shape, dtype or device.
/// @par Example
/// @code
/// const auto kernel = dm::nn::c4_group_kernel(dm::nn::ones({3, 3, 4}));
/// std::print("{}\n", kernel.size(2)); // 4
/// @endcode
/// @see c4_lift_kernel(), conv2d()
inline Tensor c4_group_kernel(const Tensor& filter) {
  if (!filter.defined() || filter.dim() != 3 || filter.size(0) != filter.size(1) ||
      filter.size(0) % 2 != 1 || filter.size(2) != 4 || !is_floating(filter.dtype()) ||
      filter.device().backend != compute::Backend::CPU)
    throw std::invalid_argument("c4_group_kernel: expected an odd square CPU filter with four relative orientations");
  auto spatial = filter;
  std::vector<Tensor> output;
  for (int g = 0; g < 4; ++g) {
    std::vector<Tensor> input;
    for (int h = 0; h < 4; ++h)
      input.push_back(select(spatial, 2, (h - g + 4) % 4));
    output.push_back(stack(input, 2));
    if (g < 3) spatial = reverse(transpose(spatial, {1, 0, 2}), {0});
  }
  return stack(output, 3);
}

/// @brief Updated coordinates and scalar node features.
/// @par Example
/// Call euclidean_graph_step() to obtain both updated tensors.
/// @see euclidean_graph_step()
struct EuclideanGraphStep {
  Tensor positions; ///< Updated positions [nodes, dimension].
  Tensor features; ///< Updated scalar features [nodes, 1].
};

/// @brief Applies one rigid-motion-equivariant scalar-message update to a dense point graph.
/// The directed message from j to i is the edge multiplier times
/// tanh(feature_i + feature_scale * feature_j - distance_scale * squared_distance).
/// Coordinates receive messages multiplied by relative position. Both updates
/// take the mean over all possible neighbors, including masked edges.
/// @param positions Finite CPU floating matrix [nodes, dimension], positive dimensions.
/// @param features Finite scalar features [nodes, 1] in the coordinate dtype.
/// @param edges Finite invariant edge multipliers [nodes, nodes, 1] in the coordinate dtype.
/// @param feature_scale Finite neighbor feature coefficient.
/// @param distance_scale Finite squared-distance coefficient.
/// @return Residually updated positions and features.
/// @throws std::invalid_argument Incompatible shapes, devices, dtypes or coefficients.
/// @par Example
/// @code
/// const auto s = dm::nn::euclidean_graph_step(dm::nn::zeros({2, 3}),
///     dm::nn::ones({2, 1}), dm::nn::zeros({2, 2, 1}));
/// std::print("{}\n", s.features.at({0, 0})); // 1
/// @endcode
/// @par References
/// Satorras, Hoogeboom and Welling, E(n) Equivariant Graph Neural Networks (2021).
inline EuclideanGraphStep euclidean_graph_step(const Tensor& positions, const Tensor& features,
                                               const Tensor& edges, double feature_scale = .7,
                                               double distance_scale = .25) {
  if (!positions.defined() || positions.dim() != 2 || positions.numel() == 0 ||
      !features.defined() || features.shape() != Shape{positions.size(0), 1} ||
      !edges.defined() || edges.shape() != Shape{positions.size(0), positions.size(0), 1} ||
      !is_floating(positions.dtype()) || features.dtype() != positions.dtype() ||
      edges.dtype() != positions.dtype() || positions.device().backend != compute::Backend::CPU ||
      features.device().backend != compute::Backend::CPU || edges.device().backend != compute::Backend::CPU ||
      !std::isfinite(feature_scale) || !std::isfinite(distance_scale))
    throw std::invalid_argument("euclidean_graph_step: incompatible CPU tensors or coefficients");
  const auto relative = expand_dims(positions, 1) - expand_dims(positions, 0);
  const auto distance = sum(square(relative), {2}, true);
  const auto scalar = expand_dims(features, 1) + feature_scale * expand_dims(features, 0);
  const auto message = edges * tanh(scalar - distance_scale * distance);
  return {positions + mean(message * relative, {1}), features + mean(message, {1})};
}

} // namespace datamunge::nn
