#pragma once

/// @file spline_edges.hpp
/// @brief Trainable edges built from uniformly spaced linear splines.

#include <datamunge/nn/ops.hpp>

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace datamunge::nn {

/// @brief Evaluates linear hat basis functions on a symmetric uniform grid.
/// Input is clipped to the grid endpoints, so edge functions extrapolate
/// as constants. The basis is differentiable away from knots and clipping
/// boundaries.
/// @param inputs Nonempty CPU Float32 or Float64 tensor.
/// @param knots Number of uniformly spaced knots, at least two.
/// @param radius Finite positive endpoint of the symmetric grid.
/// @return Tensor with the input shape and an appended knot axis.
/// @throws std::invalid_argument Invalid inputs or a grid with unusable spacing.
/// @par Example
/// @code
/// const auto basis = dm::nn::linear_spline_basis(dm::nn::zeros({1}), 3, 1.);
/// std::print("{}\n", basis.at({0, 1})); // 1
/// @endcode
/// @see spline_edge_layer()
inline Tensor linear_spline_basis(const Tensor& inputs, std::int64_t knots, double radius) {
  if (!inputs.defined() || inputs.numel() == 0 || inputs.device().backend != compute::Backend::CPU ||
      (inputs.dtype() != DType::Float32 && inputs.dtype() != DType::Float64) ||
      knots < 2 || !(radius > 0))
    throw std::invalid_argument("linear_spline_basis: invalid inputs or grid");
  const double extent = 2. * radius;
  const double spacing = extent / double(knots - 1);
  if (!(spacing > 0) || !std::isfinite(extent) ||
      (inputs.dtype() == DType::Float32 &&
       (!std::isfinite(static_cast<float>(extent)) || static_cast<float>(spacing) == 0)))
    throw std::invalid_argument("linear_spline_basis: grid spacing is outside the input dtype range");
  const auto centers = linspace(-radius, radius, knots, inputs.dtype());
  const auto distance = abs(expand_dims(clip(inputs, -radius, radius), -1) - centers);
  return relu(1. - distance / spacing);
}

/// @brief Output and basis values of a linear spline-edge layer.
/// @par Example
/// See spline_edge_layer().
struct SplineEdgeResult {
  Tensor output; ///< Sum of edge responses for each output node.
  Tensor basis;  ///< Basis indexed by row, input feature, and knot.
};

/// @brief Applies one univariate linear spline per input-to-output edge, then sums over inputs.
/// @param inputs Nonempty CPU Float32 or Float64 matrix [rows, inputs].
/// @param weights Coefficients [inputs * knots, outputs] in the same dtype and device type.
/// Rows are ordered by input feature, then knot.
/// @param knots Number of uniformly spaced knots, at least two.
/// @param radius Finite positive endpoint of the symmetric grid.
/// @return Output [rows, outputs] and basis [rows, inputs, knots].
/// @throws std::invalid_argument Invalid matrices or spline grid.
/// @par Example
/// @code
/// const auto y = dm::nn::spline_edge_layer(dm::nn::zeros({2, 1}), dm::nn::ones({3, 1}), 3, 1.);
/// std::print("{}\n", y.output.at({0, 0})); // 1
/// @endcode
/// @par References
/// - Liu et al. (2024), https://arxiv.org/abs/2404.19756. This layer uses linear splines alone.
/// @see linear_spline_basis()
inline SplineEdgeResult spline_edge_layer(const Tensor& inputs, const Tensor& weights,
                                          std::int64_t knots, double radius) {
  if (!inputs.defined() || !weights.defined() || inputs.dim() != 2 || weights.dim() != 2 ||
      weights.dtype() != inputs.dtype() || weights.device().backend != compute::Backend::CPU ||
      knots < 2 || weights.size(0) % knots != 0 || weights.size(0) / knots != inputs.size(1) ||
      weights.size(1) == 0)
    throw std::invalid_argument("spline_edge_layer: incompatible coefficient matrix");
  const auto basis = linear_spline_basis(inputs, knots, radius);
  return {matmul(reshape(basis, {inputs.size(0), weights.size(0)}), weights), basis};
}

} // namespace datamunge::nn
