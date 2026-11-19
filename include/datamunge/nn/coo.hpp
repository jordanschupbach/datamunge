/// @file
/// @brief Differentiable rectangular multiplication from sparse COO entries.
#pragma once

#include <datamunge/nn/ops.hpp>
#include <datamunge/nn/tensor.hpp>

#include <array>
#include <stdexcept>
#include <utility>

namespace datamunge::nn {

/// @brief Multiply a sparse COO matrix by a dense feature matrix.
///
/// Entries are ordered as (receiver, source). Parallel edges accumulate,
/// unreferenced receivers get zero, and coordinates may arrive in any order.
/// Materializes one feature row per entry. Gradients flow to weights and
/// features. The transpose path reverses each edge without a dense copy.
///
/// @param coordinates Int32/Int64 [E,2] tensor, zero-based (row, column).
/// @param weights Float32/Float64 [E] tensor with one value per entry.
/// @param dimensions Positive row and column counts of the original matrix.
/// @param features [columns,D] tensor matching the weights dtype, or [rows,D]
/// when transpose is true. D is positive.
/// @param transpose Use the matrix transpose if true, default false.
/// @return [rows,D] weighted feature sums, or [columns,D] for the transpose.
/// @throws std::invalid_argument For incompatible shapes, dimensions or dtypes.
/// @throws std::out_of_range For coordinates outside the declared shape.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto edges = n::from_vector(std::vector<int>{0, 1, 0, 1}, {2, 2});
/// const auto output = n::coo_matmul(edges, n::tensor({2., 3.}, {2}), {3, 2},
///                                   n::tensor({1., 4.}, {2, 1}));
/// // output contains 20, 0, 0.
/// @endcode
/// @par References
/// - TensorFlow, SparseTensorDenseMatMul, https://www.tensorflow.org/api_docs/python/tf/raw_ops/SparseTensorDenseMatMul
/// @see gather(), index_add()
inline Tensor coo_matmul(const Tensor &coordinates, const Tensor &weights,
                         std::array<std::int64_t, 2> dimensions, const Tensor &features,
                         bool transpose = false) {
  const auto [rows, columns] = dimensions;
  const auto input_rows = transpose ? rows : columns;
  const auto output_rows = transpose ? columns : rows;
  if (rows <= 0 || columns <= 0 || coordinates.dim() != 2 || coordinates.size(1) != 2 ||
      (coordinates.dtype() != DType::Int32 && coordinates.dtype() != DType::Int64) ||
      weights.dim() != 1 || weights.size(0) != coordinates.size(0) ||
      (weights.dtype() != DType::Float32 && weights.dtype() != DType::Float64) ||
      features.dim() != 2 || features.size(0) != input_rows || features.size(1) <= 0 ||
      features.dtype() != weights.dtype())
    throw std::invalid_argument("coo_matmul: incompatible coordinates, weights or features");

  const auto input = select(coordinates, 1, transpose ? 0 : 1);
  const auto output = select(coordinates, 1, transpose ? 1 : 0);
  for (const auto &[indices, extent] :
       {std::pair{input, input_rows}, std::pair{output, output_rows}}) {
    // Check before gather because the general indexing API wraps negative indices.
    const auto ids = indices.to_vector<std::int64_t>();
    for (const auto id : ids)
      if (id < 0 || id >= extent)
        throw std::out_of_range("coo_matmul: coordinate outside declared shape");
  }
  const auto messages = gather(features, input) * expand_dims(weights, 1);
  return index_add(zeros({output_rows, features.size(1)}, features.dtype(), features.context()), 0, output, messages);
}

} // namespace datamunge::nn
