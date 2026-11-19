/// @file
/// @brief Validation and projection for a reserved constant feature.
#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief A tabular feature column reserved for a fixed numeric value.
class ConstantFeature {
public:
  /// @brief Define a reserved feature.
  /// @param column Zero-based feature index.
  /// @param value Required finite value.
  /// @throws std::invalid_argument Negative column index or nonfinite value.
  explicit ConstantFeature(std::int64_t column, double value = 0.0)
      : column_(column), value_(value) {
    if (column < 0 || !std::isfinite(value)) {
      throw std::invalid_argument("ConstantFeature: invalid index or value");
    }
  }

  /// @brief Return the indices of rows satisfying this feature rule.
  /// @param x Floating-point feature matrix [rows,features].
  /// @return Increasing row indices, on the host.
  /// @throws std::invalid_argument Invalid feature matrix.
  std::vector<std::int64_t> valid_rows(const Tensor& x) const {
    check_matrix(x);
    const auto values = equal(slice_axis(x, 1, column_, column_ + 1), value_).to_vector<bool>();
    std::vector<std::int64_t> rows;
    for (std::int64_t i = 0; i < x.size(0); ++i) {
      if (values[i]) rows.push_back(i);
    }
    return rows;
  }

  /// @brief Restore the reserved column without changing the input tensor.
  /// @param x Floating-point feature matrix [rows,features].
  /// @return A matrix with the reserved column fixed to the schema value.
  /// @throws std::invalid_argument Invalid feature matrix.
  Tensor project(const Tensor& x) const {
    check_matrix(x);
    const auto reserved = full_like(slice_axis(x, 1, column_, column_ + 1), value_);
    std::vector<Tensor> columns;
    if (column_ > 0) columns.push_back(slice_axis(x, 1, 0, column_));
    columns.push_back(reserved);
    const auto remaining = x.size(1) - column_ - 1;
    if (remaining > 0) columns.push_back(slice_axis(x, 1, column_ + 1, x.size(1)));
    return concat(columns, 1);
  }

private:
  /// @brief Check that the matrix has the reserved floating-point column.
  /// @param x Matrix to validate.
  /// @throws std::invalid_argument The matrix has an incompatible shape or type.
  void check_matrix(const Tensor& x) const {
    if (!x.defined() || x.dim() != 2 || x.size(1) <= column_ ||
        (x.dtype() != DType::Float32 && x.dtype() != DType::Float64)) {
      throw std::invalid_argument("ConstantFeature: expected a floating-point feature matrix");
    }
  }

  std::int64_t column_; ///< Reserved feature index.
  double value_; ///< Required value.
};

} // namespace datamunge::nn
