#pragma once

/// @file matrix.hpp
/// @brief std::format/std::print support for Mat, SpMat and their deferred expressions.

#include <datamunge/format/linalg.hpp>
#include <datamunge/format/tex.hpp>
#include <datamunge/linalg/matrix.hpp>

namespace datamunge::format_detail {

/// @brief Format a concrete facade matrix using the existing capped, aligned grid renderer.
/// @tparam S Sparse selector.
/// @tparam T Floating-point scalar.
/// @param matrix Owned matrix to preview without copying its storage.
/// @return Header and up to ten rows/eight columns; sparse zeros appear as dots.
template<bool S, class T>
std::string matrix_text(const BasicMat<S, T>& matrix) {
  namespace grid = grid_fmt;
  auto title     = std::format("{} {} x {}", S ? "SpMat" : "Mat", matrix.rows(), matrix.cols());
  if constexpr (S)
    title += std::format(", nnz={}", matrix.nnz());
  return grid::with_body(
      title, grid::grid(matrix.rows(), matrix.cols(), [&](std::size_t i, std::size_t j) {
        const auto value = matrix(i, j);
        return S && value == 0 ? std::string(".") : grid::cell_text(value);
      }));
}

} // namespace datamunge::format_detail

/// @brief Format a writable sparse coefficient using its underlying scalar's format rules.
/// @tparam T Floating-point scalar.
/// @par Example
/// @code
/// dm::SpMat<> A(1, 1);
/// A(0, 0) = 2.5;
/// std::print("{:.1f}\n", A(0, 0)); // 2.5
/// @endcode
template<class T>
struct std::formatter<datamunge::linalg::matrix_detail::SparseCoefficientReference<T>, char>
    : std::formatter<T, char> {
  /// @brief Read and format the coefficient without inserting a sparse entry.
  /// @tparam Context Standard output-format context.
  /// @param value Coordinate proxy whose matrix must remain alive.
  /// @param context Destination format context.
  /// @return Iterator past the formatted coefficient.
  /// @throws std::out_of_range If the matrix no longer contains this coordinate.
  template<class Context>
  auto format(
      const datamunge::linalg::matrix_detail::SparseCoefficientReference<T>& value,
      Context&                                                               context) const {
    return std::formatter<T, char>::format(static_cast<T>(value), context);
  }
};

/// @brief Format native matrix expressions, evaluating a deferred computation exactly once per
/// call. Formatting owned matrices reads their capped preview directly; expressions evaluate the
/// complete computation once before previewing it. Accepts the existing string-style width,
/// alignment and precision specifications; coefficient text uses seven significant digits.
/// @tparam E Mat, SpMat or one of the facade's expression node types.
/// @par Example
/// @code
/// dm::Mat<> A = {{1, 2}}, B = {{3}, {4}};
/// std::print("{}\n", A * B);
/// // Mat 1 x 1
/// //      [,0]
/// // [0,]   11
/// @endcode
template<class E>
  requires datamunge::linalg::matrix_detail::is_expression<E>
struct std::formatter<E, char> : datamunge::format_detail::text_formatter<E> {
  /// @brief Return the owned matrix's preview or evaluate a deferred expression once.
  /// @param expression Matrix/expression whose borrowed matrices must still exist.
  /// @return Shape header and capped, column-aligned values.
  /// @throws std::invalid_argument For invalid current operand shapes or nonfinite solve inputs.
  /// @throws std::runtime_error For singular or rank-deficient solve expressions.
  static std::string text(const E& expression) {
    if constexpr (datamunge::linalg::matrix_detail::is_matrix<E>)
      return datamunge::format_detail::matrix_text(expression);
    else
      return datamunge::format_detail::matrix_text(expression.eval());
  }
};
