#pragma once

/// @file bindings.hpp
/// @brief Owning double-precision matrix expression handle for generated language bindings.

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace datamunge {

/// @brief Binding counterpart of Mat/SpMat with shared, immutable expression operands.
/// Transpose, multiplication and inverse actions defer native computation until eval(),
/// values() or text(). Handles keep temporary operands alive across garbage collection.
/// Sparse products use CSR; solves use the canonical dense LU/QR implementation.
/// Concurrent const evaluation is safe; no evaluation cache is mutated.
/// @par Example
/// @code
/// dm::LinalgMatrix A({2, 0, 0, 4}, 2, 2), b({6, 8}, 2, 1);
/// auto x = A.inverse_action().multiply(b);
/// std::print("{:.0f}\n", x.values()[0]); // 3
/// @endcode
/// @see Mat, SpMat, solve()
class LinalgMatrix {
 public:
  /// @brief Construct an owned dense or CSR matrix from row-major values.
  /// @param values Exactly rows * cols coefficients; sparse zero entries are omitted.
  /// @param rows Number of rows, including zero.
  /// @param cols Number of columns, including zero.
  /// @param sparse Select CSR storage, default false.
  /// @throws std::invalid_argument If dimensions overflow or do not match values.
  /// @par Example
  /// @code
  /// dm::LinalgMatrix A({1, 0}, 1, 2, true);
  /// std::print("{}\n", A.cols()); // 2
  /// @endcode
  LinalgMatrix(
      const std::vector<double>& values,
      std::size_t                rows,
      std::size_t                cols,
      bool                       sparse = false);

  /// @brief Read the deferred result's row count without evaluating.
  /// @return Number of rows.
  std::size_t rows() const;
  /// @brief Read the deferred result's column count without evaluating.
  /// @return Number of columns.
  std::size_t cols() const;
  /// @brief Inspect the deferred result's storage policy without evaluating.
  /// @return True for CSR results; inverse actions and solves return dense results.
  bool is_sparse() const;

  /// @brief Form a lazy transpose, retaining shared ownership of this operand.
  /// @return Deferred transpose handle.
  /// @par Example
  /// @code
  /// dm::LinalgMatrix A({1, 2}, 1, 2);
  /// std::print("{}\n", A.transpose().rows()); // 2
  /// @endcode
  LinalgMatrix transpose() const;

  /// @brief Form a lazy matrix product, fusing preceding inverse actions into a solve.
  /// An expression solve(A) * B * C evaluates B * C before one native solve.
  /// @param other Right operand, retained by shared ownership.
  /// @return Deferred product or solve handle.
  /// @throws std::invalid_argument If inner dimensions differ.
  /// @par Example
  /// @code
  /// dm::LinalgMatrix A({1, 2}, 1, 2), b({3, 4}, 2, 1);
  /// std::print("{:.0f}\n", A.multiply(b).values()[0]); // 11
  /// @endcode
  LinalgMatrix multiply(const LinalgMatrix& other) const;

  /// @brief Form a deferred square inverse action, without factorisation or inversion.
  /// Multiplication turns this into a solve; evaluating it alone forms an explicit inverse.
  /// @return Deferred inverse-action handle.
  /// @throws std::invalid_argument If this matrix is not square.
  /// @par Example
  /// @code
  /// dm::LinalgMatrix A({2}, 1, 1);
  /// std::print("{:.1f}\n", A.inverse_action().values()[0]); // 0.5
  /// @endcode
  LinalgMatrix inverse_action() const;

  /// @brief Evaluate a square system with LU or a tall least-squares system with QR.
  /// @param rhs Matrix of right-hand sides with rows() rows.
  /// @return Owned dense solution snapshot.
  /// @throws std::invalid_argument For incompatible/wide shapes or nonfinite solve inputs.
  /// @throws std::runtime_error For singular or rank-deficient systems.
  /// @par Example
  /// @code
  /// dm::LinalgMatrix A({1, 1, 1}, 3, 1), b({1, 2, 3}, 3, 1);
  /// std::print("{:.0f}\n", A.solve(b).values()[0]); // 2
  /// @endcode
  LinalgMatrix solve(const LinalgMatrix& rhs) const;

  /// @brief Evaluate once into an owned snapshot with no deferred operands.
  /// @return Materialised matrix handle, sharing immutable storage when already concrete.
  /// @throws std::invalid_argument For nonfinite solve inputs.
  /// @throws std::runtime_error For singular or rank-deficient systems.
  /// @par Example
  /// @code
  /// dm::LinalgMatrix A({1, 2}, 1, 2);
  /// auto snapshot = A.transpose().eval();
  /// std::print("{}\n", snapshot.rows()); // 2
  /// @endcode
  LinalgMatrix eval() const;

  /// @brief Evaluate and copy all coefficients in row-major order.
  /// @return Dense vector, including sparse zeros.
  /// @throws std::invalid_argument For nonfinite solve inputs.
  /// @throws std::runtime_error For singular or rank-deficient systems.
  /// @par Example
  /// @code
  /// dm::LinalgMatrix A({1, 2}, 1, 2);
  /// std::print("{:.0f}\n", A.values()[1]); // 2
  /// @endcode
  std::vector<double> values() const;

  /// @brief Evaluate once and produce the native capped Mat/SpMat preview.
  /// @return Shape, storage information and at most ten rows/eight columns.
  /// @throws std::invalid_argument For nonfinite solve inputs.
  /// @throws std::runtime_error For singular or rank-deficient systems.
  /// @par Example
  /// @code
  /// dm::LinalgMatrix A({1}, 1, 1);
  /// std::print("{}\n", A.text());
  /// // Mat 1 x 1
  /// //      [,0]
  /// // [0,]    1
  /// @endcode
  std::string text() const;

 private:
  /// @brief Internal immutable runtime expression, implemented using native matrix kernels.
  struct Node;
  /// @brief Own an existing immutable expression node.
  /// @param node Shared expression/storage lifetime.
  explicit LinalgMatrix(std::shared_ptr<const Node> node);
  std::shared_ptr<const Node> node_; ///< Shared operand tree; never borrows language objects.
};
} // namespace datamunge
