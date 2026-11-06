/// @file matrix.cpp
/// @brief Native storage/CSR kernels, canonical solves and matrix powers for the Mat facade.

#include <algorithm>
#include <cmath>
#include <datamunge/linalg/lu.hpp>
#include <datamunge/linalg/matrix.hpp>
#include <datamunge/linalg/qr.hpp>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>

namespace datamunge::linalg::matrix_detail {

template<class T>
void validate_csr(const SparseCSR<T>& s) {
  if (s.nrows == 0 && s.ncols == 0 && s.row_ptr.empty() && s.values.empty() && s.col_idx.empty())
    return;
  if (s.nrows == std::numeric_limits<std::size_t>::max() || s.row_ptr.size() != s.nrows + 1 ||
      s.row_ptr.front() != 0 || s.row_ptr.back() != s.values.size() ||
      s.col_idx.size() != s.values.size())
    throw std::invalid_argument("SpMat: malformed CSR storage");
  for (std::size_t row = 0; row < s.nrows; ++row) {
    if (s.row_ptr[row] > s.row_ptr[row + 1] || s.row_ptr[row + 1] > s.values.size())
      throw std::invalid_argument("SpMat: invalid CSR row offsets");
    for (std::size_t k = s.row_ptr[row]; k < s.row_ptr[row + 1]; ++k)
      if (s.col_idx[k] >= s.ncols || (k > s.row_ptr[row] && s.col_idx[k - 1] >= s.col_idx[k]))
        throw std::invalid_argument("SpMat: CSR columns must be in range and strictly increasing");
  }
}

template<class T>
T sparse_coefficient(const SparseCSR<T>& s, std::size_t row, std::size_t col) {
  const auto begin = s.col_idx.begin() + s.row_ptr[row],
             end   = s.col_idx.begin() + s.row_ptr[row + 1];
  const auto found = std::lower_bound(begin, end, col);
  return found != end && *found == col
             ? s.values[static_cast<std::size_t>(found - s.col_idx.begin())]
             : 0;
}

template<class T>
void sparse_set(SparseCSR<T>& s, std::size_t row, std::size_t col, T value) {
  auto       updated = s;
  const auto begin   = updated.col_idx.begin() + updated.row_ptr[row],
             end     = updated.col_idx.begin() + updated.row_ptr[row + 1];
  const auto found   = std::lower_bound(begin, end, col);
  const auto offset  = static_cast<std::size_t>(found - updated.col_idx.begin());
  const bool present = found != end && *found == col;
  if (present && value != 0)
    updated.values[offset] = value;
  else if (present) {
    updated.col_idx.erase(found);
    updated.values.erase(updated.values.begin() + offset);
    for (std::size_t i = row + 1; i < updated.row_ptr.size(); ++i)
      --updated.row_ptr[i];
  } else if (value != 0) {
    updated.col_idx.insert(found, col);
    updated.values.insert(updated.values.begin() + offset, value);
    for (std::size_t i = row + 1; i < updated.row_ptr.size(); ++i)
      ++updated.row_ptr[i];
  }
  s = std::move(updated);
}

template<class T>
DenseMatrix<T> dense_zeros(std::size_t rows, std::size_t cols) {
  if (cols && rows > std::numeric_limits<std::size_t>::max() / cols)
    throw std::length_error("Mat: matrix dimensions overflow");
  return DenseMatrix<T>(rows, cols);
}

template<class T>
DenseMatrix<T> dense_storage(const SparseFacade<T>& a) {
  auto        result = dense_zeros<T>(a.rows(), a.cols());
  const auto& s      = a.storage();
  for (std::size_t row = 0; row < a.rows(); ++row)
    for (std::size_t k = s.row_ptr[row]; k < s.row_ptr[row + 1]; ++k)
      result(row, s.col_idx[k]) = s.values[k];
  return result;
}

template<class T>
SparseCSR<T> sparse_storage(const DenseFacade<T>& a) {
  SparseCSR<T> result;
  result.nrows = a.rows();
  result.ncols = a.cols();
  result.row_ptr.push_back(0);
  for (std::size_t row = 0; row < a.rows(); ++row) {
    for (std::size_t col = 0; col < a.cols(); ++col)
      if (a(row, col) != 0) {
        result.col_idx.push_back(col);
        result.values.push_back(a(row, col));
      }
    result.row_ptr.push_back(result.nnz());
  }
  return result;
}

template<class T>
DenseFacade<T> transposed(const DenseFacade<T>& a) {
  return DenseFacade<T>(a.storage().transpose());
}

template<class T>
SparseFacade<T> transposed(const SparseFacade<T>& a) {
  SparseCSR<T> result;
  result.nrows = a.cols();
  result.ncols = a.rows();
  if (result.nrows == std::numeric_limits<std::size_t>::max())
    throw std::length_error("SpMat transpose: too many rows");
  result.row_ptr.assign(result.nrows + 1, 0);
  result.col_idx.resize(a.nnz());
  result.values.resize(a.nnz());
  const auto& s = a.storage();
  for (auto col : s.col_idx)
    ++result.row_ptr[col + 1];
  for (std::size_t row = 0; row < result.nrows; ++row)
    result.row_ptr[row + 1] += result.row_ptr[row];
  auto next = result.row_ptr;
  for (std::size_t row = 0; row < a.rows(); ++row)
    for (std::size_t k = s.row_ptr[row]; k < s.row_ptr[row + 1]; ++k) {
      const auto position      = next[s.col_idx[k]]++;
      result.col_idx[position] = row;
      result.values[position]  = s.values[k];
    }
  return SparseFacade<T>(std::move(result));
}

template<class T>
DenseFacade<T> multiply(const DenseFacade<T>& a, bool ta, const DenseFacade<T>& b, bool tb) {
  const auto rows = ta ? a.cols() : a.rows(), inner = ta ? a.rows() : a.cols(),
             cols = tb ? b.rows() : b.cols();
  auto result     = dense_zeros<T>(rows, cols);
  for (std::size_t i = 0; i < rows; ++i)
    for (std::size_t k = 0; k < inner; ++k) {
      const T value = ta ? a.storage()(k, i) : a.storage()(i, k);
      for (std::size_t j = 0; j < cols; ++j)
        result(i, j) += value * (tb ? b.storage()(j, k) : b.storage()(k, j));
    }
  return DenseFacade<T>(std::move(result));
}

template<class T>
DenseFacade<T> multiply(const SparseFacade<T>& a, bool ta, const DenseFacade<T>& b, bool tb) {
  auto        result = dense_zeros<T>(ta ? a.cols() : a.rows(), tb ? b.rows() : b.cols());
  const auto& s      = a.storage();
  for (std::size_t row = 0; row < a.rows(); ++row)
    for (std::size_t k = s.row_ptr[row]; k < s.row_ptr[row + 1]; ++k) {
      const auto i = ta ? s.col_idx[k] : row, inner = ta ? row : s.col_idx[k];
      for (std::size_t j = 0; j < result.cols(); ++j)
        result(i, j) += s.values[k] * (tb ? b.storage()(j, inner) : b.storage()(inner, j));
    }
  return DenseFacade<T>(std::move(result));
}

template<class T>
DenseFacade<T> multiply(const DenseFacade<T>& a, bool ta, const SparseFacade<T>& b, bool tb) {
  auto        result = dense_zeros<T>(ta ? a.cols() : a.rows(), tb ? b.rows() : b.cols());
  const auto& s      = b.storage();
  for (std::size_t row = 0; row < b.rows(); ++row)
    for (std::size_t k = s.row_ptr[row]; k < s.row_ptr[row + 1]; ++k) {
      const auto inner = tb ? s.col_idx[k] : row, j = tb ? row : s.col_idx[k];
      for (std::size_t i = 0; i < result.rows(); ++i)
        result(i, j) += (ta ? a.storage()(inner, i) : a.storage()(i, inner)) * s.values[k];
    }
  return DenseFacade<T>(std::move(result));
}

template<class T>
SparseFacade<T> multiply(const SparseFacade<T>& a, bool ta, const SparseFacade<T>& b, bool tb) {
  // Only CSR transposes need materialising: dense kernels consume a logical transpose directly.
  std::optional<SparseFacade<T>> at, bt;
  if (ta)
    at = transposed(a);
  if (tb)
    bt = transposed(b);
  const auto&  left  = ta ? at->storage() : a.storage();
  const auto&  right = tb ? bt->storage() : b.storage();
  SparseCSR<T> result;
  result.nrows = left.rows();
  result.ncols = right.cols();
  result.row_ptr.push_back(0);
  for (std::size_t row = 0; row < left.rows(); ++row) {
    std::map<std::size_t, T> accumulator;
    for (std::size_t k = left.row_ptr[row]; k < left.row_ptr[row + 1]; ++k) {
      const auto inner = left.col_idx[k];
      for (std::size_t j = right.row_ptr[inner]; j < right.row_ptr[inner + 1]; ++j)
        accumulator[right.col_idx[j]] += left.values[k] * right.values[j];
    }
    for (const auto& [col, value] : accumulator)
      if (value != 0) {
        result.col_idx.push_back(col);
        result.values.push_back(value);
      }
    result.row_ptr.push_back(result.nnz());
  }
  return SparseFacade<T>(std::move(result));
}

template<class T>
DenseFacade<T> identity(std::size_t n) {
  return DenseFacade<T>(DenseMatrix<T>::identity(n));
}

template<class T>
DenseFacade<T> solve_system(const DenseFacade<T>& a, const DenseFacade<T>& b) {
  if (a.rows() < a.cols() || a.rows() != b.rows())
    throw std::invalid_argument("solve(A,B): incompatible dimensions or wide coefficient matrix");
  for (T value : a.storage().data())
    if (!std::isfinite(value))
      throw std::invalid_argument("solve: coefficient matrix must be finite");
  for (T value : b.storage().data())
    if (!std::isfinite(value))
      throw std::invalid_argument("solve: right-hand sides must be finite");
  if (a.rows() == a.cols()) {
    const auto factor = lu(a.storage());
    if (factor.singular)
      throw std::runtime_error("solve: coefficient matrix is singular");
    return DenseFacade<T>(factor.solve(b.storage()));
  }
  const auto factor = qr(a.storage());
  if (!factor.is_full_rank())
    throw std::runtime_error("solve: coefficient matrix is rank-deficient; use SVD");
  return DenseFacade<T>(factor.solve(b.storage()));
}

/// @brief Explicit supported-scalar instantiation of the validate_csr backend kernel.
template void validate_csr(const SparseCSR<float>&);
/// @brief Explicit supported-scalar instantiation of the sparse_coefficient backend kernel.
template float sparse_coefficient(const SparseCSR<float>&, std::size_t, std::size_t);
/// @brief Explicit supported-scalar instantiation of the sparse_set backend kernel.
template void sparse_set(SparseCSR<float>&, std::size_t, std::size_t, float);
/// @brief Explicit supported-scalar instantiation of the dense_zeros backend kernel.
template DenseMatrix<float> dense_zeros(std::size_t, std::size_t);
/// @brief Explicit supported-scalar instantiation of the dense_storage backend kernel.
template DenseMatrix<float> dense_storage(const SparseFacade<float>&);
/// @brief Explicit supported-scalar instantiation of the sparse_storage backend kernel.
template SparseCSR<float> sparse_storage(const DenseFacade<float>&);
/// @brief Explicit supported-scalar instantiation of the transposed backend kernel.
template DenseFacade<float> transposed(const DenseFacade<float>&);
/// @brief Explicit supported-scalar instantiation of the transposed backend kernel.
template SparseFacade<float> transposed(const SparseFacade<float>&);
/// @brief Explicit supported-scalar instantiation of the identity backend kernel.
template DenseFacade<float> identity(std::size_t);
/// @brief Explicit supported-scalar instantiation of the solve_system backend kernel.
template DenseFacade<float> solve_system(const DenseFacade<float>&, const DenseFacade<float>&);
/// @brief Explicit supported-scalar instantiation of the native matrix product kernel.
template DenseFacade<float>
multiply(const DenseFacade<float>&, bool, const DenseFacade<float>&, bool);
/// @brief Explicit supported-scalar instantiation of the native matrix product kernel.
template DenseFacade<float>
multiply(const SparseFacade<float>&, bool, const DenseFacade<float>&, bool);
/// @brief Explicit supported-scalar instantiation of the native matrix product kernel.
template DenseFacade<float>
multiply(const DenseFacade<float>&, bool, const SparseFacade<float>&, bool);
/// @brief Explicit supported-scalar instantiation of the native matrix product kernel.
template SparseFacade<float>
multiply(const SparseFacade<float>&, bool, const SparseFacade<float>&, bool);
/// @brief Explicit supported-scalar instantiation of the validate_csr backend kernel.
template void validate_csr(const SparseCSR<double>&);
/// @brief Explicit supported-scalar instantiation of the sparse_coefficient backend kernel.
template double sparse_coefficient(const SparseCSR<double>&, std::size_t, std::size_t);
/// @brief Explicit supported-scalar instantiation of the sparse_set backend kernel.
template void sparse_set(SparseCSR<double>&, std::size_t, std::size_t, double);
/// @brief Explicit supported-scalar instantiation of the dense_zeros backend kernel.
template DenseMatrix<double> dense_zeros(std::size_t, std::size_t);
/// @brief Explicit supported-scalar instantiation of the dense_storage backend kernel.
template DenseMatrix<double> dense_storage(const SparseFacade<double>&);
/// @brief Explicit supported-scalar instantiation of the sparse_storage backend kernel.
template SparseCSR<double> sparse_storage(const DenseFacade<double>&);
/// @brief Explicit supported-scalar instantiation of the transposed backend kernel.
template DenseFacade<double> transposed(const DenseFacade<double>&);
/// @brief Explicit supported-scalar instantiation of the transposed backend kernel.
template SparseFacade<double> transposed(const SparseFacade<double>&);
/// @brief Explicit supported-scalar instantiation of the identity backend kernel.
template DenseFacade<double> identity(std::size_t);
/// @brief Explicit supported-scalar instantiation of the solve_system backend kernel.
template DenseFacade<double> solve_system(const DenseFacade<double>&, const DenseFacade<double>&);
/// @brief Explicit supported-scalar instantiation of the native matrix product kernel.
template DenseFacade<double>
multiply(const DenseFacade<double>&, bool, const DenseFacade<double>&, bool);
/// @brief Explicit supported-scalar instantiation of the native matrix product kernel.
template DenseFacade<double>
multiply(const SparseFacade<double>&, bool, const DenseFacade<double>&, bool);
/// @brief Explicit supported-scalar instantiation of the native matrix product kernel.
template DenseFacade<double>
multiply(const DenseFacade<double>&, bool, const SparseFacade<double>&, bool);
/// @brief Explicit supported-scalar instantiation of the native matrix product kernel.
template SparseFacade<double>
multiply(const SparseFacade<double>&, bool, const SparseFacade<double>&, bool);
/// @brief Explicit supported-scalar instantiation of the validate_csr backend kernel.
template void validate_csr(const SparseCSR<long double>&);
/// @brief Explicit supported-scalar instantiation of the sparse_coefficient backend kernel.
template long double sparse_coefficient(const SparseCSR<long double>&, std::size_t, std::size_t);
/// @brief Explicit supported-scalar instantiation of the sparse_set backend kernel.
template void sparse_set(SparseCSR<long double>&, std::size_t, std::size_t, long double);
/// @brief Explicit supported-scalar instantiation of the dense_zeros backend kernel.
template DenseMatrix<long double> dense_zeros(std::size_t, std::size_t);
/// @brief Explicit supported-scalar instantiation of the dense_storage backend kernel.
template DenseMatrix<long double> dense_storage(const SparseFacade<long double>&);
/// @brief Explicit supported-scalar instantiation of the sparse_storage backend kernel.
template SparseCSR<long double> sparse_storage(const DenseFacade<long double>&);
/// @brief Explicit supported-scalar instantiation of the transposed backend kernel.
template DenseFacade<long double> transposed(const DenseFacade<long double>&);
/// @brief Explicit supported-scalar instantiation of the transposed backend kernel.
template SparseFacade<long double> transposed(const SparseFacade<long double>&);
/// @brief Explicit supported-scalar instantiation of the identity backend kernel.
template DenseFacade<long double> identity(std::size_t);
/// @brief Explicit supported-scalar instantiation of the solve_system backend kernel.
template DenseFacade<long double> solve_system(
    const DenseFacade<long double>&,
    const DenseFacade<long double>&);
/// @brief Explicit supported-scalar instantiation of the native matrix product kernel.
template DenseFacade<long double>
multiply(const DenseFacade<long double>&, bool, const DenseFacade<long double>&, bool);
/// @brief Explicit supported-scalar instantiation of the native matrix product kernel.
template DenseFacade<long double>
multiply(const SparseFacade<long double>&, bool, const DenseFacade<long double>&, bool);
/// @brief Explicit supported-scalar instantiation of the native matrix product kernel.
template DenseFacade<long double>
multiply(const DenseFacade<long double>&, bool, const SparseFacade<long double>&, bool);
/// @brief Explicit supported-scalar instantiation of the native matrix product kernel.
template SparseFacade<long double>
multiply(const SparseFacade<long double>&, bool, const SparseFacade<long double>&, bool);

} // namespace datamunge::linalg::matrix_detail

namespace datamunge {

template<bool S, class T>
BasicMat<S, T>::BasicMat(std::size_t rows, std::size_t cols) {
  if constexpr (S) {
    if (rows == std::numeric_limits<std::size_t>::max())
      throw std::length_error("SpMat: too many rows");
    storage_.nrows = rows;
    storage_.ncols = cols;
    storage_.row_ptr.assign(rows + 1, 0);
  } else
    storage_ = linalg::matrix_detail::dense_zeros<T>(rows, cols);
}

template<bool S, class T>
BasicMat<S, T>::BasicMat(std::initializer_list<std::initializer_list<T>> rows)
    : BasicMat(rows.size(), rows.size() ? rows.begin()->size() : 0) {
  std::size_t i = 0;
  for (const auto& row : rows) {
    if (row.size() != cols())
      throw std::invalid_argument("Mat: initializer rows must have equal length");
    std::size_t j = 0;
    for (const T value : row) {
      if constexpr (S) {
        if (value != 0) {
          storage_.col_idx.push_back(j);
          storage_.values.push_back(value);
        }
      } else
        storage_(i, j) = value;
      ++j;
    }
    if constexpr (S)
      storage_.row_ptr[i + 1] = storage_.nnz();
    ++i;
  }
}

template<bool S, class T>
std::size_t BasicMat<S, T>::nnz() const {
  if constexpr (S)
    return storage_.nnz();
  else
    return static_cast<std::size_t>(std::count_if(
        storage_.data().begin(), storage_.data().end(), [](T value) { return value != 0; }));
}

template<bool S, class T>
T BasicMat<S, T>::operator()(std::size_t row, std::size_t col) const {
  if (row >= rows() || col >= cols())
    throw std::out_of_range("Mat: coefficient index out of range");
  if constexpr (S)
    return linalg::matrix_detail::sparse_coefficient(storage_, row, col);
  else
    return storage_(row, col);
}

template<bool S, class T>
void BasicMat<S, T>::set(std::size_t row, std::size_t col, T value) {
  if (row >= rows() || col >= cols())
    throw std::out_of_range("Mat::set: coefficient index out of range");
  if constexpr (S)
    linalg::matrix_detail::sparse_set(storage_, row, col, value);
  else
    storage_(row, col) = value;
}

template class BasicMat<false, float>;
template class BasicMat<true, float>;
template class BasicMat<false, double>;
template class BasicMat<true, double>;
template class BasicMat<false, long double>;
template class BasicMat<true, long double>;

} // namespace datamunge

namespace datamunge::linalg::matrix_detail {

/// @brief Validate square shape and finite coefficients before a matrix function.
/// @tparam T Floating-point scalar.
/// @param matrix Coefficient matrix.
/// @throws std::invalid_argument For nonsquare shape or nonfinite coefficients.
template<class T>
void validate_power_matrix(const DenseFacade<T>& matrix) {
  if (matrix.rows() != matrix.cols())
    throw std::invalid_argument("Mat power: matrix must be square");
  for (T value : matrix.storage().data())
    if (!std::isfinite(value))
      throw std::invalid_argument("Mat power: coefficients must be finite");
}

/// @brief Check finite matrix-function output without silently returning overflow.
/// @tparam T Floating-point scalar.
/// @param matrix Computed matrix.
/// @throws std::overflow_error For nonfinite coefficients.
template<class T>
void check_power_result(const DenseFacade<T>& matrix) {
  for (T value : matrix.storage().data())
    if (!std::isfinite(value))
      throw std::overflow_error("Mat power: result overflowed");
}

/// @brief Compute the infinity norm of a matrix minus a scalar diagonal.
/// @tparam T Floating-point scalar.
/// @param matrix Square matrix.
/// @param diagonal Scalar multiple of identity to subtract, default zero.
/// @return Maximum absolute row sum, accumulated in long double.
template<class T>
long double power_norm(const DenseFacade<T>& matrix, T diagonal = T{}) {
  long double result = 0;
  for (std::size_t i = 0; i < matrix.rows(); ++i) {
    long double row = 0;
    for (std::size_t j = 0; j < matrix.cols(); ++j)
      row += std::abs(static_cast<long double>(matrix(i, j)) - (i == j ? diagonal : T{}));
    result = std::max(result, row);
  }
  return result;
}

/// @brief Multiply through the canonical dense kernel and reject overflow.
/// @tparam T Floating-point scalar.
/// @param a Left factor.
/// @param b Right factor.
/// @return Checked dense product.
/// @throws std::overflow_error For nonfinite output.
template<class T>
DenseFacade<T> power_product(const DenseFacade<T>& a, const DenseFacade<T>& b) {
  auto result = multiply(a, false, b, false);
  check_power_result(result);
  return result;
}

template<class T>
DenseFacade<T> integer_power(DenseFacade<T> matrix, std::uintmax_t magnitude, bool negative) {
  validate_power_matrix(matrix);
  auto result = identity<T>(matrix.rows());
  if (negative && magnitude != 0) {
    matrix = solve_system(matrix, result);
    check_power_result(matrix);
  }
  while (magnitude != 0) {
    if (magnitude & 1)
      result = power_product(result, matrix);
    magnitude >>= 1;
    if (magnitude != 0)
      matrix = power_product(matrix, matrix);
  }
  return result;
}

/// @brief Approximate a principal square root with the coupled Denman-Beavers iteration.
/// Both the square-root residual and Y Z - I are checked; at most 100 LU-based
/// iterations are attempted. No eigenvalue decomposition or complex arithmetic is used.
/// @tparam T Floating-point scalar.
/// @param matrix Finite square matrix with no nonpositive real eigenvalues.
/// @return Approximate principal square root.
/// @throws std::runtime_error For singularity or failed convergence.
/// @throws std::overflow_error For nonfinite iterates.
/// @par References
/// - N. J. Higham (1997). "Stable iterations for the matrix square root".
///   Numerical Algorithms 15, 227-242.
///   https://research.manchester.ac.uk/en/publications/stable-iterations-for-the-matrix-square-root/
template<class T>
DenseFacade<T> power_square_root(const DenseFacade<T>& matrix) {
  auto              y         = matrix;
  auto              z         = identity<T>(matrix.rows());
  const auto        unit      = z;
  const long double tolerance = 32 * std::numeric_limits<T>::epsilon();
  const long double scale     = std::max(1.0L, power_norm(matrix));
  for (unsigned iteration = 0; iteration < 100; ++iteration) {
    const auto inv_y = solve_system(y, unit);
    const auto inv_z = solve_system(z, unit);
    for (std::size_t i = 0; i < y.rows(); ++i)
      for (std::size_t j = 0; j < y.cols(); ++j) {
        y(i, j) = y(i, j) / 2 + inv_z(i, j) / 2;
        z(i, j) = z(i, j) / 2 + inv_y(i, j) / 2;
      }
    check_power_result(y);
    check_power_result(z);
    auto residual = power_product(y, y);
    for (std::size_t i = 0; i < y.rows(); ++i)
      for (std::size_t j = 0; j < y.cols(); ++j)
        residual(i, j) -= matrix(i, j);
    if (power_norm(residual) <= tolerance * scale &&
        power_norm(power_product(y, z), T{1}) <= tolerance)
      return y;
  }
  throw std::runtime_error("Mat power: principal square-root iteration did not converge");
}

template<class T>
DenseFacade<T> real_power(const DenseFacade<T>& matrix, long double exponent) {
  validate_power_matrix(matrix);
  if (!std::isfinite(exponent))
    throw std::invalid_argument("Mat power: exponent must be finite");
  const long double whole         = std::trunc(exponent);
  const long double fraction      = exponent - whole;
  const long double integer_limit = std::ldexp(1.0L, std::numeric_limits<std::uintmax_t>::digits);
  if (std::abs(whole) >= integer_limit)
    throw std::overflow_error("Mat power: integral exponent exceeds uintmax_t range");
  if (fraction == 0)
    return integer_power(matrix, static_cast<std::uintmax_t>(std::abs(whole)), whole < 0);

  bool diagonal = true;
  T    maximum  = 0;
  for (std::size_t i = 0; i < matrix.rows(); ++i)
    for (std::size_t j = 0; j < matrix.cols(); ++j) {
      diagonal = diagonal && (i == j || matrix(i, j) == 0);
      maximum  = std::max(maximum, std::abs(matrix(i, j)));
    }
  if (diagonal) {
    auto result = identity<T>(matrix.rows());
    for (std::size_t i = 0; i < matrix.rows(); ++i) {
      if (matrix(i, i) < 0 || (matrix(i, i) == 0 && exponent < 0))
        throw std::domain_error(
            "Mat power: fractional diagonal power is outside the real principal domain");
      result(i, i) = static_cast<T>(std::pow(static_cast<long double>(matrix(i, i)), exponent));
    }
    check_power_result(result);
    return result;
  }

  auto reduced = matrix;
  for (std::size_t i = 0; i < reduced.rows(); ++i)
    for (std::size_t j = 0; j < reduced.cols(); ++j)
      reduced(i, j) /= maximum;
  unsigned roots = 0;
  while (power_norm(reduced, T{1}) > 0.5L) {
    if (roots == 32)
      throw std::runtime_error(
          "Mat power: could not scale matrix into the Taylor convergence region");
    reduced = power_square_root(reduced);
    ++roots;
  }
  auto h = reduced;
  for (std::size_t i = 0; i < h.rows(); ++i)
    h(i, i) -= T{1};
  const long double rho       = power_norm(h);
  auto              sum       = identity<T>(matrix.rows());
  auto              term      = sum;
  bool              converged = rho == 0;
  for (unsigned k = 1; k <= 256 && !converged; ++k) {
    term                = power_product(term, h);
    const T coefficient = static_cast<T>((fraction - (k - 1)) / k);
    for (std::size_t i = 0; i < sum.rows(); ++i)
      for (std::size_t j = 0; j < sum.cols(); ++j) {
        term(i, j) *= coefficient;
        sum(i, j) += term(i, j);
      }
    check_power_result(sum);
    // For -1 < fraction < 1 and ||H|| <= 1/2, subsequent term ratios are <= rho.
    converged = power_norm(term) * rho / (1 - rho) <=
                std::numeric_limits<T>::epsilon() * std::max(1.0L, power_norm(sum));
  }
  if (!converged)
    throw std::runtime_error("Mat power: binomial Taylor series did not converge");
  for (unsigned i = 0; i < roots; ++i)
    sum = power_product(sum, sum);
  const long double scalar = std::pow(static_cast<long double>(maximum), fraction);
  for (std::size_t i = 0; i < sum.rows(); ++i)
    for (std::size_t j = 0; j < sum.cols(); ++j)
      sum(i, j) = static_cast<T>(sum(i, j) * scalar);
  check_power_result(sum);
  if (whole != 0)
    sum = power_product(
        integer_power(matrix, static_cast<std::uintmax_t>(std::abs(whole)), whole < 0), sum);
  return sum;
}

/// @brief Instantiate checked binary powers for float matrices.
template DenseFacade<float> integer_power(DenseFacade<float>, std::uintmax_t, bool);
/// @brief Instantiate checked binary powers for double matrices.
template DenseFacade<double> integer_power(DenseFacade<double>, std::uintmax_t, bool);
/// @brief Instantiate checked binary powers for long double matrices.
template DenseFacade<long double> integer_power(DenseFacade<long double>, std::uintmax_t, bool);
/// @brief Instantiate real principal powers for float matrices.
template DenseFacade<float> real_power(const DenseFacade<float>&, long double);
/// @brief Instantiate real principal powers for double matrices.
template DenseFacade<double> real_power(const DenseFacade<double>&, long double);
/// @brief Instantiate real principal powers for long double matrices.
template DenseFacade<long double> real_power(const DenseFacade<long double>&, long double);

} // namespace datamunge::linalg::matrix_detail
