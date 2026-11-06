#pragma once

/// @file matrix.hpp
/// @brief Mat/SpMat facade and lifetime-aware expression templates for transpose, product and
/// solve.

#include <datamunge/linalg/csr.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/tex.hpp>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <initializer_list>
#include <limits>
#include <locale>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace datamunge {

template<bool Sparse, class T = double>
class BasicMat;

namespace linalg {
/// @brief Dense storage policy for la::Mat<DMat, T>.
/// @par Example
/// @code
/// dm::la::Mat<dm::Dense, float> A = {{1, 2}};
/// std::print("{} {}\n", A.rows(), A.cols());  // 1 2
/// @endcode
struct DMat {};
/// @brief CSR storage policy for la::Mat<SpMat, T>.
/// @par Example
/// @code
/// dm::la::Mat<dm::Sparse, double> A = {{1, 0}, {0, 2}};
/// std::print("{}\n", A.nnz());  // 2
/// @endcode
struct SpMat {};
/// @brief Check that a matrix storage policy is one of the supported native policies.
/// @tparam Storage DMat or SpMat.
template<class Storage>
struct MatrixStoragePolicy {
  static_assert(
      std::is_same_v<Storage, DMat> || std::is_same_v<Storage, SpMat>,
      "Mat storage must be DMat or SpMat");
  static constexpr bool sparse = std::is_same_v<Storage, SpMat>; ///< CSR selection flag.
};
} // namespace linalg

/// @brief Dense storage policy for Mat; identical to linalg::DMat.
/// @par Example
/// @code
/// dm::Mat<dm::Dense, float> A = {{1, 2}};
/// std::print("{} {}\n", A.rows(), A.cols()); // 1 2
/// @endcode
/// @see Mat, Sparse
using Dense = linalg::DMat;

/// @brief Compressed sparse row storage policy for Mat; identical to linalg::SpMat.
/// @par Example
/// @code
/// dm::Mat<dm::Sparse> A = {{1, 0}, {0, 2}};
/// std::print("{}\n", A.nnz()); // 2
/// @endcode
/// @see Mat, Dense, SpMat
using Sparse = linalg::SpMat;

/// @brief Typed matrix facade with lazy transpose/product/solve expressions.
/// Defaults to dense storage and double scalars. Float, double and long double
/// preserve their type through expressions and solves; the sparse storage policy
/// keeps products in CSR until a solve's dense boundary. This is the same facade
/// as la::Mat<Storage, T>, with owned matrix storage and lifetime-aware expressions.
/// The storage policy comes first; Mat<Sparse> uses double-precision CSR storage.
/// @tparam Storage Dense (default) or Sparse storage policy.
/// @tparam T Floating-point scalar, default double.
/// @par Example
/// @code
/// dm::Mat<> X = {{1, 0}, {1, 1}, {1, 2}};
/// dm::Mat<dm::Dense, double> y = {{1}, {3}, {5}};
/// dm::Mat<> beta = dm::solve(X.t() * X) * X.t() * y;
/// std::print("{:.0f} {:.0f}\n", beta(0, 0), beta(1, 0));  // 1 2
/// @endcode
/// @see BasicMat, SpMat, solve()
template<class Storage = Dense, class T = double>
using Mat = BasicMat<linalg::MatrixStoragePolicy<Storage>::sparse, T>;

/// @brief Owned CSR matrix; zero initializer entries consume no sparse storage.
/// Products of sparse expressions remain sparse; inverse actions produce dense solutions.
/// @tparam T Floating-point scalar, default double.
/// @par Example
/// @code
/// dm::SpMat<> A = {{2, 0}, {0, 4}};
/// dm::SpMat<> b = {{6}, {8}};
/// dm::Mat<> x = dm::solve(A) * b;
/// std::print("{} {:.0f} {:.0f}\n", A.nnz(), x(0, 0), x(1, 0));  // 2 3 2
/// @endcode
/// @see BasicMat, Mat, solve()
template<class T = double>
using SpMat = BasicMat<true, T>;

/// @brief Lifetime-aware matrix expression nodes and dispatch to canonical storage/solve kernels.
namespace linalg::matrix_detail {

/// @brief Writable CSR coefficient that routes assignment through checked sparse set().
/// Borrows its matrix, which must outlive the proxy. Reading does not insert structural
/// zeros. Assignment inserts, replaces or erases an entry while preserving CSR ordering;
/// proxies retain coordinates across other insertions, but not across matrix destruction.
/// @tparam T Floating-point scalar.
/// @par Example
/// @code
/// dm::SpMat<> A(2, 3);
/// A(0, 2) = 4;
/// A(1, 1) = A(0, 2);
/// std::print("{:.0f} {}\n", A(1, 1), A.nnz()); // 4 2
/// @endcode
template<class T>
class SparseCoefficientReference {
  BasicMat<true, T>* matrix_; ///< Borrowed matrix; no pointers into reallocatable CSR arrays.
  std::size_t        row_;    ///< Zero-based row.
  std::size_t        col_;    ///< Zero-based column.

 public:
  /// @brief Borrow one coordinate; reads and writes check its current bounds.
  /// @param matrix Matrix that must outlive this proxy.
  /// @param row Zero-based row.
  /// @param col Zero-based column.
  /// @par Example
  /// See SparseCoefficientReference.
  SparseCoefficientReference(BasicMat<true, T>& matrix, std::size_t row, std::size_t col)
      : matrix_(&matrix), row_(row), col_(col) {}
  /// @brief Copy a coordinate reference without reading or changing a coefficient.
  /// @param other Proxy to copy; both proxies borrow the same matrix.
  /// @par Example
  /// See SparseCoefficientReference.
  SparseCoefficientReference(const SparseCoefficientReference& other) = default;
  /// @brief Read the current coefficient without inserting a stored zero.
  /// @return Floating-point value, zero for a missing CSR entry.
  /// @throws std::out_of_range If the matrix no longer contains this coordinate.
  /// @par Example
  /// See SparseCoefficientReference.
  operator T() const { return std::as_const(*matrix_)(row_, col_); }
  /// @brief Assign a coefficient, erasing its CSR entry when value is zero.
  /// @param value New coefficient, including nonfinite values.
  /// @return This proxy, still referring to the same coordinate.
  /// @throws std::out_of_range If the matrix no longer contains this coordinate.
  /// @par Example
  /// See SparseCoefficientReference.
  SparseCoefficientReference& operator=(T value) {
    matrix_->set(row_, col_, value);
    return *this;
  }
  /// @brief Copy another coefficient's value without rebinding this proxy.
  /// @param other Source proxy, read before changing the destination.
  /// @return This proxy.
  /// @throws std::out_of_range If either coordinate is outside its matrix.
  /// @par Example
  /// See SparseCoefficientReference.
  SparseCoefficientReference& operator=(const SparseCoefficientReference& other) {
    return *this = static_cast<T>(other);
  }
};

/// @brief ADL marker limiting matrix expression operators to this facade.
struct ExpressionTag {};

/// @brief Whether E belongs to the facade's expression tree.
/// @tparam E Candidate type, possibly cv/ref qualified.
template<class E>
inline constexpr bool is_expression = std::is_base_of_v<ExpressionTag, std::decay_t<E>>;

/// @brief Whether E is an owned matrix leaf rather than an expression node.
/// @tparam E Unqualified candidate type.
template<class E>
inline constexpr bool is_matrix = false;
/// @brief Matrix leaves are the two BasicMat specialisations.
/// @tparam S Sparse-storage selector.
/// @tparam T Floating-point scalar.
template<bool S, class T>
inline constexpr bool is_matrix<BasicMat<S, T>> = true;

/// @brief Borrow lvalue leaves; own rvalue leaves and all expression nodes.
/// @tparam E Forwarding-reference argument type.
template<class E>
using Capture = std::conditional_t<
    is_matrix<std::decay_t<E>> && std::is_lvalue_reference_v<E>,
    std::reference_wrapper<const std::decay_t<E>>,
    std::decay_t<E>>;

/// @brief Capture an operand without borrowing temporary expression nodes.
/// @tparam E Operand type.
/// @param e Matrix or expression.
/// @return Reference wrapper for an lvalue matrix, otherwise an owned value.
template<class E>
Capture<E> capture(E&& e) {
  if constexpr (is_matrix<std::decay_t<E>> && std::is_lvalue_reference_v<E>)
    return std::cref(e);
  else
    return std::forward<E>(e);
}

/// @brief Read an owned operand.
/// @tparam E Stored operand type.
/// @param e Stored value.
/// @return Borrowed operand.
template<class E>
const E& operand(const E& e) {
  return e;
}
/// @brief Read a borrowed matrix operand.
/// @tparam E Matrix type.
/// @param e Reference wrapper.
/// @return Original matrix, which must still be alive.
template<class E>
const E& operand(const std::reference_wrapper<const E>& e) {
  return e.get();
}

template<class E>
struct Transpose;
template<class L, class R>
struct Product;
template<class E>
struct Inverse;

/// @brief Common lazy-expression interface; no results are cached.
/// @tparam D Concrete expression node or matrix type.
template<class D>
struct Expression : ExpressionTag {
  /// @brief Return this object as its concrete expression type.
  /// @return Const reference to the concrete object.
  const D& derived() const { return static_cast<const D&>(*this); }
  /// @brief Construct an owned matrix snapshot, validating current operand shapes.
  /// @return Mat, or SpMat for an entirely sparse transpose/product expression.
  /// @throws std::invalid_argument For invalid shapes or nonfinite solve inputs.
  /// @throws std::runtime_error For singular/rank-deficient solve expressions.
  /// @throws std::domain_error For fractional powers outside the supported real domain.
  /// @throws std::overflow_error For matrix-power exponent/result overflow.
  /// @par Example
  /// @code
  /// dm::Mat<> A = {{1, 2}};
  /// auto lazy = A.t();
  /// const auto snapshot = lazy.eval();
  /// std::print("{} {}\n", snapshot.rows(), snapshot.cols());  // 2 1
  /// @endcode
  auto eval() const;
  /// @brief Render every coefficient as a TeX matrix, evaluating a lazy expression once.
  ///
  /// Returns an owned Tex fragment without math-mode delimiters. Dense and sparse matrices
  /// are read without copying storage; sparse zeros are written as 0. Expressions
  /// are evaluated into one snapshot on each call. Values use seven significant
  /// digits and a locale-independent decimal point; scientific notation becomes
  /// mantissa times 10 to an integer power. Infinity and NaN use TeX math commands.
  /// A matrix with either dimension zero has an empty environment body. No rows or
  /// columns are truncated. Requires amsmath for its standard matrix environments.
  ///
  /// @param type TeX environment name, default "bmatrix"; e.g. "matrix", "pmatrix",
  ///             "Bmatrix", "vmatrix", "Vmatrix" or "smallmatrix". Must contain
  ///             ASCII letters, optionally followed by one trailing star.
  /// @return Tex begin/end environment with columns separated by & and rows by \\;
  ///         supports annotations and string conversion through Tex::str().
  /// @throws std::invalid_argument For an invalid environment name, incompatible
  ///         expression shapes or nonfinite solve inputs.
  /// @throws std::runtime_error For singular/rank-deficient solve expressions.
  /// @throws std::domain_error For fractional powers outside the supported real domain.
  /// @throws std::overflow_error For matrix-power exponent/result overflow.
  ///
  /// @par Example
  /// @code
  /// const dm::Mat<> A = {{1, 2}, {0, 3}};
  /// std::print("{}\n", A.to_tex());
  /// std::print("{}\n", (A * A).to_tex("pmatrix"));
  /// std::print("{}\n", A.to_tex().underbrace("A").display());
  /// @endcode
  /// @see eval(), Tex
  Tex to_tex(const std::string& type = "bmatrix") const;
  /// @brief Read one coefficient by evaluating the expression; use eval() once for repeated reads.
  /// @param row Zero-based row.
  /// @param col Zero-based column.
  /// @return Coefficient value.
  /// @throws std::out_of_range If the index is outside the evaluated matrix.
  /// @throws std::invalid_argument For invalid shapes or nonfinite solve inputs.
  /// @throws std::runtime_error For singular/rank-deficient solve expressions.
  /// @throws std::domain_error For fractional powers outside the supported real domain.
  /// @throws std::overflow_error For matrix-power exponent/result overflow.
  /// @par Example
  /// @code
  /// auto expression = dm::Mat<>{{1, 2}} * dm::Mat<>{{3}, {4}};
  /// std::print("{:.0f}\n", expression(0, 0));  // 11
  /// @endcode
  auto operator()(std::size_t row, std::size_t col) const {
    const auto snapshot = eval();
    return snapshot(row, col);
  }
  /// @brief Lazy transpose of an lvalue; matrices are borrowed and expression nodes copied.
  /// @return Owned transpose node; no coefficient is read.
  /// @par Example
  /// @code
  /// dm::Mat<> A = {{1, 2, 3}};
  /// auto view = A.t();
  /// std::print("{} {}\n", view.rows(), view.cols());  // 3 1
  /// @endcode
  auto t() &;
  /// @brief Lazy transpose of a const lvalue; same lifetime rules as the mutable overload.
  /// @return Owned transpose node.
  /// @par Example
  /// See t() on mutable lvalues.
  auto t() const&;
  /// @brief Lazy transpose owning an rvalue matrix or expression.
  /// @return Owned transpose node, safe beyond this full expression.
  /// @par Example
  /// See t() on mutable lvalues.
  auto t() &&;
  /// @brief Lazy transpose copying a const rvalue matrix or expression.
  /// @return Owned transpose node.
  /// @par Example
  /// See t() on mutable lvalues.
  auto t() const&&;
};

/// @brief Validate CSR shape, row offsets and strictly increasing, in-range column indices.
/// A zero-by-zero default-constructed CSR may omit its single zero row offset.
/// @param storage CSR arrays; stored zeros are allowed.
/// @throws std::invalid_argument For malformed arrays.
/// @tparam T Supported floating-point scalar.
template<class T>
void validate_csr(const SparseCSR<T>& storage);
/// @brief Read a coefficient from a validated CSR matrix by binary search.
/// @param storage CSR matrix.
/// @param row Valid row index.
/// @param col Valid column index.
/// @return Coefficient, or zero for an absent entry.
/// @tparam T Supported floating-point scalar.
template<class T>
T sparse_coefficient(const SparseCSR<T>& storage, std::size_t row, std::size_t col);
/// @brief Insert, replace or erase a CSR coefficient with a strong exception guarantee.
/// @param storage Valid CSR arrays, updated only after successful allocation.
/// @param row Valid row index.
/// @param col Valid column index.
/// @param value New value; zero erases the entry.
/// @tparam T Supported floating-point scalar.
template<class T>
void sparse_set(SparseCSR<T>& storage, std::size_t row, std::size_t col, T value);
/// @brief Allocate a zero matrix after checking dimension arithmetic.
/// @param rows Row count.
/// @param cols Column count.
/// @return Owned dense storage.
/// @throws std::length_error If rows times cols overflows size_t.
/// @tparam T Supported floating-point scalar.
template<class T>
DenseMatrix<T> dense_zeros(std::size_t rows, std::size_t cols);

/// @brief Convert between the facade's two storage types, preserving shape and values.
/// @tparam S Destination sparse selector.
/// @tparam T Destination floating-point scalar.
/// @tparam Other Source sparse selector.
/// @tparam U Source floating-point scalar.
/// @param matrix Owned source matrix, consumed when storage types agree.
/// @return Destination storage.
template<bool S, class T, bool Other, class U>
auto convert_storage(BasicMat<Other, U> matrix);

} // namespace linalg::matrix_detail

/// @brief Owned floating-point matrix backed by canonical DenseMatrix or SparseCSR storage.
/// Initialisers are rows and must be rectangular. Empty rows and zero dimensions are valid.
/// Dense reads are constant time; sparse reads use a row binary search. set() is constant
/// time for dense storage and linear in rows plus stored entries for sparse storage.
/// Expressions borrow lvalue matrices, so those matrices must outlive evaluation. They own
/// temporary matrices and copy expression nodes (including any matrices those nodes own).
/// Mutations before eval()/printing are visible;
/// construction/assignment to Mat or SpMat takes a snapshot, safely handling aliased operands.
/// Nonfinite entries are stored; dense products follow IEEE arithmetic, while sparse products
/// skip structural zeros. Solving rejects nonfinite coefficient/right-hand-side entries.
/// Concurrent reads/evaluations are safe only while all borrowed operands remain unmodified.
/// @tparam Sparse False for row-major dense storage (Mat), true for CSR storage (SpMat).
/// @tparam T Floating-point scalar: float, double (default) or long double.
/// @par Example
/// @code
/// dm::Mat<> A = {{1, 2}, {3, 4}};
/// auto expression = A.t() * A;
/// dm::Mat<> snapshot = expression;
/// A.set(0, 0, 5);
/// std::print("{:.0f} {:.0f}\n", snapshot(0, 0), expression(0, 0));  // 10 34
/// @endcode
/// @see Mat, SpMat, solve()
template<bool Sparse, class T>
class BasicMat : public linalg::matrix_detail::Expression<BasicMat<Sparse, T>> {
  static_assert(std::is_floating_point_v<T>, "Mat scalar must be float, double or long double");

 public:
  using value_type = T; ///< Floating-point scalar type.
  using storage_type =
      std::conditional_t<Sparse, linalg::SparseCSR<T>, linalg::DenseMatrix<T>>; ///< Canonical
                                                                                ///< backend.
  static constexpr bool is_sparse =
      Sparse; ///< Compile-time storage selector for expression dispatch.

  /// @brief Construct an empty 0 by 0 matrix.
  /// @par Example
  /// @code
  /// dm::Mat<> empty;
  /// std::print("{} {}\n", empty.rows(), empty.cols());  // 0 0
  /// @endcode
  BasicMat() : BasicMat(0, 0) {}

  /// @brief Copy an owned matrix snapshot; future mutations of either copy are independent.
  /// @param other Matrix to copy, including its dimensions and scalar/storage kind.
  /// @par Example
  /// @code
  /// dm::Mat<> a = {{1}};
  /// dm::Mat<> b(a);
  /// a.set(0, 0, 2);
  /// std::print("{:.0f}\n", b(0, 0));  // 1
  /// @endcode
  BasicMat(const BasicMat& other) = default;

  /// @brief Transfer storage without allocation, leaving the source empty (0 by 0).
  /// @param other Matrix whose storage is transferred.
  /// @par Example
  /// @code
  /// dm::Mat<> a = {{1, 2}};
  /// dm::Mat<> b(std::move(a));
  /// std::print("{} {}\n", a.rows(), b.cols());  // 0 2
  /// @endcode
  BasicMat(BasicMat&& other) noexcept : storage_(std::move(other.storage_)) {
    other.storage_ = storage_type{};
  }

  /// @brief Replace this matrix with an independent copy of another owned matrix.
  /// @param other Source matrix, possibly this object.
  /// @return This matrix.
  /// @par Example
  /// @code
  /// dm::Mat<> a = {{1}}, b;
  /// b = a;
  /// a.set(0, 0, 2);
  /// std::print("{:.0f}\n", b(0, 0));  // 1
  /// @endcode
  BasicMat& operator=(const BasicMat& other) = default;

  /// @brief Transfer storage without allocation; distinct sources become empty (0 by 0).
  /// @param other Source matrix; self move assignment leaves the object unchanged.
  /// @return This matrix.
  /// @par Example
  /// @code
  /// dm::SpMat<> a = {{1, 0}}, b;
  /// b = std::move(a);
  /// std::print("{} {}\n", a.cols(), b.nnz());  // 0 1
  /// @endcode
  BasicMat& operator=(BasicMat&& other) noexcept {
    if (this != &other) {
      storage_       = std::move(other.storage_);
      other.storage_ = storage_type{};
    }
    return *this;
  }

  /// @brief Allocate a zero matrix; sparse storage allocates only row offsets.
  /// @param rows Row count.
  /// @param cols Column count.
  /// @throws std::length_error For overflowing dimensions/row-offset count.
  /// @par Example
  /// @code
  /// dm::SpMat<> zeros(3, 1000);
  /// std::print("{}\n", zeros.nnz());  // 0
  /// @endcode
  BasicMat(std::size_t rows, std::size_t cols);

  /// @brief Construct from rectangular nested rows; sparse storage omits zero values.
  /// @param rows Row initialisers, including empty rows if every row is empty.
  /// @throws std::invalid_argument For ragged rows.
  /// @throws std::length_error For overflowing dimensions.
  /// @par Example
  /// @code
  /// dm::SpMat<> A = {{1, 0}, {0, 2}};
  /// std::print("{}\n", A.nnz());  // 2
  /// @endcode
  BasicMat(std::initializer_list<std::initializer_list<T>> rows);

  /// @brief Own canonical backend storage; dense storage is moved, CSR structure is validated.
  /// @param storage DenseMatrix<T> for dense storage, SparseCSR<T> for CSR storage.
  /// @throws std::invalid_argument For invalid CSR offsets or column indices.
  /// @par Example
  /// @code
  /// dm::Mat<> A(dm::linalg::DenseMatrix<double>(1, 2, {3, 4}));
  /// std::print("{:.0f}\n", A(0, 1));  // 4
  /// @endcode
  explicit BasicMat(storage_type storage) : storage_(std::move(storage)) {
    if constexpr (Sparse)
      linalg::matrix_detail::validate_csr(storage_);
  }

  /// @brief Evaluate an expression or convert another facade matrix into this storage type.
  /// @tparam E Facade expression type.
  /// @param expression Live expression whose borrowed matrices must still exist.
  /// @throws std::invalid_argument For incompatible shapes or nonfinite solve inputs.
  /// @throws std::runtime_error For singular/rank-deficient systems.
  /// @par Example
  /// @code
  /// dm::Mat<> A = {{1, 2}};
  /// dm::SpMat<> column = A.t();
  /// std::print("{} {}\n", column.rows(), column.cols());  // 2 1
  /// @endcode
  template<class E, std::enable_if_t<linalg::matrix_detail::is_expression<E>, int> = 0>
  BasicMat(const E& expression);

  /// @brief Evaluate before replacing storage, so A = A.t() and other aliases are safe.
  /// @tparam E Facade expression type.
  /// @param expression Expression or other facade matrix.
  /// @return This matrix.
  /// @throws std::invalid_argument For incompatible shapes or nonfinite solve inputs.
  /// @throws std::runtime_error For singular/rank-deficient systems.
  /// @par Example
  /// @code
  /// dm::Mat<> A = {{1, 2}, {3, 4}};
  /// A = A.t();
  /// std::print("{:.0f}\n", A(0, 1));  // 3
  /// @endcode
  template<class E, std::enable_if_t<linalg::matrix_detail::is_expression<E>, int> = 0>
  BasicMat& operator=(const E& expression) {
    BasicMat result(expression);
    storage_ = std::move(result.storage_);
    return *this;
  }

  /// @brief Number of rows.
  /// @return Row count.
  std::size_t rows() const { return storage_.rows(); }
  /// @brief Number of columns.
  /// @return Column count.
  std::size_t cols() const { return storage_.cols(); }

  /// @brief Default DataFrame column labels V1, V2, ..., without storing metadata.
  /// @return One unique, one-based label per column, independent of storage/scalar.
  /// @par Example
  /// @code
  /// const dm::Mat<> A = {{1, 2}};
  /// std::print("{}\n", A.colnames()); // ["V1", "V2"]
  /// @endcode
  std::vector<std::string> colnames() const {
    std::vector<std::string> names;
    names.reserve(cols());
    for (std::size_t j = 0; j < cols(); ++j)
      names.push_back("V" + std::to_string(j + 1));
    return names;
  }
  /// @brief Stored-entry count for sparse matrices, count of nonzero values for dense matrices.
  /// @return Count; constant time for CSR, linear in matrix size for dense storage.
  std::size_t nnz() const;
  /// @brief Read a coefficient with bounds checking; structural sparse zeros read as zero.
  /// @param row Zero-based row.
  /// @param col Zero-based column.
  /// @return Value at (row, col).
  /// @throws std::out_of_range For an out-of-bounds index.
  T operator()(std::size_t row, std::size_t col) const;
  /// @brief Access a writable coefficient using checked, zero-based indices.
  /// Dense matrices return a scalar reference, invalidated by storage replacement.
  /// Sparse matrices return a coordinate proxy: reading never inserts an entry and
  /// assigning zero removes it. The matrix must outlive its reference/proxy; sparse
  /// proxies remain valid across insertions at other coordinates. Assignment has
  /// the same storage and complexity semantics as set().
  /// @param row Zero-based row, strictly less than rows().
  /// @param col Zero-based column, strictly less than cols().
  /// @return Dense scalar reference or writable CSR coefficient proxy.
  /// @throws std::out_of_range If either index is outside the matrix.
  /// @par Example
  /// @code
  /// dm::Mat<> A(3, 5);
  /// A(2, 4) = 1.1;
  /// dm::SpMat<> S(30, 40);
  /// S(2, 8) = 2.3;
  /// std::print("{:.1f} {:.1f} {}\n", A(2, 4), S(2, 8), S.nnz()); // 1.1 2.3 1
  /// @endcode
  /// @see set()
  decltype(auto) operator()(std::size_t row, std::size_t col) {
    if (row >= rows() || col >= cols())
      throw std::out_of_range("Mat: coefficient index out of range");
    if constexpr (Sparse)
      return linalg::matrix_detail::SparseCoefficientReference<T>(*this, row, col);
    else
      return (storage_(row, col));
  }
  /// @brief Replace a coefficient, inserting/erasing sparse entries as needed.
  /// @param row Zero-based row.
  /// @param col Zero-based column.
  /// @param value New value; zero erases an existing sparse entry.
  /// @throws std::out_of_range For an out-of-bounds index.
  /// @par Example
  /// @code
  /// dm::SpMat<> A(2, 2);
  /// A.set(0, 1, 3);
  /// std::print("{} {:.0f}\n", A.nnz(), A(0, 1));  // 1 3
  /// @endcode
  void set(std::size_t row, std::size_t col, T value);
  /// @brief Borrow immutable canonical storage for existing linalg algorithms.
  /// @return Backend reference valid until mutation or destruction of this matrix.
  const storage_type& storage() const& { return storage_; }
  /// @brief Move the backend out of an rvalue matrix, leaving the source empty (0 by 0).
  /// @return Owned backend storage.
  /// @par Example
  /// @code
  /// auto native = dm::Mat<>{{1, 2}}.storage();
  /// std::print("{} {}\n", native.rows(), native.cols());  // 1 2
  /// @endcode
  storage_type storage() && {
    auto result = std::move(storage_);
    storage_    = storage_type{};
    return result;
  }

 private:
  storage_type storage_; ///< The single source of truth for matrix values and dimensions.
};

namespace linalg::matrix_detail {

/// @brief Transpose node owning its operand closure; no transposed array is allocated on
/// construction.
/// @tparam E Operand closure type.
template<class E>
struct Transpose : Expression<Transpose<E>> {
  E source; ///< Owned expression/value or borrowed matrix wrapper.
  using value_type =
      typename std::decay_t<decltype(operand(std::declval<E>()))>::value_type; ///< Scalar type.
  static constexpr bool is_sparse =
      std::decay_t<decltype(operand(std::declval<E>()))>::is_sparse; ///< Storage dispatch.

  /// @brief Capture a source without evaluating it.
  /// @param value Prepared closure.
  explicit Transpose(E value) : source(std::move(value)) {}
  /// @brief Transposed row count.
  /// @return Source column count.
  std::size_t rows() const { return operand(source).cols(); }
  /// @brief Transposed column count.
  /// @return Source row count.
  std::size_t cols() const { return operand(source).rows(); }
};

/// @brief Deferred matrix product; children are retained as lifetime-aware closures.
/// @tparam L Left closure type.
/// @tparam R Right closure type.
template<class L, class R>
struct Product : Expression<Product<L, R>> {
  L left;  ///< Left closure.
  R right; ///< Right closure.
  using value_type = typename std::decay_t<decltype(operand(
      std::declval<L>()))>::value_type; ///< Shared scalar type.
  static_assert(
      std::is_same_v<
          value_type,
          typename std::decay_t<decltype(operand(std::declval<R>()))>::value_type>,
      "Mat products require matching scalar types");
  static constexpr bool is_sparse =
      std::decay_t<decltype(operand(std::declval<L>()))>::is_sparse &&
      std::decay_t<decltype(operand(std::declval<R>()))>::is_sparse; ///< Pure sparse products stay
                                                                     ///< sparse.

  /// @brief Store both operands and validate initial dimensions without reading values.
  /// @param a Left closure.
  /// @param b Right closure.
  /// @throws std::invalid_argument For incompatible inner dimensions.
  Product(L a, R b) : left(std::move(a)), right(std::move(b)) {
    if (operand(left).cols() != operand(right).rows())
      throw std::invalid_argument("Mat product: incompatible dimensions");
  }
  /// @brief Product row count.
  /// @return Left row count.
  std::size_t rows() const { return operand(left).rows(); }
  /// @brief Product column count.
  /// @return Right column count.
  std::size_t cols() const { return operand(right).cols(); }
};

/// @brief Deferred inverse action; products consume it as a solve instead of constructing an
/// inverse.
/// @tparam E Coefficient-matrix closure type.
template<class E>
struct Inverse : Expression<Inverse<E>> {
  E source; ///< Coefficient expression or borrowed matrix.
  using value_type =
      typename std::decay_t<decltype(operand(std::declval<E>()))>::value_type; ///< Scalar type.
  static constexpr bool is_sparse = false; ///< An inverse is generally dense.

  /// @brief Store the coefficient closure and check squareness without factorisation.
  /// @param value Coefficient closure.
  /// @throws std::invalid_argument For a nonsquare expression.
  explicit Inverse(E value) : source(std::move(value)) {
    if (rows() != cols())
      throw std::invalid_argument("solve(A): A must be square");
  }
  /// @brief Inverse row count.
  /// @return Coefficient column count.
  std::size_t rows() const { return operand(source).cols(); }
  /// @brief Inverse column count.
  /// @return Coefficient row count.
  std::size_t cols() const { return operand(source).rows(); }
};

/// @brief Lazy square-matrix power; exponent -1 is consumed as an inverse action.
/// Borrows lvalue matrices and owns temporary matrices/child nodes. Integral exponents
/// use binary powering; other exponents use the real principal-power backend.
/// @tparam E Captured matrix/expression type.
/// @tparam P Integral or floating-point exponent type, excluding bool.
template<class E, class P>
struct Power : Expression<Power<E, P>> {
  E source;   ///< Owned expression or borrowed coefficient matrix.
  P exponent; ///< Original exponent, retaining exact integral values.
  using value_type =
      typename std::decay_t<decltype(operand(std::declval<E>()))>::value_type; ///< Matrix scalar.
  static constexpr bool is_sparse =
      false; ///< Negative/fractional powers generally require dense storage.

  /// @brief Capture operands and validate square shape and a finite exponent.
  /// @param value Captured coefficient matrix/expression.
  /// @param power Exponent, with no rounding of integral inputs.
  /// @throws std::invalid_argument For nonsquare input or a nonfinite exponent.
  Power(E value, P power) : source(std::move(value)), exponent(power) {
    if (rows() != cols() || !std::isfinite(static_cast<long double>(power)))
      throw std::invalid_argument("Mat power: requires a square matrix and finite exponent");
  }
  /// @brief Number of rows of the square power.
  /// @return Source row count.
  std::size_t rows() const { return operand(source).rows(); }
  /// @brief Number of columns of the square power.
  /// @return Source column count.
  std::size_t cols() const { return operand(source).cols(); }
};

/// @brief Build a lazy ordinary matrix power, with -1 and -1.0 acting as solve(A).
///
/// Integral-valued exponents use binary powering; zero returns identity. Negative
/// powers require invertibility. Nonintegral powers use a binomial Taylor series
/// after repeated Denman-Beavers square roots; sparse inputs become dense at
/// evaluation. The real principal branch requires no nonpositive real eigenvalues
/// for general matrices; nonnegative diagonal matrices also support positive powers
/// at zero. Iteration failure is reported, including difficult ill-conditioned inputs.
/// Results retain the matrix scalar type, independent of exponent type. Construction
/// performs no algebra. A leading power of -1 fuses with following products into one
/// solve. Lvalue matrices must outlive evaluation; temporaries are owned.
/// C++ gives ^ lower precedence than *, so write (A ^ -1) * B.
///
/// @tparam E Matrix or lazy expression type.
/// @tparam P Integral or floating-point exponent type, excluding bool.
/// @param matrix Square matrix/expression.
/// @param exponent Finite exponent; -1 and -1.0 both mean inverse, not exp(-A).
/// @return Deferred dense matrix power with the same floating-point scalar type.
/// @throws std::invalid_argument For nonsquare input or nonfinite exponent/coefficients.
/// @throws std::domain_error For diagonal fractional powers outside the real domain.
/// @throws std::runtime_error For singular solves or square-root/Taylor nonconvergence.
/// @throws std::overflow_error For an exponent outside the supported integer range or nonfinite
/// results.
/// @par Example
/// @code
/// const dm::Mat<> A = {{4, 0}, {0, 9}}, b = {{8}, {18}};
/// const auto x = ((A ^ -1) * b).eval();
/// const auto root = (A ^ 0.5).eval();
/// std::print("{:.0f} {:.0f} {:.0f}\n", x(0, 0), root(0, 0), root(1, 1)); // 2 2 3
/// @endcode
/// @par References
/// - N. J. Higham and L. Lin (2011). "A Schur-Pade Algorithm for Fractional Powers of a Matrix".
///   SIAM J. Matrix Anal. Appl. 32(3), 1056-1078. doi:10.1137/10081232X.
///   Inverse scaling/squaring principle; this implementation uses a checked Taylor
///   series and Denman-Beavers iteration instead of Schur reduction and Pade evaluation.
/// @see datamunge::solve(), Expression::eval()
template<
    class E,
    class P,
    std::enable_if_t<
        is_expression<E> && (std::is_integral_v<P> || std::is_floating_point_v<P>) &&
            !std::is_same_v<P, bool>,
        int> = 0>
auto operator^(E&& matrix, P exponent) {
  return Power<Capture<E>, P>(capture<E>(std::forward<E>(matrix)), exponent);
}

/// @brief Product builder found by ADL for facade matrices and expressions only.
/// @tparam L Left forwarding type.
/// @tparam R Right forwarding type.
/// @param left Left matrix/expression.
/// @param right Right matrix/expression.
/// @return Unevaluated product, borrowing lvalue matrices and owning temporaries.
/// @throws std::invalid_argument For incompatible inner dimensions.
/// @par Example
/// @code
/// dm::Mat<> A = {{1, 2}};
/// dm::Mat<> B = {{3}, {4}};
/// auto product = A * B;
/// std::print("{:.0f}\n", product(0, 0));  // 11
/// @endcode
template<class L, class R, std::enable_if_t<is_expression<L> && is_expression<R>, int> = 0>
auto operator*(L&& left, R&& right) {
  return Product<Capture<L>, Capture<R>>(
      capture<L>(std::forward<L>(left)), capture<R>(std::forward<R>(right)));
}

/// @brief Whether a node is a product; used only to reassociate a leading inverse action.
/// @tparam E Node type.
template<class E>
inline constexpr bool is_product = false;
/// @brief Product-node recognition.
/// @tparam L Left closure.
/// @tparam R Right closure.
template<class L, class R>
inline constexpr bool is_product<Product<L, R>> = true;
/// @brief Whether a node is an inverse action.
/// @tparam E Node type.
template<class E>
inline constexpr bool is_inverse = false;
/// @brief Inverse-node recognition.
/// @tparam E Coefficient closure.
template<class E>
inline constexpr bool is_inverse<Inverse<E>> = true;
/// @brief Whether a node is a deferred matrix power.
/// @tparam E Candidate expression type.
template<class E>
inline constexpr bool is_power = false;
/// @brief Recognize power nodes independently of their exponent value.
/// @tparam E Source closure.
/// @tparam P Exponent type.
template<class E, class P>
inline constexpr bool is_power<Power<E, P>> = true;
/// @brief Whether a node is a transpose.
/// @tparam E Node type.
template<class E>
inline constexpr bool is_transpose = false;
/// @brief Transpose-node recognition.
/// @tparam E Source closure.
template<class E>
inline constexpr bool is_transpose<Transpose<E>> = true;

/// @brief Whether the far-left factor can be a deferred inverse; powers need a runtime exponent check.
/// @tparam E Expression type.
template<class E>
struct HasSolve : std::bool_constant<is_inverse<E> || is_power<E>> {};
/// @brief Propagate inverse-prefix recognition down a product's left branch.
/// @tparam L Left closure.
/// @tparam R Right closure.
template<class L, class R>
struct HasSolve<Product<L, R>> : HasSolve<std::decay_t<decltype(operand(std::declval<L>()))>> {};

/// @brief Check whether a candidate leftmost factor is currently an inverse action.
/// @tparam E Expression type.
/// @param expression Matrix/expression whose exponent may equal -1.
/// @return True for solve(A) or A raised to -1 (including floating-point -1).
template<class E>
bool has_solve(const E& expression) {
  if constexpr (is_power<E>) {
    if constexpr (std::is_unsigned_v<decltype(expression.exponent)>)
      return false;
    else
      return expression.exponent == -1;
  } else if constexpr (is_product<E>)
    return has_solve(operand(expression.left));
  else
    return is_inverse<E>;
}

/// @brief Recheck all dimensions at evaluation, including operands resized after capture.
/// @tparam E Expression type.
/// @param e Expression to validate recursively.
/// @throws std::invalid_argument For an incompatible product or nonsquare inverse action.
template<class E>
void validate(const E& e) {
  if constexpr (is_product<E>) {
    validate(operand(e.left));
    validate(operand(e.right));
    if (operand(e.left).cols() != operand(e.right).rows())
      throw std::invalid_argument("Mat product: operand dimensions changed");
  } else if constexpr (is_inverse<E> || is_power<E>) {
    validate(operand(e.source));
    if (e.rows() != e.cols())
      throw std::invalid_argument("solve(A): coefficient dimensions changed");
  } else if constexpr (is_transpose<E>)
    validate(operand(e.source));
}

/// @brief Dense facade with a specified scalar type.
/// @tparam T Floating-point scalar.
template<class T>
using DenseFacade = BasicMat<false, T>;
/// @brief Sparse facade with a specified scalar type.
/// @tparam T Floating-point scalar.
template<class T>
using SparseFacade = BasicMat<true, T>;

/// @brief Dense-dense multiplication with logical transpose flags; never allocates transpose
/// arrays.
/// @param a Left matrix.
/// @param ta Logically transpose a.
/// @param b Right matrix.
/// @param tb Logically transpose b.
/// @return Owned product.
/// @tparam T Supported floating-point scalar.
template<class T>
DenseFacade<T> multiply(const DenseFacade<T>& a, bool ta, const DenseFacade<T>& b, bool tb);
/// @brief Sparse-dense multiplication visiting only stored left entries.
/// @param a Left CSR matrix.
/// @param ta Logically transpose a.
/// @param b Right dense matrix.
/// @param tb Logically transpose b.
/// @return Dense product.
/// @tparam T Supported floating-point scalar.
template<class T>
DenseFacade<T> multiply(const SparseFacade<T>& a, bool ta, const DenseFacade<T>& b, bool tb);
/// @brief Dense-sparse multiplication visiting only stored right entries.
/// @param a Left dense matrix.
/// @param ta Logically transpose a.
/// @param b Right CSR matrix.
/// @param tb Logically transpose b.
/// @return Dense product.
/// @tparam T Supported floating-point scalar.
template<class T>
DenseFacade<T> multiply(const DenseFacade<T>& a, bool ta, const SparseFacade<T>& b, bool tb);
/// @brief CSR product with sparse row accumulators; no dense input or output grids are created.
/// @param a Left CSR matrix.
/// @param ta Logically transpose a.
/// @param b Right CSR matrix.
/// @param tb Logically transpose b.
/// @return CSR product, with sorted columns and zero sums omitted.
/// @tparam T Supported floating-point scalar.
template<class T>
SparseFacade<T> multiply(const SparseFacade<T>& a, bool ta, const SparseFacade<T>& b, bool tb);
/// @brief Convert a CSR facade to row-major storage at an explicit dense boundary.
/// @param a Source CSR matrix.
/// @return Owned dense storage.
/// @tparam T Supported floating-point scalar.
template<class T>
DenseMatrix<T> dense_storage(const SparseFacade<T>& a);
/// @brief Convert dense facade values to CSR, omitting exact zeros.
/// @param a Source dense matrix.
/// @return Owned CSR storage.
/// @tparam T Supported floating-point scalar.
template<class T>
SparseCSR<T> sparse_storage(const DenseFacade<T>& a);
/// @brief Materialise a standalone logical transpose while retaining storage kind.
/// @param a Dense source.
/// @return Dense transpose.
/// @tparam T Supported floating-point scalar.
template<class T>
DenseFacade<T> transposed(const DenseFacade<T>& a);
/// @brief Materialise a standalone CSR transpose in O(rows + cols + nnz) time/storage.
/// @param a CSR source.
/// @return CSR transpose.
/// @tparam T Supported floating-point scalar.
template<class T>
SparseFacade<T> transposed(const SparseFacade<T>& a);
/// @brief Solve all right-hand sides with one canonical LU (square) or QR (tall) factorisation.
/// @param a Coefficient matrix with rows at least cols.
/// @param b Right-hand sides, with b.rows() equal to a.rows().
/// @return Dense solution matrix, one column per right-hand side.
/// @throws std::invalid_argument For shape mismatch, wide systems or nonfinite entries.
/// @throws std::runtime_error For singular or numerically rank-deficient a.
/// @tparam T Supported floating-point scalar.
template<class T>
DenseFacade<T> solve_system(const DenseFacade<T>& a, const DenseFacade<T>& b);
/// @brief Construct the dense identity used only when explicitly evaluating an inverse.
/// @param n Matrix order.
/// @return Owned identity matrix.
/// @tparam T Supported floating-point scalar.
template<class T>
DenseFacade<T> identity(std::size_t n);

/// @brief Evaluate an integer matrix power by squaring, inverting once when requested.
/// @tparam T Supported floating-point matrix scalar.
/// @param matrix Finite square dense coefficient matrix.
/// @param magnitude Absolute exponent, including the full unsigned integer range.
/// @param negative Whether to invert before powering.
/// @return Owned dense matrix power, identity for magnitude zero.
/// @throws std::invalid_argument For nonfinite coefficients or nonsquare shape.
/// @throws std::runtime_error For singular negative powers.
/// @throws std::overflow_error For nonfinite products.
template<class T>
DenseFacade<T> integer_power(DenseFacade<T> matrix, std::uintmax_t magnitude, bool negative);

/// @brief Evaluate a real principal matrix power using inverse scaling and a checked binomial
/// Taylor series.
/// @tparam T Supported floating-point matrix scalar.
/// @param matrix Finite square coefficient matrix, generally with no nonpositive real eigenvalues.
/// @param exponent Finite real exponent; integral values reuse integer_power().
/// @return Owned dense principal power with scalar type T.
/// @throws std::invalid_argument For nonfinite input or nonsquare shape.
/// @throws std::domain_error For invalid real diagonal powers.
/// @throws std::runtime_error For singularity or nonconvergent square-root/Taylor iteration.
/// @throws std::overflow_error For excessive exponents or nonfinite results.
template<class T>
DenseFacade<T> real_power(const DenseFacade<T>& matrix, long double exponent);

/// @brief Evaluate a validated expression; overloads below control fusion and storage.
/// @tparam E Expression type.
/// @param e Valid expression.
/// @return Owned result retaining sparse storage where possible.
template<class E>
auto materialize(const E& e);

/// @brief Expose a matrix and logical transpose to a kernel, evaluating complex children once.
/// @tparam E Child expression.
/// @tparam F Kernel callback type.
/// @param e Child expression.
/// @param f Callback accepting (matrix, transpose flag), invoked before temporary destruction.
/// @return Callback result by value.
template<class E, class F>
auto with_operand(const E& e, F&& f) {
  if constexpr (is_transpose<E>) {
    return with_operand(operand(e.source), [&](const auto& matrix, bool transpose) {
      return f(matrix, !transpose);
    });
  } else if constexpr (is_matrix<E>)
    return f(e, false);
  else {
    auto value = materialize(e);
    return f(value, false);
  }
}

template<class E>
auto materialize(const E& e) {
  if constexpr (is_matrix<E>)
    return e;
  else if constexpr (is_transpose<E>) {
    return with_operand(e, [](const auto& matrix, bool transpose) {
      return transpose ? transposed(matrix) : matrix;
    });
  } else if constexpr (is_inverse<E>) {
    DenseFacade<typename E::value_type> a = materialize(operand(e.source));
    return solve_system(a, identity<typename E::value_type>(a.rows()));
  } else if constexpr (is_power<E>) {
    DenseFacade<typename E::value_type> a = materialize(operand(e.source));
    using P                               = decltype(e.exponent);
    if constexpr (std::is_integral_v<P>) {
      using U             = std::make_unsigned_t<P>;
      const bool negative = std::is_signed_v<P> && e.exponent < 0;
      const U    magnitude =
          negative ? U(U(0) - static_cast<U>(e.exponent)) : static_cast<U>(e.exponent);
      return integer_power(std::move(a), static_cast<std::uintmax_t>(magnitude), negative);
    } else
      return real_power(a, static_cast<long double>(e.exponent));
  } else {
    const auto& left  = operand(e.left);
    const auto& right = operand(e.right);
    using L           = std::decay_t<decltype(left)>;
    if constexpr (is_inverse<L> || is_power<L>) {
      if (has_solve(left)) {
        DenseFacade<typename E::value_type> a = materialize(operand(left.source));
        DenseFacade<typename E::value_type> b = materialize(right);
        return solve_system(a, b);
      }
    } else if constexpr (is_product<L> && HasSolve<L>::value) {
      if (has_solve(left))
        return materialize(operand(left.left) * (operand(left.right) * right));
    }
    return with_operand(left, [&](const auto& a, bool ta) {
      return with_operand(right, [&](const auto& b, bool tb) { return multiply(a, ta, b, tb); });
    });
  }
}

template<bool S, class T, bool Other, class U>
auto convert_storage(BasicMat<Other, U> matrix) {
  if constexpr (std::is_same_v<T, U>) {
    if constexpr (S == Other)
      return std::move(matrix).storage();
    else if constexpr (S)
      return sparse_storage(matrix);
    else
      return dense_storage(matrix);
  } else {
    BasicMat<S, U> same_kind(std::move(matrix));
    if constexpr (S) {
      SparseCSR<T> result;
      const auto&  source = same_kind.storage();
      result.nrows        = source.nrows;
      result.ncols        = source.ncols;
      result.row_ptr      = source.row_ptr;
      result.col_idx      = source.col_idx;
      result.values.assign(source.values.begin(), source.values.end());
      return result;
    } else {
      auto result = dense_zeros<T>(same_kind.rows(), same_kind.cols());
      std::copy(
          same_kind.storage().data().begin(),
          same_kind.storage().data().end(),
          result.data().begin());
      return result;
    }
  }
}

template<class D>
auto Expression<D>::eval() const {
  validate(derived());
  return materialize(derived());
}

template<class D>
Tex Expression<D>::to_tex(const std::string& type) const {
  const auto letters = type.size() - (!type.empty() && type.back() == '*');
  if (letters == 0 ||
      type.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ", 0) < letters)
    throw std::invalid_argument(
        "Mat::to_tex: expected an environment name of letters with an optional trailing star");
  if constexpr (!is_matrix<D>) {
    return eval().to_tex(type);
  } else {
    const auto&        matrix = derived();
    std::string        result = "\\begin{" + type + "}\n";
    std::ostringstream cell;
    cell.imbue(std::locale::classic());
    cell << std::setprecision(7);
    if (matrix.cols() != 0) {
      for (std::size_t i = 0; i < matrix.rows(); ++i) {
        if (i != 0)
          result += " \\\\\n";
        for (std::size_t j = 0; j < matrix.cols(); ++j) {
          if (j != 0)
            result += " & ";
          const auto value = matrix(i, j);
          if (std::isnan(value))
            result += "\\mathrm{NaN}";
          else if (std::isinf(value))
            result += value < 0 ? "-\\infty" : "\\infty";
          else {
            cell.str("");
            cell.clear();
            cell << value;
            const auto text     = cell.str();
            const auto exponent = text.find_first_of("eE");
            if (exponent == std::string::npos)
              result += text;
            else
              result += text.substr(0, exponent) + " \\times 10^{" +
                        std::to_string(std::stoi(text.substr(exponent + 1))) + "}";
          }
        }
      }
      if (matrix.rows() != 0)
        result += '\n';
    }
    return Tex(result + "\\end{" + type + "}");
  }
}

template<class D>
auto Expression<D>::t() & {
  return Transpose<Capture<D&>>(capture<D&>(static_cast<D&>(*this)));
}
template<class D>
auto Expression<D>::t() const& {
  return Transpose<Capture<const D&>>(capture<const D&>(derived()));
}
template<class D>
auto Expression<D>::t() && {
  return Transpose<D>(static_cast<D&&>(*this));
}
template<class D>
auto Expression<D>::t() const&& {
  return Transpose<D>(derived());
}

} // namespace linalg::matrix_detail

template<bool S, class T>
template<class E, std::enable_if_t<linalg::matrix_detail::is_expression<E>, int>>
BasicMat<S, T>::BasicMat(const E& expression)
    : storage_(linalg::matrix_detail::convert_storage<S, T>(expression.eval())) {}

/// @brief Defer an inverse action; multiplying by a right-hand side performs a linear solve.
/// No factorisation, transpose, product or inverse is computed here. eval()/printing of a
/// standalone solve(A) explicitly computes A inverse by solving against the identity. In
/// solve(A)*B*C the right-hand-side product B*C is evaluated before one factorisation/solve;
/// it avoids an inverse and an A.cols() by B.cols() solved intermediate. This reassociation
/// may change floating-point rounding. The written normal equations still square the design's
/// condition number; solve(X,y) uses QR directly for tall systems. Sparse inputs/products
/// retain CSR storage until the coefficient matrix/right-hand sides enter the dense solve
/// backend; this is not a sparse LU implementation. Lvalue matrices must outlive evaluation.
/// @tparam E Facade expression type.
/// @param coefficient Square coefficient matrix/expression.
/// @return Owned lazy inverse-action expression.
/// @throws std::invalid_argument Immediately for a nonsquare expression, at evaluation for
///         incompatible shapes or nonfinite solve inputs.
/// @throws std::runtime_error At evaluation for singular coefficients.
/// @par Example
/// @code
/// dm::Mat<> A = {{2, 0}, {0, 4}}, b = {{6}, {8}};
/// auto solution = dm::solve(A) * b;
/// std::print("{:.0f} {:.0f}\n", solution(0, 0), solution(1, 0));  // 3 2
/// @endcode
/// @see linalg::lu(), linalg::qr()
template<class E, std::enable_if_t<linalg::matrix_detail::is_expression<E>, int> = 0>
auto solve(E&& coefficient) {
  namespace md = linalg::matrix_detail;
  return md::Inverse<md::Capture<E>>(md::capture<E>(std::forward<E>(coefficient)));
}

/// @brief Immediately solve A X = B, or full-column-rank least squares when A is tall.
/// Square systems reuse canonical pivoted LU; tall systems use canonical Householder QR,
/// including its explicit rows-by-rows Q storage. Sparse inputs are converted to dense here.
/// Multiple right-hand sides reuse one factorisation. Wide and rank-deficient systems are
/// rejected; use the existing SVD API for a minimum-norm solution.
/// @tparam A Coefficient expression type.
/// @tparam B Right-hand-side expression type.
/// @param coefficient Coefficient matrix with rows at least cols.
/// @param rhs Right-hand sides with matching rows.
/// @return Dense solution, one column per right-hand side.
/// @throws std::invalid_argument For dimension mismatch, wide systems or nonfinite values.
/// @throws std::runtime_error For singular/rank-deficient coefficients.
/// @par Example
/// @code
/// dm::Mat<> X = {{1, 0}, {1, 1}, {1, 2}}, y = {{1}, {2}, {4}};
/// const auto beta = dm::solve(X, y);
/// std::print("{:.6f} {:.6f}\n", beta(0, 0), beta(1, 0));  // 0.833333 1.500000
/// @endcode
/// @see linalg::least_squares()
template<
    class A,
    class B,
    std::enable_if_t<
        linalg::matrix_detail::is_expression<A> && linalg::matrix_detail::is_expression<B>,
        int> = 0>
auto solve(const A& coefficient, const B& rhs) {
  static_assert(
      std::is_same_v<typename A::value_type, typename B::value_type>,
      "solve(A,B) requires matching scalar types");
  using Dense = BasicMat<false, typename A::value_type>;
  return linalg::matrix_detail::solve_system(Dense(coefficient), Dense(rhs));
}

namespace linalg {
/// @brief Alias for dm::Mat<Storage, T>; defaults to dense storage and double scalars.
/// Supports float, double and long double. Products and solves require equal scalar types;
/// constructing another Mat<Storage, T> explicitly converts scalar/storage type. Expression
/// evaluation preserves the scalar and storage kinds until an inverse action yields dense output.
/// @tparam Storage dm::Dense (default) or dm::Sparse storage policy.
/// @tparam T Floating-point scalar (default double).
/// @par Example
/// @code
/// using dm::la::Mat;
/// Mat<> X = {{1, 0}, {1, 1}, {1, 2}};
/// Mat<dm::Dense, double> y = {{1}, {3}, {5}};
/// auto beta = dm::solve(X.t() * X) * X.t() * y;
/// std::print("{:.0f} {:.0f}\n", beta(0, 0), beta(1, 0));  // 1 2
/// @endcode
/// @see datamunge::Mat<>, datamunge::SpMat<>, datamunge::solve()
template<class Storage = datamunge::Dense, class T = double>
using Mat = datamunge::Mat<Storage, T>;
} // namespace linalg

/// @brief Short namespace spelling for the native linear algebra API.
namespace la = linalg;

/// @brief Create a lazy matrix transpose using the lowercase, R-style t(A) spelling.
/// Forwards to A.t() without reading coefficients or allocating transposed storage.
/// Lvalue matrices are borrowed and must outlive evaluation; temporary matrices
/// are owned. Expression nodes are copied/moved with their existing ownership rules.
/// Dense/CSR storage and scalar type are preserved on evaluation.
/// @tparam E Matrix or lazy matrix expression type, including const and rvalue inputs.
/// @param matrix Matrix/expression to transpose without mutation.
/// @return Deferred transpose node, equivalent to the corresponding matrix.t() overload.
/// @par Example
/// @code
/// const dm::Mat<> A = {{1, 2, 3}};
/// const auto transposed = dm::t(A).eval();
/// std::print("{} {} {:.0f}\n", transposed.rows(), transposed.cols(), transposed(2, 0)); // 3 1 3
/// @endcode
/// @see BasicMat, linalg::matrix_detail::Expression::t()
template<class E, std::enable_if_t<linalg::matrix_detail::is_expression<E>, int> = 0>
[[nodiscard]] auto t(E&& matrix) {
  return std::forward<E>(matrix).t();
}

/// @brief Create an owned, zero-initialised dense matrix.
/// Allocates rows times cols scalar entries. Empty dimensions are valid; indices
/// are zero-based. The returned matrix owns its storage and supports lazy algebra.
/// @tparam T Floating-point scalar, default double; e.g. mat<float>(rows, cols).
/// @param rows Number of rows.
/// @param cols Number of columns.
/// @return Mat<Dense, T> with every coefficient zero.
/// @throws std::length_error If the dimensions overflow.
/// @par Example
/// @code
/// auto A = dm::mat(3, 5);
/// A(2, 4) = 1.1;
/// std::print("{} {} {:.1f}\n", A.rows(), A.cols(), A(2, 4)); // 3 5 1.1
/// @endcode
/// @see spmat(), Mat, BasicMat::set()
template<class T = double>
[[nodiscard]] Mat<Dense, T> mat(std::size_t rows, std::size_t cols) {
  return Mat<Dense, T>(rows, cols);
}

/// @brief Create an owned, zero-initialised compressed sparse row matrix.
/// Allocates only rows plus one row offsets; no coefficient entries are stored.
/// Empty dimensions are valid and indices are zero-based. Coefficient assignment
/// inserts nonzeros or removes stored zeros using the canonical CSR mutation path.
/// @tparam T Floating-point scalar, default double; e.g. spmat<float>(rows, cols).
/// @param rows Number of rows.
/// @param cols Number of columns, without a dense rows-times-cols allocation.
/// @return Mat<Sparse, T> with zero stored entries.
/// @throws std::length_error If the row-offset count overflows.
/// @par Example
/// @code
/// auto A = dm::spmat(30, 40);
/// A(2, 8) = 2.3;
/// A(4, 5) = 1.8;
/// std::print("{} {} {}\n", A.rows(), A.cols(), A.nnz()); // 30 40 2
/// @endcode
/// @see mat(), SpMat, BasicMat::set()
template<class T = double>
[[nodiscard]] Mat<Sparse, T> spmat(std::size_t rows, std::size_t cols) {
  return Mat<Sparse, T>(rows, cols);
}

/// @brief Bind a numeric vector before matrix columns, like R's cbind(y, X).
/// Copies both inputs into dense storage, preserving the matrix scalar type.
/// dm::Vector<T> implicitly supplies its underlying standard vector.
/// @tparam S Sparse-storage selector of the input matrix.
/// @tparam T Floating-point matrix scalar.
/// @param first First column, with matrix.rows() elements.
/// @param matrix Remaining columns, unchanged by the copy.
/// @return Owned dense matrix with one additional first column.
/// @throws std::invalid_argument If row counts differ.
/// @throws std::length_error If the result dimensions overflow.
/// @par Example
/// @code
/// const auto A = dm::cbind(dm::c(1, 2), dm::Mat<>{{3}, {4}});
/// std::print("{:.0f} {:.0f}\n", A(0, 0), A(1, 1)); // 1 4
/// @endcode
/// @see BasicMat::colnames(), dstruct::DataFrame
template<bool S, class T>
BasicMat<false, T> cbind(
    const std::vector<typename BasicMat<S, T>::value_type>& first,
    const BasicMat<S, T>&                                   matrix) {
  if (first.size() != matrix.rows())
    throw std::invalid_argument("cbind: row counts differ");
  if (matrix.cols() == std::numeric_limits<std::size_t>::max())
    throw std::length_error("cbind: column count overflows");
  BasicMat<false, T> result(matrix.rows(), matrix.cols() + 1);
  for (std::size_t i = 0; i < matrix.rows(); ++i) {
    result.set(i, 0, first[i]);
    for (std::size_t j = 0; j < matrix.cols(); ++j)
      result.set(i, j + 1, matrix(i, j));
  }
  return result;
}

} // namespace datamunge
