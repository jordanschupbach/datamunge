#pragma once

/// @file vector.hpp
/// @brief Owned typed vectors and lazy numeric expressions with R-style selection and recycling.
///
/// Provides dm::Vector (an owned std::vector with R semantics), the R-style constructors
/// c(), seql(), factor(), element-wise arithmetic and math with recycling, summary
/// statistics (sum(), mean(), sd()), and seeded random sampling (runif(), rnorm()).
/// Arithmetic and element-wise math build read-only expression templates. Named Vector
/// leaves are borrowed and must outlive evaluation; temporary vectors and child nodes
/// are owned. Use Vector<> destination = expression or expression.eval() to keep a
/// snapshot. Random sampling is eager and happens once, even when its result is owned
/// by a lazy expression; subsequent evaluations reuse those sampled values.
/// Functions returning expressions must forward arguments or keep borrowed leaves
/// alive: returning an expression borrowing a by-value local Vector is invalid.
/// Vectors format with `std::print("{}", v)` (as `[1, 2, 3]`) and convert implicitly to
/// `const std::vector<T>&`, so they can be passed straight to the plotting API.
///
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// #include <print>
///
/// dm::set_seed(42);
/// const auto x = dm::unif(0, 1, 100);
/// const auto y = dm::sin(20 / (x + 0.25)) + dm::rnorm(0, 0.2, 100);
/// std::print("{} {}\n", dm::length(y), dm::round(dm::mean(x), 2));
/// const std::string svg = dm::Plot().points(x, y).to_svg();   // or dm::Plot(x, y)
/// @endcode

#include <datamunge/plot/plot.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

/// @brief Core library APIs and their algorithm, data, and numerical submodules.
namespace datamunge {

template<typename T = double> class Vector;

/// @brief ADL marker for native vector expressions, distinct from matrix expressions.
struct VectorExpressionTag {};

/// @brief Lifetime-aware vector expression closures and pointwise evaluation helpers.
namespace vector_expr {
/// @brief Whether an unqualified type is an owned Vector leaf.
/// @tparam E Unqualified candidate type.
template<class E> inline constexpr bool is_vector = false;
/// @brief Recognise owned Vector leaves for borrowing and temporary ownership.
/// @tparam T Stored element type.
template<class T> inline constexpr bool is_vector<Vector<T>> = true;

/// @brief Detect numeric vector expressions without accepting matrices or scalar-only calls.
/// @tparam E Candidate type, possibly cv/ref-qualified.
/// @tparam Enable Internal detection parameter.
template<class E, class Enable = void> struct Numeric : std::false_type {};
/// @brief Numeric vectors/nodes have the vector marker and double-convertible elements.
/// @tparam E Candidate type with a value_type member.
template<class E> struct Numeric<E, std::void_t<typename std::decay_t<E>::value_type>>
    : std::bool_constant<std::is_base_of_v<VectorExpressionTag, std::decay_t<E>> &&
                         std::is_convertible_v<typename std::decay_t<E>::value_type, double>> {};
/// @brief Whether E is a numeric vector leaf or deferred node.
/// @tparam E Candidate type.
template<class E> inline constexpr bool numeric = Numeric<E>::value;
/// @brief Whether E is a deferred numeric node rather than owned storage.
/// @tparam E Candidate type.
template<class E> inline constexpr bool is_node = numeric<E> && !is_vector<std::decay_t<E>>;
/// @brief Enable arithmetic only for vector expressions paired with numeric expressions/scalars.
/// @tparam L Left operand type.
/// @tparam R Right operand type.
template<class L, class R> inline constexpr bool binary_enabled =
    (numeric<L> || numeric<R>) && (numeric<L> || std::is_convertible_v<L, double>) &&
    (numeric<R> || std::is_convertible_v<R, double>);

/// @brief Read an owned operand closure.
/// @tparam E Closure type.
/// @param value Owned operand.
/// @return Borrowed const operand.
template<class E> const E& operand(const E& value) { return value; }
/// @brief Read a borrowed Vector leaf.
/// @tparam E Leaf type.
/// @param value Reference closure whose referent must remain alive.
/// @return Borrowed const leaf.
template<class E> const E& operand(const std::reference_wrapper<const E>& value) { return value.get(); }

/// @brief Read-only input iterator computing a node's elements on dereference.
/// @tparam E Expression type whose lifetime must exceed the iterator's.
template<class E> class Iterator {
  const E* owner_ = nullptr; ///< Expression containing all operand closures.
  std::size_t index_ = 0; ///< Zero-based position.
public:
  using value_type = double; ///< Numeric expressions evaluate to double, as the eager API did.
  using reference = double; ///< Each dereference computes a value, not a mutable reference.
  using pointer = void; ///< Expression values have no stable element address.
  using difference_type = std::ptrdiff_t; ///< Standard iterator difference type.
  using iterator_category = std::input_iterator_tag; ///< Single-pass readable range contract.

  /// @brief Construct a singular iterator.
  Iterator() = default;
  /// @brief Construct an iterator into a live expression.
  /// @param owner Expression borrowed for the iterator's lifetime.
  /// @param index Initial zero-based position, including size() for the end.
  Iterator(const E* owner, std::size_t index) : owner_(owner), index_(index) {}
  /// @brief Compute the current element; the iterator must not equal end().
  /// @return Current numeric value.
  double operator*() const { return (*owner_)[index_]; }
  /// @brief Advance by one position.
  /// @return This iterator.
  Iterator& operator++() { ++index_; return *this; }
  /// @brief Advance, returning the previous position.
  /// @return Previous iterator.
  Iterator operator++(int) { auto copy = *this; ++*this; return copy; }
  /// @brief Compare expression identity and position.
  /// @param other Iterator to compare.
  /// @return True for the same expression and index.
  bool operator==(const Iterator& other) const { return owner_ == other.owner_ && index_ == other.index_; }
  /// @brief Compare expression identity and position for inequality.
  /// @param other Iterator to compare.
  /// @return True if identity or position differs.
  bool operator!=(const Iterator& other) const { return !(*this == other); }
};

/// @brief Shared read-only range and materialisation API for lazy numeric nodes.
/// Lvalue Vector leaves are borrowed; temporary Vectors and all child nodes are
/// owned. Borrowed leaves must outlive evaluation. Reads see their current values
/// and lengths. Copies of a node copy any temporary vectors that it owns.
/// @tparam D Concrete node type.
template<class D> class Expression : public VectorExpressionTag {
  /// @brief Access the concrete node.
  /// @return Borrowed const node.
  const D& derived() const { return static_cast<const D&>(*this); }
public:
  using value_type = double; ///< Preserve the numeric result type of the existing arithmetic API.

  /// @brief Whether the current expression has zero length.
  /// @return True when size() is zero.
  bool empty() const { return derived().size() == 0; }
  /// @brief Begin read-only iteration without materialising storage.
  /// @return Iterator at the first element.
  Iterator<D> begin() const { return {&derived(), 0}; }
  /// @brief End read-only iteration without materialising storage.
  /// @return Iterator one past the current result length.
  Iterator<D> end() const { return {&derived(), derived().size()}; }
  /// @brief Compute one owned snapshot in a single pointwise pass.
  /// @return Owned numeric Vector; later operand mutations do not affect it.
  /// @par Example
  /// @code
  /// auto x = dm::c(1, 2);
  /// const auto lazy = x + 1;
  /// const auto snapshot = lazy.eval();
  /// x[0] = 10;
  /// std::print("{} {}\n", lazy[0], snapshot[0]); // 11 2
  /// @endcode
  Vector<double> eval() const;
  /// @brief Materialise standard storage for APIs requiring std::vector<double>.
  /// @return Owned values, recomputed on each call; no cache or operand mutation.
  /// @par Example
  /// @code
  /// const auto expression = dm::c(1, 2) + 1;
  /// const std::vector<double> values = expression.values();
  /// std::print("{}\n", values); // [2, 3]
  /// @endcode
  std::vector<double> values() const {
    std::vector<double> result;
    const auto n = derived().size(); result.reserve(n);
    for (std::size_t i = 0; i < n; ++i) result.push_back(derived()[i]);
    return result;
  }
  /// @brief Materialise once at a standard-vector API boundary.
  /// @return Owned standard storage, with lifetime extended by a receiving const reference.
  /// @par Example
  /// @code
  /// const std::vector<double> values = dm::sin(dm::c(0, 0));
  /// std::print("{}\n", values); // [0, 0]
  /// @endcode
  operator std::vector<double>() const { return values(); }
  /// @brief Materialise before R-style selection.
  /// @tparam I Index/mask element type supported by Vector::select().
  /// @param indices One-based positions, exclusions or logical mask.
  /// @return Owned selection snapshot.
  template<class I> auto select(const std::vector<I>& indices) const;
  /// @brief Materialise before selection by a native index/mask vector.
  /// @tparam I Index/mask element type.
  /// @param indices One-based indices or logical mask.
  /// @return Owned selection snapshot.
  template<class I> auto select(const Vector<I>& indices) const;
  /// @brief Materialise before selection by deferred numeric indices.
  /// @tparam E Deferred numeric index expression.
  /// @param indices One-based positions or exclusions.
  /// @return Owned selection snapshot.
  template<class E, std::enable_if_t<is_node<E>, int> = 0>
  auto select(const E& indices) const;
  /// @brief Materialise before selection by an index list.
  /// @param indices One-based positions or exclusions.
  /// @return Owned selection snapshot.
  Vector<double> select(std::initializer_list<std::ptrdiff_t> indices) const;
  /// @brief Materialise before recycling to an explicit length.
  /// @param n Requested result length.
  /// @return Owned recycled snapshot.
  /// @throws std::invalid_argument For nonzero n and an empty source.
  Vector<double> recycled(std::size_t n) const;
};

/// @brief Owned scalar broadcast as a length-one operand.
struct Scalar {
  double value; ///< Scalar snapshot, never a borrowed scalar reference.

  /// @brief Return the broadcast operand length.
  /// @return One.
  std::size_t size() const { return 1; }
  /// @brief Read the constant scalar.
  /// @param index Unused recycled index.
  /// @return Stored value.
  double operator[](std::size_t index) const { (void)index; return value; }
};

/// @brief Borrow lvalue leaves; own temporary leaves, child expressions and scalar values.
/// @tparam E Forwarded operand type.
/// @param value Operand whose temporary ownership is transferred when possible.
/// @return Typed lifetime closure.
template<class E> auto capture(E&& value) {
  if constexpr (is_vector<std::decay_t<E>> && std::is_lvalue_reference_v<E>) return std::cref(value);
  else if constexpr (numeric<E>) return std::decay_t<E>(std::forward<E>(value));
  else return Scalar{static_cast<double>(std::forward<E>(value))};
}

/// @brief Deferred unary numeric operation.
/// @tparam E Captured operand closure.
/// @tparam F Pointwise numeric operation.
template<class E, class F> class Unary : public Expression<Unary<E, F>> {
  E source_; ///< Borrowed leaf or owned child.
  F operation_; ///< Owned numeric operation and any scalar parameters.
public:
  /// @brief Retain an operand and operation without evaluating elements.
  /// @param source Lifetime closure.
  /// @param operation Numeric operation.
  Unary(E source, F operation) : source_(std::move(source)), operation_(std::move(operation)) {}
  /// @brief Read the current source length.
  /// @return Source length, without invoking the numeric operation.
  std::size_t size() const { return operand(source_).size(); }
  /// @brief Compute one element; index must be below size().
  /// @param index Zero-based index.
  /// @return Transformed source value.
  double operator[](std::size_t index) const { return operation_(static_cast<double>(operand(source_)[index])); }
};

/// @brief Deferred binary operation retaining recycling at each expression node.
/// @tparam L Left closure type.
/// @tparam R Right closure type.
/// @tparam F Binary numeric operation.
template<class L, class R, class F> class Binary : public Expression<Binary<L, R, F>> {
  L left_; ///< Left lifetime closure.
  R right_; ///< Right lifetime closure.
  F operation_; ///< Owned operation.
public:
  /// @brief Retain operands without computing their elements.
  /// @param left Left lifetime closure.
  /// @param right Right lifetime closure.
  /// @param operation Numeric operation.
  Binary(L left, R right, F operation) : left_(std::move(left)), right_(std::move(right)), operation_(std::move(operation)) {}
  /// @brief Recompute the recycled result length from current operand lengths.
  /// @return Zero for any empty operand; otherwise the longer length.
  std::size_t size() const {
    const auto l = operand(left_).size(), r = operand(right_).size();
    return l == 0 || r == 0 ? 0 : std::max(l, r);
  }
  /// @brief Compute a recycled pair; index must be below size().
  /// @param index Zero-based result index.
  /// @return Operation applied to each child's own recycled value.
  double operator[](std::size_t index) const {
    const auto& l = operand(left_); const auto& r = operand(right_);
    return operation_(static_cast<double>(l[index % l.size()]), static_cast<double>(r[index % r.size()]));
  }
};

/// @brief Construct a unary node using lifetime-safe operand capture.
/// @tparam E Operand type.
/// @tparam F Numeric operation type.
/// @param value Forwarded operand.
/// @param operation Owned pointwise operation.
/// @return Deferred unary expression.
template<class E, class F> auto unary(E&& value, F operation) {
  auto source = capture(std::forward<E>(value));
  return Unary<decltype(source), F>(std::move(source), std::move(operation));
}
/// @brief Construct a binary node using lifetime-safe operand capture.
/// @tparam L Left operand type.
/// @tparam R Right operand type.
/// @tparam F Numeric operation type.
/// @param left Forwarded left operand.
/// @param right Forwarded right operand.
/// @param operation Owned numeric operation.
/// @return Deferred binary expression.
template<class L, class R, class F> auto binary(L&& left, R&& right, F operation) {
  auto l = capture(std::forward<L>(left)); auto r = capture(std::forward<R>(right));
  return Binary<decltype(l), decltype(r), F>(std::move(l), std::move(r), std::move(operation));
}

/// @brief Sum numeric values in left-to-right order without intermediate arrays.
/// @tparam E Numeric vector or node.
/// @param values Read-only numeric input.
/// @return Sum, or zero for empty input.
template<class E> double sum_values(const E& values) {
  double result = 0; const auto n = values.size();
  for (std::size_t i = 0; i < n; ++i) result += static_cast<double>(values[i]);
  return result;
}
/// @brief Welford sample deviation shared by owned and deferred numeric inputs.
/// @tparam E Numeric vector or node.
/// @param values Read-only numeric input.
/// @return Sample deviation, or NaN for fewer than two/nonfinite values.
template<class E> double sample_sd(const E& values) {
  const auto n = values.size();
  if (n < 2) return std::numeric_limits<double>::quiet_NaN();
  long double center = 0, squared_deviations = 0; std::size_t count = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const double value = static_cast<double>(values[i]);
    if (!std::isfinite(value)) return std::numeric_limits<double>::quiet_NaN();
    const long double delta = static_cast<long double>(value) - center;
    center += delta / ++count;
    squared_deviations += delta * (static_cast<long double>(value) - center);
  }
  return static_cast<double>(std::sqrt(squared_deviations / (n - 1)));
}
} // namespace vector_expr

/// @brief An owned std::vector<T> with R-style selection and recycling (R's atomic vector).
///
/// Numeric binary operations recycle the shorter operand to the longer length,
/// including partial cycles without a warning. An empty operand gives an empty result.
/// Storage access and operator[] use C++ zero-based indexing; select() is one-based.
/// Concurrent reads are safe; callers synchronize mutation of a shared instance.
///
/// Vectors are usually built with dm::c(), dm::seql(), dm::runif() or dm::rnorm() rather
/// than constructed directly. They are ranges (begin()/end()), so `std::print("{}", v)`
/// prints `[a, b, c]` (strings quoted), and they convert implicitly to
/// `const std::vector<T>&` for APIs such as dm::Plot. Free arithmetic and math
/// functions build lazy double-valued expressions for numeric inputs. `auto y =
/// x + 1` retains a read-only expression; `Vector<> y = x + 1` computes an owned,
/// mutable snapshot. Compound assignments remain eager and alias-safe.
///
/// @tparam T  Element type; defaults to double for existing numeric workflows.
///
/// @par Example
/// @code
/// dm::Vector<double> x{1.5, 2.5, 3.5};
/// x[0] = 10;                                     // zero-based storage access
/// std::print("{} {}\n", x, x.size());            // [10, 2.5, 3.5] 3
/// std::print("{}\n", x.select({1, 3}));          // [10, 3.5]   (one-based)
/// std::print("{}\n", x * 2.0);                   // [20, 5, 7]
/// @endcode
/// @see dm::c(), dm::length(), dm::apply()
template <typename T>
class Vector : public VectorExpressionTag {
 public:
  /// @brief Type of each owned element.
  using value_type = T;
  /// @brief Constructs an empty vector.
  Vector() = default;
  /// @brief Constructs @p n copies of @p value, like R's `rep(value, n)`.
  ///
  /// @param n  Number of elements.
  /// @param value  Value copied into every element (value-initialized `T{}` by default).
  ///
  /// @par Example
  /// @code
  /// dm::Vector<int> z(3, 7);
  /// std::print("{}\n", z);                       // [7, 7, 7]
  /// @endcode
  explicit Vector(std::size_t n, T value = T{}) : data_(n, value) {}
  /// @brief Copies the supplied elements into owned storage.
  ///
  /// Class template argument deduction turns brace-initialized arithmetic values into a
  /// `Vector<double>` (see the deduction guide), so `dm::Vector v = {1, 2, 3};` is numeric.
  ///
  /// @param values  Elements to copy, in order.
  ///
  /// @par Example
  /// @code
  /// const dm::Vector v = {1, 2, 3};               // Vector<double>
  /// std::print("{}\n", v * 2.0);                 // [2, 4, 6]
  /// @endcode
  Vector(std::initializer_list<T> values) : data_(values) {}
  /// @brief Takes ownership of supplied storage (copies lvalues and moves rvalues).
  ///
  /// @param values  Storage to adopt; pass an rvalue (or std::move) to avoid a copy.
  ///
  /// @par Example
  /// @code
  /// dm::Vector w(std::vector<std::string>{"a", "b"});   // Vector<std::string>
  /// std::print("{}\n", w);                              // ["a", "b"]
  /// @endcode
  Vector(std::vector<T> values) : data_(std::move(values)) {}
  /// @brief Copies elements from a vector whose element type converts implicitly to T.
  ///
  /// This is what lets a `Vector<int>` (e.g. a dm::factor() result) be passed to the
  /// `Vector<double>` arithmetic operators and math functions.
  ///
  /// @tparam U  Source element type, implicitly convertible to T.
  /// @param values  Source vector; it is not modified.
  ///
  /// @par Example
  /// @code
  /// dm::Vector<double> d = dm::Vector<int>(2, 4);
  /// std::print("{}\n", d);                       // [4, 4]
  /// @endcode
  template <typename U, std::enable_if_t<std::is_convertible_v<U, T>, int> = 0>
  Vector(const Vector<U>& values) : data_(values.begin(), values.end()) {}

  /// @brief Evaluate a lazy numeric expression once into owned contiguous storage.
  /// Computes each output element in a single fused pass. Copies values into T;
  /// the default numeric result type is double. No expression references are retained.
  /// @tparam E Deferred numeric expression with elements convertible to T.
  /// @param expression Expression whose borrowed vector leaves must still exist.
  /// @par Example
  /// @code
  /// const auto x = dm::c(1, 2, 3);
  /// const dm::Vector<> values = x * 2 + 1;
  /// dm::Vector inferred = dm::sin(x * 0);
  /// std::print("{} {}\n", values, inferred); // [3, 5, 7] [0, 0, 0]
  /// @endcode
  template<class E, std::enable_if_t<vector_expr::is_node<E> &&
      std::is_convertible_v<typename std::decay_t<E>::value_type, T>, int> = 0>
  Vector(const E& expression) {
    const auto n = expression.size();
    data_.reserve(n);
    for (std::size_t i = 0; i < n; ++i) data_.push_back(static_cast<T>(expression[i]));
  }

  /// @brief Evaluate before replacing storage, including aliased/recycled assignments.
  /// Leaves this vector unchanged if evaluation throws.
  /// @tparam E Numeric expression with elements convertible to T.
  /// @param expression Deferred input, possibly borrowing this vector.
  /// @return This owned vector after evaluation.
  /// @par Example
  /// @code
  /// auto x = dm::c(1, 2);
  /// x = x + dm::c(10, 20, 30);
  /// std::print("{}\n", x); // [11, 22, 31]
  /// @endcode
  template<class E, std::enable_if_t<vector_expr::is_node<E> &&
      std::is_convertible_v<typename std::decay_t<E>::value_type, T>, int> = 0>
  Vector& operator=(const E& expression) {
    Vector result(expression);
    data_.swap(result.data_);
    return *this;
  }

  /// @brief Copy an owned vector snapshot, matching lazy expressions' eval() spelling.
  /// @return Independent Vector<T> copy.
  /// @par Example
  /// @code
  /// auto x = dm::c(1, 2);
  /// const auto snapshot = x.eval();
  /// x[0] = 3;
  /// std::print("{}\n", snapshot); // [1, 2]
  /// @endcode
  Vector eval() const { return *this; }

  /// @brief Returns the number of elements (see also dm::length()).
  std::size_t size() const noexcept { return data_.size(); }
  /// @brief Returns whether storage is empty.
  bool empty() const noexcept { return data_.empty(); }
  /// @brief Returns a mutable element (a proxy for bool); @p i must be below size().
  ///
  /// Indexing is zero-based and unchecked, unlike the one-based select().
  /// @param i  Zero-based position, less than size().
  /// @return  A reference to element @p i (std::vector<bool>::reference for bool).
  typename std::vector<T>::reference operator[](std::size_t i) { return data_[i]; }
  /// @brief Returns a read-only element; @p i must be below size().
  /// @param i  Zero-based position, less than size().
  /// @return  A const reference to element @p i (a bool value for packed bool storage).
  typename std::vector<T>::const_reference operator[](std::size_t i) const { return data_[i]; }
  /// @brief Returns a mutable iterator to the first element.
  auto begin() noexcept { return data_.begin(); }
  /// @brief Returns a mutable iterator one past the last element.
  auto end() noexcept { return data_.end(); }
  /// @brief Returns a read-only iterator to the first element.
  auto begin() const noexcept { return data_.begin(); }
  /// @brief Returns a read-only iterator one past the last element.
  auto end() const noexcept { return data_.end(); }
  /// @brief Borrows storage for this object's lifetime; mutation may invalidate element references.
  ///
  /// @return  A const reference to the backing std::vector; it dangles once this Vector is
  ///          destroyed, so do not call it on a temporary and keep the reference.
  ///
  /// @par Example
  /// @code
  /// const auto x = dm::c(1, 2, 3);
  /// const std::vector<double>& raw = x.values();
  /// std::print("{}\n", raw.size());              // 3
  /// @endcode
  const std::vector<T>& values() const noexcept { return data_; }
  /// @brief Borrows the backing std::vector with the same lifetime as values().
  /// @return  A const reference to the owned storage.
  operator const std::vector<T>&() const noexcept { return data_; }

  /// @brief Selects a copy using one-based numeric indices or a recycled logical mask (R's `x[i]`).
  ///
  /// Positive indices retain order and duplicates; zero is omitted. Negative indices
  /// exclude positions and may only be mixed with zero. Fractional indices truncate
  /// toward zero. Negative indices beyond size() are ignored. Logical masks recycle
  /// to size(); a longer mask may only select positions that exist. An empty index
  /// list returns an empty vector.
  ///
  /// A common use is mapping a dm::factor() code vector through a palette to get one
  /// color per observation.
  ///
  /// @tparam I  Arithmetic index type, or bool for a logical mask.
  /// @param indices  One-based positions (positive to keep, negative to drop) or a mask.
  /// @return  A new owned vector; this vector is unchanged.
  /// @throws std::invalid_argument For non-finite indices or mixed positive/negative indices.
  /// @throws std::out_of_range For a selected position beyond size() (there is no NA sentinel).
  ///
  /// @par Example
  /// @code
  /// const auto x = dm::c(10, 20, 30, 40);
  /// std::print("{}\n", x.select(std::vector<double>{1.9, 0, 3}));   // [10, 30]
  /// std::print("{}\n", x.select(std::vector<bool>{true, false}));   // [10, 30]
  ///
  /// const auto palette = dm::c("red", "green", "blue");
  /// const auto codes = dm::factor(dm::c("b", "a", "b", "c"));        // [2, 1, 2, 3]
  /// std::print("{}\n", palette.select(codes));   // ["green", "red", "green", "blue"]
  /// @endcode
  /// @see recycled(), dm::factor()
  template <typename I>
  Vector select(const std::vector<I>& indices) const {
    static_assert(std::is_arithmetic_v<I>, "selection requires numeric indices or logical values");
    std::vector<T> result;
    if constexpr (std::is_same_v<I, bool>) {
      if (indices.empty()) return Vector{};
      for (std::size_t i = 0; i < std::max(size(), indices.size()); ++i) {
        if (!indices[i % indices.size()]) continue;
        if (i >= size()) throw std::out_of_range("Vector::select: logical index beyond size");
        result.push_back(data_[i]);
      }
    } else {
      bool positive = false, negative = false;
      for (const auto index : indices) {
        const long double value = std::trunc(static_cast<long double>(index));
        if (!std::isfinite(value)) throw std::invalid_argument("Vector::select: non-finite index");
        positive = positive || value > 0;
        negative = negative || value < 0;
      }
      if (positive && negative) throw std::invalid_argument("Vector::select: mixed index signs");
      if (negative) {
        std::vector<bool> excluded(size(), false);
        for (const auto index : indices) {
          const long double value = -std::trunc(static_cast<long double>(index));
          if (value >= 1 && value <= static_cast<long double>(size()))
            excluded[static_cast<std::size_t>(value) - 1] = true;
        }
        for (std::size_t i = 0; i < size(); ++i) if (!excluded[i]) result.push_back(data_[i]);
      } else {
        for (const auto index : indices) {
          const long double value = std::trunc(static_cast<long double>(index));
          if (value == 0) continue;
          if (value > static_cast<long double>(size()))
            throw std::out_of_range("Vector::select: index beyond size");
          result.push_back(data_[static_cast<std::size_t>(value) - 1]);
        }
      }
    }
    return Vector(std::move(result));
  }
  /// @brief Selects using indices owned by another Vector; see the std::vector overload.
  ///
  /// Same rules as select(const std::vector<I>&); a `Vector<bool>` (e.g. from
  /// `dm::c(true, false)`) acts as a logical mask.
  ///
  /// @tparam I  Arithmetic index type, or bool for a logical mask.
  /// @param indices  One-based positions or a logical mask.
  /// @return  A new owned vector; this vector is unchanged.
  /// @throws std::invalid_argument For non-finite indices or mixed positive/negative indices.
  /// @throws std::out_of_range For a selected position beyond size().
  ///
  /// @par Example
  /// @code
  /// const auto x = dm::c(10, 20, 30, 40);
  /// std::print("{}\n", x.select(dm::c(true, false, true, true)));   // [10, 30, 40]
  /// std::print("{}\n", x.select(dm::c(4, 1)));                      // [40, 10]
  /// @endcode
  template <typename I>
  Vector select(const Vector<I>& indices) const { return select(indices.values()); }

  /// @brief Materialise deferred numeric indices before one-based selection.
  /// @tparam E Numeric index expression.
  /// @param indices One-based positions or exclusions, evaluated once.
  /// @return Owned selected values of this vector's element type.
  /// @throws std::invalid_argument For nonfinite or mixed-sign indices.
  /// @throws std::out_of_range For a selected position beyond size().
  /// @par Example
  /// @code
  /// const auto labels = dm::c("a", "b", "c");
  /// std::print("{}\n", labels.select(dm::c(0, 2) + 1)); // ["a", "c"]
  /// @endcode
  template<class E, std::enable_if_t<vector_expr::is_node<E>, int> = 0>
  Vector select(const E& indices) const { return select(indices.values()); }
  /// @brief Selects one-based integral positions supplied in braces.
  ///
  /// Same rules as select(const std::vector<I>&).
  ///
  /// @param indices  One-based positions; all non-negative or all non-positive.
  /// @return  A new owned vector; this vector is unchanged.
  /// @throws std::invalid_argument For mixed positive/negative indices.
  /// @throws std::out_of_range For a selected position beyond size().
  ///
  /// @par Example
  /// @code
  /// const auto x = dm::c(10, 20, 30, 40);
  /// std::print("{}\n", x.select({2, 2, 4}));     // [20, 20, 40]
  /// std::print("{}\n", x.select({-1, -3}));      // [20, 40]   (drop 1st and 3rd)
  /// @endcode
  Vector select(std::initializer_list<std::ptrdiff_t> indices) const {
    return select(std::vector<std::ptrdiff_t>(indices));
  }
  /// @brief Repeats elements to exactly @p n entries, truncating the final cycle (R's `rep_len()`).
  ///
  /// @param n  Requested length; zero returns an empty vector.
  /// @return  A new owned vector of length @p n.
  /// @throws std::invalid_argument If a nonempty result is requested from empty storage.
  ///
  /// @par Example
  /// @code
  /// std::print("{}\n", dm::c(1, 2, 3).recycled(7));   // [1, 2, 3, 1, 2, 3, 1]
  /// @endcode
  /// @see select()
  Vector recycled(std::size_t n) const {
    if (empty() && n != 0) throw std::invalid_argument("Vector::recycled: empty source");
    std::vector<T> result;
    result.reserve(n);
    for (std::size_t i = 0; i < n; ++i) result.push_back(data_[i % size()]);
    return Vector(std::move(result));
  }

  /// @brief Adds rhs in place, recycling both inputs to the longer length; empty inputs give empty.
  ///
  /// The result length is the longer of the two lengths (this vector may grow), or zero
  /// if either is empty. Safe when @p rhs aliases `*this` (e.g. `x += x`).
  ///
  /// @param rhs  Vector added element-wise, recycled as needed.
  /// @return  `*this`.
  ///
  /// @par Example
  /// @code
  /// auto x = dm::c(1, 2, 3, 4);
  /// x += dm::c(10, 20);
  /// std::print("{}\n", x);                       // [11, 22, 13, 24]
  /// @endcode
  /// @see dm::operator+(Vector<>, const Vector<>&)
  Vector& operator+=(const Vector& rhs) { return combine(rhs, [](T a, T b) { return a + b; }); }
  /// @brief Subtracts rhs in place with the recycling rules of operator+=.
  /// @param rhs  Vector subtracted element-wise, recycled as needed.
  /// @return  `*this`.
  Vector& operator-=(const Vector& rhs) { return combine(rhs, [](T a, T b) { return a - b; }); }
  /// @brief Multiplies by rhs in place with the recycling rules of operator+=.
  ///
  /// @param rhs  Vector multiplied element-wise, recycled as needed.
  /// @return  `*this`.
  ///
  /// @par Example
  /// @code
  /// auto x = dm::c(1, 2, 3, 4);
  /// x *= dm::c(2, 0.5);
  /// std::print("{}\n", x);                       // [2, 1, 6, 2]
  /// @endcode
  Vector& operator*=(const Vector& rhs) { return combine(rhs, [](T a, T b) { return a * b; }); }
  /// @brief Divides by rhs in place with recycling; numeric zero divisors are not checked.
  /// @param rhs  Vector of divisors, recycled as needed (zero gives inf/NaN for doubles).
  /// @return  `*this`.
  Vector& operator/=(const Vector& rhs) { return combine(rhs, [](T a, T b) { return a / b; }); }
  /// @brief Adds a scalar to every element in place.
  ///
  /// The arithmetic happens in the element type, so for an integral `T` the result is
  /// truncated (`Vector<int>{1, 2} += 1.5` gives `[2, 3]`).
  ///
  /// @param rhs  Scalar added to each element.
  /// @return  `*this`.
  ///
  /// @par Example
  /// @code
  /// auto x = dm::c(1, 2, 3);
  /// x += 1;
  /// x *= 10;
  /// std::print("{}\n", x);                       // [20, 30, 40]
  /// @endcode
  Vector& operator+=(double rhs) { for (std::size_t i = 0; i < size(); ++i) data_[i] += rhs; return *this; }
  /// @brief Subtracts a scalar from every element in place.
  /// @param rhs  Scalar subtracted from each element.
  /// @return  `*this`.
  Vector& operator-=(double rhs) { for (std::size_t i = 0; i < size(); ++i) data_[i] -= rhs; return *this; }
  /// @brief Multiplies every element by a scalar in place.
  /// @param rhs  Scalar multiplier.
  /// @return  `*this`.
  Vector& operator*=(double rhs) { for (std::size_t i = 0; i < size(); ++i) data_[i] *= rhs; return *this; }
  /// @brief Divides every element by a scalar in place; numeric zero divisors are not checked.
  /// @param rhs  Scalar divisor.
  /// @return  `*this`.
  Vector& operator/=(double rhs) { for (std::size_t i = 0; i < size(); ++i) data_[i] /= rhs; return *this; }

 private:
  /// @brief Computes into fresh storage so recycling remains safe even when rhs aliases this vector.
  /// @tparam F  Binary callable `T(T, T)` applied to each recycled pair.
  /// @param rhs  Right-hand operand, recycled to the longer length.
  /// @param operation  Element operation (lhs element, rhs element).
  /// @return  `*this`, holding max(size(), rhs.size()) results, or none if either is empty.
  template <typename F>
  Vector& combine(const Vector& rhs, F operation) {
    const std::size_t n = empty() || rhs.empty() ? 0 : std::max(size(), rhs.size());
    std::vector<T> result;
    result.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
      result.push_back(operation(data_[i % size()], rhs[i % rhs.size()]));
    data_ = std::move(result);
    return *this;
  }
  /// @brief Owned contiguous elements (packed bits for bool).
  std::vector<T> data_;
};

namespace vector_expr {
template<class D> Vector<double> Expression<D>::eval() const { return Vector<double>(derived()); }
template<class D> template<class I>
auto Expression<D>::select(const std::vector<I>& indices) const { return eval().select(indices); }
template<class D> template<class I>
auto Expression<D>::select(const Vector<I>& indices) const { return eval().select(indices); }
template<class D> template<class E, std::enable_if_t<is_node<E>, int>>
auto Expression<D>::select(const E& indices) const { return eval().select(indices); }
template<class D>
Vector<double> Expression<D>::select(std::initializer_list<std::ptrdiff_t> indices) const { return eval().select(indices); }
template<class D> Vector<double> Expression<D>::recycled(std::size_t n) const { return eval().recycled(n); }
} // namespace vector_expr

/// @brief Deduce an owned double Vector from a lazy numeric expression.
/// @tparam E Deferred node type.
/// @par Example
/// See Vector's expression constructor.
template<class E, std::enable_if_t<vector_expr::is_node<E>, int> = 0>
Vector(const E&) -> Vector<double>;

/// @brief Read a deferred vector's current length without evaluating elements.
/// @tparam E Deferred numeric expression.
/// @param values Expression whose leaves remain alive.
/// @return Current recycled result length.
/// @par Example
/// @code
/// std::print("{}\n", dm::length(dm::c(1, 2) + 1)); // 2
/// @endcode
template<class E, std::enable_if_t<vector_expr::is_node<E>, int> = 0>
std::size_t length(const E& values) { return values.size(); }

/// @brief Deduces element types when constructing from owned standard storage.
template <typename T> Vector(std::vector<T>) -> Vector<T>;
/// @brief Keeps brace-initialized numeric vectors double-valued for existing arithmetic APIs.
template <typename T> Vector(std::initializer_list<T>) -> Vector<std::conditional_t<std::is_arithmetic_v<T> && !std::is_same_v<T, bool>, double, T>>;

/// @brief Returns the number of elements in a vector, including zero for an empty vector (R's `length()`).
///
/// @tparam T  Element type.
/// @param values  Vector to measure.
/// @return  values.size().
///
/// @par Example
/// @code
/// std::print("{} {}\n", dm::length(dm::c(1, 2, 3)), dm::length(dm::c()));   // 3 0
/// @endcode
/// @see Vector::size()
template <typename T>
std::size_t length(const Vector<T>& values) noexcept { return values.size(); }

/// @brief Owns a callable for use with apply(), preserving generic lambdas and captured state.
///
/// This mirrors R's `function(x) ...` spelling; it simply returns a decayed copy of the
/// callable. Lvalues are copied and rvalues are moved; no type erasure or invocation occurs.
/// Reference captures still borrow their referents, which must outlive invocation.
/// The result can be called directly or passed to dm::apply().
///
/// @tparam F  Any callable type (lambda, function object, function pointer).
/// @param callable  The callable to store.
/// @return  A `std::decay_t<F>` copy of @p callable.
///
/// @par Example
/// @code
/// const auto square = dm::function([](auto v) { return v * v; });
/// std::print("{}\n", square(1.5));                     // 2.25
/// std::print("{}\n", dm::apply(dm::c(1, 2, 3), square));   // [1, 4, 9]
/// @endcode
/// @see dm::apply()
template <typename F>
std::decay_t<F> function(F&& callable) {
  return std::forward<F>(callable);
}

/// @brief Calls a function once per element in order and returns an owned vector of results (R's `sapply()`).
///
/// The callable receives read-only elements; the input is unchanged. Result references
/// are copied into owned values. Empty input returns an empty vector without invoking
/// the callable. Captures may be mutated, and callable exceptions propagate to the caller.
/// The result element type is whatever the callable returns, so apply() can change type.
///
/// @tparam T Input element type.
/// @tparam F Callable accepting a const element reference (a value for packed bool storage).
/// @param values  Input vector.
/// @param callable  Function applied to each element, in order.
/// @return  A `Vector<R>` of the same length, where R is the decayed callable result type.
///
/// @par Example
/// @code
/// const auto x = dm::c(0.1, 0.5);
/// const auto y = dm::apply(x, dm::function([](auto v) { return dm::round(v * 10, 0); }));
/// std::print("{}\n", y);                                   // [1, 5]
/// const auto lens = dm::apply(dm::c("a", "bb"), [](const std::string& s) { return s.size(); });
/// std::print("{}\n", lens);                                // [1, 2]
/// @endcode
/// @see dm::function()
template <typename T, typename F>
auto apply(const Vector<T>& values, F&& callable) {
  using Result = std::decay_t<std::invoke_result_t<F&, typename std::vector<T>::const_reference>>;
  std::vector<Result> result;
  result.reserve(values.size());
  for (const auto& value : values) result.push_back(std::invoke(callable, value));
  return Vector<Result>(std::move(result));
}

/// @brief Materialise a numeric expression before applying an arbitrary eager callback.
/// Preserves apply()'s side-effect order: input values are computed first, then the
/// callback runs once per element. Generic vector arithmetic should call a vector-
/// aware function directly (for example doppler(x)) to retain pointwise fusion.
/// @tparam E Deferred numeric node.
/// @tparam F Callable accepted by apply(Vector, callable).
/// @param values Deferred input, evaluated once before the callback.
/// @param callable Function applied to each materialised value.
/// @return Owned vector of callback results.
/// @par Example
/// @code
/// const auto values = dm::apply(dm::c(1, 2) + 1, [](double x) { return x * x; });
/// std::print("{}\n", values); // [4, 9]
/// @endcode
template<class E, class F, std::enable_if_t<vector_expr::is_node<E>, int> = 0>
auto apply(const E& values, F&& callable) {
  return apply(values.eval(), std::forward<F>(callable));
}

/// @brief Internal adapters for R-style concatenation and coercion.
namespace vector_detail {
/// @brief Converts numeric scalars to locale-independent, compact text for mixed c() calls.
/// @param value  Number to format (shortest round-trip form; NaN/Inf/-Inf spelled like R).
/// @return  The text, e.g. "1", "2.5", "1e+20", "Inf".
std::string scalar_text(double value);
/// @brief Accepts numeric, logical, and string elements in a heterogeneous initializer list.
struct StringValue {
  /// @brief Owned textual representation of the supplied scalar.
  std::string value;
  /// @brief Copies or moves a string into the combined vector.
  /// @param text  String stored as-is.
  StringValue(std::string text) : value(std::move(text)) {}
  /// @brief Copies a non-null C string into owned storage.
  /// @param text  Null-terminated string; must not be null.
  StringValue(const char* text) : value(text) {}
  /// @brief Formats logical values as R's TRUE and FALSE.
  /// @param logical  Value converted to "TRUE" or "FALSE".
  StringValue(bool logical) : value(logical ? "TRUE" : "FALSE") {}
  /// @brief Formats an arithmetic scalar without a trailing decimal for integral values.
  /// @tparam N  Arithmetic, non-bool type.
  /// @param number  Value formatted with scalar_text().
  template <typename N, std::enable_if_t<std::is_arithmetic_v<N> && !std::is_same_v<N, bool>, int> = 0>
  StringValue(N number) : value(scalar_text(static_cast<double>(number))) {}
};
/// @brief Selects owned strings, doubles, or the original generic element type for c().
template <typename T>
using combined_type = std::conditional_t<std::is_convertible_v<T, std::string>, std::string,
                      std::conditional_t<std::is_arithmetic_v<T> && !std::is_same_v<T, bool>, double, T>>;
} // namespace vector_detail

/// @brief Combines a homogeneous list; text is owned, arithmetic becomes double, bool stays logical.
///
/// Brace form of dm::c(). Element types convertible to std::string become
/// `Vector<std::string>`, arithmetic types become `Vector<double>`, bool stays
/// `Vector<bool>`, and any other type (e.g. dm::RGB) is kept as-is.
///
/// @tparam T  Common element type deduced from the braces.
/// @param values  Elements to copy, in order.
/// @return  An owned vector of the combined type.
///
/// @par Example
/// @code
/// std::print("{}\n", dm::c({"a", "b"}));       // ["a", "b"]
/// std::print("{}\n", dm::c({1, 2}));           // [1, 2]   (Vector<double>)
/// @endcode
template <typename T>
Vector<vector_detail::combined_type<T>> c(std::initializer_list<T> values) {
  using R = vector_detail::combined_type<T>;
  return Vector<R>(std::vector<R>(values.begin(), values.end()));
}
/// @brief Combines mixed numeric types as doubles, including an empty list.
///
/// @param values  Numbers to copy, in order.
/// @return  An owned `Vector<double>`.
///
/// @par Example
/// @code
/// std::print("{}\n", dm::c({1, 2.5}));         // [1, 2.5]
/// @endcode
Vector<> c(std::initializer_list<double> values);
/// @brief Coerces a list mixing text with numeric/logical scalars to owned strings, like R's c().
///
/// Numbers use their shortest round-trip text ("1", "2.5", "1e+20", "Inf", "NaN") and
/// logicals become "TRUE"/"FALSE".
///
/// @param values  Mixed scalars, each converted through vector_detail::StringValue.
/// @return  An owned `Vector<std::string>`.
///
/// @par Example
/// @code
/// std::print("{}\n", dm::c({"a", 1, 2.5}));    // ["a", "1", "2.5"]
/// @endcode
template <typename = void>
Vector<std::string> c(std::initializer_list<vector_detail::StringValue> values) {
  std::vector<std::string> result;
  result.reserve(values.size());
  for (const auto& value : values) result.push_back(value.value);
  return result;
}
/// @brief Copies standard storage into a typed Vector without changing its element type.
///
/// Commonly used to wrap a DataFrame column, e.g.
/// `dm::c(df.get_column<std::string>("Species").values())`.
///
/// @tparam T  Element type (kept, so `std::vector<int>` gives `Vector<int>`).
/// @param values  Storage to adopt (moved when passed an rvalue).
/// @return  An owned `Vector<T>`.
///
/// @par Example
/// @code
/// const auto v = dm::c(std::vector<int>{1, 2});   // Vector<int>
/// std::print("{}\n", v);                           // [1, 2]
/// @endcode
template <typename T>
Vector<T> c(std::vector<T> values) { return Vector<T>(std::move(values)); }
/// @brief Combines scalar arguments with the same coercion as the initializer-list overloads (R's `c()`).
///
/// A single lazy expression is materialised as an owned numeric vector.
/// Text plus numeric/logical scalars becomes strings; other inputs need a common C++ type.
/// Arithmetic arguments become a `Vector<double>`, all-bool arguments a `Vector<bool>`,
/// and other types (such as dm::RGB colors) a vector of that type.
///
/// @tparam First  Type of the first argument.
/// @tparam Rest  Types of the remaining arguments.
/// @param first  First element.
/// @param rest  Remaining elements, in order.
/// @return  An owned vector of the combined element type.
///
/// @par Example
/// @code
/// std::print("{}\n", dm::c(1, 2, 3));                // [1, 2, 3]
/// std::print("{}\n", dm::c(true, false));            // [true, false]
/// std::print("{}\n", dm::c("x", 1, true, 2.5));      // ["x", "1", "TRUE", "2.5"]
/// const auto palette = dm::c(dm::RGB{255, 0, 0}, dm::RGB{0, 0, 255});   // Vector<RGB>
/// @endcode
/// @see dm::factor(), Vector::select()
template <typename First, typename... Rest>
auto c(First first, Rest... rest) {
  if constexpr (sizeof...(Rest) == 0 && vector_expr::is_node<First>) {
    return first.eval();
  } else if constexpr (std::is_convertible_v<First, std::string> || (std::is_convertible_v<Rest, std::string> || ...)) {
    return c(std::initializer_list<vector_detail::StringValue>{first, rest...});
  } else {
    using R = vector_detail::combined_type<std::common_type_t<First, Rest...>>;
    return Vector<R>(std::vector<R>{static_cast<R>(first), static_cast<R>(rest)...});
  }
}
/// @brief Returns an empty numeric vector.
/// @return  An empty `Vector<double>`.
inline Vector<> c() { return {}; }
/// @brief Encodes strings as one-based indices into sorted unique labels, like R's factor().
///
/// Levels are the distinct values sorted lexicographically (byte order), so codes match
/// R's default `as.integer(factor(x))` for ASCII labels. The codes are designed to feed
/// Vector::select(), e.g. to give each observation a color from a palette. The levels
/// themselves are not returned.
///
/// @param values  Labels to encode.
/// @return  A `Vector<int>` of codes in [1, number of distinct labels], same length as @p values.
/// @throws std::length_error  If there are more distinct labels than `int` can index.
///
/// @par Example
/// @code
/// const auto codes = dm::factor(dm::c("lo", "hi", "mid", "hi"));
/// std::print("{}\n", codes);                   // [2, 1, 3, 1]   (hi=1, lo=2, mid=3)
/// const auto palette = dm::c(dm::RGB{255, 0, 0}, dm::RGB{0, 255, 0}, dm::RGB{0, 0, 255});
/// const auto colors = palette.select(codes);   // one RGB per observation
/// @endcode
/// @see Vector::select(), dm::c()
Vector<int> factor(const Vector<std::string>& values);

/// @brief Returns the distinct values in order of first appearance, like R's unique().
///
/// Unlike dm::factor()'s levels, the result is not sorted: each value appears once, at the
/// position where it first occurs.
///
/// @tparam T  Element type; must be equality-comparable.
/// @param values  Values to de-duplicate.
/// @return  The distinct values, first-appearance order.
///
/// @par Example
/// @code
/// std::print("{}\n", dm::unique(std::vector<int>{3, 1, 3, 2, 1}));   // [3, 1, 2]
/// @endcode
/// @see unique(const Vector<T>&), dm::factor()
template <typename T>
[[nodiscard]] std::vector<T> unique(const std::vector<T>& values) {
  std::vector<T> result;
  for (const auto& value : values) {
    if (std::find(result.begin(), result.end(), value) == result.end()) result.push_back(value);
  }
  return result;
}

/// @brief unique() for a dm::Vector; returns a dm::Vector.
/// @tparam T  Element type; must be equality-comparable.
/// @param values  Values to de-duplicate.
/// @return  The distinct values, first-appearance order.
///
/// @par Example
/// @code
/// std::print("{}\n", dm::unique(dm::c("b", "a", "b")));   // ["b", "a"]
/// @endcode
template <typename T>
[[nodiscard]] Vector<T> unique(const Vector<T>& values) {
  return Vector<T>(unique(values.values()));
}

/// @brief Materialise a lazy numeric expression before collecting distinct values.
/// @tparam E Deferred numeric node.
/// @param values Deferred input, unchanged by evaluation.
/// @return Owned numeric vector of distinct values in first-appearance order.
/// @par Example
/// @code
/// std::print("{}\n", dm::unique(dm::c(1, 1, 2) + 1)); // [2, 3]
/// @endcode
/// @see unique(const Vector<T>&)
template<class E, std::enable_if_t<vector_expr::is_node<E>, int> = 0>
Vector<double> unique(const E& values) { return unique(values.eval()); }

/// @brief Form a lazy element-wise negation without allocating vector storage.
/// @tparam E Numeric Vector or expression type.
/// @param v Borrowed lvalue leaf or owned temporary/expression.
/// @return Deferred double-valued negation.
/// @par Example
/// @code
/// std::print("{}\n", -dm::c(1, 2, 3)); // [-1, -2, -3]
/// @endcode
template<class E, std::enable_if_t<vector_expr::numeric<E>, int> = 0>
auto operator-(E&& v) { return vector_expr::unary(std::forward<E>(v), std::negate<double>{}); }

/// @brief Form a lazy element-wise sum with R-style recycling and scalar broadcasting.
/// An empty operand gives an empty result; partial recycling is allowed without a
/// warning. Each child retains its own recycling length. Numeric values are promoted
/// to double, as in the earlier eager API. No elements are computed at construction.
/// Lvalue Vector leaves are borrowed; temporaries and child nodes are owned.
/// @tparam L Numeric vector/expression or scalar; at least one operand is a vector/expression.
/// @tparam R Numeric vector/expression or scalar.
/// @param lhs Forwarded left operand.
/// @param rhs Forwarded right operand.
/// @return Deferred sum; eval() or an owned Vector/standard-vector consumer materialises it.
/// @par Example
/// @code
/// const auto x = dm::c(1, 2, 3, 4);
/// std::print("{}\n", x + dm::c(10, 20)); // [11, 22, 13, 24]
/// std::print("{}\n", x + dm::c(1, 2, 3)); // [2, 4, 6, 5]
/// std::print("{} {}\n", x + 1, 1 + x); // [2, 3, 4, 5] [2, 3, 4, 5]
/// @endcode
/// @see Vector::eval(), vector_expr::Expression::eval()
template<class L, class R, std::enable_if_t<vector_expr::binary_enabled<L, R>, int> = 0>
auto operator+(L&& lhs, R&& rhs) {
  return vector_expr::binary(std::forward<L>(lhs), std::forward<R>(rhs), std::plus<double>{});
}

/// @brief Form a lazy difference with the recycling/ownership rules of operator+.
/// @tparam L Numeric vector/expression or scalar; at least one operand is a vector/expression.
/// @tparam R Numeric vector/expression or scalar.
/// @param lhs Forwarded left operand.
/// @param rhs Forwarded right operand.
/// @return Deferred double-valued difference.
/// @par Example
/// @code
/// const auto x = dm::c(1, 2, 3);
/// std::print("{} {}\n", x - 1, 10 - x); // [0, 1, 2] [9, 8, 7]
/// @endcode
template<class L, class R, std::enable_if_t<vector_expr::binary_enabled<L, R>, int> = 0>
auto operator-(L&& lhs, R&& rhs) {
  return vector_expr::binary(std::forward<L>(lhs), std::forward<R>(rhs), std::minus<double>{});
}

/// @brief Form a lazy product with the recycling/ownership rules of operator+.
/// @tparam L Numeric vector/expression or scalar; at least one operand is a vector/expression.
/// @tparam R Numeric vector/expression or scalar.
/// @param lhs Forwarded left operand.
/// @param rhs Forwarded right operand.
/// @return Deferred double-valued product.
/// @par Example
/// @code
/// const auto x = dm::c(1, 2, 3);
/// std::print("{} {}\n", x * 2, 2 * x); // [2, 4, 6] [2, 4, 6]
/// @endcode
template<class L, class R, std::enable_if_t<vector_expr::binary_enabled<L, R>, int> = 0>
auto operator*(L&& lhs, R&& rhs) {
  return vector_expr::binary(std::forward<L>(lhs), std::forward<R>(rhs), std::multiplies<double>{});
}

/// @brief Form a lazy quotient with the recycling/ownership rules of operator+.
/// Zero divisors follow double arithmetic, producing infinity/NaN without checks.
/// @tparam L Numeric vector/expression or scalar; at least one operand is a vector/expression.
/// @tparam R Numeric vector/expression or scalar.
/// @param lhs Forwarded numerator.
/// @param rhs Forwarded divisor.
/// @return Deferred double-valued quotient.
/// @par Example
/// @code
/// const auto x = dm::c(1, 2, 4);
/// std::print("{} {}\n", x / 2, 8 / x); // [0.5, 1, 2] [8, 4, 2]
/// @endcode
template<class L, class R, std::enable_if_t<vector_expr::binary_enabled<L, R>, int> = 0>
auto operator/(L&& lhs, R&& rhs) {
  return vector_expr::binary(std::forward<L>(lhs), std::forward<R>(rhs), std::divides<double>{});
}

/// @brief Prints the vector as `[a, b, c]`.
///
/// Elements are written with their own `operator<<`, so strings are unquoted here
/// (`[a, b]`), whereas `std::print("{}", v)` quotes them (`["a", "b"]`).
///
/// @tparam T  Element type with an `operator<<`.
/// @param out  Destination stream.
/// @param v  Vector to print.
/// @return  @p out.
///
/// @par Example
/// @code
/// std::cout << dm::c(1.5, 2) << ' ' << dm::c("a", "b") << '\n';   // [1.5, 2] [a, b]
/// @endcode
template <typename T>
std::ostream& operator<<(std::ostream& out, const Vector<T>& v) {
  out << '[';
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (i) out << ", ";
    out << v[i];
  }
  return out << ']';
}

/// @brief Stream a lazy expression without materialising vector storage.
/// @tparam E Deferred numeric node.
/// @param out Destination stream.
/// @param v Read-only expression with live borrowed leaves.
/// @return Destination stream after writing bracketed numeric values.
/// @par Example
/// @code
/// std::cout << (dm::c(1, 2) + 1) << '\n'; // [2, 3]
/// @endcode
template<class E, std::enable_if_t<vector_expr::is_node<E>, int> = 0>
std::ostream& operator<<(std::ostream& out, const E& v) {
  out << '[';
  for (std::size_t i = 0; i < v.size(); ++i) { if (i) out << ", "; out << v[i]; }
  return out << ']';
}

// Keep the scalar overloads visible alongside the Vector<> ones, so unqualified calls such as
// `sin(0.5)` inside namespace datamunge still resolve to the <cmath> functions. round() is the
// exception: the scalar dm::round(double, int = 0) below follows R, so `dm::round(2.5)` is 2.
using std::abs;
using std::cos;
using std::exp;
using std::log;
using std::pow;
using std::sin;
using std::sqrt;
using std::tan;

/// @name Element-wise math
/// Vector overloads build lazy nodes without transforming elements yet. The result
/// has the same length as its input and evaluates pointwise at a consuming boundary. Scalar
/// calls such as `dm::sin(0.5)` or `dm::pow(2.0, 3)` resolve to the std:: functions.
/// @{

/// @brief Returns the sine of each element, interpreting input angles in radians.
/// @tparam E Numeric Vector or expression, borrowed for lvalues and owned for temporaries.
/// @param v  Angles in radians.
/// @return Deferred double-valued expression computing `[sin(v1), sin(v2), ...]`.
///
/// @par Example
/// @code
/// const double pi = 3.141592653589793;
/// std::print("{}\n", dm::round(dm::sin(dm::c(0, pi / 2, pi)), 6));   // [0, 1, 0]
/// @endcode
/// @see dm::cos(), dm::tan()
template<class E, std::enable_if_t<vector_expr::numeric<E>, int> = 0>
auto sin(E&& v) {
  return vector_expr::unary(std::forward<E>(v), [](double value) { return std::sin(value); });
}
/// @brief Returns the cosine of each element, interpreting input angles in radians.
/// @tparam E Numeric Vector or expression, borrowed for lvalues and owned for temporaries.
/// @param v  Angles in radians.
/// @return Deferred double-valued expression computing `[cos(v1), cos(v2), ...]`.
///
/// @par Example
/// @code
/// const double pi = 3.141592653589793;
/// std::print("{}\n", dm::round(dm::cos(dm::c(0, pi / 2, pi)), 6));   // [1, 0, -1]
/// @endcode
template<class E, std::enable_if_t<vector_expr::numeric<E>, int> = 0>
auto cos(E&& v) {
  return vector_expr::unary(std::forward<E>(v), [](double value) { return std::cos(value); });
}
/// @brief Returns the tangent of each element, interpreting input angles in radians.
/// @tparam E Numeric Vector or expression, borrowed for lvalues and owned for temporaries.
/// @param v  Angles in radians.
/// @return Deferred double-valued expression computing `[tan(v1), tan(v2), ...]`.
///
/// @par Example
/// @code
/// const double pi = 3.141592653589793;
/// std::print("{}\n", dm::round(dm::tan(dm::c(0, pi / 4)), 6));   // [0, 1]
/// @endcode
template<class E, std::enable_if_t<vector_expr::numeric<E>, int> = 0>
auto tan(E&& v) {
  return vector_expr::unary(std::forward<E>(v), [](double value) { return std::tan(value); });
}
/// @brief Returns the base-e exponential of each element using std::exp.
/// @tparam E Numeric Vector or expression, borrowed for lvalues and owned for temporaries.
/// @param v  Exponents.
/// @return Deferred double-valued expression computing `[e^v1, e^v2, ...]`.
///
/// @par Example
/// @code
/// std::print("{}\n", dm::round(dm::exp(dm::c(0, 1, 2)), 4));   // [1, 2.7183, 7.3891]
/// @endcode
/// @see dm::log()
template<class E, std::enable_if_t<vector_expr::numeric<E>, int> = 0>
auto exp(E&& v) {
  return vector_expr::unary(std::forward<E>(v), [](double value) { return std::exp(value); });
}
/// @brief Returns the natural logarithm of each element using std::log, without domain checks.
///
/// Zero gives -inf and negative inputs give NaN (no exception).
///
/// @tparam E Numeric Vector or expression, borrowed for lvalues and owned for temporaries.
/// @param v  Positive inputs.
/// @return Deferred double-valued expression computing `[ln(v1), ln(v2), ...]`.
///
/// @par Example
/// @code
/// std::print("{}\n", dm::round(dm::log(dm::exp(dm::c(1, 2))), 6));   // [1, 2]
/// @endcode
/// @see dm::exp()
template<class E, std::enable_if_t<vector_expr::numeric<E>, int> = 0>
auto log(E&& v) {
  return vector_expr::unary(std::forward<E>(v), [](double value) { return std::log(value); });
}
/// @brief Returns each element's square root using std::sqrt, without domain checks.
///
/// Negative inputs give NaN (no exception).
///
/// @tparam E Numeric Vector or expression, borrowed for lvalues and owned for temporaries.
/// @param v  Non-negative inputs.
/// @return Deferred double-valued expression computing `[sqrt(v1), sqrt(v2), ...]`.
///
/// @par Example
/// @code
/// std::print("{}\n", dm::sqrt(dm::c(4, 9)));   // [2, 3]
/// @endcode
template<class E, std::enable_if_t<vector_expr::numeric<E>, int> = 0>
auto sqrt(E&& v) {
  return vector_expr::unary(std::forward<E>(v), [](double value) { return std::sqrt(value); });
}
/// @brief Returns the absolute value of each element.
/// @tparam E Numeric Vector or expression, borrowed for lvalues and owned for temporaries.
/// @param v  Input values.
/// @return Deferred double-valued expression computing `[|v1|, |v2|, ...]`.
///
/// @par Example
/// @code
/// std::print("{}\n", dm::abs(dm::c(-1.5, 0, 2)));   // [1.5, 0, 2]
/// @endcode
template<class E, std::enable_if_t<vector_expr::numeric<E>, int> = 0>
auto abs(E&& v) {
  return vector_expr::unary(std::forward<E>(v), [](double value) { return std::abs(value); });
}
/// @brief Raises each element to @p exponent using std::pow, without domain checks.
///
/// Like R's `x^p`. Negative bases with fractional exponents give NaN.
///
/// @tparam E Numeric Vector or expression.
/// @param v  Bases.
/// @param exponent  Exponent applied to every element.
/// @return Deferred expression computing `[v1^exponent, v2^exponent, ...]`.
///
/// @par Example
/// @code
/// std::print("{}\n", dm::pow(dm::c(1, 2, 3), 2));   // [1, 4, 9]
/// const auto x = dm::c(1, 2, 3, 4);
/// const auto m3 = dm::sum(dm::pow(x - dm::mean(x), 3)) / dm::length(x);   // 3rd moment
/// @endcode
/// @see dm::sqrt()
template<class E, std::enable_if_t<vector_expr::numeric<E>, int> = 0>
auto pow(E&& v, double exponent) {
  return vector_expr::unary(std::forward<E>(v), [exponent](double value) { return std::pow(value, exponent); });
}

/// @}

/// @brief Rounds to decimal places exactly like R's round() (R >= 4.0.0).
///
/// Negative @p digits round to tens, hundreds, etc. Non-finite values pass through.
/// Precision beyond double's decimal range leaves the value unchanged for positive
/// digits and returns signed zero for negative digits; results beyond double's range overflow.
/// Of the two candidates with @p digits decimals that bracket @p value, the one nearer to
/// the value as stored in binary wins; an exact tie goes to the even last digit. So
/// `round(0.125, 2)` is `0.12` (a true tie), while `round(0.15, 1)` is `0.1` and
/// `round(2.675, 2)` is `2.67` because those inputs are stored slightly below the half.
///
/// @param value  Number to round.
/// @param digits  Decimal places to keep (default 0); negative values round left of the decimal point.
/// @return  The rounded number.
///
/// @par Example
/// @code
/// std::print("{} {}\n", dm::round(3.14159, 2), dm::round(1234.5, -2));   // 3.14 1200
/// std::print("{} {}\n", dm::round(0.125, 2), dm::round(2.5));            // 0.12 2
/// @endcode
/// @see dm::round(Vector<>, int)
double round(double value, int digits = 0);

/// @brief Rounds each element to @p digits decimal places; see the scalar round() overload.
///
/// Matches R's round(): exact halves go to the even digit (`2.5` -> `2`, `-2.5` -> `-2`).
///
/// @tparam E Numeric Vector or expression.
/// @param v  Values to round.
/// @param digits  Decimal places; negative rounds to tens, hundreds, ... (default 0).
/// @return Deferred double-valued rounded expression.
///
/// @par Example
/// @code
/// std::print("{}\n", dm::round(dm::c(1.2345, 2.71828), 2));   // [1.23, 2.72]
/// std::print("{}\n", dm::round(dm::c(2.5, -2.5, 3.5)));       // [2, -2, 4]
/// std::print("{}\n", dm::round(dm::c(1234.5, 1251), -2));     // [1200, 1300]
/// @endcode
/// @see dm::round(double, int)
template<class E, std::enable_if_t<vector_expr::numeric<E>, int> = 0>
auto round(E&& v, int digits = 0) {
  return vector_expr::unary(std::forward<E>(v), [digits](double value) { return datamunge::round(value, digits); });
}


/// @brief Evenly spaced values from @p from to @p to inclusive, like R's `seq(from, to, length.out = n)`.
///
/// @p to may be less than @p from for a decreasing sequence.
///
/// @param from First value (also the only value when @p n is 1).
/// @param to Last value when @p n is at least 2.
/// @param n Number of output values; zero returns an empty vector.
/// @return An owned vector of @p n values.
///
/// @par Example
/// @code
/// std::print("{}\n", dm::seql(0, 1, 5));       // [0, 0.25, 0.5, 0.75, 1]
/// std::print("{}\n", dm::seql(1, 0, 3));       // [1, 0.5, 0]
/// @endcode
Vector<> seql(double from, double to, std::size_t n);

/// @brief Sum of the elements, or zero for an empty vector (R's `sum()`).
///
/// NaN propagates; there is no `na.rm`.
///
/// @param v  Values to add.
/// @return  The total.
///
/// @par Example
/// @code
/// std::print("{} {}\n", dm::sum(dm::c(1, 2, 3.5)), dm::sum(dm::c()));   // 6.5 0
/// @endcode
/// @see dm::mean()
double sum(const Vector<>& v);
/// @brief Arithmetic mean of the elements (NaN for an empty vector), like R's `mean()`.
///
/// NaN in the input propagates; there is no `na.rm`.
///
/// @param v  Values to average.
/// @return  sum(v) / length(v).
///
/// @par Example
/// @code
/// std::print("{}\n", dm::mean(dm::c(1, 2, 3, 4)));   // 2.5
/// @endcode
/// @see dm::sum(), dm::sd()
double mean(const Vector<>& v);
/// @brief Sample standard deviation, using the n - 1 denominator (R's `sd()`).
///
/// Returns NaN for fewer than two observations or any non-finite observation.
/// Constant finite input returns zero; the input is unchanged. Uses Welford's
/// centered update in long double for numerical stability.
///
/// @param v  Observations.
/// @return  sqrt(sum((v - mean(v))^2) / (n - 1)).
///
/// @par Example
/// @code
/// std::print("{}\n", dm::round(dm::sd(dm::c(1, 2, 3, 4)), 4));   // 1.291
/// std::print("{}\n", dm::sd(dm::c(3, 3, 3)));                    // 0
/// @endcode
/// @see dm::mean()
double sd(const Vector<>& v);

/// @brief Sum a lazy numeric expression directly in a single pass without a result array.
/// @tparam E Deferred numeric node.
/// @param v Expression with live borrowed leaves.
/// @return Left-to-right sum, or zero for empty input; NaN/Inf follow double arithmetic.
/// @par Example
/// @code
/// std::print("{}\n", dm::sum(dm::c(1, 2, 3) * 2)); // 12
/// @endcode
/// @see sum(const Vector<>&)
template<class E, std::enable_if_t<vector_expr::is_node<E>, int> = 0>
double sum(const E& v) { return vector_expr::sum_values(v); }

/// @brief Mean of a lazy numeric expression without allocating an intermediate vector.
/// @tparam E Deferred numeric node.
/// @param v Expression with live borrowed leaves.
/// @return Sum divided by length, or NaN for empty input.
/// @par Example
/// @code
/// std::print("{}\n", dm::mean(dm::c(1, 2, 3) + 1)); // 3
/// @endcode
/// @see mean(const Vector<>&)
template<class E, std::enable_if_t<vector_expr::is_node<E>, int> = 0>
double mean(const E& v) {
  const auto n = v.size();
  return n ? vector_expr::sum_values(v) / static_cast<double>(n) : std::numeric_limits<double>::quiet_NaN();
}

/// @brief Welford sample standard deviation of a lazy expression, evaluated in one pass.
/// Uses the same long-double accumulation as sd(Vector), without a result array.
/// @tparam E Deferred numeric node.
/// @param v Expression with live borrowed leaves.
/// @return Sample deviation (denominator n-1), or NaN for fewer than two/nonfinite values.
/// @par Example
/// @code
/// std::print("{:.0f}\n", dm::sd(dm::c(1, 2, 3) * 2)); // 2
/// @endcode
/// @see sd(const Vector<>&)
template<class E, std::enable_if_t<vector_expr::is_node<E>, int> = 0>
double sd(const E& v) { return vector_expr::sample_sd(v); }



/// @brief Reseeds the generator shared by unif() and rnorm(), like R's `set.seed()`.
///
/// Until this is called, the generator is seeded from std::random_device. The same
/// seed reproduces the same subsequent sequence of runif()/unif()/rnorm() draws.
/// Calls using this shared generator must be externally synchronized across threads.
///
/// @param seed  Any 64-bit seed.
///
/// @par Example
/// @code
/// dm::set_seed(1);
/// const auto a = dm::runif(0, 1, 3);
/// dm::set_seed(1);
/// const auto b = dm::runif(0, 1, 3);
/// std::print("{}\n", a.values() == b.values());   // true
/// @endcode
/// @see dm::runif(), dm::rnorm()
void set_seed(std::uint64_t seed);

/// @brief @p n draws from Uniform[@p lower, @p upper), like R's `runif(n, lower, upper)`.
///
/// Uses the shared generator described by set_seed(). Note the argument order: bounds
/// first, count last.
///
/// @param lower  Inclusive lower bound.
/// @param upper  Exclusive upper bound; must exceed @p lower.
/// @param n  Number of draws; zero returns an empty vector (bounds are then not checked).
/// @return  An owned vector of @p n draws.
/// @throws std::invalid_argument For non-finite bounds or @p upper <= @p lower when @p n > 0.
///
/// @par Example
/// @code
/// dm::set_seed(42);
/// const auto x = dm::unif(-1, 1, 1000);
/// std::print("{}\n", dm::length(x));            // 1000, all in [-1, 1)
/// @endcode
/// @see dm::runif(), dm::set_seed()
Vector<> unif(double lower, double upper, std::size_t n);

/// @brief Alias for unif(lower, upper, n), using the same bounds, validation, and shared generator.
///
/// Argument order is lower bound, upper bound, count; zero count returns an empty vector.
///
/// @param lower  Inclusive lower bound.
/// @param upper  Exclusive upper bound; must exceed @p lower.
/// @param n  Number of draws.
/// @return  An owned vector of @p n draws from Uniform[@p lower, @p upper).
/// @throws std::invalid_argument For non-finite bounds or @p upper <= @p lower when @p n > 0.
///
/// @par Example
/// @code
/// dm::set_seed(42);
/// const auto x = dm::runif(0, 1, 1000);
/// const auto y = dm::sin(20 / (x + 0.25)) + dm::rnorm(0, 0.2, 1000);   // noisy Doppler
/// @endcode
/// @see dm::unif(), dm::rnorm(), dm::set_seed()
Vector<> runif(double lower, double upper, std::size_t n);

/// @brief @p n draws from Normal(@p mean, @p sd), like R's `rnorm(n, mean, sd)`.
///
/// Uses the shared generator described by set_seed(). Note the argument order:
/// parameters first, count last.
///
/// @param mean  Distribution mean (finite).
/// @param sd  Standard deviation (finite, > 0).
/// @param n  Number of draws; zero returns an empty vector (parameters are then not checked).
/// @return  An owned vector of @p n draws.
/// @throws std::invalid_argument For non-finite parameters or @p sd <= 0 when @p n > 0.
///
/// @par Example
/// @code
/// dm::set_seed(7);
/// const auto z = dm::rnorm(10, 2, 100000);
/// std::print("{} {}\n", dm::round(dm::mean(z), 1), dm::round(dm::sd(z), 1));   // 10 2
/// @endcode
/// @see dm::runif(), dm::set_seed()
Vector<> rnorm(double mean, double sd, std::size_t n);

/// @brief Short name for graphics::RPlot, the R-base-graphics-style plot.
///
/// @par Example
/// @code
/// const std::string svg = dm::Plot(dm::c(1, 2, 3), dm::c(4, 5, 6)).to_svg();
/// @endcode
/// @see graphics::RPlot
using Plot = graphics::RPlot;
/// @brief Short name for graphics::RGB, a plotting color (0-255 channels plus optional alpha).
/// @see graphics::RGB
using RGB = graphics::RGB;
/// @brief Short name for graphics::RGBA, the same type as RGB, spelled for translucent colors.
/// @see graphics::RGB
using RGBA = graphics::RGBA;

/// @brief Create an empty native plot for adding points and lines through chained calls.
/// Owns its layers and style. Construction opens no display; add a data layer before
/// rendering or saving. Equivalent to Plot() with the normal canvas and axis defaults.
/// @return Owned empty RPlot, ready for title(), points(), lines(), xlim() and ylim().
/// @par Example
/// @code
/// auto plt = dm::plot().title("Fit").points(dm::c(0, 1), dm::c(1, 2), "data", "blue")
///                .ylim(-4, 4).xlim(0, 1);
/// std::print("{} {}\n", plt.series().size(), plt.y_min()); // 1 -4
/// @endcode
/// @see plot(std::vector<double>, std::vector<double>, std::string, RGBA), graphics::RPlot
inline Plot plot() { return {}; }

/// @brief Create a native scatter plot, like R's plot(x, y), ready for chained lines().
/// Copies coordinates into owned plot storage; no GUI opens until show() is requested.
/// @param x Horizontal coordinates.
/// @param y Vertical coordinates, same length as x.
/// @param label Legend label, default "Observed".
/// @param color Point color, default translucent blue.
/// @return Owned RPlot that can be saved as SVG or PNG.
/// @throws std::invalid_argument For mismatched coordinate lengths.
/// @par Example
/// @code
/// auto figure = dm::plot(dm::c(0, 1), dm::c(1, 2));
/// std::print("{}\n", figure.series().size()); // 1
/// @endcode
/// @see graphics::RPlot::plot(), graphics::RPlot::lines()
inline Plot plot(std::vector<double> x, std::vector<double> y, std::string label = "Observed",
                 RGBA color = {37, 99, 235, 0.3}) {
  return Plot::plot(std::move(x), std::move(y), "p", std::move(label), color);
}

} // namespace datamunge
