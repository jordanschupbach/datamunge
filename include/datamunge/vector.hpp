#pragma once

/// @file vector.hpp
/// @brief Owned typed vectors with R-style selection, recycling, and numeric operations.
///
/// Provides dm::Vector (an owned std::vector with R semantics), the R-style constructors
/// c(), seql(), factor(), element-wise arithmetic and math with recycling, summary
/// statistics (sum(), mean(), sd()), and seeded random sampling (runif(), rnorm()).
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
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

/// @brief Core library APIs and their algorithm, data, and numerical submodules.
namespace datamunge {

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
/// `const std::vector<T>&` for APIs such as dm::Plot. The free arithmetic operators
/// and element-wise math functions are defined for `Vector<double>` (the default `T`).
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
template <typename T = double>
class Vector {
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
  if constexpr (std::is_convertible_v<First, std::string> || (std::is_convertible_v<Rest, std::string> || ...)) {
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

/// @brief Returns element-wise negation of the by-value vector.
///
/// @param v  Vector to negate.
/// @return  `[-v1, -v2, ...]`.
///
/// @par Example
/// @code
/// std::print("{}\n", -dm::c(1, 2, 3));         // [-1, -2, -3]
/// @endcode
Vector<> operator-(Vector<> v);

/// @brief Returns pairwise sums, recycling the shorter vector; empty inputs give empty.
///
/// Like R, the shorter operand is recycled to the longer length, with no warning when
/// the longer length is not a multiple of the shorter one.
///
/// @param lhs  Left operand (taken by value and reused as the result's storage).
/// @param rhs  Right operand.
/// @return  A vector of length max(lhs.size(), rhs.size()), or empty if either is empty.
///
/// @par Example
/// @code
/// const auto x = dm::c(1, 2, 3, 4);
/// std::print("{}\n", x + dm::c(10, 20));       // [11, 22, 13, 24]
/// std::print("{}\n", x + dm::c(1, 2, 3));      // [2, 4, 6, 5]   (partial recycle)
/// @endcode
/// @see Vector::operator+=()
Vector<> operator+(Vector<> lhs, const Vector<>& rhs);
/// @brief Returns pairwise differences with the same recycling as operator+.
/// @param lhs  Minuend vector.
/// @param rhs  Subtrahend vector, recycled as needed.
/// @return  `lhs - rhs` element-wise.
///
/// @par Example
/// @code
/// std::print("{}\n", dm::c(5, 6, 7) - dm::c(1));   // [4, 5, 6]
/// @endcode
Vector<> operator-(Vector<> lhs, const Vector<>& rhs);
/// @brief Returns pairwise products with the same recycling as operator+.
/// @param lhs  Left factor.
/// @param rhs  Right factor, recycled as needed.
/// @return  `lhs * rhs` element-wise.
///
/// @par Example
/// @code
/// std::print("{}\n", dm::c(1, 2, 3, 4) * dm::c(2, 3, 4));   // [2, 6, 12, 8]
/// @endcode
Vector<> operator*(Vector<> lhs, const Vector<>& rhs);
/// @brief Returns pairwise quotients with the same recycling as operator+.
///
/// Zero divisors are not checked; IEEE rules apply (`1 / 0` is inf, `0 / 0` is NaN).
///
/// @param lhs  Dividends.
/// @param rhs  Divisors, recycled as needed.
/// @return  `lhs / rhs` element-wise.
///
/// @par Example
/// @code
/// std::print("{}\n", dm::c(1, 2, 3, 4) / dm::c(2));   // [0.5, 1, 1.5, 2]
/// @endcode
Vector<> operator/(Vector<> lhs, const Vector<>& rhs);

/// @brief Returns each element of @p lhs plus the scalar @p rhs.
/// @param lhs  Vector operand.
/// @param rhs  Scalar added to every element.
/// @return  A vector the same length as @p lhs.
///
/// @par Example
/// @code
/// std::print("{}\n", dm::c(1, 2, 3) + 1);      // [2, 3, 4]
/// @endcode
Vector<> operator+(Vector<> lhs, double rhs);
/// @brief Returns each element of @p lhs minus the scalar @p rhs.
/// @param lhs  Vector operand.
/// @param rhs  Scalar subtracted from every element.
/// @return  A vector the same length as @p lhs.
///
/// @par Example
/// @code
/// const auto x = dm::c(1, 2, 3);
/// std::print("{}\n", x - dm::mean(x));         // [-1, 0, 1]
/// @endcode
Vector<> operator-(Vector<> lhs, double rhs);
/// @brief Returns each element of @p lhs multiplied by the scalar @p rhs.
/// @param lhs  Vector operand.
/// @param rhs  Scalar multiplier.
/// @return  A vector the same length as @p lhs.
///
/// @par Example
/// @code
/// std::print("{}\n", dm::c(1, 2, 3) * 2);      // [2, 4, 6]
/// @endcode
Vector<> operator*(Vector<> lhs, double rhs);
/// @brief Returns each element of @p lhs divided by @p rhs, without checking zero.
/// @param lhs  Vector of dividends.
/// @param rhs  Scalar divisor.
/// @return  A vector the same length as @p lhs.
///
/// @par Example
/// @code
/// std::print("{}\n", dm::c(1, 2, 3) / 2);      // [0.5, 1, 1.5]
/// @endcode
Vector<> operator/(Vector<> lhs, double rhs);

/// @brief Returns the scalar @p lhs plus each element of @p rhs.
/// @param lhs  Scalar added to every element.
/// @param rhs  Vector operand.
/// @return  A vector the same length as @p rhs.
///
/// @par Example
/// @code
/// std::print("{}\n", 1 + dm::c(1, 2, 3));      // [2, 3, 4]
/// @endcode
Vector<> operator+(double lhs, Vector<> rhs);
/// @brief Returns the scalar @p lhs minus each element of @p rhs.
/// @param lhs  Scalar minuend.
/// @param rhs  Vector of subtrahends.
/// @return  A vector the same length as @p rhs.
///
/// @par Example
/// @code
/// std::print("{}\n", 10 - dm::c(1, 2, 3));     // [9, 8, 7]
/// @endcode
Vector<> operator-(double lhs, Vector<> rhs);
/// @brief Returns the scalar @p lhs multiplied by each element of @p rhs.
/// @param lhs  Scalar multiplier.
/// @param rhs  Vector operand.
/// @return  A vector the same length as @p rhs.
///
/// @par Example
/// @code
/// std::print("{}\n", 2 * dm::c(1, 2, 3));      // [2, 4, 6]
/// @endcode
Vector<> operator*(double lhs, Vector<> rhs);
/// @brief Returns @p lhs divided by each element of @p rhs, without checking zero.
/// @param lhs  Scalar dividend.
/// @param rhs  Vector of divisors (zero gives inf/NaN).
/// @return  A vector the same length as @p rhs.
///
/// @par Example
/// @code
/// const auto x = dm::c(0.5, 0.75);
/// std::print("{}\n", 20 / (x + 0.25));         // [26.666666666666668, 20]
/// @endcode
Vector<> operator/(double lhs, Vector<> rhs);

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
/// Vector overloads of the standard math functions. Each takes its input by value, transforms
/// it in place, and returns it, so the result has the same length as the input. Scalar
/// calls such as `dm::sin(0.5)` or `dm::pow(2.0, 3)` resolve to the std:: functions.
/// @{

/// @brief Returns the sine of each element, interpreting input angles in radians.
/// @param v  Angles in radians.
/// @return  `[sin(v1), sin(v2), ...]`.
///
/// @par Example
/// @code
/// const double pi = 3.141592653589793;
/// std::print("{}\n", dm::round(dm::sin(dm::c(0, pi / 2, pi)), 6));   // [0, 1, 0]
/// @endcode
/// @see dm::cos(), dm::tan()
Vector<> sin(Vector<> v);
/// @brief Returns the cosine of each element, interpreting input angles in radians.
/// @param v  Angles in radians.
/// @return  `[cos(v1), cos(v2), ...]`.
///
/// @par Example
/// @code
/// const double pi = 3.141592653589793;
/// std::print("{}\n", dm::round(dm::cos(dm::c(0, pi / 2, pi)), 6));   // [1, 0, -1]
/// @endcode
Vector<> cos(Vector<> v);
/// @brief Returns the tangent of each element, interpreting input angles in radians.
/// @param v  Angles in radians.
/// @return  `[tan(v1), tan(v2), ...]`.
///
/// @par Example
/// @code
/// const double pi = 3.141592653589793;
/// std::print("{}\n", dm::round(dm::tan(dm::c(0, pi / 4)), 6));   // [0, 1]
/// @endcode
Vector<> tan(Vector<> v);
/// @brief Returns the base-e exponential of each element using std::exp.
/// @param v  Exponents.
/// @return  `[e^v1, e^v2, ...]`.
///
/// @par Example
/// @code
/// std::print("{}\n", dm::round(dm::exp(dm::c(0, 1, 2)), 4));   // [1, 2.7183, 7.3891]
/// @endcode
/// @see dm::log()
Vector<> exp(Vector<> v);
/// @brief Returns the natural logarithm of each element using std::log, without domain checks.
///
/// Zero gives -inf and negative inputs give NaN (no exception).
///
/// @param v  Positive inputs.
/// @return  `[ln(v1), ln(v2), ...]`.
///
/// @par Example
/// @code
/// std::print("{}\n", dm::round(dm::log(dm::exp(dm::c(1, 2))), 6));   // [1, 2]
/// @endcode
/// @see dm::exp()
Vector<> log(Vector<> v);
/// @brief Returns each element's square root using std::sqrt, without domain checks.
///
/// Negative inputs give NaN (no exception).
///
/// @param v  Non-negative inputs.
/// @return  `[sqrt(v1), sqrt(v2), ...]`.
///
/// @par Example
/// @code
/// std::print("{}\n", dm::sqrt(dm::c(4, 9)));   // [2, 3]
/// @endcode
Vector<> sqrt(Vector<> v);
/// @brief Returns the absolute value of each element.
/// @param v  Input values.
/// @return  `[|v1|, |v2|, ...]`.
///
/// @par Example
/// @code
/// std::print("{}\n", dm::abs(dm::c(-1.5, 0, 2)));   // [1.5, 0, 2]
/// @endcode
Vector<> abs(Vector<> v);
/// @brief Raises each element to @p exponent using std::pow, without domain checks.
///
/// Like R's `x^p`. Negative bases with fractional exponents give NaN.
///
/// @param v  Bases.
/// @param exponent  Exponent applied to every element.
/// @return  `[v1^exponent, v2^exponent, ...]`.
///
/// @par Example
/// @code
/// std::print("{}\n", dm::pow(dm::c(1, 2, 3), 2));   // [1, 4, 9]
/// const auto x = dm::c(1, 2, 3, 4);
/// const auto m3 = dm::sum(dm::pow(x - dm::mean(x), 3)) / dm::length(x);   // 3rd moment
/// @endcode
/// @see dm::sqrt()
Vector<> pow(Vector<> v, double exponent);
/// @brief Rounds each element to @p digits decimal places; see the scalar round() overload.
///
/// Matches R's round(): exact halves go to the even digit (`2.5` -> `2`, `-2.5` -> `-2`).
///
/// @param v  Values to round.
/// @param digits  Decimal places; negative rounds to tens, hundreds, ... (default 0).
/// @return  The rounded values.
///
/// @par Example
/// @code
/// std::print("{}\n", dm::round(dm::c(1.2345, 2.71828), 2));   // [1.23, 2.72]
/// std::print("{}\n", dm::round(dm::c(2.5, -2.5, 3.5)));       // [2, -2, 4]
/// std::print("{}\n", dm::round(dm::c(1234.5, 1251), -2));     // [1200, 1300]
/// @endcode
/// @see dm::round(double, int)
Vector<> round(Vector<> v, int digits = 0);
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

} // namespace datamunge
