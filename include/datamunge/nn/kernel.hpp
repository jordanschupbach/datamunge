#pragma once

/// @file kernel.hpp
/// @brief Building blocks for writing tensor kernels outside the library: floating-dtype
///        dispatch, a parallel elementwise map, and summation with a chosen accuracy.
///
/// A custom operation (an nn::Function) needs a forward kernel that reads and writes raw
/// memory. These helpers cover the parts every such kernel repeats. dispatch_floating() turns a
/// runtime dtype into a compile-time element type, and map_elements() applies a scalar function
/// over a packed tensor on nn's thread pool. sum_with() adds a row of numbers by one of four
/// methods whose rounding error differs by orders of magnitude, and
/// summation_error_bound() states the classical bound for each.

#include "datamunge/nn/parallel.hpp"
#include "datamunge/nn/tensor.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace datamunge::nn {

/// @brief Calls `f(T{})` with `T` = float for Float32 and double for Float64.
///
/// The argument is a value of the element type, used only for its type
/// (`using T = decltype(tag);` inside a generic lambda).
/// @param dtype  The runtime dtype.
/// @param what  Operation name for the error message.
/// @param f  A generic callable.
/// @return Whatever f returns.
/// @throws std::invalid_argument  For a non-floating dtype.
template <typename F>
decltype(auto) dispatch_floating(DType dtype, const std::string& what, F&& f) {
  switch (dtype) {
    case DType::Float32: return f(float{});
    case DType::Float64: return f(double{});
    default: break;
  }
  throw std::invalid_argument(what + ": expected float32 or float64, got " + dtype_name(dtype));
}

/// @brief A new tensor with `f` applied to every element, computed in parallel chunks.
///
/// The input is packed first if it is a strided view. `f` is called as `f(v)` with `v` of the
/// element type (float or double) and must return a value convertible to it. It must be
/// thread-safe. Each element is written by exactly one call, so the result does not depend on
/// the thread count. No history is recorded. Wrap the call in an nn::Function to give it a gradient.
/// @param x  A float32 or float64 tensor.
/// @param f  The scalar function.
/// @param grain  Minimum elements per chunk before work is split across threads.
/// @return A tensor of x's shape and dtype.
/// @throws std::invalid_argument  For a non-floating dtype.
template <typename F>
[[nodiscard]] Tensor map_elements(const Tensor& x, F f, std::int64_t grain = 32768) {
  const Tensor packed = x.detach().contiguous();
  Tensor       out    = empty(packed.shape(), packed.dtype());
  dispatch_floating(packed.dtype(), "map_elements", [&](auto tag) {
    using T     = decltype(tag);
    const T* in = packed.data_ptr<T>();
    T*       o  = out.data_ptr<T>();
    parallel_for(0, packed.numel(), grain, [&](std::int64_t lo, std::int64_t hi) {
      for (std::int64_t i = lo; i < hi; ++i) o[i] = static_cast<T>(f(in[i]));
    });
    return 0;
  });
  return out;
}

/// @brief How sum_with() adds a sequence of numbers.
enum class Summation {
  /// Left to right in the element type: \f$s\leftarrow s+x_i\f$.
  Naive,
  /// Recursive halving: sum each half, then add the two (blocks of 8 are summed naively).
  Pairwise,
  /// Kahan's compensated summation in the element type, carrying the lost low-order part.
  Kahan,
  /// Left to right in double, rounded to the element type once at the end (what nn::sum does).
  WideAccumulator,
};

/// @brief Display name of a summation method ("naive", "pairwise", "kahan", "wide accumulator").
/// @param method  The method.
/// @return The name.
[[nodiscard]] std::string summation_name(Summation method);

/// @brief Sums `n` values by the chosen method.
/// @tparam T  float or double.
/// @param values  Pointer to n contiguous values.
/// @param n  Count (0 gives 0).
/// @param method  Summation method.
/// @return The sum, rounded to T.
template <typename T>
[[nodiscard]] T sum_with(const T* values, std::int64_t n, Summation method) {
  static_assert(std::is_floating_point_v<T>, "sum_with needs a floating type");
  switch (method) {
    case Summation::Naive: {
      T s = 0;
      for (std::int64_t i = 0; i < n; ++i) s += values[i];
      return s;
    }
    case Summation::Pairwise: {
      if (n <= 8) return sum_with(values, n, Summation::Naive);
      const std::int64_t half = n / 2;
      return sum_with(values, half, Summation::Pairwise) + sum_with(values + half, n - half, Summation::Pairwise);
    }
    case Summation::Kahan: {
      // volatile keeps the compensation from being optimised away under reassociation.
      T s = 0, c = 0;
      for (std::int64_t i = 0; i < n; ++i) {
        const T          y = values[i] - c;
        const volatile T t = s + y;
        c                  = (t - s) - y;
        s                  = t;
      }
      return s;
    }
    case Summation::WideAccumulator: {
      double s = 0;
      for (std::int64_t i = 0; i < n; ++i) s += static_cast<double>(values[i]);
      return static_cast<T>(s);
    }
  }
  throw std::invalid_argument("sum_with: unknown method");
}

/// @brief The classical worst-case bound on \f$\lvert\hat s-s\rvert/\sum_i\lvert x_i\rvert\f$ for summing n values in `dtype`.
///
/// With unit roundoff u (\f$2^{-24}\f$ for float32, \f$2^{-53}\f$ for float64) and \f$\gamma_k=ku/(1-ku)\f$ (Higham, 2002):
/// naive \f$\gamma_{n-1}\f$; pairwise \f$\gamma_{\lceil\log_2 n\rceil+7}\f$ (the naive blocks of 8 add 7); Kahan \f$2u+O(nu^2)\f$, returned as
/// \f$2u+2nu^2\f$; wide accumulator \f$u+\gamma_{n-1}(2^{-53})\f$, the final rounding plus the double sum's error.
/// The bound for the result is this factor times \f$\sum_i\lvert x_i\rvert\f$, and typical errors are far smaller.
/// @param method  Summation method.
/// @param n  Number of terms (at least 1).
/// @param dtype  Float32 or Float64, the type the values are stored and returned in.
/// @return The bound factor.
/// @throws std::invalid_argument  For a non-floating dtype or n < 1.
[[nodiscard]] double summation_error_bound(Summation method, std::int64_t n, DType dtype);

}  // namespace datamunge::nn
