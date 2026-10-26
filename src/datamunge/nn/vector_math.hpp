#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#if defined(__APPLE__) && defined(DATAMUNGE_HAVE_CBLAS)
#include <Accelerate/Accelerate.h>
#endif

namespace datamunge::nn::detail {
#if defined(__aarch64__) && defined(__ARM_NEON)
inline double sum_float_values(const float* values, std::int64_t count) {
  // Widen before every addition. Eight vector accumulators hide conversion/add
  // latency and cover sixteen values per iteration without Float32 summation.
  float64x2_t accumulators[8];
  for (auto& accumulator : accumulators)
    accumulator = vdupq_n_f64(0);
  std::int64_t i = 0;
  for (; i + 15 < count; i += 16) {
    for (int lane = 0; lane < 8; lane += 2) {
      const auto packed      = vld1q_f32(values + i + lane * 2);
      accumulators[lane]     = vaddq_f64(accumulators[lane], vcvt_f64_f32(vget_low_f32(packed)));
      accumulators[lane + 1] = vaddq_f64(accumulators[lane + 1], vcvt_high_f64_f32(packed));
    }
  }
  for (int half = 4; half; half /= 2)
    for (int lane = 0; lane < half; ++lane)
      accumulators[lane] = vaddq_f64(accumulators[lane], accumulators[lane + half]);
  double total = vaddvq_f64(accumulators[0]);
  for (; i < count; ++i)
    total += static_cast<double>(values[i]);
  return total;
}
#endif

template<typename T>
double sum_values(const T* values, std::int64_t count) {
#if defined(__aarch64__) && defined(__ARM_NEON)
  if constexpr (std::is_same<T, float>::value)
    if (count >= 32)
      return sum_float_values(values, count);
  if constexpr (std::is_same<T, double>::value) {
    // Short rows already vectorize well through the scalar lane loop below.
    if (count >= 256) {
      // Explicit vectors avoid strict-FP scalar reduction code generation.
      // Each lane still accumulates in double, without fast-math flags.
      float64x2_t accumulators[8];
      for (auto& accumulator : accumulators)
        accumulator = vdupq_n_f64(0);
      std::int64_t i = 0;
      for (; i + 15 < count; i += 16)
        for (int lane = 0; lane < 8; ++lane)
          accumulators[lane] = vaddq_f64(accumulators[lane], vld1q_f64(values + i + lane * 2));
      for (int half = 4; half; half /= 2)
        for (int lane = 0; lane < half; ++lane)
          accumulators[lane] = vaddq_f64(accumulators[lane], accumulators[lane + half]);
      double total = vaddvq_f64(accumulators[0]);
      for (; i < count; ++i)
        total += values[i];
      return total;
    }
  }
#endif
  double       a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
  std::int64_t i = 0;
  for (; i + 7 < count; i += 8) {
    a0 += values[i];
    a1 += values[i + 1];
    a2 += values[i + 2];
    a3 += values[i + 3];
    a4 += values[i + 4];
    a5 += values[i + 5];
    a6 += values[i + 6];
    a7 += values[i + 7];
  }
  double total = ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
  for (; i < count; ++i)
    total += values[i];
  return total;
}

// ARM FMAX (unlike FMAXNM) propagates NaNs, matching the reduction API.
// Other platforms use the same NaN-propagating scalar comparison.
template<typename T>
T max_values(const T* source, std::int64_t count) {
  T            largest = -std::numeric_limits<T>::infinity();
  std::int64_t i       = 0;
#if defined(__aarch64__) && defined(__ARM_NEON)
  if constexpr (std::is_same<T, float>::value) {
    auto packed = vdupq_n_f32(largest);
    for (; i + 3 < count; i += 4)
      packed = vmaxq_f32(packed, vld1q_f32(source + i));
    largest = vmaxvq_f32(packed);
  } else {
    auto packed = vdupq_n_f64(largest);
    for (; i + 1 < count; i += 2)
      packed = vmaxq_f64(packed, vld1q_f64(source + i));
    largest = vmaxvq_f64(packed);
  }
#endif
  for (; i < count; ++i)
    if (source[i] > largest || std::isnan(source[i]))
      largest = source[i];
  return largest;
}

// Contiguous vector exponential, including exact in-place operation. Reuse the
// already-linked Accelerate backend; other builds retain standard scalar math.
template<typename T>
void exp_values(const T* source, T* dest, std::int64_t count) {
#if defined(__APPLE__) && defined(DATAMUNGE_HAVE_CBLAS)
  if (count >= 32) {
    while (count > 0) {
      const int n =
          static_cast<int>(std::min<std::int64_t>(count, std::numeric_limits<int>::max()));
      if constexpr (std::is_same<T, float>::value)
        vvexpf(dest, source, &n);
      else
        vvexp(dest, source, &n);
      source += n;
      dest += n;
      count -= n;
    }
    return;
  }
#endif
  for (std::int64_t i = 0; i < count; ++i)
    dest[i] = std::exp(source[i]);
}
// Uses the existing platform vector library, preserving scalar fallback builds.
template<typename T>
void tanh_values(const T* source, T* dest, std::int64_t count) {
#if defined(__APPLE__) && defined(DATAMUNGE_HAVE_CBLAS)
  if (count >= 32) {
    while (count > 0) {
      const int n =
          static_cast<int>(std::min<std::int64_t>(count, std::numeric_limits<int>::max()));
      if constexpr (std::is_same<T, float>::value)
        vvtanhf(dest, source, &n);
      else
        vvtanh(dest, source, &n);
      source += n;
      dest += n;
      count -= n;
    }
    return;
  }
#endif
  for (std::int64_t i = 0; i < count; ++i)
    dest[i] = std::tanh(source[i]);
}

// Fresh, non-aliasing output serves as exponential scratch. The magnitude form
// avoids overflow at either tail, just like the scalar stable sigmoid.
template<typename T>
void sigmoid_values(const T* source, T* dest, std::int64_t count) {
#if defined(__APPLE__) && defined(DATAMUNGE_HAVE_CBLAS)
  for (std::int64_t i = 0; i < count; ++i)
    dest[i] = -std::abs(source[i]);
  exp_values(dest, dest, count);
  for (std::int64_t i = 0; i < count; ++i) {
    const T e = dest[i], denominator = T(1) + e;
    dest[i] = source[i] >= 0 ? T(1) / denominator : e / denominator;
  }
#else
  for (std::int64_t i = 0; i < count; ++i) {
    const T value = source[i];
    if (value >= 0) {
      const T e = std::exp(-value);
      dest[i] = T(1) / (T(1) + e);
    } else {
      const T e = std::exp(value);
      dest[i] = e / (T(1) + e);
    }
  }
#endif
}

} // namespace datamunge::nn::detail
