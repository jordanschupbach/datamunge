#pragma once

/// @file int8_dense.hpp
/// @brief Symmetric signed quantization, integer dense inference, and fake quantization.

#include <datamunge/nn/autograd.hpp>
#include <datamunge/nn/ops.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace datamunge::nn {

/// @brief Packed signed int8 values and their real-valued scale.
/// @par Example
/// @code
/// const auto q = dm::nn::quantize_int8(dm::nn::tensor({-1., 1.}), 1.);
/// std::println("{}", int(q.values.front())); // -1
/// @endcode
/// @see quantize_int8()
struct SymmetricInt8 {
  std::vector<std::int8_t> values; ///< Row-major integer codes in [-127, 127].
  Shape shape; ///< Logical tensor dimensions.
  double scale; ///< Positive multiplier for converting one code to real units.
};

namespace detail {

/// @brief Validate a real CPU input before symmetric quantization.
inline void check_int8_input(const Tensor& x, double scale) {
  if (!x.defined() || x.device().backend != compute::Backend::CPU ||
      (x.dtype() != DType::Float32 && x.dtype() != DType::Float64) ||
      !(scale > 0.) || !std::isfinite(127. * scale) ||
      !all(is_finite(x)).item<bool>())
    throw std::invalid_argument("int8 quantization: expected a finite CPU float tensor and valid scale");
}

/// @brief Compute integer grid coordinates with saturation and ties-to-even rounding.
inline Tensor symmetric_int8_codes(const Tensor& x, double scale) {
  return round(maximum(full({}, -127., DType::Float64),
                       minimum(full({}, 127., DType::Float64), x / scale)));
}

} // namespace detail

/// @brief Calibrate a symmetric int8 scale using the largest absolute input value.
/// @param training Finite nonempty CPU Float32 or Float64 calibration tensor.
/// @return Maximum absolute value divided by 127, or one for all-zero values.
/// Exceptional endpoints are adjusted to preserve a positive scale and finite clipping bounds.
/// @throws std::invalid_argument Invalid calibration input.
/// @par Example
/// @code
/// std::println("{}", dm::nn::calibrate_int8_scale(dm::nn::tensor({-127., 2.}))); // 1
/// @endcode
/// @see quantize_int8(), fake_quantize_int8()
[[nodiscard]] inline double calibrate_int8_scale(const Tensor& training) {
  detail::check_int8_input(training, 1.);
  if (training.numel() == 0) throw std::invalid_argument("calibrate_int8_scale: empty tensor");
  const double peak = max(abs(training)).item();
  if (peak == 0.) return 1.;
  const double scale = std::max(peak / 127., std::numeric_limits<double>::denorm_min());
  return std::isfinite(127. * scale) ? scale : std::nextafter(scale, 0.);
}

/// @brief Round a real tensor to packed symmetric int8 values.
/// @param x Finite CPU Float32 or Float64 tensor.
/// @param scale Positive scale with finite clipping bounds.
/// @return Packed row-major codes, saturated to [-127, 127], rounded ties to even.
/// @throws std::invalid_argument Invalid input or scale.
/// @par Example
/// @code
/// const auto q = dm::nn::quantize_int8(dm::nn::tensor({-.5, 1.5, 200.}), 1.);
/// std::println("{}", int(q.values.back())); // 127
/// @endcode
/// @see calibrate_int8_scale(), int8_dense()
[[nodiscard]] inline SymmetricInt8 quantize_int8(const Tensor& x, double scale) {
  detail::check_int8_input(x, scale);
  auto codes = detail::symmetric_int8_codes(x.detach().to(DType::Float64), scale).to_vector<std::int32_t>();
  return {std::vector<std::int8_t>(codes.begin(), codes.end()), x.shape(), scale};
}

/// @brief Simulate the int8 grid with a clipped straight-through gradient.
/// @param x Finite CPU Float32 or Float64 tensor.
/// @param scale Positive scale with finite clipping bounds.
/// @return Dequantized values with the input dtype. The backward multiplier
/// is one inside the open clipping interval and zero at or outside either bound.
/// @throws std::invalid_argument Invalid input or scale.
/// @par Example
/// @code
/// const auto q = dm::nn::fake_quantize_int8(dm::nn::tensor({1.4, 200.}), 1.);
/// std::println("{}", q.values_str()); // [1, 127]
/// @endcode
/// @par References
/// - Jacob et al. (2018), Quantization and Training of Neural Networks for Efficient
///   Integer-Arithmetic-Only Inference, https://arxiv.org/abs/1712.05877.
///   This example uses symmetric zero-centered scales.
/// @see quantize_int8(), stop_gradient()
[[nodiscard]] inline Tensor fake_quantize_int8(const Tensor& x, double scale) {
  detail::check_int8_input(x, scale);
  const auto wide = x.to(DType::Float64);
  const auto lo = full({}, -127. * scale, DType::Float64);
  const auto hi = full({}, 127. * scale, DType::Float64);
  const auto clipped = maximum(lo, minimum(hi, wide));
  const auto grid = detail::symmetric_int8_codes(wide, scale) * scale;
  return (clipped + stop_gradient(grid - clipped)).to(x.dtype());
}

/// @brief Multiply two packed int8 matrices, retaining their exact int32 accumulator.
/// @param features Packed [rows, inputs] matrix with codes in [-127, 127].
/// @param weights Packed [inputs, outputs] matrix with codes in [-127, 127].
/// @return Int32 matrix of sums and Float64 predictions scaled by both operands' scales.
/// @throws std::invalid_argument Invalid shape, storage, or scale, or possible int32 overflow.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto x = n::quantize_int8(n::tensor({2.}, {1,1}), 1.);
/// const auto [sum, prediction] = n::int8_dense(x, x);
/// std::println("{} {}", sum.item<int>(), prediction.item()); // 4 4
/// @endcode
/// @see quantize_int8(), matmul()
[[nodiscard]] inline std::pair<Tensor, Tensor> int8_dense(
    const SymmetricInt8& features, const SymmetricInt8& weights) {
  /// @brief Validate shape, payload size, codes and scale for an int32-safe multiplication.
  auto valid = [](const SymmetricInt8& q) {
    return q.shape.size() == 2 && q.shape[0] > 0 && q.shape[1] > 0 &&
           static_cast<std::uint64_t>(q.shape[0]) <= q.values.size() / static_cast<std::uint64_t>(q.shape[1]) &&
           static_cast<std::uint64_t>(q.shape[0]) * static_cast<std::uint64_t>(q.shape[1]) == q.values.size() &&
           q.scale > 0. && std::isfinite(127. * q.scale) &&
           std::ranges::none_of(q.values, [](auto code) { return code == -128; });
  };
  if (!valid(features) || !valid(weights) || features.shape[1] != weights.shape[0] ||
      features.shape[1] > std::numeric_limits<std::int32_t>::max() / (127 * 127) ||
      !std::isfinite(features.scale * weights.scale))
    throw std::invalid_argument("int8_dense: incompatible operands or possible accumulator overflow");

  const auto x = from_vector(std::vector<std::int32_t>(features.values.begin(), features.values.end()), features.shape);
  const auto w = from_vector(std::vector<std::int32_t>(weights.values.begin(), weights.values.end()), weights.shape);
  auto accumulator = matmul(x, w);
  auto prediction = accumulator.to(DType::Float64) * (features.scale * weights.scale);
  return {std::move(accumulator), std::move(prediction)};
}

} // namespace datamunge::nn
