#pragma once

/// @file dtype.hpp
/// @brief Element types for `datamunge::nn` tensors.

#include <cstddef>
#include <cstdint>
#include <string>

namespace datamunge::nn {

/// Element type of an `nn::Tensor`. Bool is stored as one byte per element.
enum class DType { Bool, Int32, Int64, Float32, Float64 };

/// Short lowercase name of a dtype ("float32", "int64", ...).
std::string dtype_name(DType dtype);

/// Size of one element in bytes.
std::size_t dtype_size(DType dtype);

/// True for Float32 and Float64.
constexpr bool is_floating(DType dtype) { return dtype == DType::Float32 || dtype == DType::Float64; }

/// True for Int32 and Int64.
constexpr bool is_integral(DType dtype) { return dtype == DType::Int32 || dtype == DType::Int64; }

/// The dtype both operands of a binary op are converted to: the "larger" of
/// the two in the order bool < int32 < int64 < float32 < float64.
constexpr DType promote_types(DType a, DType b) { return static_cast<int>(a) >= static_cast<int>(b) ? a : b; }

/// The float dtype a non-float input is converted to by float-valued ops
/// (exp, sqrt, mean, ...): float tensors keep their dtype, others use Float32.
constexpr DType to_floating(DType dtype) { return is_floating(dtype) ? dtype : DType::Float32; }

/// Maps a C++ element type to its dtype: `dtype_of<float>::value == DType::Float32`.
template <typename T>
struct dtype_of;
template <>
struct dtype_of<bool> {
  static constexpr DType value = DType::Bool;
};
template <>
struct dtype_of<std::int32_t> {
  static constexpr DType value = DType::Int32;
};
template <>
struct dtype_of<std::int64_t> {
  static constexpr DType value = DType::Int64;
};
template <>
struct dtype_of<float> {
  static constexpr DType value = DType::Float32;
};
template <>
struct dtype_of<double> {
  static constexpr DType value = DType::Float64;
};

}  // namespace datamunge::nn
