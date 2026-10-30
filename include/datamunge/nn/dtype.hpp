#pragma once

/// @file dtype.hpp
/// @brief Element types for `datamunge::nn` tensors.

#include <cstddef>
#include <cstdint>
#include <string>

/// @brief Tensor storage, differentiation, neural network operations, and training APIs.
namespace datamunge::nn {

/// Element type of an `nn::Tensor`. Bool is stored as one byte per element.
enum class DType {
  Bool,     ///< Boolean values stored in one byte, not packed bits.
  Int32,    ///< Signed 32-bit integers.
  Int64,    ///< Signed 64-bit integers.
  Float32,  ///< C++ float elements (4 bytes on supported platforms).
  Float64   ///< C++ double elements (8 bytes on supported platforms).
};

/// Short lowercase name of a dtype ("float32", "int64", ...), or "unknown" for an invalid value.
std::string dtype_name(DType dtype);

/// Size of one element in bytes, or zero for an invalid enum value.
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
/// @brief Maps C++ bool storage to the tensor boolean dtype.
template <>
struct dtype_of<bool> {
  /// @brief Dtype for a C++ bool element.
  static constexpr DType value = DType::Bool;
};
/// @brief Maps signed 32-bit C++ integers to tensor Int32 storage.
template <>
struct dtype_of<std::int32_t> {
  /// @brief Dtype for a std::int32_t element.
  static constexpr DType value = DType::Int32;
};
/// @brief Maps signed 64-bit C++ integers to tensor Int64 storage.
template <>
struct dtype_of<std::int64_t> {
  /// @brief Dtype for a std::int64_t element.
  static constexpr DType value = DType::Int64;
};
/// @brief Maps C++ float elements to tensor Float32 storage.
template <>
struct dtype_of<float> {
  /// @brief Dtype for a C++ float element.
  static constexpr DType value = DType::Float32;
};
/// @brief Maps C++ double elements to tensor Float64 storage.
template <>
struct dtype_of<double> {
  /// @brief Dtype for a C++ double element.
  static constexpr DType value = DType::Float64;
};

}  // namespace datamunge::nn
