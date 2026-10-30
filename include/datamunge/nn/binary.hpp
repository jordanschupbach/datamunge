// Bounded binary file reads and checked dimension arithmetic for data loaders
// (port of tflow's common/data/binary.h).
#pragma once

/// @file binary.hpp
/// @brief Bounded binary file access and checked loader arithmetic.

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

/// @brief Validated host-side formats and fitted preprocessing for neural-network inputs.
namespace datamunge::nn::data {

/// @brief Owned sequence of eight-bit bytes.
using Bytes = std::vector<std::uint8_t>;

/// @brief Read an entire binary file subject to a byte limit.
///
/// The size is checked before allocation. This is not a concurrent-writer snapshot guarantee.
/// @param path Input file; no expansion of shell variables.
/// @param max_bytes Maximum file length, default 512 MiB.
/// @return Owned bytes, including an empty vector for an empty file.
/// @throws std::runtime_error On open/read errors, a growing file or an exceeded limit.
///
/// @par Example
/// @code
/// namespace d = dm::nn::data;
/// const auto bytes = d::ReadBinary("weights.npz");
/// @endcode
/// @see WriteBinary()
inline Bytes ReadBinary(
    const std::filesystem::path& path,
    std::size_t                  max_bytes = 512ull * 1024 * 1024) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in)
    throw std::runtime_error("cannot open " + path.string());
  const auto size = in.tellg();
  if (size < 0 || std::uint64_t(size) > max_bytes)
    throw std::runtime_error("file exceeds byte limit: " + path.string());
  Bytes result(static_cast<std::size_t>(size));
  in.seekg(0);
  if (!result.empty() &&
      !in.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(result.size())))
    throw std::runtime_error("short read: " + path.string());
  if (in.peek() != std::char_traits<char>::eof())
    throw std::runtime_error("file grew while reading: " + path.string());
  return result;
}

/// @brief Replace a binary file with exactly the supplied bytes.
/// @param path Output file; its parent must exist.
/// @param bytes Owned input bytes; unchanged.
/// @throws std::runtime_error On open, write or close failure.
///
/// @par Example
/// @code
/// namespace d = dm::nn::data;
/// d::WriteBinary("empty.bin", {});
/// @endcode
/// @see ReadBinary()
inline void WriteBinary(const std::filesystem::path& path, const Bytes& bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out)
    throw std::runtime_error("cannot write " + path.string());
  if (!bytes.empty())
    out.write(
        reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  out.close();
  if (!out)
    throw std::runtime_error("failed writing " + path.string());
}

/// @brief Multiply loader dimensions with overflow detection.
/// @param a First unsigned dimension.
/// @param b Second unsigned dimension.
/// @return Exact product; zero if either factor is zero.
/// @throws std::invalid_argument If the product exceeds size_t.
///
/// @par Example
/// @code
/// namespace d = dm::nn::data;
/// const auto pixels = d::Product(32, 32); // 1024
/// @endcode
/// @see BigEndian32()
inline std::size_t Product(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b)
    throw std::invalid_argument("dataset dimensions overflow");
  return a * b;
}

/// @brief Decode a four-byte big-endian unsigned integer.
/// @param bytes Input storage, unchanged.
/// @param offset Zero-based position of its first byte.
/// @return Decoded 32-bit value.
/// @throws std::invalid_argument If fewer than four bytes remain.
///
/// @par Example
/// @code
/// namespace d = dm::nn::data;
/// const auto rank = d::BigEndian32({0,0,8,3}, 0); // 2051
/// @endcode
/// @see Product()
inline std::uint32_t BigEndian32(const Bytes& bytes, std::size_t offset) {
  if (offset > bytes.size() || bytes.size() - offset < 4)
    throw std::invalid_argument("truncated integer");
  return (std::uint32_t(bytes[offset]) << 24) | (std::uint32_t(bytes[offset + 1]) << 16) |
         (std::uint32_t(bytes[offset + 2]) << 8) | std::uint32_t(bytes[offset + 3]);
}

} // namespace datamunge::nn::data
