// Bounded binary file reads and checked dimension arithmetic for data loaders
// (port of tflow's common/data/binary.h).
#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace data {

using Bytes = std::vector<std::uint8_t>;

inline Bytes ReadBinary(const std::filesystem::path& path, std::size_t max_bytes = 512ull * 1024 * 1024) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("cannot open " + path.string());
  const auto size = in.tellg();
  if (size < 0 || std::uint64_t(size) > max_bytes) throw std::runtime_error("file exceeds byte limit: " + path.string());
  Bytes result(static_cast<std::size_t>(size));
  in.seekg(0);
  if (!result.empty() && !in.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(result.size())))
    throw std::runtime_error("short read: " + path.string());
  if (in.peek() != std::char_traits<char>::eof()) throw std::runtime_error("file grew while reading: " + path.string());
  return result;
}

inline void WriteBinary(const std::filesystem::path& path, const Bytes& bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("cannot write " + path.string());
  if (!bytes.empty()) out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  out.close();
  if (!out) throw std::runtime_error("failed writing " + path.string());
}

inline std::size_t Product(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b) throw std::invalid_argument("dataset dimensions overflow");
  return a * b;
}

inline std::uint32_t BigEndian32(const Bytes& bytes, std::size_t offset) {
  if (offset > bytes.size() || bytes.size() - offset < 4) throw std::invalid_argument("truncated integer");
  return (std::uint32_t(bytes[offset]) << 24) | (std::uint32_t(bytes[offset + 1]) << 16) |
         (std::uint32_t(bytes[offset + 2]) << 8) | std::uint32_t(bytes[offset + 3]);
}

}  // namespace data
