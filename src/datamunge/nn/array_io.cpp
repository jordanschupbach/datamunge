/// @file array_io.cpp
/// @brief Native numeric interchange writers for report preparation and standalone consumers.
#include <algorithm>
#include <cmath>
#include <datamunge/image/detail/crc32.hpp>
#include <datamunge/image/detail/zlib_codec.hpp>
#include <datamunge/nn/array_io.hpp>
#include <fstream>
#include <limits>
#include <set>

namespace datamunge::nn {
/// @brief Private byte-order, checked-file and ZIP-header helpers.
namespace {
using Bytes = std::vector<unsigned char>; ///< Owned interchange document bytes.

/// @brief Write an owned byte buffer without text transcoding.
/// @param path Destination file.
/// @param bytes Complete encoded document.
void write(const std::string& path, const Bytes& bytes) {
  std::ofstream out(path, std::ios::binary);
  if (!out)
    throw std::runtime_error("cannot write " + path);
  out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  out.close();
  if (!out)
    throw std::runtime_error("write failed: " + path);
}
/// @brief Append a little-endian two-byte ZIP field.
/// @param out Destination buffer.
/// @param x Field value.
void le16(Bytes& out, std::uint32_t x) {
  out.push_back(x & 255);
  out.push_back((x >> 8) & 255);
}
/// @brief Append a little-endian four-byte ZIP field.
/// @param out Destination buffer.
/// @param x Field value.
void le32(Bytes& out, std::uint32_t x) {
  le16(out, x & 65535);
  le16(out, x >> 16);
}
} // namespace
std::vector<unsigned char> encode_npy(const Tensor& tensor, bool fortran_order, bool big_endian) {
  auto              out   = encode_npy(tensor);
  const std::size_t start = 10 + out[8] + (std::size_t(out[9]) << 8),
                    width = dtype_size(tensor.dtype());
  std::string header(out.begin() + 10, out.begin() + start);
  if (fortran_order) {
    const auto at = header.find("'fortran_order': False");
    if (at == std::string::npos)
      throw std::runtime_error("unexpected native NPY header");
    header.replace(at + 17, 5, "True ");
  }
  if (big_endian && width > 1) {
    const auto at = header.find("'descr': '<");
    if (at == std::string::npos)
      throw std::runtime_error("unexpected native NPY dtype");
    header[at + 10] = '>';
  }
  std::copy(header.begin(), header.end(), out.begin() + 10);
  const Bytes c_order(out.begin() + start, out.end());
  for (std::int64_t i = 0; i < tensor.numel(); ++i) {
    std::int64_t logical = i;
    if (fortran_order) {
      std::int64_t remaining = i;
      logical                = 0;
      for (auto dimension : tensor.shape()) {
        const auto coordinate = remaining % dimension;
        remaining /= dimension;
        logical = logical * dimension + coordinate;
      }
    }
    for (std::size_t k = 0; k < width; ++k)
      out[start + std::size_t(i) * width + k] =
          c_order[std::size_t(logical) * width + (big_endian ? width - 1 - k : k)];
  }
  return out;
}
void save_npy(const std::string& path, const Tensor& tensor, bool fortran_order, bool big_endian) {
  write(path, encode_npy(tensor, fortran_order, big_endian));
}
std::vector<unsigned char> encode_npz(const NamedTensors& arrays, bool compressed) {
  std::set<std::string> names;
  for (const auto& [name, _] : arrays)
    if (name.empty() || name.size() > 65531 || !names.insert(name).second)
      throw std::invalid_argument("NPZ names must be nonempty, unique and at most65531 bytes");
  if (!compressed)
    return encode_npz(arrays);
  /// @brief ZIP32 central directory bookkeeping for one NPY member.
  struct Entry {
    std::string name;      ///< NPY member filename, including its suffix.
    std::uint32_t crc;     ///< CRC32 of the uncompressed NPY document.
    std::uint32_t size;    ///< Uncompressed NPY byte count.
    std::uint32_t packed;  ///< DEFLATE payload byte count.
    std::uint32_t offset;  ///< Local member header offset from the archive start.
  };
  Bytes              out;
  std::vector<Entry> entries;
  for (const auto& [name, tensor] : arrays) {
    const auto data = encode_npy(tensor), payload = image::detail::deflate_compress(data, 6);
    if (data.size() >= 0xffffffffULL || payload.size() >= 0xffffffffULL ||
        out.size() >= 0xffffffffULL)
      throw std::runtime_error("NPZ exceeds ZIP32 limits");
    Entry e{
        name + ".npy",
        image::detail::crc32_update(0, data.data(), data.size()),
        std::uint32_t(data.size()),
        std::uint32_t(payload.size()),
        std::uint32_t(out.size())};
    le32(out, 0x04034b50);
    le16(out, 20);
    le16(out, 0);
    le16(out, 8);
    le16(out, 0);
    le16(out, 0x21);
    le32(out, e.crc);
    le32(out, e.packed);
    le32(out, e.size);
    le16(out, e.name.size());
    le16(out, 0);
    out.insert(out.end(), e.name.begin(), e.name.end());
    out.insert(out.end(), payload.begin(), payload.end());
    entries.push_back(e);
  }
  const auto directory = out.size();
  for (const auto& e : entries) {
    le32(out, 0x02014b50);
    le16(out, 20);
    le16(out, 20);
    le16(out, 0);
    le16(out, 8);
    le16(out, 0);
    le16(out, 0x21);
    le32(out, e.crc);
    le32(out, e.packed);
    le32(out, e.size);
    le16(out, e.name.size());
    le16(out, 0);
    le16(out, 0);
    le16(out, 0);
    le16(out, 0);
    le32(out, 0);
    le32(out, e.offset);
    out.insert(out.end(), e.name.begin(), e.name.end());
  }
  const auto size = out.size() - directory;
  if (entries.size() > 65535 || out.size() >= 0xffffffffULL)
    throw std::runtime_error("NPZ exceeds ZIP32 limits");
  le32(out, 0x06054b50);
  le16(out, 0);
  le16(out, 0);
  le16(out, entries.size());
  le16(out, entries.size());
  le32(out, size);
  le32(out, directory);
  le16(out, 0);
  return out;
}
void save_npz(const std::string& path, const NamedTensors& arrays, bool compressed) {
  write(path, encode_npz(arrays, compressed));
}
void write_idx(const std::string& path, const Tensor& tensor, bool unsigned_byte) {
  if (tensor.device().backend != compute::Backend::CPU)
    throw std::logic_error("write_idx requires explicit CPU storage");
  if (tensor.dim() < 1 || tensor.dim() > 255)
    throw std::invalid_argument("IDX requires rank1..255");
  unsigned_byte |= tensor.dtype() == DType::Bool;
  int type;
  if (unsigned_byte)
    type = 8;
  else if (tensor.dtype() == DType::Int32)
    type = 12;
  else if (tensor.dtype() == DType::Float32)
    type = 13;
  else if (tensor.dtype() == DType::Float64)
    type = 14;
  else
    throw std::invalid_argument("unsupported IDX dtype");
  Bytes out{0, 0, static_cast<unsigned char>(type), static_cast<unsigned char>(tensor.dim())};
  for (auto d : tensor.shape()) {
    if (d <= 0 || std::uint64_t(d) > 0xffffffffULL)
      throw std::invalid_argument("IDX dimensions require positive UINT32 values");
    for (int k = 3; k >= 0; --k)
      out.push_back((std::uint64_t(d) >> (8 * k)) & 255);
  }
  if (unsigned_byte) {
    for (double value : tensor.to_vector<double>()) {
      if (!std::isfinite(value) || value < 0 || value > 255 || value != std::floor(value))
        throw std::invalid_argument("IDX byte values require integers in0..255");
      out.push_back(static_cast<unsigned char>(value));
    }
  } else {
    const auto        npy   = encode_npy(tensor, false, true);
    const std::size_t start = 10 + npy[8] + (std::size_t(npy[9]) << 8);
    out.insert(out.end(), npy.begin() + start, npy.end());
  }
  write(path, out);
}
} // namespace datamunge::nn
