// NumPy .npy / .npz reading and writing.

#include "datamunge/nn/io.hpp"

#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "datamunge/image/detail/crc32.hpp"
#include "datamunge/image/detail/zlib_codec.hpp"
#include "datamunge/nn/ops.hpp"
#include "detail.hpp"

namespace datamunge::nn {

namespace {

using Bytes = std::vector<unsigned char>;

bool host_is_little_endian() {
  const std::uint16_t one = 1;
  unsigned char       first;
  std::memcpy(&first, &one, 1);
  return first == 1;
}

Bytes read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open '" + path + "' for reading");
  in.seekg(0, std::ios::end);
  const auto size = static_cast<std::size_t>(in.tellg());
  in.seekg(0, std::ios::beg);
  Bytes bytes(size);
  if (size && !in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size)))
    throw std::runtime_error("failed reading '" + path + "'");
  return bytes;
}

void write_file(const std::string& path, const Bytes& bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("cannot open '" + path + "' for writing");
  out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if (!out) throw std::runtime_error("failed writing '" + path + "'");
}

std::string npy_descr(DType dtype) {
  switch (dtype) {
    case DType::Bool: return "|b1";
    case DType::Int32: return "<i4";
    case DType::Int64: return "<i8";
    case DType::Float32: return "<f4";
    case DType::Float64: return "<f8";
  }
  throw std::logic_error("npy: unknown dtype");
}

// ---- header parsing ---------------------------------------------------------------

struct NpyHeader {
  char         kind{'f'};  // f, i, u, b
  std::size_t  item_size{4};
  bool         big_endian{false};
  bool         fortran{false};
  Shape        shape;
};

[[noreturn]] void bad_header(const std::string& why) { throw std::runtime_error("npy: malformed header: " + why); }

/// The text after `'key':` in the header dict.
std::size_t find_value(const std::string& header, const std::string& key) {
  std::size_t at = header.find("'" + key + "'");
  if (at == std::string::npos) at = header.find("\"" + key + "\"");
  if (at == std::string::npos) bad_header("missing key " + key);
  at = header.find(':', at);
  if (at == std::string::npos) bad_header("missing ':' after " + key);
  ++at;
  while (at < header.size() && std::isspace(static_cast<unsigned char>(header[at]))) ++at;
  return at;
}

NpyHeader parse_header(const std::string& header) {
  NpyHeader h;
  // descr
  std::size_t at = find_value(header, "descr");
  if (at >= header.size() || (header[at] != '\'' && header[at] != '"')) bad_header("descr is not a string");
  const char        quote = header[at];
  const std::size_t end   = header.find(quote, at + 1);
  if (end == std::string::npos) bad_header("unterminated descr");
  const std::string descr = header.substr(at + 1, end - at - 1);
  if (descr.size() < 3) bad_header("unsupported descr '" + descr + "'");
  const char order = descr[0];
  if (order == '>')
    h.big_endian = true;
  else if (order == '<' || order == '|')
    h.big_endian = false;
  else if (order == '=')
    h.big_endian = !host_is_little_endian();
  else
    bad_header("unsupported byte order in '" + descr + "'");
  h.kind = descr[1];
  try {
    h.item_size = static_cast<std::size_t>(std::stoul(descr.substr(2)));
  } catch (const std::exception&) {
    bad_header("unsupported descr '" + descr + "'");
  }
  // fortran_order
  at = find_value(header, "fortran_order");
  if (header.compare(at, 4, "True") == 0)
    h.fortran = true;
  else if (header.compare(at, 5, "False") == 0)
    h.fortran = false;
  else
    bad_header("fortran_order is not True/False");
  // shape
  at = find_value(header, "shape");
  if (at >= header.size() || header[at] != '(') bad_header("shape is not a tuple");
  const std::size_t close = header.find(')', at);
  if (close == std::string::npos) bad_header("unterminated shape");
  const std::string inner = header.substr(at + 1, close - at - 1);
  std::size_t       i     = 0;
  while (i < inner.size()) {
    while (i < inner.size() && (std::isspace(static_cast<unsigned char>(inner[i])) || inner[i] == ',')) ++i;
    if (i >= inner.size()) break;
    std::size_t j = i;
    while (j < inner.size() && std::isdigit(static_cast<unsigned char>(inner[j]))) ++j;
    if (j == i) bad_header("bad shape '(" + inner + ")'");
    // Python 2 era files may write "3L".
    h.shape.push_back(static_cast<std::int64_t>(std::stoll(inner.substr(i, j - i))));
    i = j;
    if (i < inner.size() && inner[i] == 'L') ++i;
  }
  return h;
}

/// Target dtype for a stored element type.
DType target_dtype(const NpyHeader& h) {
  const auto unsupported = [&]() -> DType {
    throw std::runtime_error("npy: unsupported element type '" + std::string(1, h.kind) + std::to_string(h.item_size) +
                             "'");
  };
  switch (h.kind) {
    case 'f':
      if (h.item_size == 4) return DType::Float32;
      if (h.item_size == 8) return DType::Float64;
      return unsupported();
    case 'i':
      if (h.item_size == 1 || h.item_size == 2 || h.item_size == 4) return DType::Int32;
      if (h.item_size == 8) return DType::Int64;
      return unsupported();
    case 'u':
      if (h.item_size == 1 || h.item_size == 2) return DType::Int32;
      if (h.item_size == 4) return DType::Int64;
      return unsupported();
    case 'b':
      if (h.item_size == 1) return DType::Bool;
      return unsupported();
    default: return unsupported();
  }
}

/// Reads one element at `p` (already byte-order corrected copy) as the target type.
template <typename T>
T read_element(const unsigned char* p, const NpyHeader& h, bool swap) {
  unsigned char buf[8];
  for (std::size_t k = 0; k < h.item_size; ++k) buf[k] = swap ? p[h.item_size - 1 - k] : p[k];
  switch (h.kind) {
    case 'f':
      if (h.item_size == 4) {
        float v;
        std::memcpy(&v, buf, 4);
        return static_cast<T>(v);
      } else {
        double v;
        std::memcpy(&v, buf, 8);
        return static_cast<T>(v);
      }
    case 'b': return static_cast<T>(buf[0] != 0);
    case 'i':
      switch (h.item_size) {
        case 1: {
          std::int8_t v;
          std::memcpy(&v, buf, 1);
          return static_cast<T>(v);
        }
        case 2: {
          std::int16_t v;
          std::memcpy(&v, buf, 2);
          return static_cast<T>(v);
        }
        case 4: {
          std::int32_t v;
          std::memcpy(&v, buf, 4);
          return static_cast<T>(v);
        }
        default: {
          std::int64_t v;
          std::memcpy(&v, buf, 8);
          return static_cast<T>(v);
        }
      }
    default:  // 'u'
      switch (h.item_size) {
        case 1: return static_cast<T>(buf[0]);
        case 2: {
          std::uint16_t v;
          std::memcpy(&v, buf, 2);
          return static_cast<T>(v);
        }
        default: {
          std::uint32_t v;
          std::memcpy(&v, buf, 4);
          return static_cast<T>(v);
        }
      }
  }
}

// ---- little-endian integer helpers for ZIP -----------------------------------------------

void put16(Bytes& out, std::uint32_t v) {
  out.push_back(static_cast<unsigned char>(v & 0xff));
  out.push_back(static_cast<unsigned char>((v >> 8) & 0xff));
}
void put32(Bytes& out, std::uint32_t v) {
  put16(out, v & 0xffff);
  put16(out, (v >> 16) & 0xffff);
}
std::uint32_t get16(const Bytes& b, std::size_t at) {
  if (at + 2 > b.size()) throw std::runtime_error("npz: truncated archive");
  return static_cast<std::uint32_t>(b[at]) | (static_cast<std::uint32_t>(b[at + 1]) << 8);
}
std::uint32_t get32(const Bytes& b, std::size_t at) { return get16(b, at) | (get16(b, at + 2) << 16); }
std::uint64_t get64(const Bytes& b, std::size_t at) {
  return static_cast<std::uint64_t>(get32(b, at)) | (static_cast<std::uint64_t>(get32(b, at + 4)) << 32);
}

std::uint32_t crc32(const Bytes& data) {
  return image::detail::crc32_update(0, data.data(), data.size());
}

}  // namespace

// ---- npy -------------------------------------------------------------------------------------

std::vector<unsigned char> encode_npy(const Tensor& tensor) {
  if (tensor.device().backend != compute::Backend::CPU)
    throw std::logic_error("encode_npy: transfer GPU tensors to an explicit CPU context first");
  const Tensor packed = tensor.detach().contiguous();
  std::string  header = "{'descr': '" + npy_descr(packed.dtype()) + "', 'fortran_order': False, 'shape': (";
  for (std::size_t i = 0; i < packed.shape().size(); ++i) {
    header += std::to_string(packed.shape()[i]);
    if (packed.shape().size() == 1 || i + 1 < packed.shape().size()) header += ",";
    if (i + 1 < packed.shape().size()) header += " ";
  }
  header += "), }";
  // Pad with spaces so magic + version + length + header + '\n' is a multiple of 64.
  const std::size_t prefix = 10;
  while ((prefix + header.size() + 1) % 64 != 0) header += ' ';
  header += '\n';
  if (header.size() > 0xffff) throw std::runtime_error("npy: header too long");

  Bytes out = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
  put16(out, static_cast<std::uint32_t>(header.size()));
  out.insert(out.end(), header.begin(), header.end());

  const std::size_t   count = static_cast<std::size_t>(packed.numel());
  const std::size_t   size  = dtype_size(packed.dtype());
  const unsigned char* src  = packed.impl()->storage->bytes.host_data() + packed.offset() * static_cast<std::int64_t>(size);
  const std::size_t   start = out.size();
  out.resize(start + count * size);
  if (host_is_little_endian() || size == 1) {
    if (count) std::memcpy(out.data() + start, src, count * size);
  } else {
    for (std::size_t i = 0; i < count; ++i)
      for (std::size_t k = 0; k < size; ++k) out[start + i * size + k] = src[i * size + size - 1 - k];
  }
  return out;
}

Tensor decode_npy(const std::vector<unsigned char>& bytes) {
  static const unsigned char magic[6] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
  if (bytes.size() < 10 || std::memcmp(bytes.data(), magic, 6) != 0) throw std::runtime_error("npy: bad magic");
  const int   major = bytes[6];
  std::size_t header_len, header_start;
  if (major == 1) {
    header_len   = bytes[8] | (static_cast<std::size_t>(bytes[9]) << 8);
    header_start = 10;
  } else if (major == 2 || major == 3) {
    if (bytes.size() < 12) throw std::runtime_error("npy: truncated header");
    header_len = bytes[8] | (static_cast<std::size_t>(bytes[9]) << 8) | (static_cast<std::size_t>(bytes[10]) << 16) |
                 (static_cast<std::size_t>(bytes[11]) << 24);
    header_start = 12;
  } else {
    throw std::runtime_error("npy: unsupported format version " + std::to_string(major));
  }
  if (header_start + header_len > bytes.size()) throw std::runtime_error("npy: truncated header");
  const NpyHeader h = parse_header(std::string(bytes.begin() + static_cast<std::ptrdiff_t>(header_start),
                                               bytes.begin() + static_cast<std::ptrdiff_t>(header_start + header_len)));
  const DType        dtype = target_dtype(h);
  const std::int64_t count = shape_numel(h.shape);
  const std::size_t  data  = header_start + header_len;
  if (bytes.size() - data != static_cast<std::size_t>(count) * h.item_size)
    throw std::runtime_error("npy: data size " + std::to_string(bytes.size() - data) + " does not match shape " +
                             shape_str(h.shape) + " of " + std::to_string(h.item_size) + "-byte elements");

  // Fortran order: the data is the transpose's C-order layout.
  Shape stored = h.shape;
  if (h.fortran) stored.assign(h.shape.rbegin(), h.shape.rend());
  Tensor     out  = empty(stored, dtype);
  const bool swap = h.item_size > 1 && h.big_endian == host_is_little_endian();
  detail::dispatch(dtype, [&](auto tag) {
    using T = typename decltype(tag)::type;
    T* dst  = out.data_ptr<T>();
    for (std::int64_t i = 0; i < count; ++i)
      dst[i] = read_element<T>(bytes.data() + data + static_cast<std::size_t>(i) * h.item_size, h, swap);
  });
  if (h.fortran && h.shape.size() > 1) out = transpose(out).contiguous();
  return out;
}

void   save_npy(const std::string& path, const Tensor& tensor) { write_file(path, encode_npy(tensor)); }
Tensor load_npy(const std::string& path) {
  try {
    return decode_npy(read_file(path));
  } catch (const std::runtime_error& e) {
    throw std::runtime_error(std::string(e.what()) + " (in '" + path + "')");
  }
}

// ---- npz (ZIP archive of .npy members) -------------------------------------------------------

std::vector<unsigned char> encode_npz(const NamedTensors& arrays) {
  struct Entry {
    std::string   name;
    std::uint32_t crc, size, offset;
  };
  Bytes              out;
  std::vector<Entry> entries;
  for (const auto& [name, tensor] : arrays) {
    if (name.empty()) throw std::invalid_argument("save_npz: empty array name");
    const Bytes       data   = encode_npy(tensor);
    const std::string member = name + ".npy";
    if (data.size() >= 0xffffffffULL || out.size() >= 0xffffffffULL)
      throw std::runtime_error("save_npz: archives over 4 GiB (ZIP64) are not supported");
    Entry e{member, crc32(data), static_cast<std::uint32_t>(data.size()), static_cast<std::uint32_t>(out.size())};
    put32(out, 0x04034b50);  // local file header
    put16(out, 20);          // version needed
    put16(out, 0);           // flags
    put16(out, 0);           // method: stored
    put16(out, 0);           // time
    put16(out, 0x21);        // date: 1980-01-01
    put32(out, e.crc);
    put32(out, e.size);
    put32(out, e.size);
    put16(out, static_cast<std::uint32_t>(member.size()));
    put16(out, 0);  // extra length
    out.insert(out.end(), member.begin(), member.end());
    out.insert(out.end(), data.begin(), data.end());
    entries.push_back(e);
  }
  const std::size_t directory = out.size();
  for (const auto& e : entries) {
    put32(out, 0x02014b50);  // central directory header
    put16(out, 20);          // version made by
    put16(out, 20);          // version needed
    put16(out, 0);
    put16(out, 0);
    put16(out, 0);
    put16(out, 0x21);
    put32(out, e.crc);
    put32(out, e.size);
    put32(out, e.size);
    put16(out, static_cast<std::uint32_t>(e.name.size()));
    put16(out, 0);  // extra
    put16(out, 0);  // comment
    put16(out, 0);  // disk
    put16(out, 0);  // internal attributes
    put32(out, 0);  // external attributes
    put32(out, e.offset);
    out.insert(out.end(), e.name.begin(), e.name.end());
  }
  const std::size_t directory_size = out.size() - directory;
  if (entries.size() > 0xffff || out.size() >= 0xffffffffULL)
    throw std::runtime_error("save_npz: too many members or too large for a non-ZIP64 archive");
  put32(out, 0x06054b50);  // end of central directory
  put16(out, 0);
  put16(out, 0);
  put16(out, static_cast<std::uint32_t>(entries.size()));
  put16(out, static_cast<std::uint32_t>(entries.size()));
  put32(out, static_cast<std::uint32_t>(directory_size));
  put32(out, static_cast<std::uint32_t>(directory));
  put16(out, 0);
  return out;
}

void save_npz(const std::string& path, const NamedTensors& arrays) { write_file(path, encode_npz(arrays)); }

namespace {

NamedTensors decode_npz_impl(const Bytes& b, const std::string& source) {
  const auto fail = [&](const std::string& why) -> std::runtime_error {
    return std::runtime_error("npz: " + why + (source.empty() ? std::string() : " (in '" + source + "')"));
  };
  // Find the end-of-central-directory record (it may be followed by a comment).
  if (b.size() < 22) throw fail("file too small to be a ZIP archive");
  std::size_t eocd   = std::string::npos;
  std::size_t lowest = b.size() > 22 + 0xffff ? b.size() - 22 - 0xffff : 0;
  for (std::size_t at = b.size() - 22 + 1; at-- > lowest;)
    if (get32(b, at) == 0x06054b50) {
      eocd = at;
      break;
    }
  if (eocd == std::string::npos) throw fail("no ZIP end-of-central-directory record");
  std::uint64_t count  = get16(b, eocd + 10);
  std::uint64_t offset = get32(b, eocd + 16);
  // ZIP64 end record, as written by Python's zipfile for large archives.
  if ((count == 0xffff || offset == 0xffffffff) && eocd >= 20 && get32(b, eocd - 20) == 0x07064b50) {
    const std::uint64_t record = get64(b, eocd - 20 + 8);
    if (record + 56 > b.size() || get32(b, record) != 0x06064b50) throw fail("bad ZIP64 end record");
    count  = get64(b, record + 32);
    offset = get64(b, record + 48);
  }

  NamedTensors result;
  std::size_t  at = static_cast<std::size_t>(offset);
  for (std::uint64_t n = 0; n < count; ++n) {
    if (get32(b, at) != 0x02014b50) throw fail("bad central directory entry");
    const std::uint32_t flags        = get16(b, at + 8);
    const std::uint32_t method       = get16(b, at + 10);
    const std::uint32_t crc          = get32(b, at + 16);
    std::uint64_t       csize        = get32(b, at + 20);
    std::uint64_t       usize        = get32(b, at + 24);
    const std::uint32_t name_len     = get16(b, at + 28);
    const std::uint32_t extra_len    = get16(b, at + 30);
    const std::uint32_t comment_len  = get16(b, at + 32);
    std::uint64_t       local_offset = get32(b, at + 42);
    if (at + 46 + name_len + extra_len > b.size()) throw fail("truncated central directory");
    const std::string name(b.begin() + static_cast<std::ptrdiff_t>(at + 46),
                           b.begin() + static_cast<std::ptrdiff_t>(at + 46 + name_len));
    // ZIP64 extra field: the 0xffffffff fields follow in a fixed order.
    for (std::size_t x = at + 46 + name_len; x + 4 <= at + 46 + name_len + extra_len;) {
      const std::uint32_t id = get16(b, x), len = get16(b, x + 2);
      if (id == 0x0001) {
        std::size_t field = x + 4;
        if (usize == 0xffffffff) usize = get64(b, field), field += 8;
        if (csize == 0xffffffff) csize = get64(b, field), field += 8;
        if (local_offset == 0xffffffff) local_offset = get64(b, field);
      }
      x += 4 + len;
    }
    if (flags & 1u) throw fail("encrypted member '" + name + "'");
    if (get32(b, static_cast<std::size_t>(local_offset)) != 0x04034b50) throw fail("bad local header for '" + name + "'");
    const std::size_t data = static_cast<std::size_t>(local_offset) + 30 +
                             get16(b, static_cast<std::size_t>(local_offset) + 26) +
                             get16(b, static_cast<std::size_t>(local_offset) + 28);
    if (data + csize > b.size()) throw fail("member '" + name + "' runs past the end of the file");
    Bytes contents;
    if (method == 0) {
      if (csize != usize) throw fail("stored member '" + name + "' has mismatched sizes");
      contents.assign(b.begin() + static_cast<std::ptrdiff_t>(data),
                      b.begin() + static_cast<std::ptrdiff_t>(data + csize));
    } else if (method == 8) {
      contents = image::detail::deflate_decompress(b.data() + data, static_cast<std::size_t>(csize),
                                                   static_cast<std::size_t>(usize));
    } else {
      throw fail("member '" + name + "' uses unsupported compression method " + std::to_string(method));
    }
    if (crc32(contents) != crc) throw fail("CRC-32 mismatch in member '" + name + "'");
    std::string key = name;
    if (key.size() > 4 && key.compare(key.size() - 4, 4, ".npy") == 0) key.resize(key.size() - 4);
    try {
      result.emplace_back(key, decode_npy(contents));
    } catch (const std::runtime_error& e) {
      throw fail(std::string(e.what()) + " in member '" + name + "'");
    }
    at += 46 + name_len + extra_len + comment_len;
  }
  return result;
}

}  // namespace

NamedTensors decode_npz(const std::vector<unsigned char>& bytes) { return decode_npz_impl(bytes, ""); }

NamedTensors load_npz(const std::string& path) { return decode_npz_impl(read_file(path), path); }

}  // namespace datamunge::nn
