// TFRecord streaming I/O and a minimal tf.Example codec (port of tflow's
// common/data/tfrecord.h, which wraps TensorFlow's RecordReader/RecordWriter
// and the generated Example protobuf; nn links neither, so both are written
// out here from the published formats).
//
// TFRecord framing: uint64 little-endian payload length, masked CRC-32C of
// those 8 bytes, the payload, masked CRC-32C of the payload. Compression is
// explicit: "", "GZIP" (one gzip member around the whole stream) or "ZLIB"
// (one zlib stream). The record payload may be any bytes, including empty
// strings or serialized tf.Example protos. EOF is distinct from an empty
// record; checksum and truncation failures throw std::runtime_error, an
// unknown compression name std::invalid_argument.
//
// Both GZIP and ZLIB readers accept stored, fixed and dynamic DEFLATE blocks;
// the shared codec bounds unknown-size output with its default InflateLimit.
//
// tf.Example: Example{features=1: Features{feature=1: map<string, Feature>}},
// Feature oneof {bytes_list=1, float_list=2, int64_list=3}, each a repeated
// field 1 (floats and int64s packed, as proto3 writes them). Map entries are
// written in key order. The parser accepts packed and unpacked numbers and
// skips unknown fields.
#pragma once

/// @file tfrecord.hpp
/// @brief TFRecord framing, compression and a minimal tf.Example codec.

#include <array>
#include <cstdint>
#include <cstring>
#include <datamunge/image/detail/crc32.hpp>
#include <datamunge/image/detail/zlib_codec.hpp>
#include <datamunge/nn/ops.hpp>
#include <filesystem>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "datamunge/nn/binary.hpp"

namespace datamunge::nn::data {

// ---- CRC-32C (Castagnoli), masked as TensorFlow's crc32c::Mask ----------------

/// @brief Compute reflected CRC-32C (Castagnoli), including initial/final complements.
/// @param data Readable byte span; may be null only at zero length.
/// @param size Number of bytes.
/// @return CRC-32C; empty span gives zero.
///
/// @par Example
/// @code
/// namespace d = dm::nn::data;
/// const std::string s="123456789";
/// const auto crc=d::Crc32c(reinterpret_cast<const uint8_t*>(s.data()),s.size()); // 0xe3069283
/// @endcode
///
/// @par References
/// - TensorFlow TFRecord framing, https://www.tensorflow.org/tutorials/load_data/tfrecord
/// @see MaskedCrc()
inline std::uint32_t Crc32c(const std::uint8_t* data, std::size_t size) {
  // Immutable reflected Castagnoli lookup table; thread-safe static initialization.
  static const auto table = [] {
    std::array<std::uint32_t, 256> t{};
    for (std::uint32_t i = 0; i < 256; ++i) {
      std::uint32_t c = i;
      for (int k = 0; k < 8; ++k)
        c = c & 1 ? 0x82F63B78u ^ (c >> 1) : c >> 1;
      t[i] = c;
    }
    return t;
  }();
  std::uint32_t crc = 0xFFFFFFFFu;
  for (std::size_t i = 0; i < size; ++i)
    crc = table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
  return crc ^ 0xFFFFFFFFu;
}
/// @brief Apply TFRecord rotate-and-offset masking to CRC-32C.
/// @param data Readable bytes, unchanged.
/// @param size Byte count.
/// @return Masked checksum modulo 2^32.
///
/// @par Example
/// @code
/// namespace d = dm::nn::data;
/// const auto crc=d::MaskedCrc(nullptr,0); // 0xa282ead8
/// @endcode
/// @see Crc32c()
inline std::uint32_t MaskedCrc(const std::uint8_t* data, std::size_t size) {
  const std::uint32_t crc = Crc32c(data, size);
  return ((crc >> 15) | (crc << 17)) + 0xa282ead8u;
}

/// @brief Reject unsupported TFRecord compression names.
/// @param compression Exactly empty, GZIP or ZLIB; case-sensitive.
/// @throws std::invalid_argument On any other name.
///
/// @par Example
/// @code
/// namespace d = dm::nn::data;
/// d::ValidateCompression("GZIP");
/// @endcode
/// @see ReadTFRecords()
inline void ValidateCompression(const std::string& compression) {
  if (!compression.empty() && compression != "GZIP" && compression != "ZLIB")
    throw std::invalid_argument("TFRecord compression must be empty, GZIP or ZLIB");
}

/// @brief Byte-order and compression internals of the TFRecord codec.
namespace detail {

/// @brief Append the low bytes of an unsigned integer.
/// @param out Destination byte buffer.
/// @param value Unsigned integer.
/// @param bytes Byte count from 0 through 8; internal precondition.
inline void PutLE(Bytes& out, std::uint64_t value, int bytes) {
  for (int i = 0; i < bytes; ++i)
    out.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
}
/// @brief Read an unchecked little-endian integer after caller bounds checks.
/// @param in Input bytes.
/// @param at First byte index.
/// @param bytes Count from 0 through 8, within input bounds.
/// @return Decoded value.
inline std::uint64_t GetLE(const Bytes& in, std::size_t at, int bytes) {
  std::uint64_t v = 0;
  for (int i = 0; i < bytes; ++i)
    v |= std::uint64_t(in[at + i]) << (8 * i);
  return v;
}

/// The raw DEFLATE stream inside a zlib stream (2-byte header, 4-byte trailer).
/// @brief Remove the wrapper from a freshly compressed zlib stream.
/// @param raw Uncompressed bytes.
/// @return Raw DEFLATE bytes.
inline Bytes Deflate(const Bytes& raw) {
  const auto z = datamunge::image::detail::zlib_compress(raw);
  return Bytes(z.begin() + 2, z.end() - 4);
}

/// @brief Wrap a DEFLATE stream with a deterministic gzip header and trailer.
/// @param raw Uncompressed bytes.
/// @return One gzip member with CRC-32 and size trailer.
inline Bytes Gzip(const Bytes& raw) {
  Bytes       out{0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 255}; // no flags, no mtime, unknown OS
  const Bytes body = Deflate(raw);
  out.insert(out.end(), body.begin(), body.end());
  PutLE(out, datamunge::image::detail::crc32_update(0, raw.data(), raw.size()), 4);
  PutLE(out, raw.size() & 0xFFFFFFFFu, 4);
  return out;
}

/// @brief Decode one gzip member with bounds and trailer checks.
/// @param b Complete gzip member.
/// @return Uncompressed bytes; runtime_error on bad format or trailer.
inline Bytes Gunzip(const Bytes& b) {
  if (b.size() < 18 || b[0] != 0x1f || b[1] != 0x8b || b[2] != 8)
    throw std::runtime_error("GZIP TFRecord: bad header");
  const unsigned flags = b[3];
  std::size_t    at    = 10;
  if (flags & 0x04u) {
    if (at + 2 > b.size())
      throw std::runtime_error("GZIP TFRecord: truncated header");
    at += 2 + (b[at] | (std::size_t(b[at + 1]) << 8));
  }
  for (unsigned bit : {0x08u, 0x10u}) {
    if (!(flags & bit))
      continue;
    while (at < b.size() && b[at])
      ++at;
    ++at;
  }
  if (flags & 0x02u)
    at += 2;
  if (at + 8 > b.size())
    throw std::runtime_error("GZIP TFRecord: truncated stream");
  const std::size_t          trailer = b.size() - 8;
  const auto                 crc     = static_cast<std::uint32_t>(GetLE(b, trailer, 4));
  const std::size_t          isize   = static_cast<std::size_t>(GetLE(b, trailer + 4, 4));
  std::vector<unsigned char> out;
  try {
    out = datamunge::image::detail::deflate_decompress(b.data() + at, trailer - at, isize);
  } catch (const std::runtime_error& e) {
    throw std::runtime_error(std::string("GZIP TFRecord: ") + e.what());
  }
  if (datamunge::image::detail::crc32_update(0, out.data(), out.size()) != crc)
    throw std::runtime_error("GZIP TFRecord: CRC-32 mismatch");
  return Bytes(out.begin(), out.end());
}

} // namespace detail

/// @brief Replace a file with framed records and optional whole-stream compression.
///
/// Materialises the framed stream in memory; not a lazy streaming writer. CRCs
/// protect framing and payload against accidental corruption, not malicious changes.
/// @param path Output path; parent must exist.
/// @param records Ordered arbitrary payloads, including empty and NUL-containing strings.
/// @param compression Empty for plain, GZIP or ZLIB.
/// @throws std::runtime_error On I/O failure; std::invalid_argument on unknown compression.
///
/// @par Example
/// @code
/// namespace d = dm::nn::data;
/// d::WriteTFRecords("records.bin", {"", "abc"}, "GZIP");
/// @endcode
///
/// @par References
/// - TensorFlow TFRecord format, https://www.tensorflow.org/tutorials/load_data/tfrecord
/// @see ReadTFRecords()
inline void WriteTFRecords(
    const std::filesystem::path&    path,
    const std::vector<std::string>& records,
    const std::string&              compression = "") {
  ValidateCompression(compression);
  Bytes stream;
  for (const auto& record : records) {
    Bytes length;
    detail::PutLE(length, record.size(), 8);
    stream.insert(stream.end(), length.begin(), length.end());
    detail::PutLE(stream, MaskedCrc(length.data(), 8), 4);
    stream.insert(stream.end(), record.begin(), record.end());
    detail::PutLE(
        stream, MaskedCrc(reinterpret_cast<const std::uint8_t*>(record.data()), record.size()), 4);
  }
  if (compression == "GZIP")
    stream = detail::Gzip(stream);
  if (compression == "ZLIB") {
    const auto z = datamunge::image::detail::zlib_compress(stream);
    stream.assign(z.begin(), z.end());
  }
  WriteBinary(path, stream);
}

/// @brief Read a complete file and validate every frame checksum before returning payloads.
///
/// The complete input and output records reside in memory. An empty payload
/// is distinct from EOF. Decompression uses the shared codec output limit.
/// @param path File bounded to 512 MiB compressed/plain input.
/// @param compression Explicit empty, GZIP or ZLIB; not autodetected.
/// @return Ordered owned payloads; an empty file gives no records.
/// @throws std::runtime_error On I/O, truncation, decompression or checksum errors;
/// std::invalid_argument on compression name.
///
/// @par Example
/// @code
/// namespace d = dm::nn::data;
/// d::WriteTFRecords("records.bin", {"", "abc"});
/// const auto records=d::ReadTFRecords("records.bin"); // two records
/// @endcode
/// @see WriteTFRecords()
inline std::vector<std::string> ReadTFRecords(
    const std::filesystem::path& path,
    const std::string&           compression = "") {
  ValidateCompression(compression);
  Bytes stream = ReadBinary(path);
  if (compression == "GZIP")
    stream = detail::Gunzip(stream);
  if (compression == "ZLIB") {
    std::vector<unsigned char> z(stream.begin(), stream.end()), out;
    out = datamunge::image::detail::zlib_decompress(z);
    stream.assign(out.begin(), out.end());
  }
  std::vector<std::string> records;
  for (std::size_t at = 0; at < stream.size();) {
    if (stream.size() - at < 12)
      throw std::runtime_error("truncated TFRecord length header");
    if (MaskedCrc(stream.data() + at, 8) != detail::GetLE(stream, at + 8, 4))
      throw std::runtime_error("corrupted TFRecord length (CRC-32C mismatch)");
    const std::uint64_t length = detail::GetLE(stream, at, 8);
    at += 12;
    if (length > stream.size() - at || stream.size() - at - length < 4)
      throw std::runtime_error("truncated TFRecord payload");
    const auto* data = stream.data() + at;
    if (MaskedCrc(data, length) != detail::GetLE(stream, at + length, 4))
      throw std::runtime_error("corrupted TFRecord payload (CRC-32C mismatch)");
    records.emplace_back(reinterpret_cast<const char*>(data), length);
    at += length + 4;
  }
  return records;
}

// ---- tf.Example ---------------------------------------------------------------

/// One tf.train.Feature: exactly one of the three lists is used.
/// @brief One typed tf.Example feature list.
///
/// Only the vector selected by kind is serialized; other vector fields are ignored.
///
/// @par Example
/// @code
/// namespace d = dm::nn::data;
/// const auto f=d::Feature::Floats({1,.5f});
/// @endcode
/// @see SerializeExample()
struct Feature {
  /// @brief Feature oneof tags from the published protobuf.
  enum class Kind {
    kBytes = 1, ///< List of arbitrary byte strings.
    kFloat = 2, ///< List of IEEE binary32 values.
    kInt64 = 3  ///< List of signed 64-bit integers.
  };
  Kind                      kind{Kind::kFloat}; ///< Active list kind, Float32 by default.
  std::vector<std::string>  bytes;              ///< Active byte strings when kind is kBytes.
  std::vector<float>        floats;             ///< Active values when kind is kFloat.
  std::vector<std::int64_t> ints;               ///< Active values when kind is kInt64.

                                                /// @brief Construct a byte-string feature list.
                                                /// @param v Owned values, moved into the feature.
                                                /// @return Typed feature value.
                                                ///
                                                /// @par Example
                                                /// @code
                                                /// namespace d = dm::nn::data;
                                                /// const auto f=d::Feature::Bytes({"id"});
                                                /// @endcode
                                                /// @see Feature
  static Feature Bytes(std::vector<std::string> v) {
    Feature f;
    f.kind  = Kind::kBytes;
    f.bytes = std::move(v);
    return f;
  }
  /// @brief Construct a Float32 feature list.
  /// @param v Owned values, moved into the feature.
  /// @return Typed feature value.
  ///
  /// @par Example
  /// @code
  /// namespace d = dm::nn::data;
  /// const auto f=d::Feature::Floats({1,.5f});
  /// @endcode
  /// @see Feature
  static Feature Floats(std::vector<float> v) {
    Feature f;
    f.kind   = Kind::kFloat;
    f.floats = std::move(v);
    return f;
  }
  /// @brief Construct a Int64 feature list.
  /// @param v Owned values, moved into the feature.
  /// @return Typed feature value.
  ///
  /// @par Example
  /// @code
  /// namespace d = dm::nn::data;
  /// const auto f=d::Feature::Int64s({7});
  /// @endcode
  /// @see Feature
  static Feature Int64s(std::vector<std::int64_t> v) {
    Feature f;
    f.kind = Kind::kInt64;
    f.ints = std::move(v);
    return f;
  }
};
/// @brief Name-sorted map of feature lists; keys are serialized deterministically.
using Example = std::map<std::string, Feature>;

/// @brief Minimal protobuf wire-format internals, not a general protobuf runtime.
namespace pb {

/// @brief Append one unsigned protobuf varint.
/// @param out Destination bytes.
/// @param v Value; signed int64 is encoded via unsigned conversion.
inline void Varint(std::string& out, std::uint64_t v) {
  while (v >= 0x80) {
    out.push_back(static_cast<char>((v & 0x7F) | 0x80));
    v >>= 7;
  }
  out.push_back(static_cast<char>(v));
}
/// @brief Append a protobuf field/wire tag.
/// @param out Destination bytes.
/// @param field Positive field number (internal precondition).
/// @param wire Supported wire type.
inline void Tag(std::string& out, int field, int wire) {
  Varint(out, (std::uint64_t(field) << 3) | wire);
}
/// @brief Append a length-delimited protobuf field.
/// @param out Destination bytes.
/// @param field Positive field number.
/// @param payload Raw field bytes.
inline void Delimited(std::string& out, int field, const std::string& payload) {
  Tag(out, field, 2);
  Varint(out, payload.size());
  out += payload;
}

/// @brief Borrowed protobuf byte reader with checked field lengths.
struct Reader {
  const std::string& s;     ///< Borrowed bytes, outliving this reader.
  std::size_t        at{0}; ///< First unread byte position.

                            /// @brief Check whether the input cursor is at its end.
                            /// @return True when no unread bytes remain.
  bool done() const { return at >= s.size(); }
  /// @brief Consume one bounded unsigned varint.
  /// @return Decoded value.
  /// @throws std::invalid_argument On truncation or an overlong varint.
  std::uint64_t varint() {
    std::uint64_t v = 0;
    for (int shift = 0; shift < 64; shift += 7) {
      if (at >= s.size())
        throw std::invalid_argument("tf.Example: truncated varint");
      const auto b = static_cast<std::uint8_t>(s[at++]);
      v |= std::uint64_t(b & 0x7F) << shift;
      if (!(b & 0x80))
        return v;
    }
    throw std::invalid_argument("tf.Example: varint too long");
  }
  /// @brief Consume a length-delimited field.
  /// @return Owned field contents.
  /// @throws std::invalid_argument On bad length or truncation.
  std::string bytes() {
    const auto n = varint();
    if (n > s.size() - at)
      throw std::invalid_argument("tf.Example: truncated field");
    std::string out = s.substr(at, n);
    at += n;
    return out;
  }
  /// @brief Skip one unknown supported field.
  /// @param wire Wire type 0, 1, 2 or 5.
  /// @throws std::invalid_argument On unsupported wire type or truncated data.
  void skip(int wire) {
    if (wire == 0)
      varint();
    else if (wire == 1)
      at += 8;
    else if (wire == 2)
      bytes();
    else if (wire == 5)
      at += 4;
    else
      throw std::invalid_argument("tf.Example: unsupported wire type");
    if (at > s.size())
      throw std::invalid_argument("tf.Example: truncated field");
  }
};

} // namespace pb

/// @brief Encode name-sorted typed lists as a tf.Example protobuf.
/// @param example Feature map; only the vector selected by each kind is used.
/// @return Owned protobuf bytes; arbitrary byte-string feature contents are retained.
///
/// @par Example
/// @code
/// namespace d = dm::nn::data;
/// const auto s=d::SerializeExample({{"label",d::Feature::Int64s({7})}});
/// @endcode
///
/// @par References
/// - Protocol Buffers encoding, https://protobuf.dev/programming-guides/encoding/
/// @see ParseExample()
inline std::string SerializeExample(const Example& example) {
  std::string features;
  for (const auto& [key, feature] : example) {
    std::string list;
    if (feature.kind == Feature::Kind::kBytes) {
      for (const auto& b : feature.bytes)
        pb::Delimited(list, 1, b);
    } else if (feature.kind == Feature::Kind::kFloat) {
      if (!feature.floats.empty()) {
        std::string packed(feature.floats.size() * 4, '\0');
        for (std::size_t i = 0; i < feature.floats.size(); ++i) {
          std::uint32_t bits;
          std::memcpy(&bits, &feature.floats[i], 4);
          for (int k = 0; k < 4; ++k)
            packed[4 * i + k] = static_cast<char>(bits >> (8 * k));
        }
        pb::Delimited(list, 1, packed);
      }
    } else {
      if (!feature.ints.empty()) {
        std::string packed;
        for (auto v : feature.ints)
          pb::Varint(packed, static_cast<std::uint64_t>(v));
        pb::Delimited(list, 1, packed);
      }
    }
    std::string value, entry;
    pb::Delimited(value, static_cast<int>(feature.kind), list);
    pb::Delimited(entry, 1, key);
    pb::Delimited(entry, 2, value);
    pb::Delimited(features, 1, entry);
  }
  std::string out;
  pb::Delimited(out, 1, features);
  return out;
}

/// @brief Decode one Feature protobuf, accepting packed or unpacked numeric fields.
/// @param s Complete Feature payload bytes.
/// @return Typed feature list.
/// @throws std::invalid_argument On unsupported/malformed wire data.
///
/// @par Example
/// @code
/// namespace d = dm::nn::data;
/// const auto e=d::ParseExample(d::SerializeExample({{"x",d::Feature::Floats({1})}}));
/// @endcode
/// @see ParseExample()
inline Feature ParseFeature(const std::string& s) {
  Feature    f;
  pb::Reader r{s};
  while (!r.done()) {
    const auto tag   = r.varint();
    const int  field = int(tag >> 3), wire = int(tag & 7);
    if (field < 1 || field > 3 || wire != 2) {
      r.skip(wire);
      continue;
    }
    f.kind = static_cast<Feature::Kind>(field);
    f.bytes.clear(), f.floats.clear(), f.ints.clear();
    const std::string list = r.bytes();
    pb::Reader        lr{list};
    while (!lr.done()) {
      const auto ltag   = lr.varint();
      const int  lfield = int(ltag >> 3), lwire = int(ltag & 7);
      if (lfield != 1) {
        lr.skip(lwire);
        continue;
      }
      if (field == 1) {
        if (lwire != 2)
          throw std::invalid_argument("tf.Example: bad bytes_list");
        f.bytes.push_back(lr.bytes());
      } else if (field == 2) {
        auto take = [&](const std::string& b, std::size_t at) {
          std::uint32_t bits = 0;
          for (int k = 0; k < 4; ++k)
            bits |= std::uint32_t(static_cast<std::uint8_t>(b[at + k])) << (8 * k);
          float v;
          std::memcpy(&v, &bits, 4);
          f.floats.push_back(v);
        };
        if (lwire == 2) {
          const std::string packed = lr.bytes();
          if (packed.size() % 4)
            throw std::invalid_argument("tf.Example: bad packed float_list");
          for (std::size_t i = 0; i < packed.size(); i += 4)
            take(packed, i);
        } else if (lwire == 5) {
          if (lr.at + 4 > list.size())
            throw std::invalid_argument("tf.Example: truncated float");
          take(list, lr.at);
          lr.at += 4;
        } else {
          throw std::invalid_argument("tf.Example: bad float_list");
        }
      } else {
        if (lwire == 2) {
          const std::string packed = lr.bytes();
          pb::Reader        pr{packed};
          while (!pr.done())
            f.ints.push_back(static_cast<std::int64_t>(pr.varint()));
        } else if (lwire == 0) {
          f.ints.push_back(static_cast<std::int64_t>(lr.varint()));
        } else {
          throw std::invalid_argument("tf.Example: bad int64_list");
        }
      }
    }
  }
  return f;
}

/// Parses a serialized tf.Example; throws std::invalid_argument when malformed.
/// @brief Decode a tf.Example map and skip supported unknown fields.
/// @param serialized Complete Example bytes; empty bytes form an empty Example.
/// @return Owned name-to-feature map.
/// @throws std::invalid_argument On malformed protobuf.
///
/// @par Example
/// @code
/// namespace d = dm::nn::data;
/// const auto e=d::ParseExample(d::SerializeExample({{"x",d::Feature::Floats({1})}}));
/// @endcode
/// @see SerializeExample()
inline Example ParseExample(const std::string& serialized) {
  Example    example;
  pb::Reader r{serialized};
  while (!r.done()) {
    const auto tag = r.varint();
    if ((tag >> 3) != 1 || (tag & 7) != 2) {
      r.skip(int(tag & 7));
      continue;
    }
    const std::string features = r.bytes();
    pb::Reader        fr{features};
    while (!fr.done()) {
      const auto ftag = fr.varint();
      if ((ftag >> 3) != 1 || (ftag & 7) != 2) {
        fr.skip(int(ftag & 7));
        continue;
      }
      const std::string entry = fr.bytes();
      pb::Reader        er{entry};
      std::string       key;
      Feature           value;
      while (!er.done()) {
        const auto etag = er.varint();
        if ((etag >> 3) == 1 && (etag & 7) == 2)
          key = er.bytes();
        else if ((etag >> 3) == 2 && (etag & 7) == 2)
          value = ParseFeature(er.bytes());
        else
          er.skip(int(etag & 7));
      }
      example[key] = std::move(value); // later entries win, as in protobuf maps
    }
  }
  return example;
}

/// A fixed-length (dense) feature for ParseSingleExample: `size` values of
/// `kind`, or `default_value` when the key is absent (none = required).
/// @brief Required or defaulted fixed-length feature schema entry.
///
/// A default must have the declared kind and length; defaults are used verbatim
/// for absent keys. Present values are checked for exact kind and count.
///
/// @par Example
/// @code
/// namespace d = dm::nn::data;
/// d::DenseSpec s{"x",d::Feature::Kind::kFloat,2,std::nullopt};
/// @endcode
/// @see ParseSingleExample()
struct DenseSpec {
  std::string            key;           ///< Exact Example map key.
  Feature::Kind          kind;          ///< Expected active list type.
  std::int64_t           size;          ///< Exact expected present-value count.
  std::optional<Feature> default_value; ///< Absent-key replacement; none means required.
};

/// TF ParseSingleExample for dense features: one Feature per spec, in spec
/// order. A missing required key, a type mismatch or a wrong value count
/// throws std::invalid_argument.
/// @brief Decode dense feature values in schema order.
/// @param serialized Complete tf.Example protobuf.
/// @param specs Ordered feature requirements; defaults must match their declared kind and size.
/// @return One owned feature per spec, including defaults for absent optional keys.
/// @throws std::invalid_argument On malformed protobuf, missing required keys or wrong present
/// type/count.
///
/// @par Example
/// @code
/// namespace d = dm::nn::data;
/// const auto s=d::SerializeExample({{"x",d::Feature::Floats({1,2})}});
/// const auto x=d::ParseSingleExample(s,{{"x",d::Feature::Kind::kFloat,2,std::nullopt}});
/// @endcode
/// @see DenseSpec
inline std::vector<Feature> ParseSingleExample(
    const std::string&            serialized,
    const std::vector<DenseSpec>& specs) {
  const Example        example = ParseExample(serialized);
  std::vector<Feature> out;
  for (const auto& spec : specs) {
    const auto it = example.find(spec.key);
    if (it == example.end()) {
      if (!spec.default_value)
        throw std::invalid_argument(
            "Feature: " + spec.key + " (data type) is required but could not be found.");
      out.push_back(*spec.default_value);
      continue;
    }
    const Feature& f = it->second;
    const auto     n = f.kind == Feature::Kind::kBytes   ? f.bytes.size()
                       : f.kind == Feature::Kind::kFloat ? f.floats.size()
                                                         : f.ints.size();
    if (f.kind != spec.kind)
      throw std::invalid_argument("Feature: " + spec.key + " has the wrong data type");
    if (std::int64_t(n) != spec.size)
      throw std::invalid_argument(
          "Feature: " + spec.key + " has " + std::to_string(n) + " values, expected " +
          std::to_string(spec.size));
    out.push_back(f);
  }
  return out;
}

} // namespace datamunge::nn::data
