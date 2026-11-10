#pragma once
/// @file learned_image_codec.hpp
/// @brief Fitting-range byte quantization and the bounded self-contained TFLCODE1 format.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn::models {
/// @brief Private validation and explicit little-endian scalar operations for the teaching codec.
namespace codec_detail {
/// @brief Validate positive-axis finite floating CPU storage without changing paths.
/// @param value Defined positive-rank Float32/64 CPU tensor with positive axes.
/// @throws std::invalid_argument For invalid geometry, dtype or nonfinite values.
/// @throws std::logic_error For non-CPU storage.
inline void Floating(const Tensor& value) {
  if (!value.defined() || value.dim() < 1 ||
      (value.dtype() != DType::Float32 && value.dtype() != DType::Float64))
    throw std::invalid_argument("codec requires finite floating tensors");
  for (auto d : value.shape())
    if (d <= 0)
      throw std::invalid_argument("codec axes must be positive");
  for (double x : value.contiguous().to_vector<double>())
    if (!std::isfinite(x))
      throw std::invalid_argument("codec values must be finite");
}
/// @brief Append one unsigned scalar in least-significant-byte-first order.
/// @param bytes Destination sequence, extended by four bytes.
/// @param value Unsigned 32-bit scalar; every bit pattern is supported.
inline void AppendU32(std::vector<std::uint8_t>& bytes, std::uint32_t value) {
  for (int j = 0; j < 4; ++j)
    bytes.push_back((value >> (8 * j)) & 255);
}
/// @brief Append the exact IEEE754 binary32 bits in little-endian order.
/// @param bytes Destination sequence, extended by four bytes.
/// @param value Binary32 scalar; callers validate finiteness before serialization.
inline void AppendFloat(std::vector<std::uint8_t>& bytes, float value) {
  static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
  std::uint32_t bits;
  std::memcpy(&bits, &value, 4);
  AppendU32(bytes, bits);
}
/// @brief Read a little-endian unsigned scalar and advance the checked cursor.
/// @param bytes Source bytes, unchanged.
/// @param cursor Offset advanced by four on success; unchanged on truncation.
/// @return Unsigned scalar with exactly the stored bits.
/// @throws std::invalid_argument If fewer than four bytes remain or cursor is past end.
inline std::uint32_t ReadU32(const std::vector<std::uint8_t>& bytes, std::size_t& cursor) {
  if (cursor > bytes.size() || bytes.size() - cursor < 4)
    throw std::invalid_argument("truncated codec scalar");
  std::uint32_t value = 0;
  for (int j = 0; j < 4; ++j)
    value |= std::uint32_t(bytes[cursor++]) << (8 * j);
  return value;
}
/// @brief Read exact finite binary32 bits from little-endian input.
/// @param bytes Source bytes, unchanged.
/// @param cursor Offset advanced by four when a scalar is available.
/// @return Finite Float32 value, retaining signed-zero bits.
/// @throws std::invalid_argument For truncation or a nonfinite encoded scalar.
inline float ReadFloat(const std::vector<std::uint8_t>& bytes, std::size_t& cursor) {
  const auto bits = ReadU32(bytes, cursor);
  float      value;
  std::memcpy(&value, &bits, 4);
  if (!std::isfinite(value))
    throw std::invalid_argument("nonfinite codec parameter");
  return value;
}
} // namespace codec_detail

/// @brief Own explicit per-coordinate Float32 ranges for a uniform byte quantizer.
///
/// minimum and maximum must have the same positive width D, with finite ordered
/// entries. Public vectors are independently copied when the record is copied;
/// every consuming operation revalidates them. No fitting, implicit normalization,
/// gradient history or RNG is stored. Synchronize external mutation.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const dm::nn::models::CodecQuantizer range{{-1.f, .25f},{1.f,.25f}};
/// // Two coordinates: a signed interval and a constant fitting coordinate.
/// @endcode
/// @see FitCodecQuantizer(), QuantizeCodec(), DequantizeCodec()
struct CodecQuantizer {
  std::vector<float> minimum; ///< Owning finite fitting minima[D], D>0.
  std::vector<float> maximum; ///< Owning finite fitting maxima[D], maximum[j]>=minimum[j].
};
namespace codec_detail {
/// @brief Validate explicit finite ranges and return their shared positive width.
/// @param q Owning range record, unchanged.
/// @return Positive coordinate count D, representable by Int64 tensor geometry.
/// @throws std::invalid_argument For empty, unequal, nonfinite, reversed or excessive ranges.
inline std::int64_t Width(const CodecQuantizer& q) {
  if (q.minimum.empty() || q.minimum.size() != q.maximum.size() ||
      q.minimum.size() > std::size_t(std::numeric_limits<std::int64_t>::max()))
    throw std::invalid_argument("codec ranges need equal positive widths");
  for (std::size_t j = 0; j < q.minimum.size(); ++j)
    if (!std::isfinite(q.minimum[j]) || !std::isfinite(q.maximum[j]) || q.minimum[j] > q.maximum[j])
      throw std::invalid_argument("codec ranges must be finite and ordered");
  return std::int64_t(q.minimum.size());
}
} // namespace codec_detail

/// @brief Fit per-coordinate byte ranges using only the supplied Float32 sample rows.
///
/// Finite signed CPU code[N,D], N,D>0, may be strided. Scans each coordinate in
/// sample order with Float32 min/max, preserving the original example's equal
/// value/signed-zero convention. No labels, validation/test fitting, scaling or
/// optimization is implicit. Returns owning vectors with stopped fitting paths;
/// changing input later cannot change them. Work O(N*D), host storage O(N*D+D).
/// @param code Finite Float32 CPU matrix[N,D] of caller-selected fitting samples.
/// @return Independent per-coordinate fitting extrema, including constant ranges.
/// @throws std::invalid_argument For non-Float32, nonfinite or invalid matrix geometry.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const auto q=dm::nn::models::FitCodecQuantizer(dm::nn::tensor({-1.,.25,1.,.25},{2,2}));
/// // q.minimum=[-1,.25], q.maximum=[1,.25]; the second coordinate is constant.
/// @endcode
/// @see CodecQuantizer, QuantizeCodec()
inline CodecQuantizer FitCodecQuantizer(const Tensor& code) {
  if (!code.defined() || code.dim() != 2 || code.dtype() != DType::Float32)
    throw std::invalid_argument("quantizer fitting needs a Float32 matrix");
  codec_detail::Floating(code);
  const auto     x = code.contiguous().to_vector<float>();
  CodecQuantizer q{
      std::vector<float>(code.size(1), INFINITY), std::vector<float>(code.size(1), -INFINITY)};
  for (std::int64_t j = 0; j < code.size(1); ++j)
    for (std::int64_t i = 0; i < code.size(0); ++i) {
      q.minimum[j] = std::min(q.minimum[j], x[i * code.size(1) + j]);
      q.maximum[j] = std::max(q.maximum[j], x[i * code.size(1) + j]);
    }
  return q;
}

/// @brief Quantize finite Float32 rows with explicit fitting ranges and nearest half-up bytes.
///
/// For each coordinate widens v,low,high to double, evaluates
/// floor(clamp((v-low)/(high-low)*255,0,255)+.5), then converts to a byte.
/// Constant ranges emit0 for every finite query, even outside that constant.
/// Clipping is part of quantization; no range refit or straight-through gradient
/// is implied. Strided inputs retain logical row order. Input/ranges unchanged;
/// output owns detached bytes. Work/storage O(N*D), plus finite-value validation.
/// @param code Finite signed Float32 CPU matrix[N,D], N,D>0.
/// @param q Finite ordered D-coordinate ranges; need not come from this query.
/// @return Owning row-major bytes, exactly N*D entries in[0,255].
/// @throws std::invalid_argument For invalid input/ranges or a width mismatch.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const dm::nn::models::CodecQuantizer q{{0.f},{1.f}};
/// const auto s=dm::nn::models::QuantizeCodec(dm::nn::tensor({-.2,.5,1.2},{3,1}),q);
/// // s contains 0,128,255: clipping and a half-up midpoint.
/// @endcode
/// @see FitCodecQuantizer(), DequantizeCodec(), SymbolTensor()
inline std::vector<std::uint8_t> QuantizeCodec(const Tensor& code, const CodecQuantizer& q) {
  const auto width = codec_detail::Width(q);
  if (!code.defined() || code.dim() != 2 || code.dtype() != DType::Float32 || code.size(1) != width)
    throw std::invalid_argument("quantization needs matching Float32 rows");
  codec_detail::Floating(code);
  const auto                x = code.contiguous().to_vector<float>();
  std::vector<std::uint8_t> symbols(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    const auto   j   = i % std::size_t(width);
    const double low = q.minimum[j], high = q.maximum[j], v = x[i];
    symbols[i] = std::uint8_t(
        high > low ? int(std::floor(std::clamp((v - low) / (high - low) * 255, 0., 255.) + .5))
                   : 0);
  }
  return symbols;
}

/// @brief Dequantize complete byte rows to detached Float32 coordinate values.
///
/// Computes Float32(low+symbol*(high-low)/255) with explicit double operations,
/// including constant ranges. Symbols need no clipping because their type bounds
/// them. Output owns contiguous CPU storage; no fitting or continuous/straight-
/// through path is created. Work/storage O(N*D); inputs unchanged.
/// @param symbols Nonempty row-major byte values, with length divisible by D.
/// @param q Finite ordered ranges with positive width D.
/// @return Owning detached Float32[N,D] reconstruction of quantized coordinates.
/// @throws std::invalid_argument For invalid ranges, partial/empty rows or excessive length.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const auto x=dm::nn::models::DequantizeCodec({0,128,255},{{0.f},{1.f}});
/// // Float32[3,1] values are 0, approximately .5019608, and 1.
/// @endcode
/// @see QuantizeCodec(), LearnedCodecDecode()
inline Tensor DequantizeCodec(const std::vector<std::uint8_t>& symbols, const CodecQuantizer& q) {
  const auto width = codec_detail::Width(q);
  if (symbols.empty() || symbols.size() % std::size_t(width) ||
      symbols.size() > std::size_t(std::numeric_limits<std::int64_t>::max()))
    throw std::invalid_argument("dequantization needs complete nonempty symbol rows");
  std::vector<float> code(symbols.size());
  for (std::size_t i = 0; i < symbols.size(); ++i) {
    const auto   j   = i % std::size_t(width);
    const double low = q.minimum[j], high = q.maximum[j];
    code[i] = float(low + symbols[i] * (high - low) / 255.);
  }
  return from_vector(code, {std::int64_t(symbols.size()) / width, width});
}

/// @brief Copy byte rows into a detached Int32 tensor for native array archives.
///
/// No quantization or fitting occurs. Each byte becomes its exact nonnegative
/// Int32 value. Result owns contiguous CPU storage; input unchanged. O(length)
/// work/storage. Width defaults to32 for the original teaching architecture.
/// @param symbols Nonempty complete row-major byte rows.
/// @param width Positive row width D, default32, dividing the byte count exactly.
/// @return Owning detached Int32[N,D] values in[0,255].
/// @throws std::invalid_argument For nonpositive width, partial/empty rows or excessive length.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const auto t=dm::nn::models::SymbolTensor({0,128,255},3);
/// // t is Int32[1,3] containing 0,128,255 without conversion loss.
/// @endcode
/// @see QuantizeCodec(), DequantizeCodec()
inline Tensor SymbolTensor(const std::vector<std::uint8_t>& symbols, std::int64_t width = 32) {
  if (width <= 0 || symbols.empty() || symbols.size() % std::size_t(width) ||
      symbols.size() > std::size_t(std::numeric_limits<std::int64_t>::max()))
    throw std::invalid_argument("symbol tensor needs complete nonempty rows");
  return from_vector(
      std::vector<int>(symbols.begin(), symbols.end()),
      {std::int64_t(symbols.size()) / width, width});
}

/// @brief Decode explicit floating codes with sigmoid(codes*weights+bias), without an encoder.
///
/// Finite matching Float32/64 CPU codes[N,D], weights[D,F] and bias[F] have
/// positive geometry and may be strided/signed. The original operation grouping
/// is retained. Owning output keeps applicable code/weight/bias gradients and
/// higher paths; no store, fitting, RNG, clipping or quantizer is implicit.
/// Inputs unchanged. Work O(N*D*F), activation storage O(N*F); validation also
/// reads inputs. Nonfinite computed logits are rejected before sigmoid.
/// @param codes Finite matching floating CPU matrix[N,D], N,D>0.
/// @param weights Finite matching floating CPU decoder matrix[D,F], F>0.
/// @param bias Finite matching floating CPU vector[F].
/// @return Owning live floating[N,F] probabilities, in[0,1].
/// @throws std::invalid_argument For invalid geometry/dtype/values or nonfinite computed logits.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const auto p=dm::nn::models::LearnedCodecDecode(dm::nn::zeros({2,3}),
///     dm::nn::ones({3,4}),dm::nn::zeros({4}));
/// // p is Float32[2,4] filled with .5; no encoder parameters are needed.
/// @endcode
/// @see DecodeCodec(), DequantizeCodec(), sigmoid()
inline Tensor LearnedCodecDecode(const Tensor& codes, const Tensor& weights, const Tensor& bias) {
  codec_detail::Floating(codes);
  codec_detail::Floating(weights);
  codec_detail::Floating(bias);
  if (codes.dim() != 2 || weights.dim() != 2 || bias.dim() != 1 ||
      codes.size(1) != weights.size(0) || weights.size(1) != bias.size(0) ||
      codes.dtype() != weights.dtype() || codes.dtype() != bias.dtype())
    throw std::invalid_argument("decoder requires matching code, weight and bias geometry/dtype");
  auto logits = matmul(codes, weights) + bias;
  codec_detail::Floating(logits);
  return sigmoid(logits);
}

/// @brief Hold all values needed by a bounded TFLCODE1 decoder, without encoder state.
///
/// The fixed teaching format uses32 code coordinates and784 grayscale outputs.
/// Vectors own their values; Tensor fields retain shared storage/path handles
/// supplied by the caller. UnpackCodec creates independent detached tensors.
/// PackCodec/DecodeCodec validate every field; callers synchronize mutation.
/// This record itself neither fits nor reads files. There is no checksum,
/// entropy model, version negotiation or corruption-detection guarantee.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// dm::nn::models::CodecPacket p;
/// p.quantizer={std::vector<float>(32,0),std::vector<float>(32,1)};
/// p.weights=dm::nn::zeros({32,784});p.bias=dm::nn::zeros({784});
/// p.source_ids={7};p.symbols=std::vector<std::uint8_t>(32,0);
/// // Fully specified one-image packet; a fresh receiver needs no encoder.
/// @endcode
/// @see PackCodec(), UnpackCodec(), DecodeCodec()
struct CodecPacket {
  CodecQuantizer quantizer; ///< Owning finite ordered Float32 bounds[32].
  Tensor
      weights; ///< Finite Float32[32,784] decoder handle; aliases caller storage until serialized.
  Tensor bias; ///< Finite Float32[784] decoder bias handle.
  std::vector<int>
      source_ids; ///< Owning nonnegative Int32 identities[N], 1<=N<=10000; repeats allowed.
  std::vector<std::uint8_t> symbols; ///< Owning row-major byte symbols[N*32].
};
namespace codec_detail {
/// @brief Validate all fixed packet fields before packing or decoding.
/// @param p Packet, unchanged; finite Float32 CPU decoder and32 valid ranges required.
/// @throws std::invalid_argument For invalid count, symbols, identities, range or decoder
/// values/geometry.
/// @throws std::logic_error For non-CPU decoder storage.
inline void Packet(const CodecPacket& p) {
  if (Width(p.quantizer) != 32 || p.source_ids.empty() || p.source_ids.size() > 10000 ||
      p.symbols.size() != p.source_ids.size() * 32)
    throw std::invalid_argument("TFLCODE1 needs32 coordinates and1..10000 complete images");
  Floating(p.weights);
  Floating(p.bias);
  if (p.weights.dtype() != DType::Float32 || p.bias.dtype() != DType::Float32 ||
      p.weights.shape() != Shape({32, 784}) || p.bias.shape() != Shape({784}))
    throw std::invalid_argument("TFLCODE1 decoder needs Float32[32,784] and[784]");
  for (auto id : p.source_ids)
    if (id < 0 || std::uint64_t(id) > std::uint64_t(INT32_MAX))
      throw std::invalid_argument("TFLCODE1 identities must be nonnegative Int32");
}
} // namespace codec_detail

/// @brief Serialize the complete fixed TFLCODE1 learned-transform packet as exact little-endian
/// bytes.
///
/// Layout:8 ASCII magic bytes TFLCODE1; fiveU32LE(N,32,28,28,1);32 interleaved
/// F32LE(min,max) pairs; row-major decoder W[32,784] then b[784] F32LE;
/// N U32LE source IDs; N*32 U8 symbols. Exactly103772+36*N bytes are emitted.
/// IEEE754 binary32 bits, including signed zero, are preserved. Requires all
/// finite/ordered Float32 fields and N in[1,10000]; no file IO, encoder, labels,
/// fitting, checksum or entropy coding. Inputs unchanged; output owns bytes.
/// Work/storage O(32*784+N*32). No gradient path crosses serialization.
/// @param p Complete valid packet with nonnegative Int32 identities.
/// @return Owning complete TFLCODE1 byte sequence, compatible with the original example.
/// @throws std::invalid_argument For invalid count, symbol length, identities, bounds or decoder
/// fields.
/// @throws std::logic_error For non-CPU decoder storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const dm::nn::models::CodecPacket p{{std::vector<float>(32,0),std::vector<float>(32,1)},
///   dm::nn::zeros({32,784}),dm::nn::zeros({784}),{7},std::vector<std::uint8_t>(32,0)};
/// const auto bytes=dm::nn::models::PackCodec(p);
/// // bytes.size()==103808: all decoder/range/identity overhead is included.
/// @endcode
/// @see CodecPacket, UnpackCodec(), DecodeCodec()
inline std::vector<std::uint8_t> PackCodec(const CodecPacket& p) {
  codec_detail::Packet(p);
  std::vector<std::uint8_t> bytes{'T', 'F', 'L', 'C', 'O', 'D', 'E', '1'};
  bytes.reserve(103772 + p.source_ids.size() * 36);
  for (std::uint32_t v : {std::uint32_t(p.source_ids.size()), 32u, 28u, 28u, 1u})
    codec_detail::AppendU32(bytes, v);
  for (int j = 0; j < 32; ++j) {
    codec_detail::AppendFloat(bytes, p.quantizer.minimum[j]);
    codec_detail::AppendFloat(bytes, p.quantizer.maximum[j]);
  }
  for (const auto& t : {p.weights, p.bias})
    for (float v : t.contiguous().to_vector<float>())
      codec_detail::AppendFloat(bytes, v);
  for (int id : p.source_ids)
    codec_detail::AppendU32(bytes, std::uint32_t(id));
  bytes.insert(bytes.end(), p.symbols.begin(), p.symbols.end());
  return bytes;
}

/// @brief Parse and validate an exact bounded TFLCODE1 packet into owning detached values.
///
/// Checks short header/magic, fixed architecture, N in[1,10000] and exact
/// length103772+36*N before allocating decoder arrays. Every encoded float must
/// be finite, bounds ordered, and IDs <=INT32_MAX. Truncation, extra trailing
/// bytes and incompatible shapes are rejected. No source assets, encoder,
/// parameter store or file access occurs. Vectors and contiguous CPU Float32
/// tensors own independent storage, with stopped paths; input unchanged.
/// Work/storage O(32*784+N*32), bounded by10000 images. This structural parser
/// cannot detect all bit corruption: the teaching format has no checksum.
/// @param bytes Complete binary TFLCODE1 sequence, unchanged.
/// @return Independent packet, preserving all finite Float32 bits and identities.
/// @throws std::invalid_argument For malformed magic/header/count/length/scalars/ranges/identities.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const dm::nn::models::CodecPacket p{{std::vector<float>(32,0),std::vector<float>(32,1)},
///   dm::nn::zeros({32,784}),dm::nn::zeros({784}),{7},std::vector<std::uint8_t>(32,0)};
/// const auto bytes=dm::nn::models::PackCodec(p);
/// const auto copy=dm::nn::models::UnpackCodec(bytes);
/// // PackCodec(copy)==bytes; copy.weights/bias are independently owned.
/// @endcode
/// @see PackCodec(), DecodeCodec()
inline CodecPacket UnpackCodec(const std::vector<std::uint8_t>& bytes) {
  if (bytes.size() < 28 || std::string(bytes.begin(), bytes.begin() + 8) != "TFLCODE1")
    throw std::invalid_argument("invalid TFLCODE1 header/magic");
  std::size_t cursor = 8;
  const auto  n = codec_detail::ReadU32(bytes, cursor), d = codec_detail::ReadU32(bytes, cursor),
             h = codec_detail::ReadU32(bytes, cursor), w = codec_detail::ReadU32(bytes, cursor),
             c = codec_detail::ReadU32(bytes, cursor);
  if (n == 0 || n > 10000 || d != 32 || h != 28 || w != 28 || c != 1 ||
      bytes.size() != 103772 + std::size_t(n) * 36)
    throw std::invalid_argument("invalid TFLCODE1 count/architecture/exact length");
  CodecPacket p;
  p.quantizer = {std::vector<float>(32), std::vector<float>(32)};
  for (int j = 0; j < 32; ++j) {
    p.quantizer.minimum[j] = codec_detail::ReadFloat(bytes, cursor);
    p.quantizer.maximum[j] = codec_detail::ReadFloat(bytes, cursor);
  }
  codec_detail::Width(p.quantizer);
  std::vector<float> weights(32 * 784), bias(784);
  for (auto& v : weights)
    v = codec_detail::ReadFloat(bytes, cursor);
  for (auto& v : bias)
    v = codec_detail::ReadFloat(bytes, cursor);
  p.weights = from_vector(weights, {32, 784});
  p.bias    = from_vector(bias, {784});
  p.source_ids.reserve(n);
  for (std::uint32_t i = 0; i < n; ++i) {
    const auto id = codec_detail::ReadU32(bytes, cursor);
    if (id > std::uint32_t(INT32_MAX))
      throw std::invalid_argument("invalid TFLCODE1 source identity");
    p.source_ids.push_back(int(id));
  }
  p.symbols.assign(bytes.begin() + std::ptrdiff_t(cursor), bytes.end());
  return p;
}

/// @brief Decode only packet ranges, symbols and serialized decoder values, with stopped paths.
///
/// Fully validates all fields, dequantizes symbols to Float32[N,32], then uses
/// LearnedCodecDecode with the packet's own W/b. Requires neither original
/// model/encoder nor source images. Owning contiguous Float32[N,784] sigmoid
/// probabilities use no per-image normalization or clipping. NoGradGuard stops
/// even caller-supplied trainable decoder handles without changing their flags.
/// No fitting/file IO/RNG; inputs unchanged. Work O(N*32*784), output storage
/// O(N*784); quantization and host validation storage are additional.
/// @param packet Complete finite valid fixed-architecture packet, N in[1,10000].
/// @return Independent contiguous detached Float32[N,784] image probabilities.
/// @throws std::invalid_argument For invalid packet or nonfinite computed decoder logits.
/// @throws std::logic_error For non-CPU decoder storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const dm::nn::models::CodecPacket p{{std::vector<float>(32,0),std::vector<float>(32,1)},
///   dm::nn::zeros({32,784}),dm::nn::zeros({784}),{7},std::vector<std::uint8_t>(32,0)};
/// const auto images=dm::nn::models::DecodeCodec(p);
/// // images is detached Float32[1,784], every probability .5.
/// @endcode
/// @see LearnedCodecDecode(), PackCodec(), UnpackCodec()
inline Tensor DecodeCodec(const CodecPacket& packet) {
  codec_detail::Packet(packet);
  NoGradGuard guard;
  return LearnedCodecDecode(
             DequantizeCodec(packet.symbols, packet.quantizer), packet.weights, packet.bias)
      .contiguous();
}
} // namespace datamunge::nn::models
