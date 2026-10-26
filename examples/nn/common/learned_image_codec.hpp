// Port of tflow's common/{nn,ml}/learned_image_codec.h: an actual little-endian lossy
// bitstream holding the quantizer, the decoder weights and 8-bit symbols of 32-D codes
// (not coordinate-count-only compression), decoded by a fresh decoder from the file only.
// nn has no uint8 dtype: symbols are std::vector<uint8_t> here and int32 tensors in .npz files.
#pragma once

#include <cstring>
#include <limits>

#include "common/autoencoder.hpp"

namespace ae {

/// Decoder-only learned transform: sigmoid(codes W + b) from serialized weights.
inline nn::Tensor LearnedCodecDecode(const nn::Tensor& codes, const nn::Tensor& weights, const nn::Tensor& bias) {
  return nn::sigmoid(nn::matmul(codes, weights) + bias);
}

struct CodecQuantizer {
  std::vector<float> minimum, maximum;  // [32] each, fit-only per-channel range
};

inline CodecQuantizer FitCodecQuantizer(const nn::Tensor& code) {
  check::Require(code.dtype() == nn::DType::Float32 && code.dim() == 2 && code.size(1) == 32, "fit32DF32quantizer");
  const auto     x = code.contiguous().to_vector<float>();
  CodecQuantizer q{std::vector<float>(32), std::vector<float>(32)};
  for (int j = 0; j < 32; ++j) {
    float lo = INFINITY, hi = -INFINITY;
    for (std::int64_t i = 0; i < code.size(0); ++i) {
      const float v = x[i * 32 + j];
      check::Require(std::isfinite(v), "finitefittingcodes");
      lo = std::min(lo, v);
      hi = std::max(hi, v);
    }
    q.minimum[j] = lo;
    q.maximum[j] = hi;
  }
  return q;
}

/// Uniform 8-bit symbols, nearest half-up with clamping; constant channels encode 0.
inline std::vector<std::uint8_t> QuantizeCodec(const nn::Tensor& code, const CodecQuantizer& q) {
  const auto                x = code.contiguous().to_vector<float>();
  std::vector<std::uint8_t> symbols(x.size());
  for (std::int64_t b = 0; b < code.size(0); ++b)
    for (int j = 0; j < 32; ++j) {
      const double low = q.minimum[j], high = q.maximum[j], v = x[b * 32 + j];
      check::Require(std::isfinite(v), "finitecodetoquantize");
      symbols[b * 32 + j] =
          std::uint8_t(high > low ? int(std::floor(std::clamp((v - low) / (high - low) * 255, 0., 255.) + .5)) : 0);
    }
  return symbols;
}

inline nn::Tensor DequantizeCodec(const std::vector<std::uint8_t>& symbols, const CodecQuantizer& q) {
  const auto         n = std::int64_t(symbols.size() / 32);
  std::vector<float> code(symbols.size());
  for (std::int64_t b = 0; b < n; ++b)
    for (int j = 0; j < 32; ++j) {
      const double low = q.minimum[j], high = q.maximum[j];
      code[b * 32 + j] = float(low + symbols[b * 32 + j] * (high - low) / 255.);
    }
  return nn::from_vector(code, {n, 32});
}

/// Symbols as an int32 tensor [n, 32] for .npz records.
inline nn::Tensor SymbolTensor(const std::vector<std::uint8_t>& symbols) {
  return nn::from_vector(std::vector<int>(symbols.begin(), symbols.end()), {std::int64_t(symbols.size() / 32), 32});
}

inline void CodecU32(std::vector<std::uint8_t>& bytes, std::uint32_t value) {
  for (int j = 0; j < 4; ++j) bytes.push_back((value >> (8 * j)) & 255);
}

inline void CodecFloat(std::vector<std::uint8_t>& bytes, float value) {
  static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "IEEE754F32bitstream");
  std::uint32_t bits;
  std::memcpy(&bits, &value, 4);
  CodecU32(bytes, bits);
}

struct CodecPacket {
  CodecQuantizer            quantizer;
  nn::Tensor                weights, bias;  // float32 [32, 784], [784]
  std::vector<int>          source_ids;
  std::vector<std::uint8_t> symbols;  // [n * 32]
};

/// TFLCODE1 magic; U32LE n/32/28/28/1; 32 F32LE min/max pairs; decoder W, b F32LE; n U32LE ids; n*32 U8 symbols.
inline std::vector<std::uint8_t> PackCodec(const CodecPacket& p) {
  const auto n = p.source_ids.size();
  check::Require(n > 0 && n <= 10000 && p.symbols.size() == n * 32 && p.weights.shape() == nn::Shape({32, 784}) &&
                     p.bias.shape() == nn::Shape({784}) && p.quantizer.minimum.size() == 32 &&
                     p.quantizer.maximum.size() == 32,
                 "fixedcodecarchitectureandboundedsamplecount");
  std::vector<std::uint8_t> bytes{'T', 'F', 'L', 'C', 'O', 'D', 'E', '1'};
  for (std::uint32_t v : {std::uint32_t(n), 32u, 28u, 28u, 1u}) CodecU32(bytes, v);
  for (int j = 0; j < 32; ++j) {
    CodecFloat(bytes, p.quantizer.minimum[j]);
    CodecFloat(bytes, p.quantizer.maximum[j]);
  }
  for (const auto& t : {p.weights, p.bias})
    for (float v : t.contiguous().to_vector<float>()) CodecFloat(bytes, v);
  for (int id : p.source_ids) CodecU32(bytes, std::uint32_t(id));
  bytes.insert(bytes.end(), p.symbols.begin(), p.symbols.end());
  return bytes;
}

inline CodecPacket UnpackCodec(const std::vector<std::uint8_t>& bytes) {
  check::Require(bytes.size() >= 28 && std::string(bytes.begin(), bytes.begin() + 8) == "TFLCODE1",
                 "validcodec8bytemagic/header");
  std::size_t cursor = 8;
  auto        u32    = [&]() {
    check::Require(cursor + 4 <= bytes.size(), "untruncatedcodecscalar");
    std::uint32_t v = 0;
    for (int j = 0; j < 4; ++j) v |= std::uint32_t(bytes[cursor++]) << (8 * j);
    return v;
  };
  const std::uint32_t n = u32(), dimensions = u32(), height = u32(), width = u32(), channels = u32();
  check::Require(n > 0 && n <= 10000 && dimensions == 32 && height == 28 && width == 28 && channels == 1,
                 "boundedfixedcodecshape");
  check::Require(bytes.size() == 28 + 256 + 103488 + std::size_t(n) * 36, "EXACTcodecbytelengthnobadpayload/trailingdata");
  auto scalar = [&]() {
    const std::uint32_t bits = u32();
    float               value;
    std::memcpy(&value, &bits, 4);
    check::Require(std::isfinite(value), "finitequantizer/decoderF32parameters");
    return value;
  };
  CodecPacket p;
  p.quantizer = {std::vector<float>(32), std::vector<float>(32)};
  for (int j = 0; j < 32; ++j) {
    p.quantizer.minimum[j] = scalar();
    p.quantizer.maximum[j] = scalar();
    check::Require(p.quantizer.minimum[j] <= p.quantizer.maximum[j], "orderedquantizerbounds");
  }
  std::vector<float> w(32 * 784), b(784);
  for (auto& v : w) v = scalar();
  for (auto& v : b) v = scalar();
  p.weights = nn::from_vector(w, {32, 784});
  p.bias    = nn::from_vector(b, {784});
  for (std::uint32_t i = 0; i < n; ++i) {
    const auto source = u32();
    check::Require(source <= std::uint32_t(std::numeric_limits<int>::max()), "validsourceidentitymetadata");
    p.source_ids.push_back(int(source));
  }
  p.symbols.assign(bytes.begin() + std::ptrdiff_t(cursor), bytes.end());
  return p;
}

/// FRESH decoder from the bitstream only: no original model, encoder or images.
inline nn::Tensor DecodeCodec(const CodecPacket& packet) {
  nn::NoGradGuard guard;
  return LearnedCodecDecode(DequantizeCodec(packet.symbols, packet.quantizer), packet.weights, packet.bias).contiguous();
}

}  // namespace ae
