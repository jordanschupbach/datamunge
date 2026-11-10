#include <gtest/gtest.h>

#include <bit>
#include <datamunge/nn/gradcheck.hpp>
#include <datamunge/nn/learned_image_codec.hpp>
namespace n = datamunge::nn;
namespace m = n::models;
namespace {
m::CodecPacket Packet(int count = 2) {
  m::CodecPacket p{
      {std::vector<float>(32, -1), std::vector<float>(32, 1)},
      n::zeros({32, 784}),
      n::zeros({784}),
      {},
      std::vector<std::uint8_t>(count * 32)};
  for (int i = 0; i < count; ++i)
    p.source_ids.push_back(i == 0 ? 7 : INT32_MAX);
  for (std::size_t i = 0; i < p.symbols.size(); ++i)
    p.symbols[i] = std::uint8_t(i);
  return p;
}
void Put(std::vector<std::uint8_t>& bytes, std::size_t offset, std::uint32_t value) {
  for (int j = 0; j < 4; ++j)
    bytes[offset + j] = std::uint8_t(value >> (8 * j));
}
} // namespace
TEST(NnLearnedCodec, RangeFittingStridesIndependentValuesAndStoppedFitting) {
  auto fit = n::transpose(n::tensor({-2., 1., 4., .25, .25, .25}, {2, 3}), {1, 0}).requires_grad_();
  const auto q = m::FitCodecQuantizer(fit);
  EXPECT_EQ(q.minimum, (std::vector<float>{-2, .25}));
  EXPECT_EQ(q.maximum, (std::vector<float>{4, .25}));
  {
    n::NoGradGuard guard;
    fit.fill_(77);
  }
  EXPECT_EQ(q.minimum[0], -2);
  auto copy       = q;
  copy.minimum[0] = 0;
  EXPECT_EQ(q.minimum[0], -2);
  EXPECT_FALSE(m::DequantizeCodec({0, 0, 255, 255}, q).requires_grad());
  auto zero = n::from_vector(std::vector<float>{-0.f, +0.f}, {2, 1});
  auto z    = m::FitCodecQuantizer(zero);
  EXPECT_TRUE(std::signbit(z.minimum[0]));
  EXPECT_TRUE(std::signbit(z.maximum[0]));
}
TEST(NnLearnedCodec, ExactHalfUpClippingSignedRangesConstantsAndLogicalRows) {
  const m::CodecQuantizer q{{0, -1, .25}, {1, 1, .25}};
  const auto query = n::transpose(n::tensor({-.2, .5, 1.2, -3, 0, 3, -8, .25, 9}, {3, 3}), {1, 0})
                         .requires_grad_();
  const auto s = m::QuantizeCodec(query, q);
  EXPECT_EQ(s, (std::vector<std::uint8_t>{0, 0, 0, 128, 128, 0, 255, 255, 0}));
  auto deq = m::DequantizeCodec(s, q);
  EXPECT_EQ(deq.dtype(), n::DType::Float32);
  EXPECT_EQ(deq.shape(), (n::Shape{3, 3}));
  EXPECT_FALSE(deq.requires_grad());
  const auto v = deq.to_vector<float>();
  EXPECT_EQ(v[0], 0);
  EXPECT_EQ(v[1], -1);
  EXPECT_EQ(v[4], float(-1. + 128. * 2. / 255.));
  EXPECT_EQ(v[6], 1);
  EXPECT_EQ(v[7], 1);
  for (int i = 0; i < 3; ++i)
    EXPECT_EQ(v[i * 3 + 2], .25);
  const auto t = m::SymbolTensor(s, 3);
  EXPECT_EQ(t.dtype(), n::DType::Int32);
  EXPECT_EQ(t.to_vector<int>(), (std::vector<int>{0, 0, 0, 128, 128, 0, 255, 255, 0}));
  // Half-integer normalized coordinates must go upward, not ties-to-even.
  const auto halfway = n::tensor({.5, 1.5, 2.5, 253.5}, {4, 1});
  EXPECT_EQ(m::QuantizeCodec(halfway, {{0}, {255}}), (std::vector<std::uint8_t>{1, 2, 3, 254}));
}
TEST(NnLearnedCodec, GenericDecoderKeepsAllContinuousFirstAndSecondPaths) {
  const auto t    = n::DType::Float64;
  auto       code = n::tensor({-.4, .8, .2, -.1}, {2, 2}, t).requires_grad_();
  auto       w    = n::tensor({.3, -.2, .7, .4}, {2, 2}, t).requires_grad_();
  auto       b    = n::tensor({.1, -.3}, {2}, t).requires_grad_();
  const auto f    = [](const std::vector<n::Tensor>& x) {
    return m::LearnedCodecDecode(x[0], x[1], x[2]);
  };
  EXPECT_TRUE(n::gradcheck(f, {code, w, b}).ok);
  EXPECT_TRUE(n::gradgradcheck(f, {code, w, b}).ok);
  const auto result = f({code, w, b});
  EXPECT_EQ(result.dtype(), t);
  EXPECT_TRUE(result.requires_grad());
  auto st = n::transpose(w, {1, 0});
  EXPECT_TRUE(
      n::equal_bits(m::LearnedCodecDecode(code, st, b), n::sigmoid(n::matmul(code, st) + b)));
}
TEST(NnLearnedCodec, IndependentLittleEndianOffsetsSignedZeroAndExactRoundTrip) {
  auto p                     = Packet();
  p.quantizer.minimum[0]     = -0.f;
  p.quantizer.maximum[0]     = +0.f;
  p.weights.data_ptr<float>()[0] = -0.f;
  p.weights.data_ptr<float>()[1] = 1.5f;
  p.bias.data_ptr<float>()[0]    = -.5f;
  const auto bytes           = m::PackCodec(p);
  ASSERT_EQ(bytes.size(), 103772 + 72);
  EXPECT_EQ(std::string(bytes.begin(), bytes.begin() + 8), "TFLCODE1");
  const auto u32 = [&](std::size_t o) {
    return std::uint32_t(bytes[o]) | (std::uint32_t(bytes[o + 1]) << 8) |
           (std::uint32_t(bytes[o + 2]) << 16) | (std::uint32_t(bytes[o + 3]) << 24);
  };
  EXPECT_EQ(u32(8), 2);
  EXPECT_EQ(u32(12), 32);
  EXPECT_EQ(u32(16), 28);
  EXPECT_EQ(u32(20), 28);
  EXPECT_EQ(u32(24), 1);
  EXPECT_EQ(u32(28), 0x80000000u);
  EXPECT_EQ(u32(32), 0);
  EXPECT_EQ(u32(284), 0x80000000u);
  EXPECT_EQ(u32(288), 0x3fc00000u);
  EXPECT_EQ(u32(100636), 0xbf000000u);
  EXPECT_EQ(u32(103772), 7);
  EXPECT_EQ(u32(103776), 0x7fffffffu);
  EXPECT_TRUE(std::equal(bytes.begin() + 103780, bytes.end(), p.symbols.begin()));
  auto copy = m::UnpackCodec(bytes);
  EXPECT_EQ(m::PackCodec(copy), bytes);
  EXPECT_FALSE(copy.weights.requires_grad());
  EXPECT_FALSE(copy.bias.requires_grad());
  copy.weights.fill_(9);
  EXPECT_EQ(p.weights.at({0, 1}), 1.5);
  copy.quantizer.minimum[1] = -2;
  EXPECT_EQ(p.quantizer.minimum[1], -1);
}
TEST(NnLearnedCodec, FreshPacketOnlyDecoderIsDetachedAndDoesNotChangeCallerFlagsOrValues) {
  auto p = Packet(3);
  p.weights.requires_grad_();
  p.bias.requires_grad_();
  const auto before = p.weights.clone().detach();
  const auto out    = m::DecodeCodec(p);
  EXPECT_EQ(out.shape(), (n::Shape{3, 784}));
  EXPECT_FALSE(out.requires_grad());
  EXPECT_TRUE(out.is_contiguous());
  EXPECT_EQ(out.to_vector<float>(), std::vector<float>(3 * 784, .5));
  EXPECT_TRUE(p.weights.requires_grad());
  EXPECT_TRUE(p.bias.requires_grad());
  EXPECT_TRUE(n::equal_bits(before, p.weights));
  auto fresh = m::UnpackCodec(m::PackCodec(p));
  p.weights  = n::Tensor();
  p.bias     = n::Tensor();
  p.symbols.clear();
  EXPECT_TRUE(n::equal_bits(out, m::DecodeCodec(fresh)));
  auto owned = m::DecodeCodec(fresh);
  fresh.weights.fill_(7);
  EXPECT_EQ(owned.at({0, 0}), .5);
}
TEST(NnLearnedCodec, ParserRejectsEveryFixedHeaderFieldTruncationTrailingNonfiniteBoundsAndIds) {
  const auto bytes = m::PackCodec(Packet());
  for (std::size_t length : {0u, 7u, 8u, 27u, 28u, 103771u, 103843u}) {
    auto b = std::vector<std::uint8_t>(bytes.begin(), bytes.begin() + length);
    EXPECT_THROW(m::UnpackCodec(b), std::invalid_argument);
  }
  auto b = bytes;
  b[0]   = 'X';
  EXPECT_THROW(m::UnpackCodec(b), std::invalid_argument);
  b = bytes;
  b.push_back(0);
  EXPECT_THROW(m::UnpackCodec(b), std::invalid_argument);
  for (auto offset : {8u, 12u, 16u, 20u, 24u}) {
    b = bytes;
    Put(b, offset, 0);
    EXPECT_THROW(m::UnpackCodec(b), std::invalid_argument);
  }
  b = bytes;
  Put(b, 8, 10001);
  EXPECT_THROW(m::UnpackCodec(b), std::invalid_argument);
  for (auto offset : {28u, 32u, 284u, 100636u})
    for (auto bits : {0x7f800000u, 0xff800000u, 0x7fc00001u}) {
      b = bytes;
      Put(b, offset, bits);
      EXPECT_THROW(m::UnpackCodec(b), std::invalid_argument);
    }
  b = bytes;
  Put(b, 28, std::bit_cast<std::uint32_t>(2.f));
  EXPECT_THROW(m::UnpackCodec(b), std::invalid_argument);
  b = bytes;
  Put(b, 103772, 0x80000000u);
  EXPECT_THROW(m::UnpackCodec(b), std::invalid_argument);
  // Structurally valid corruption is not covered by a nonexistent checksum.
  b = bytes;
  b.back() ^= 1;
  EXPECT_NO_THROW(m::UnpackCodec(b));
}
TEST(NnLearnedCodec, PackingAndDecodeValidateAllCallerSuppliedFields) {
  auto p          = Packet();
  p.source_ids[0] = -1;
  EXPECT_THROW(m::PackCodec(p), std::invalid_argument);
  EXPECT_THROW(m::DecodeCodec(p), std::invalid_argument);
  p = Packet();
  p.source_ids.clear();
  EXPECT_THROW(m::PackCodec(p), std::invalid_argument);
  p = Packet();
  p.source_ids.resize(10001);
  EXPECT_THROW(m::PackCodec(p), std::invalid_argument);
  p = Packet();
  p.symbols.pop_back();
  EXPECT_THROW(m::PackCodec(p), std::invalid_argument);
  p = Packet();
  p.quantizer.maximum.pop_back();
  EXPECT_THROW(m::PackCodec(p), std::invalid_argument);
  p                      = Packet();
  p.quantizer.minimum[0] = 2;
  EXPECT_THROW(m::PackCodec(p), std::invalid_argument);
  p                      = Packet();
  p.quantizer.minimum[0] = INFINITY;
  EXPECT_THROW(m::PackCodec(p), std::invalid_argument);
  p         = Packet();
  p.weights = n::zeros({32, 784}, n::DType::Float64);
  EXPECT_THROW(m::PackCodec(p), std::invalid_argument);
  p      = Packet();
  p.bias = n::zeros({1, 784});
  EXPECT_THROW(m::PackCodec(p), std::invalid_argument);
  p                           = Packet();
  p.weights.data_ptr<float>()[17] = NAN;
  EXPECT_THROW(m::PackCodec(p), std::invalid_argument);
  p                        = Packet();
  p.bias.data_ptr<float>()[17] = INFINITY;
  EXPECT_THROW(m::DecodeCodec(p), std::invalid_argument);
}
TEST(NnLearnedCodec, QuantizerAndContinuousDecoderRejectMalformedAndOverflowInputs) {
  for (const auto& x :
       {n::Tensor(),
        n::ones({2}),
        n::ones({0, 2}),
        n::ones({2, 2}, n::DType::Float64),
        n::ones({2, 2}, n::DType::Int32),
        n::full({2, 2}, INFINITY)})
    EXPECT_THROW(m::FitCodecQuantizer(x), std::invalid_argument);
  for (const auto& q :
       std::vector<m::CodecQuantizer>{{}, {{0}, {1, 2}}, {{2}, {1}}, {{NAN}, {1}}}) {
    EXPECT_THROW(m::QuantizeCodec(n::zeros({1, 1}), q), std::invalid_argument);
    EXPECT_THROW(m::DequantizeCodec({0}, q), std::invalid_argument);
  }
  EXPECT_THROW(m::QuantizeCodec(n::ones({2, 2}), {{0}, {1}}), std::invalid_argument);
  EXPECT_THROW(
      m::QuantizeCodec(n::ones({2, 1}, n::DType::Float64), {{0}, {1}}), std::invalid_argument);
  EXPECT_THROW(m::DequantizeCodec({}, {{0}, {1}}), std::invalid_argument);
  EXPECT_THROW(m::DequantizeCodec({1, 2, 3}, {{0, 0}, {1, 1}}), std::invalid_argument);
  EXPECT_THROW(m::SymbolTensor({1}, 0), std::invalid_argument);
  EXPECT_THROW(m::SymbolTensor({1, 2, 3}, 2), std::invalid_argument);
  EXPECT_THROW(
      m::LearnedCodecDecode(n::ones({1, 2}), n::ones({3, 1}), n::zeros({1})),
      std::invalid_argument);
  EXPECT_THROW(
      m::LearnedCodecDecode(n::ones({1, 2}), n::ones({2, 1}, n::DType::Float64), n::zeros({1})),
      std::invalid_argument);
  const auto huge = n::full({1, 1}, std::numeric_limits<double>::max(), n::DType::Float64);
  EXPECT_THROW(
      m::LearnedCodecDecode(huge, huge, n::zeros({1}, n::DType::Float64)), std::invalid_argument);
}
