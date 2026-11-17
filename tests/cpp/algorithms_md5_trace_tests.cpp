#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/md5.hpp>
#include <random>
#include <stdexcept>
using namespace datamunge::algorithms;
TEST(MD5Trace, AllPublishedReferenceVectors) {
  for (auto& [message, expected] : md5_reference_vectors()) {
    EXPECT_EQ(md5_hex(message), expected);
    auto t = md5_trace(message);
    EXPECT_EQ(to_hex(t.analysis.digest), expected);
    EXPECT_EQ(t.analysis.steps, 64 * t.analysis.blocks);
  }
}
TEST(MD5Trace, ActualPaddingWordDecodeStepStatesAndFeedForwardAcrossBoundaries) {
  std::mt19937 rng(7);
  for (std::size_t n = 0; n <= 256; ++n) {
    std::string message(n, '\0');
    for (char& c : message)
      c = static_cast<char>(rng());
    auto                       t = md5_trace(message);
    std::vector<unsigned char> padded(message.begin(), message.end());
    padded.push_back(128);
    while (padded.size() % 64 != 56)
      padded.push_back(0);
    auto bitlength = static_cast<std::uint64_t>(n) * 8;
    for (unsigned i = 0; i < 8; ++i)
      padded.push_back(static_cast<unsigned char>(bitlength >> (8 * i)));
    ASSERT_EQ(t.padded.size(), padded.size());
    for (std::size_t i = 0; i < padded.size(); ++i)
      EXPECT_EQ(static_cast<unsigned char>(t.padded[i]), padded[i]);
    std::array<std::uint32_t, 4> chain{0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476};
    std::size_t                  step         = 0;
    constexpr unsigned           shifts[4][4] = {
        {7, 12, 17, 22}, {5, 9, 14, 20}, {4, 11, 16, 23}, {6, 10, 15, 21}};
    for (std::size_t off = 0; off < padded.size(); off += 64) {
      auto& block = t.blocks[off / 64];
      EXPECT_EQ(block.offset, off);
      EXPECT_EQ(block.before, chain);
      std::array<std::uint32_t, 16> words{};
      for (std::size_t i = 0; i < 16; ++i)
        for (unsigned j = 0; j < 4; ++j)
          words[i] += std::uint32_t(padded[off + 4 * i + j]) << (8 * j);
      EXPECT_EQ(block.words, words);
      auto state = chain;
      for (std::size_t i = 0; i < 64; ++i) {
        auto& s = t.steps[step++];
        EXPECT_EQ(s.before, state);
        EXPECT_EQ(s.block, off / 64);
        EXPECT_EQ(s.index, i);
        std::size_t   round = i / 16, g = round == 0   ? i
                                          : round == 1 ? (5 * i + 1) % 16
                                          : round == 2 ? (3 * i + 5) % 16
                                                       : (7 * i) % 16;
        std::uint32_t f        = round == 0   ? ((state[1] & state[2]) | (~state[1] & state[3]))
                                 : round == 1 ? ((state[3] & state[1]) | (~state[3] & state[2]))
                                 : round == 2 ? (state[1] ^ state[2] ^ state[3])
                                              : (state[2] ^ (state[1] | ~state[3]));
        std::uint32_t constant = static_cast<std::uint32_t>(
            std::floor(4294967296.0 * std::abs(std::sin(double(i + 1)))));
        unsigned      shift   = shifts[round][i % 4];
        std::uint32_t sum     = state[0] + f + constant + words[g];
        std::uint32_t rotated = (sum << shift) | (sum >> (32 - shift));
        EXPECT_EQ(s.function, f);
        EXPECT_EQ(s.word_index, g);
        EXPECT_EQ(s.word, words[g]);
        EXPECT_EQ(s.constant, constant);
        EXPECT_EQ(s.shift, shift);
        EXPECT_EQ(s.sum, sum);
        EXPECT_EQ(s.rotated, rotated);
        state = {state[3], state[1] + rotated, state[1], state[2]};
        EXPECT_EQ(s.after, state);
      }
      EXPECT_EQ(block.working_after, state);
      for (std::size_t i = 0; i < 4; ++i)
        chain[i] += state[i];
      EXPECT_EQ(block.after, chain);
    }
    MD5Digest expected{};
    for (std::size_t w = 0; w < 4; ++w)
      for (unsigned j = 0; j < 4; ++j)
        expected[4 * w + j] = static_cast<std::uint8_t>(chain[w] >> (8 * j));
    EXPECT_EQ(t.analysis.digest, expected);
    EXPECT_EQ(md5(message), expected);
    EXPECT_EQ(t.captured_entries, n + padded.size() + 65 * t.blocks.size() + 16);
  }
}
TEST(MD5Trace, EveryOriginalMutationActualHammingSummariesAndCaps) {
  auto base = md5_demo_message();
  auto a    = md5_avalanche(base);
  EXPECT_EQ(base.size(), 9);
  ASSERT_EQ(a.flips.size(), 72);
  double   sum     = 0;
  unsigned minimum = 128, maximum = 0;
  for (std::size_t bit = 0; bit < 72; ++bit) {
    auto m = base;
    m[bit / 8] ^= static_cast<char>(1u << (bit % 8));
    auto     digest  = md5(m);
    unsigned changed = 0;
    for (std::size_t byte = 0; byte < 16; ++byte)
      for (unsigned j = 0; j < 8; ++j)
        changed += ((a.baseline[byte] ^ digest[byte]) >> j) & 1;
    EXPECT_EQ(a.flips[bit].bit, bit);
    EXPECT_EQ(a.flips[bit].changed, changed);
    EXPECT_EQ(md5_hamming(a.baseline, digest), changed);
    sum += changed;
    minimum = std::min(minimum, changed);
    maximum = std::max(maximum, changed);
  }
  EXPECT_DOUBLE_EQ(a.mean_changed, sum / 72);
  EXPECT_EQ(a.minimum_changed, minimum);
  EXPECT_EQ(a.maximum_changed, maximum);
  EXPECT_NO_THROW(md5_trace("abc", 148));
  EXPECT_THROW(md5_trace("abc", 147), std::length_error);
  EXPECT_NO_THROW(md5_trace("abc", 0));
  EXPECT_NO_THROW(md5_avalanche("a", 8));
  EXPECT_THROW(md5_avalanche("a", 7), std::length_error);
  EXPECT_NO_THROW(md5_avalanche("a", 0));
  auto empty = md5_avalanche("");
  EXPECT_EQ(empty.baseline, md5(""));
  EXPECT_TRUE(empty.flips.empty());
  EXPECT_DOUBLE_EQ(empty.mean_changed, 0);
  EXPECT_EQ(empty.minimum_changed, 0);
  EXPECT_EQ(empty.maximum_changed, 0);
  auto message = std::string("a\0\xff", 3);
  auto owned   = md5_trace(message);
  message[0]   = 'x';
  EXPECT_EQ(owned.input[0], 'a');
  MD5Digest zero{}, ones;
  ones.fill(255);
  EXPECT_EQ(md5_hamming(zero, ones), 128);
}
