#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <climits>
#include <datamunge/algorithms/fletcher16.hpp>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <type_traits>
namespace {
using namespace datamunge::algorithms;
// Independent right-to-left weighted formula, without a running second sum.
std::uint16_t oracle(std::string_view s) {
  std::uint64_t a = 0, b = 0;
  for (std::size_t i = 0; i < s.size(); ++i) {
    const auto x = static_cast<unsigned char>(s[i]);
    a            = (a + x) % 255;
    b            = (b + ((s.size() - i) % 255) * x) % 255;
  }
  return static_cast<std::uint16_t>((b << 8) | a);
}
TEST(FletcherNative, EveryOneAndTwoByteMessageAndStateMultiplicity) {
  std::array<std::size_t, 65536> counts{};
  for (unsigned a = 0; a < 256; ++a) {
    const std::string one(1, static_cast<char>(a));
    EXPECT_EQ(fletcher16(one), oracle(one));
    for (unsigned b = 0; b < 256; ++b) {
      std::string s = one;
      s.push_back(static_cast<char>(b));
      const auto tag = fletcher16(s);
      EXPECT_EQ(tag, oracle(s));
      ++counts[tag];
    }
  }
  std::size_t once = 0, twice = 0, four = 0, other = 0;
  for (auto n : counts) {
    if (n == 1)
      ++once;
    else if (n == 2)
      ++twice;
    else if (n == 4)
      ++four;
    else if (n)
      ++other;
  }
  EXPECT_EQ(once, 64516);
  EXPECT_EQ(twice, 508);
  EXPECT_EQ(four, 1);
  EXPECT_EQ(other, 0);
  EXPECT_EQ(counts[0], 4);
}
TEST(FletcherNative, AllCanonicalStatesAndEveryInvalidPackedLane) {
  for (unsigned value = 0; value < 65536; ++value) {
    const auto state = static_cast<std::uint16_t>(value);
    const bool valid = (value & 255) < 255 && (value >> 8) < 255;
    if (valid) {
      EXPECT_EQ(fletcher16_update(state, {}), state);
      EXPECT_EQ(fletcher16_combine(state, 0, 0), state);
    } else {
      EXPECT_THROW(fletcher16_update(state, {}), std::invalid_argument);
      EXPECT_THROW(fletcher16_combine(state, 0, 0), std::invalid_argument);
      EXPECT_THROW(fletcher16_combine(0, state, 0), std::invalid_argument);
    }
  }
}
TEST(FletcherNative, OriginalGoldensBinaryChunksAndFullLengthModulus) {
  EXPECT_EQ(fletcher16("abcde"), 0xC8F0);
  EXPECT_EQ(fletcher16("ABCD"), 38411);
  EXPECT_EQ(fletcher16("ABDC"), 38667);
  EXPECT_EQ(byte_sum32("ABCD"), 266);
  EXPECT_EQ(byte_sum32("ABDC"), 266);
  std::mt19937_64 rng(114);
  for (int fixture = 0; fixture < 1600; ++fixture) {
    std::string s(rng() % 513, '\0');
    for (auto& c : s)
      c = static_cast<char>(rng());
    const auto expected = oracle(s);
    EXPECT_EQ(fletcher16(s), expected);
    for (int cut = 0; cut < 4; ++cut) {
      auto             k = s.empty() ? 0 : rng() % (s.size() + 1);
      std::string_view left(s.data(), k), right(s.data() + k, s.size() - k);
      const auto       a = fletcher16_update(0, left), b = fletcher16_update(0, right);
      EXPECT_EQ(fletcher16_update(a, right), expected);
      EXPECT_EQ(fletcher16_combine(a, b, right.size()), expected);
    }
  }
  const auto a = fletcher16("hello"), b = fletcher16("world");
  for (std::uint64_t length : std::vector<std::uint64_t>{0, 1, 254, 255, 256, 65535, UINT64_MAX}) {
    const auto expected = static_cast<std::uint16_t>(
        (((a >> 8) + (b >> 8) + (static_cast<unsigned __int128>(length) * (a & 255)) % 255) % 255)
            << 8 |
        (((a & 255) + (b & 255)) % 255));
    EXPECT_EQ(fletcher16_combine(a, b, length), expected);
  }
}
TEST(FletcherNative, ActualByteStatesOwnedCopiesAndCompleteTraceQuota) {
  std::string input("\1\2", 2);
  const auto  t = fletcher16_trace(input, 2);
  ASSERT_EQ(t.steps.size(), 2);
  EXPECT_EQ(t.checksum, 0x0403);
  EXPECT_EQ(t.steps[0].before_a, 0);
  EXPECT_EQ(t.steps[0].before_b, 0);
  EXPECT_EQ(t.steps[0].after_a, 1);
  EXPECT_EQ(t.steps[0].after_b, 1);
  EXPECT_EQ(t.steps[1].before_a, 1);
  EXPECT_EQ(t.steps[1].before_b, 1);
  EXPECT_EQ(t.steps[1].after_a, 3);
  EXPECT_EQ(t.steps[1].after_b, 4);
  input[0]             = 99;
  auto edited          = t;
  edited.steps[0].byte = 99;
  EXPECT_EQ(t.steps[0].byte, 1);
  EXPECT_THROW(fletcher16_trace("abcde", 4), std::length_error);
  EXPECT_EQ(fletcher16_trace("abcde", 5).checksum, 0xC8F0);
  EXPECT_EQ(fletcher16_trace("abcde", 0).checksum, 0xC8F0);
  EXPECT_TRUE(fletcher16_trace({}).steps.empty());
  const std::string all(10000, static_cast<char>(255));
  const auto        q = fletcher16_trace(all);
  for (std::size_t i = 0; i < q.steps.size(); ++i) {
    EXPECT_EQ(q.steps[i].index, i);
    EXPECT_EQ(q.steps[i].byte, 255);
    EXPECT_EQ(q.steps[i].after_a, 0);
    EXPECT_EQ(q.steps[i].after_b, 0);
  }
  EXPECT_EQ(q.checksum, 0);
}
TEST(FletcherNative, CompleteBitScansAndIndependentWeightedDeltas) {
  const auto original = fletcher16_flip_scan("abracadabra", 979);
  EXPECT_EQ(original.flips.size(), 88);
  EXPECT_EQ(original.detected, 88);
  EXPECT_EQ(original.bytes_processed, 979);
  for (unsigned a = 0; a < 256; ++a) {
    const std::string s(1, static_cast<char>(a));
    const auto        scan = fletcher16_flip_scan(s);
    EXPECT_EQ(scan.detected, 8);
    for (const auto& r : scan.flips) {
      std::string changed = s;
      changed[0]          = static_cast<char>(a ^ (1U << r.bit));
      EXPECT_EQ(r.checksum, oracle(changed));
      EXPECT_TRUE(r.detected);
    }
  }
  std::mt19937 rng(18114);
  for (int fixture = 0; fixture < 120; ++fixture) {
    std::string s(1 + rng() % 40, '\0');
    for (auto& c : s)
      c = static_cast<char>(rng());
    const auto scan = fletcher16_flip_scan(s);
    EXPECT_EQ(scan.detected, s.size() * 8);
    for (const auto& r : scan.flips) {
      auto changed = s;
      changed[r.bit / 8] =
          static_cast<char>(static_cast<unsigned char>(changed[r.bit / 8]) ^ (1U << (r.bit % 8)));
      const auto expected = oracle(changed);
      EXPECT_EQ(r.checksum, expected);
      EXPECT_EQ(r.delta_a, ((expected & 255) + 255 - (scan.checksum & 255)) % 255);
      EXPECT_EQ(r.delta_b, ((expected >> 8) + 255 - (scan.checksum >> 8)) % 255);
    }
  }
  const auto long_scan = fletcher16_flip_scan(std::string(255, '\0'));
  EXPECT_EQ(long_scan.flips[0].delta_b, 0);
  EXPECT_NE(long_scan.flips[0].delta_a, 0);
  EXPECT_EQ(long_scan.detected, 2040);
}
TEST(FletcherNative, EveryOneByteReplacementAndContextAliases) {
  for (unsigned a = 0; a < 256; ++a) {
    const auto c = fletcher16_replacement_scan(std::string(1, static_cast<char>(a)), 0);
    EXPECT_EQ(c.replacements.size(), 256);
    EXPECT_EQ(c.changed, 255);
    EXPECT_EQ(c.detected, (a == 0 || a == 255) ? 254 : 255);
    EXPECT_EQ(c.missed, (a == 0 || a == 255) ? 1 : 0);
    for (unsigned b = 0; b < 256; ++b) {
      const auto& r = c.replacements[b];
      EXPECT_EQ(r.replacement, b);
      EXPECT_EQ(r.checksum, oracle(std::string(1, static_cast<char>(b))));
      EXPECT_EQ(r.changed, a != b);
      EXPECT_EQ(r.detected, r.checksum != c.checksum);
    }
  }
  std::mt19937 rng(7114);
  for (int fixture = 0; fixture < 100; ++fixture) {
    std::string s(1 + rng() % 32, '\0');
    for (auto& c : s)
      c = static_cast<char>(rng());
    const auto index = rng() % s.size();
    const auto c     = fletcher16_replacement_scan(s, index);
    for (const auto& r : c.replacements) {
      auto changed   = s;
      changed[index] = static_cast<char>(r.replacement);
      EXPECT_EQ(r.checksum, oracle(changed));
    }
  }
}
TEST(FletcherNative, TranspositionsCompositeModulusAndIndependentRecomputation) {
  for (unsigned a = 0; a < 256; ++a)
    for (unsigned b = 0; b < 256; ++b) {
      std::string s;
      s.push_back(static_cast<char>(a));
      s.push_back(static_cast<char>(b));
      const auto changed = checksum_swap_bytes(s, 0, 1);
      const auto before = oracle(s), after = oracle(changed);
      EXPECT_EQ(
          fletcher16_transposition_delta(1, a, b), ((after >> 8) + 255 - (before >> 8)) % 255);
      EXPECT_EQ((before & 255), (after & 255));
    }
  for (auto [distance, delta] :
       std::vector<std::pair<std::size_t, unsigned>>{{3, 85}, {5, 51}, {17, 15}, {255, 1}}) {
    std::string s(distance + 1, '\0');
    s[0]        = 1;
    s[distance] = static_cast<char>(1 + delta);
    EXPECT_EQ(fletcher16(s), fletcher16(checksum_swap_bytes(s, 0, distance)));
    EXPECT_EQ(fletcher16_transposition_delta(distance, 1, 1 + delta), 0);
  }
  EXPECT_EQ(fletcher16(std::string("\1\0\0\126", 4)), 0x5A57);
  EXPECT_EQ(fletcher16(std::string("\126\0\0\1", 4)), 0x5A57);
  std::mt19937_64 rng(45114);
  for (int fixture = 0; fixture < 1000; ++fixture) {
    std::string s(2 + rng() % 511, '\0');
    for (auto& c : s)
      c = static_cast<char>(rng());
    auto i = rng() % s.size(), j = rng() % s.size();
    if (i > j)
      std::swap(i, j);
    const auto before = fletcher16(s), after = fletcher16(checksum_swap_bytes(s, i, j));
    const auto delta = fletcher16_transposition_delta(
        j - i, static_cast<unsigned char>(s[i]), static_cast<unsigned char>(s[j]));
    EXPECT_EQ(delta, ((after >> 8) + 255 - (before >> 8)) % 255);
    EXPECT_EQ(before & 255, after & 255);
  }
  for (unsigned a : {0, 1, 17, 128, 254, 255})
    for (unsigned b : {0, 1, 85, 170, 254, 255}) {
      const auto d = UINT64_MAX;
      const auto expected =
          static_cast<unsigned>((static_cast<unsigned __int128>(d) * ((b + 255 - a) % 255)) % 255);
      EXPECT_EQ(fletcher16_transposition_delta(d, a, b), expected);
      EXPECT_EQ(fletcher16_transposition_delta(0, a, b), 0);
    }
}
TEST(FletcherNative, PlainModuloSumAndBinaryOwnedSwap) {
  EXPECT_EQ(byte_sum32({}), 0);
  EXPECT_EQ(byte_sum32(std::string("\0\377", 2)), 255);
  const std::string wrapping(16843010, static_cast<char>(255));
  EXPECT_EQ(byte_sum32(wrapping), 254);
  const std::string input("\0A\377B", 4);
  const auto        swapped = checksum_swap_bytes(input, 0, 2);
  EXPECT_EQ(static_cast<unsigned char>(swapped[0]), 255);
  EXPECT_EQ(swapped[2], 0);
  EXPECT_EQ(input[0], 0);
  EXPECT_EQ(checksum_swap_bytes(input, 1, 1), input);
  EXPECT_THROW(checksum_swap_bytes(input, 0, 4), std::invalid_argument);
  EXPECT_THROW(checksum_swap_bytes(input, 4, 0), std::invalid_argument);
  EXPECT_THROW(checksum_swap_bytes({}, 0, 0), std::invalid_argument);
}
TEST(FletcherNative, CompleteScanBudgetsEmptyAndInvalidBytePositions) {
  EXPECT_EQ(fletcher16_flip_scan("A", 9).bytes_processed, 9);
  EXPECT_THROW(fletcher16_flip_scan("A", 8), std::length_error);
  EXPECT_EQ(fletcher16_flip_scan("A", 0).detected, 8);
  const auto empty = fletcher16_flip_scan({});
  EXPECT_EQ(empty.bytes_processed, 0);
  EXPECT_EQ(empty.detected, 0);
  EXPECT_EQ(empty.checksum, 0);
  EXPECT_TRUE(empty.flips.empty());
  EXPECT_EQ(fletcher16_replacement_scan("A", 0, 257).bytes_processed, 257);
  EXPECT_THROW(fletcher16_replacement_scan("A", 0, 256), std::length_error);
  EXPECT_EQ(fletcher16_replacement_scan("A", 0, 0).detected, 255);
  EXPECT_THROW(fletcher16_replacement_scan({}, 0), std::invalid_argument);
  EXPECT_THROW(fletcher16_replacement_scan("A", 1), std::invalid_argument);
}
TEST(FletcherNative, OwnedScansAndCanonicalBinaryMessageSemantics) {
  std::string input("\0\377A", 3);
  const auto  f = fletcher16_flip_scan(input);
  const auto  r = fletcher16_replacement_scan(input, 0);
  input.assign(3, 'x');
  auto edited_f                       = f;
  edited_f.flips[0].checksum          = 123;
  auto edited_r                       = r;
  edited_r.replacements[255].checksum = 123;
  EXPECT_NE(f.flips[0].checksum, 123);
  EXPECT_EQ(r.replacements[255].checksum, r.checksum);
  EXPECT_EQ(r.original, 0);
  for (int n : {0, 1, 2, 255, 256, 5552, 10000}) {
    EXPECT_EQ(fletcher16(std::string(n, '\0')), 0);
    EXPECT_EQ(fletcher16(std::string(n, static_cast<char>(255))), 0);
  }
  EXPECT_EQ(fletcher16(std::string("\0abcde", 6)), fletcher16("abcde"));
  EXPECT_EQ(fletcher16(std::string("\377abcde", 6)), fletcher16("abcde"));
}
TEST(FletcherNative, LegacyFunctionPointerAndStreamingAssociativity) {
  using Old = std::uint16_t (*)(const std::string&);
  static_assert(std::is_same_v<decltype(&fletcher16), Old>);
  EXPECT_EQ(fletcher16(""), 0);
  const std::string a("\0A", 2), b("\377B", 2), c("\0\377", 2);
  const auto        ab = fletcher16_combine(fletcher16(a), fletcher16(b), b.size());
  const auto        bc = fletcher16_combine(fletcher16(b), fletcher16(c), c.size());
  EXPECT_EQ(fletcher16_combine(ab, fletcher16(c), c.size()), fletcher16(a + b + c));
  EXPECT_EQ(fletcher16_combine(fletcher16(a), bc, b.size() + c.size()), fletcher16(a + b + c));
}
} // namespace
