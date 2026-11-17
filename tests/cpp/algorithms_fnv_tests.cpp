#include <gtest/gtest.h>

#include <datamunge/algorithms/fnv_analysis.hpp>
#include <random>

namespace {
namespace a = datamunge::algorithms;
// Independent wider multiplication with an explicit reduction after each byte.
std::uint64_t oracle(const std::string& s, unsigned bits) {
  unsigned __int128       h    = bits == 32 ? 2166136261ULL : 14695981039346656037ULL;
  const unsigned __int128 mask = (static_cast<unsigned __int128>(1) << bits) - 1;
  for (unsigned char b : s)
    h = ((h ^ b) * (bits == 32 ? 16777619ULL : 1099511628211ULL)) & mask;
  return static_cast<std::uint64_t>(h);
}
TEST(FNV, ExhaustiveShortBinaryStringsAndWideOracle) {
  EXPECT_EQ(a::fnv1a_32("a"), 0xe40c292cU);
  EXPECT_EQ(a::fnv1a_64("a"), 0xaf63dc4c8601ec8cULL);
  for (unsigned bits : {32u, 64u}) {
    EXPECT_EQ(a::fnv1a_trace("", bits).value, oracle("", bits));
    for (unsigned x = 0; x < 256; ++x)
      for (unsigned y = 0; y < 256; ++y) {
        std::string s;
        s.push_back(char(x));
        s.push_back(char(y));
        EXPECT_EQ(bits == 32 ? std::uint64_t(a::fnv1a_32(s)) : a::fnv1a_64(s), oracle(s, bits));
      }
  }
}
TEST(FNV, ActualRecordsAndArbitraryBinaryInputs) {
  std::mt19937_64 rng(104);
  for (int n = 0; n < 400; ++n) {
    std::string s;
    for (std::size_t j = 0, len = rng() % 80; j < len; ++j)
      s.push_back(char(rng()));
    for (unsigned bits : {32u, 64u}) {
      const auto t = a::fnv1a_trace(s, bits);
      ASSERT_EQ(t.steps.size(), s.size());
      EXPECT_EQ(t.value, oracle(s, bits));
      for (std::size_t j = 0; j < s.size(); ++j) {
        const auto& e = t.steps[j];
        EXPECT_EQ(e.index, j);
        EXPECT_EQ(e.byte, static_cast<unsigned char>(s[j]));
        EXPECT_EQ(e.before, oracle(s.substr(0, j), bits));
        EXPECT_EQ(e.xored, e.before ^ e.byte);
        EXPECT_EQ(e.after, oracle(s.substr(0, j + 1), bits));
        EXPECT_EQ(e.after & 1, (e.before ^ e.byte) & 1);
      }
      auto copy = t;
      if (!copy.steps.empty()) {
        copy.steps[0].after ^= 1;
        EXPECT_NE(copy.steps[0].after, t.steps[0].after);
      }
    }
  }
}
TEST(FNV, CompleteTraceQuotaAndEmptyAndInvalidWidth) {
  EXPECT_THROW(a::fnv1a_trace("ab", 32, 1), std::length_error);
  EXPECT_EQ(a::fnv1a_trace("ab", 32, 2).steps.size(), 2u);
  EXPECT_EQ(a::fnv1a_trace("ab", 64, 0).steps.size(), 2u);
  EXPECT_TRUE(a::fnv1a_trace("", 32, 1).steps.empty());
  EXPECT_THROW(a::fnv1a_trace("ab", 16, 1), std::invalid_argument);
  EXPECT_EQ(
      a::fnv_trace_table(a::fnv1a_trace("a", 64)).get_column<std::string>("after").values()[0],
      "AF63DC4C8601EC8C");
}
TEST(FNV, OriginalFixtureAndIndependentBucketPairs) {
  const std::vector<std::size_t> expected{1, 2, 1, 1, 0, 4, 1, 2, 1, 2, 3, 2, 3, 0, 2, 7};
  const auto                     words = a::fnv_example_words();
  const auto                     p     = a::fnv_bucket_profile(words);
  EXPECT_EQ(p.counts, expected);
  EXPECT_EQ(p.keys, 32u);
  EXPECT_EQ(p.distinct_keys, 32u);
  std::uint64_t pairs = 0;
  for (std::size_t i = 0; i < words.size(); ++i)
    for (std::size_t j = 0; j < i; ++j)
      pairs += (oracle(words[i], 64) % 16 == oracle(words[j], 64) % 16);
  EXPECT_EQ(p.bucket_pairs, pairs);
  EXPECT_EQ(p.hash_pairs, 0u);
  EXPECT_EQ(p.table().nrows(), 16u);
  EXPECT_EQ(p.summary().nrows(), 1u);
}
TEST(FNV, AllByteResidueClassesAndExactOccupancies) {
  for (unsigned k = 0; k <= 8; ++k)
    for (unsigned residue = 0; residue < (1u << k); ++residue) {
      const auto keys = a::fnv_byte_keys(k, residue);
      EXPECT_EQ(keys.size(), 256u / (1u << k));
      for (const auto& key : keys)
        EXPECT_EQ(static_cast<unsigned char>(key[0]) & ((1u << k) - 1), residue);
    }
  for (unsigned bits : {32u, 64u}) {
    auto all = a::fnv_bucket_profile(a::fnv_byte_keys(), 16, bits);
    EXPECT_EQ(all.counts, std::vector<std::size_t>(16, 16));
    EXPECT_EQ(all.hash_pairs, 0u);
    auto low = a::fnv_bucket_profile(a::fnv_byte_keys(4, 1), 16, bits);
    EXPECT_EQ(low.counts[12], 16u);
    EXPECT_EQ(low.bucket_pairs, 120u);
    EXPECT_EQ(low.hash_pairs, 0u);
  }
  EXPECT_THROW(a::fnv_byte_keys(9, 0), std::invalid_argument);
  EXPECT_THROW(a::fnv_byte_keys(4, 16), std::invalid_argument);
}
TEST(FNV, DuplicateRecordsAreNotDistinctHashCollisions) {
  auto p = a::fnv_bucket_profile({"costarring", "liquid", "costarring"}, 16, 32);
  EXPECT_EQ(p.distinct_keys, 2u);
  EXPECT_EQ(p.hash_pairs, 1u);
  EXPECT_EQ(p.bucket_pairs, 3u);
  EXPECT_EQ(a::fnv_bucket_profile({"costarring", "liquid"}, 16, 64).hash_pairs, 0u);
  EXPECT_THROW(a::fnv_bucket_profile({}, 0), std::invalid_argument);
  EXPECT_THROW(a::fnv_bucket_profile({}, 1, 8), std::invalid_argument);
  EXPECT_EQ(a::fnv_bucket_profile({}, 16).keys, 0u);
}
TEST(FNV, BitFlipCountsAndWholeRowAdmission) {
  const std::string s("a\0\xff", 3);
  for (unsigned bits : {32u, 64u}) {
    const auto t = a::fnv_bit_flips(s, bits, 24);
    EXPECT_EQ(t.nrows(), 24u);
    for (std::size_t i = 0; i < 24; ++i) {
      auto changed   = s;
      changed[i / 8] = char(static_cast<unsigned char>(changed[i / 8]) ^ (1u << (i % 8)));
      EXPECT_EQ(
          t.get_column<double>("changed_output_bits").values()[i],
          std::popcount(oracle(s, bits) ^ oracle(changed, bits)));
    }
  }
  EXPECT_THROW(a::fnv_bit_flips(s, 32, 23), std::length_error);
  EXPECT_EQ(a::fnv_bit_flips(s, 32, 0).nrows(), 24u);
  EXPECT_THROW(a::fnv_bit_flips(s, 8), std::invalid_argument);
  EXPECT_EQ(a::fnv_bit_flips("", 32).nrows(), 0u);
}
} // namespace
