#include <gtest/gtest.h>

#include <datamunge/algorithms/levenshtein_coding.hpp>
#include <limits>
using namespace datamunge::algorithms;
TEST(LevCodingTrace, FullRangeBoundariesAndActualSpans) {
  std::vector<std::uint64_t> v{0, 1, 2, 4, 7, 16, 100};
  for (int k = 0; k < 64; ++k) {
    v.push_back(std::uint64_t{1} << k);
    if (k)
      v.push_back((std::uint64_t{1} << k) - 1);
  }
  v.push_back(std::numeric_limits<std::uint64_t>::max());
  auto e = levenshtein_encode_trace(v), d = levenshtein_decode_trace(e.bits, int(v.size()));
  EXPECT_EQ(e.bits, levenshtein_encode(v));
  EXPECT_EQ(d.values, v);
  EXPECT_EQ(e.work.bit_visits, e.bits.size());
  EXPECT_EQ(d.work.bit_visits, e.bits.size());
  EXPECT_EQ(e.captured_entries, d.captured_entries);
  std::size_t pos = 0;
  for (std::size_t i = 0; i < v.size(); ++i) {
    EXPECT_EQ(e.words[i].begin, pos);
    EXPECT_EQ(e.words[i].end, d.words[i].end);
    EXPECT_EQ(e.words[i].value, v[i]);
    EXPECT_EQ(e.words[i].blocks.size(), d.words[i].blocks.size());
    for (std::size_t j = 0; j < e.words[i].blocks.size(); ++j) {
      EXPECT_EQ(e.words[i].blocks[j].width, d.words[i].blocks[j].width);
      EXPECT_EQ(e.words[i].blocks[j].value, d.words[i].blocks[j].value);
      EXPECT_EQ(e.words[i].blocks[j].begin, d.words[i].blocks[j].begin);
    }
    pos = e.words[i].end;
  }
  EXPECT_EQ(pos, e.bits.size());
}
TEST(LevCodingTrace, OriginalRangeAndNoPrefixCollisions) {
  std::vector<std::uint64_t> v;
  for (std::uint64_t n = 0; n <= 10000; ++n)
    v.push_back(n);
  auto b = levenshtein_encode(v);
  EXPECT_EQ(levenshtein_decode(b, int(v.size())), v);
  std::vector<std::vector<bool>> words;
  for (std::uint64_t n = 0; n < 256; ++n)
    words.push_back(levenshtein_encode({n}));
  for (std::size_t i = 0; i < words.size(); ++i)
    for (std::size_t j = 0; j < words.size(); ++j)
      if (i != j && words[i].size() <= words[j].size())
        EXPECT_FALSE(std::equal(words[i].begin(), words[i].end(), words[j].begin()));
}
TEST(LevCodingTrace, EveryTruncatedNonemptyWordAndMalformedPrefix) {
  for (auto n : {1ULL, 2ULL, 10ULL, 1000ULL, 18446744073709551615ULL}) {
    auto b = levenshtein_encode({n});
    for (std::size_t i = 0; i < b.size(); ++i) {
      std::vector<bool> prefix(b.begin(), b.begin() + i);
      EXPECT_THROW(levenshtein_decode(prefix, 1), std::invalid_argument);
    }
  }
  EXPECT_THROW(levenshtein_decode({}, -1), std::invalid_argument);
  EXPECT_THROW(levenshtein_decode({}, 1), std::invalid_argument);
  EXPECT_THROW(
      levenshtein_decode({true, true, true, true, true, true, false}, 1), std::invalid_argument);
  EXPECT_TRUE(levenshtein_decode({true}, 0).empty());
  auto d = levenshtein_decode_trace({false, true}, 1);
  EXPECT_EQ(d.values, (std::vector<std::uint64_t>{0}));
  EXPECT_EQ(d.consumed_bits, 1);
  EXPECT_EQ(d.work.bit_visits, 1);
  EXPECT_EQ(d.bits.size(), 2);
}
TEST(LevCodingTrace, QuotaAppendEmptyAndOwnership) {
  auto t = levenshtein_encode_trace({10});
  EXPECT_THROW(levenshtein_encode_trace({10}, t.captured_entries - 1), std::length_error);
  EXPECT_EQ(levenshtein_encode_trace({10}, t.captured_entries).bits, t.bits);
  EXPECT_THROW(levenshtein_decode_trace(t.bits, 1, t.captured_entries - 1), std::length_error);
  EXPECT_EQ(
      levenshtein_decode_trace(t.bits, 1, t.captured_entries).values,
      (std::vector<std::uint64_t>{10}));
  std::vector<bool> b{true};
  levenshtein_encode_value(10, b);
  EXPECT_TRUE(b.front());
  EXPECT_TRUE(std::equal(t.bits.begin(), t.bits.end(), b.begin() + 1));
  auto d = levenshtein_decode_trace(b, 0, 0);
  b[0]   = false;
  EXPECT_TRUE(d.bits[0]);
  EXPECT_TRUE(levenshtein_encode_trace({}).bits.empty());
}
