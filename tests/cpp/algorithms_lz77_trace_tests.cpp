#include <gtest/gtest.h>

#include <algorithm>
#include <datamunge/algorithms/lz77.hpp>
#include <limits>
using namespace datamunge::algorithms;
namespace {
std::vector<LZ77Token> reference(const std::string& data, int window, int cap) {
  std::vector<LZ77Token> out;
  for (std::size_t i = 0; i < data.size();) {
    std::size_t best = 0, offset = 0;
    for (std::size_t j = i > std::size_t(window) ? i - window : 0; j < i; ++j)
      for (std::size_t length = 1; length <= std::min(std::size_t(cap), data.size() - i); ++length)
        if (data.substr(j, length) == data.substr(i, length) && length > best) {
          best   = length;
          offset = i - j;
        }
    bool literal = best < data.size() - i;
    out.push_back({int(offset), int(best), literal ? data[i + best] : '\0', literal});
    i += best + literal;
  }
  return out;
}
void same(const std::vector<LZ77Token>& a, const std::vector<LZ77Token>& b) {
  ASSERT_EQ(a.size(), b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    EXPECT_EQ(a[i].offset, b[i].offset);
    EXPECT_EQ(a[i].length, b[i].length);
    EXPECT_EQ(a[i].next, b[i].next);
    EXPECT_EQ(a[i].literal, b[i].literal);
  }
}
} // namespace
TEST(LZ77Trace, ExhaustiveShortBinaryInputsAgainstSubstringEnumeration) {
  for (int n = 0; n <= 7; ++n)
    for (int word = 0; word < (1 << n); ++word) {
      std::string data(n, 'a');
      for (int i = 0; i < n; ++i)
        data[i] += (word >> i) & 1;
      for (int window : {0, 1, 3, 16})
        for (int cap : {0, 1, 4, 16}) {
          auto t = lz77_encode_trace(data, window, cap);
          same(t.analysis.tokens, reference(data, window, cap));
          same(t.analysis.tokens, lz77_encode(data, window, cap));
          EXPECT_EQ(lz77_decode(t.analysis.tokens), data);
          auto d = lz77_decode_trace(t.analysis.tokens);
          EXPECT_EQ(d.analysis.data, data);
          EXPECT_EQ(d.bytes.size(), data.size());
          for (auto& byte : d.bytes) {
            EXPECT_EQ(byte.value, static_cast<unsigned char>(data[byte.position]));
            if (byte.source != SIZE_MAX) {
              EXPECT_LT(byte.source, byte.position);
              EXPECT_EQ(byte.source, byte.position - t.analysis.tokens[byte.token].offset);
            }
          }
          EXPECT_EQ(t.analysis.work.comparisons, t.comparisons.size());
          EXPECT_EQ(t.analysis.work.candidates, t.candidates.size());
          EXPECT_EQ(t.analysis.work.emissions, t.emissions.size());
          EXPECT_EQ(
              t.captured_entries,
              data.size() + t.comparisons.size() + t.candidates.size() + t.emissions.size());
          for (auto& c : t.comparisons) {
            EXPECT_EQ(c.left, static_cast<unsigned char>(data[c.candidate + c.length]));
            EXPECT_EQ(c.right, static_cast<unsigned char>(data[c.input + c.length]));
            EXPECT_EQ(c.equal, c.left == c.right);
          }
          EXPECT_EQ(d.captured_entries, t.analysis.tokens.size() + data.size());
        }
    }
}
TEST(LZ77Trace, ActualOverlapOldestTieOriginalAndBytes) {
  auto t = lz77_encode_trace("aaaaaa");
  ASSERT_EQ(t.analysis.tokens.size(), 2);
  EXPECT_EQ(t.analysis.tokens[1].offset, 1);
  EXPECT_EQ(t.analysis.tokens[1].length, 5);
  EXPECT_FALSE(t.analysis.tokens[1].literal);
  auto d = lz77_decode_trace(t.analysis.tokens);
  EXPECT_EQ(d.analysis.overlapping_reads, 4);
  EXPECT_EQ(d.analysis.copied, 5);
  EXPECT_EQ(d.analysis.literals, 1);
  auto original = lz77_encode(lz77_demo_text(), 128, 64);
  EXPECT_EQ(lz77_demo_text().size(), 89);
  EXPECT_EQ(original.size(), 33);
  EXPECT_EQ(lz77_decode(original), lz77_demo_text());
  for (auto s : {std::string("a\0a\0a\0", 6), std::string("\xff\0\xff\0", 4)})
    EXPECT_EQ(lz77_decode(lz77_encode(s)), s);
  EXPECT_EQ(lz77_decode({{0, 0, 'a', true}, {1, 3, 'x', false}, {0, 0, 'b', true}}), "aaaab");
}
TEST(LZ77Trace, CompleteMalformedGuardsAndCaptureAdmission) {
  EXPECT_THROW(lz77_encode("x", -1, 1), std::invalid_argument);
  EXPECT_THROW(lz77_encode("x", 1, -1), std::invalid_argument);
  for (auto token : std::vector<LZ77Token>{
           {-1, 1, 0, false},
           {1, -1, 0, false},
           {1, 1, 0, false},
           {1, 0, 0, true},
           {0, 0, 0, false},
           {0, 1, 0, true}})
    EXPECT_THROW(lz77_decode({token}), std::invalid_argument);
  EXPECT_THROW(
      lz77_decode_trace({{0, 0, 'a', true}, {2, 1, 0, false}}, 0, 1), std::invalid_argument);
  auto t = lz77_encode_trace("aaaaaa");
  EXPECT_THROW(lz77_encode_trace("aaaaaa", 4096, 255, t.captured_entries - 1), std::length_error);
  EXPECT_EQ(
      lz77_encode_trace("aaaaaa", 4096, 255, t.captured_entries).captured_entries,
      t.captured_entries);
  auto d = lz77_decode_trace(t.analysis.tokens);
  EXPECT_THROW(lz77_decode_trace(t.analysis.tokens, 5), std::length_error);
  EXPECT_THROW(lz77_decode_trace(t.analysis.tokens, 0, d.captured_entries - 1), std::length_error);
  EXPECT_EQ(lz77_decode_trace(t.analysis.tokens, 6, d.captured_entries).analysis.data, "aaaaaa");
  std::string data = "aaaaaa";
  auto        own  = lz77_encode_trace(data, 4096, 255, 0);
  data[0]          = 'b';
  EXPECT_EQ(own.input, "aaaaaa");
  auto tokens    = own.analysis.tokens;
  auto decoded   = lz77_decode_trace(tokens, 0, 0);
  tokens[0].next = 'b';
  EXPECT_EQ(decoded.input[0].next, 'a');
  EXPECT_EQ(decoded.analysis.data, "aaaaaa");
  EXPECT_TRUE(lz77_encode_trace("").analysis.tokens.empty());
  EXPECT_TRUE(lz77_decode_trace({}).analysis.data.empty());
}
