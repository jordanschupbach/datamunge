#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <datamunge/algorithms/golomb_analysis.hpp>
#include <datamunge/algorithms/golomb_coding.hpp>
#include <random>
#include <string>

using namespace datamunge::algorithms;
namespace {
// Independent small-value dictionary construction: assign the short binary
// leaves first, then the remaining leaves of the full depth-b tree.
std::string reference(std::uint64_t n, int M) {
  unsigned      b     = 0;
  std::uint64_t power = 1;
  while (power < static_cast<unsigned>(M)) {
    power *= 2;
    ++b;
  }
  auto        q = n / M, r = n % M, shorts = power - M;
  std::string word(q, '1');
  word += '0';
  unsigned      digits = b;
  std::uint64_t label  = r;
  if (r < shorts)
    --digits;
  else
    label += shorts;
  std::string tail(digits, '0');
  for (unsigned i = 0; i < digits; ++i) {
    tail[digits - 1 - i] = '0' + label % 2;
    label /= 2;
  }
  return word + tail;
}
std::vector<bool> bits(const std::string& s) {
  std::vector<bool> b;
  for (char c : s)
    b.push_back(c == '1');
  return b;
}
} // namespace
TEST(GolombChecked, DictionaryAndNativeRecords) {
  for (int M = 1; M <= 65; ++M) {
    std::vector<std::uint64_t> values;
    std::string                expected;
    for (std::uint64_t n = 0; n < 512; ++n) {
      values.push_back(n);
      auto s = reference(n, M);
      expected += s;
      EXPECT_EQ(golomb_code_length(n, M), s.size());
    }
    auto t = golomb_encode_trace(values, M, 512);
    ASSERT_EQ(golomb_bit_string(t.bits), expected);
    EXPECT_EQ(golomb_encode(values, M), t.bits);
    EXPECT_EQ(golomb_decode(t.bits, M, INT_MAX), values);
    auto d = golomb_decode_trace(t.bits, M, INT_MAX, 512);
    ASSERT_EQ(t.words.size(), values.size());
    ASSERT_EQ(d.words.size(), values.size());
    std::size_t pos = 0, unary = 0, rem = 0;
    for (std::size_t i = 0; i < values.size(); ++i) {
      const auto &w = t.words[i], &v = d.words[i];
      auto        word = reference(values[i], M);
      EXPECT_EQ(w.value, values[i]);
      EXPECT_EQ(w.quotient, values[i] / M);
      EXPECT_EQ(w.remainder, values[i] % M);
      EXPECT_EQ(w.begin, pos);
      EXPECT_EQ(w.remainder_begin, pos + w.quotient + 1);
      EXPECT_EQ(w.end, pos + word.size());
      EXPECT_EQ(v.begin, w.begin);
      EXPECT_EQ(v.remainder_begin, w.remainder_begin);
      EXPECT_EQ(v.end, w.end);
      EXPECT_EQ(v.value, w.value);
      EXPECT_EQ(v.quotient, w.quotient);
      EXPECT_EQ(v.remainder, w.remainder);
      unary += w.remainder_begin - w.begin;
      rem += w.end - w.remainder_begin;
      pos = w.end;
    }
    EXPECT_EQ(t.unary_bits, unary);
    EXPECT_EQ(t.remainder_bits, rem);
    EXPECT_EQ(d.unary_bits, unary);
    EXPECT_EQ(d.remainder_bits, rem);
    EXPECT_EQ(d.bits, t.bits);
    EXPECT_EQ(unary + rem, t.bits.size());
  }
}
TEST(GolombChecked, ExhaustiveMalformedAndFraming) {
  for (int M = 1; M <= 9; ++M)
    for (unsigned len = 0; len <= 12; ++len)
      for (unsigned mask = 0; mask < (1u << len); ++mask) {
        std::string s(len, '0');
        for (unsigned j = 0; j < len; ++j)
          s[j] = '0' + ((mask >> j) & 1);
        if (s.empty()) {
          EXPECT_TRUE(golomb_decode({}, M, 1).empty());
          continue;
        }
        // A finite bit string either begins with exactly one dictionary word or is
        // a proper prefix. A quotient can't exceed this finite string's length.
        bool matched = false;
        for (std::uint64_t n = 0; n <= static_cast<unsigned>((len + 1) * M); ++n) {
          auto word = reference(n, M);
          if (s.starts_with(word)) {
            auto d = golomb_decode_trace(bits(s), M, 1);
            ASSERT_EQ(d.words.size(), 1);
            EXPECT_EQ(d.words[0].value, n);
            EXPECT_EQ(d.bits, bits(word));
            matched = true;
            break;
          }
        }
        if (!matched)
          EXPECT_THROW(golomb_decode(bits(s), M, 1), std::invalid_argument);
        EXPECT_TRUE(golomb_decode(bits(s), M, 0).empty());
      }
}
TEST(GolombChecked, DomainQuotaOverflowAndOwnership) {
  for (int M : {0, -1, INT_MIN}) {
    EXPECT_THROW(golomb_encode({}, M), std::invalid_argument);
    EXPECT_THROW(golomb_decode({}, M, 0), std::invalid_argument);
    EXPECT_THROW(golomb_encode_trace({}, M, 1), std::invalid_argument);
  }
  EXPECT_THROW(golomb_decode({}, 3, -1), std::invalid_argument);
  for (int k : {-1, 31, INT_MAX}) {
    EXPECT_THROW(rice_encode({}, k), std::invalid_argument);
    EXPECT_THROW(rice_decode({}, k, 0), std::invalid_argument);
  }
  for (int k = 0; k <= 30; ++k) {
    std::vector<std::uint64_t> v{0, 1, 5, 99};
    auto                       b = rice_encode(v, k);
    EXPECT_EQ(b, golomb_encode(v, 1 << k));
    EXPECT_EQ(rice_decode(b, k, 4), v);
  }
  for (int M : {INT_MAX, INT_MAX - 1, 1 << 30}) {
    auto t = golomb_encode_trace({0, 1, static_cast<unsigned>(M) - 1}, M);
    EXPECT_EQ(
        golomb_decode(t.bits, M, 3),
        (std::vector<std::uint64_t>{0, 1, static_cast<unsigned>(M) - 1}));
  }
  EXPECT_THROW(golomb_code_length(UINT64_MAX, 1), std::length_error);
  std::vector<bool> prefix{true, false};
  auto              original = prefix;
  EXPECT_THROW(golomb_encode_value(UINT64_MAX, 1, prefix), std::length_error);
  EXPECT_EQ(prefix, original);
  golomb_encode_value(8, 3, prefix);
  EXPECT_EQ(golomb_bit_string(prefix), "1011011");
  auto t = golomb_encode_trace({8, 0}, 3, 2);
  EXPECT_THROW(golomb_encode_trace({8, 0}, 3, 1), std::length_error);
  EXPECT_THROW(golomb_decode_trace(t.bits, 3, 2, 1), std::length_error);
  EXPECT_NO_THROW(golomb_decode_trace(t.bits, 3, 1, 1));
  EXPECT_NO_THROW(golomb_decode_trace(t.bits, 3, 2, 0));
  EXPECT_TRUE(golomb_decode_trace({true}, 3, 0, 1).bits.empty());
  auto copy        = t;
  t.words[0].value = 99;
  t.bits[0]        = false;
  EXPECT_EQ(copy.words[0].value, 8);
  EXPECT_TRUE(copy.bits[0]);
}

TEST(GolombChecked, VirtualReconstructionBoundaries) {
  for (int M : {1, 2, 3, 5, INT_MAX}) {
    const std::uint64_t q = UINT64_MAX / M, r = UINT64_MAX % M;
    EXPECT_EQ(golomb_detail::reconstruct(q, r, M), UINT64_MAX);
    if (r + 1 < static_cast<unsigned>(M)) {
      EXPECT_THROW(golomb_detail::reconstruct(q, r + 1, M), std::overflow_error);
    }
    if (q < UINT64_MAX) {
      EXPECT_THROW(golomb_detail::reconstruct(q + 1, 0, M), std::overflow_error);
    }
  }
}
TEST(GolombChecked, GeometricExpectationAndPresentation) {
  for (double a : {0., .25, .5, .8, .95})
    for (int M = 1; M <= 32; ++M) {
      long double expected = 0, weight = 1 - a;
      for (std::uint64_t n = 0; n < 20000; ++n) {
        expected += weight * reference(n, M).size();
        weight *= a;
      }
      EXPECT_NEAR(golomb_geometric_expected(a, M), double(expected), 1e-11);
    }
  EXPECT_EQ(golomb_length_table({0, 8}, {3, 4}).nrows(), 4);
  EXPECT_EQ(golomb_geometric_table(.5, {1, 2}, 200).nrows(), 2);
  for (double a :
       {-1.,
        1.,
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_THROW(golomb_geometric_expected(a, 3), std::invalid_argument);
    EXPECT_THROW(golomb_geometric_table(a, {}), std::invalid_argument);
  }
  EXPECT_THROW(golomb_geometric_table(.5, {}, 0), std::invalid_argument);
  EXPECT_THROW(golomb_length_table({}, {0}), std::invalid_argument);
  EXPECT_THROW(golomb_length_plot({}, {3}), std::invalid_argument);
}
