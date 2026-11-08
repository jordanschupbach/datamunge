#include <gtest/gtest.h>

#include <algorithm>
#include <datamunge/algorithms/fibonacci_coding.hpp>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
namespace {
using namespace datamunge::algorithms;
std::vector<bool> bits(const std::string& s) {
  std::vector<bool> b;
  for (char c : s)
    b.push_back(c == '1');
  return b;
}
std::string text(const std::vector<bool>& b) {
  std::string s;
  for (bool x : b)
    s += x ? '1' : '0';
  return s;
}
void work(const FibonacciWork& a, const FibonacciWork& b) {
  EXPECT_EQ(a.values, b.values);
  EXPECT_EQ(a.payload_bits, b.payload_bits);
  EXPECT_EQ(a.terminator_bits, b.terminator_bits);
  EXPECT_EQ(a.bit_visits, b.bit_visits);
}
void check(const std::vector<std::uint64_t>& v) {
  const auto e = fibonacci_encode_trace(v, 0), d = fibonacci_decode_trace(e.bits, 0);
  EXPECT_EQ(e.values, v);
  EXPECT_EQ(d.values, v);
  EXPECT_EQ(e.bits, d.bits);
  EXPECT_EQ(e.bits, fibonacci_encode(v));
  EXPECT_EQ(v, fibonacci_decode(e.bits));
  EXPECT_TRUE(e.encoding);
  EXPECT_FALSE(d.encoding);
  EXPECT_EQ(e.words.size(), v.size());
  EXPECT_EQ(d.words.size(), v.size());
  EXPECT_EQ(e.captured_entries, e.bits.size() + 2 * v.size());
  EXPECT_EQ(d.captured_entries, e.captured_entries);
  work(e.work, d.work);
  work(e.work, fibonacci_encode_work(v).work);
  work(d.work, fibonacci_decode_work(e.bits).work);
  EXPECT_EQ(e.work.values, v.size());
  EXPECT_EQ(e.work.terminator_bits, v.size());
  EXPECT_EQ(e.work.bit_visits, e.bits.size());
  EXPECT_EQ(e.work.payload_bits + e.work.terminator_bits, e.work.bit_visits);
  std::size_t begin = 0;
  for (std::size_t i = 0; i < v.size(); ++i) {
    const auto& a = e.words[i];
    const auto& b = d.words[i];
    EXPECT_EQ(a.value, v[i]);
    EXPECT_EQ(a.begin, begin);
    EXPECT_EQ(a.begin, b.begin);
    EXPECT_EQ(a.end, b.end);
    EXPECT_EQ(a.delimiter_begin, a.end - 2);
    EXPECT_EQ(a.delimiter_begin, b.delimiter_begin);
    EXPECT_EQ(a.end - a.begin, fibonacci_code_length(v[i]));
    ASSERT_LE(a.end, e.bits.size());
    EXPECT_TRUE(e.bits[a.end - 1]);
    EXPECT_TRUE(e.bits[a.end - 2]);
    unsigned __int128 sum = 0;
    for (std::size_t p = a.begin; p < a.end - 1; ++p) {
      if (p + 1 < a.end - 1)
        EXPECT_FALSE(e.bits[p] && e.bits[p + 1]);
      if (e.bits[p])
        sum += fibonacci_coding_weight(p - a.begin);
    }
    EXPECT_EQ(sum, v[i]);
    begin = a.end;
  }
  EXPECT_EQ(begin, e.bits.size());
}
TEST(FibonacciTrace, KnownTinyWords) {
  const std::vector<std::pair<std::uint64_t, std::string>> gold{
      {1, "11"},
      {2, "011"},
      {3, "0011"},
      {4, "1011"},
      {10, "010011"},
      {20, "0101011"},
      {50, "001001011"},
      {100, "00101000011"}};
  for (auto& [n, s] : gold) {
    EXPECT_EQ(text(fibonacci_encode({n})), s);
    EXPECT_EQ(fibonacci_decode(bits(s)), std::vector<std::uint64_t>{n});
  }
  check({1, 2, 3, 4, 7, 10, 50, 100, 1000, 100000});
  check({});
}
TEST(FibonacciTrace, IndependentNonAdjacentEnumeration) {
  std::vector<std::uint64_t> f{1, 2};
  for (int i = 2; i < 20; ++i)
    f.push_back(f[i - 1] + f[i - 2]);
  std::vector<bool> seen(17711);
  std::size_t       count = 0;
  for (std::size_t length = 1; length <= 20; ++length) {
    std::string                                           digits(length, '0');
    std::function<void(std::size_t, bool, std::uint64_t)> visit =
        [&](std::size_t pos, bool previous, std::uint64_t sum) {
          if (pos == length) {
            if (digits.back() != '1')
              return;
            ASSERT_LT(sum, seen.size());
            EXPECT_FALSE(seen[sum]);
            seen[sum] = true;
            ++count;
            const auto code = digits + '1';
            EXPECT_EQ(text(fibonacci_encode({sum})), code);
            EXPECT_EQ(fibonacci_decode(bits(code)), std::vector<std::uint64_t>{sum});
            return;
          }
          digits[pos] = '0';
          visit(pos + 1, false, sum);
          if (!previous) {
            digits[pos] = '1';
            visit(pos + 1, true, sum + f[pos]);
          }
        };
    visit(0, false, 0);
  }
  EXPECT_EQ(count, 17710);
  for (std::size_t n = 1; n < seen.size(); ++n)
    EXPECT_TRUE(seen[n]);
}
TEST(FibonacciTrace, AllWeightBoundariesAndMaximum) {
  unsigned __int128 a = 1, b = 2;
  for (std::size_t i = 0; i < 92; ++i) {
    const auto w = fibonacci_coding_weight(i);
    EXPECT_EQ(w, a);
    check({w});
    if (w > 1)
      check({w - 1});
    if (w < UINT64_MAX)
      check({w + 1});
    const auto next = a + b;
    a               = b;
    b               = next;
  }
  EXPECT_GT(a, UINT64_MAX);
  EXPECT_EQ(fibonacci_code_length(UINT64_MAX), 93);
  check({7540113804746346429ULL, 12200160415121876738ULL, UINT64_MAX});
  std::vector<bool> old_boundary(90, false);
  old_boundary.push_back(true);
  old_boundary.push_back(true);
  EXPECT_EQ(fibonacci_decode(old_boundary), std::vector<std::uint64_t>{7540113804746346429ULL});
}
TEST(FibonacciTrace, SeededFullWidthRoundTrips) {
  std::mt19937_64 rng(108);
  for (int batch = 0; batch < 100; ++batch) {
    std::vector<std::uint64_t> v;
    for (int i = 0; i < 100; ++i)
      v.push_back(rng() | 1);
    check(v);
  }
  for (int batch = 0; batch < 600; ++batch) {
    std::vector<std::uint64_t> v;
    for (std::size_t i = 0, k = rng() % 30; i < k; ++i)
      v.push_back(1 + rng() % 1000000);
    check(v);
  }
}
TEST(FibonacciTrace, StrictTailsAndOverflow) {
  for (auto s : {"0", "1", "10", "110", "111", "001", "00000"}) {
    EXPECT_THROW(fibonacci_decode(bits(s)), std::invalid_argument);
    EXPECT_THROW(fibonacci_decode_work(bits(s)), std::invalid_argument);
    EXPECT_THROW(fibonacci_decode_trace(bits(s)), std::invalid_argument);
  }
  std::vector<bool> too_long(93, false);
  too_long.push_back(true);
  too_long.push_back(true);
  EXPECT_THROW(fibonacci_decode(too_long), std::overflow_error);
  std::vector<bool> maxplus = bits(
      "00001000010100010100000100010101000100100010010000000010010001001000100010100000100010100101"
      "1");
  EXPECT_THROW(fibonacci_decode(maxplus), std::overflow_error);
  const auto e = fibonacci_encode_trace({1, 2, 10, UINT64_MAX}, 0);
  for (std::size_t end = 1; end < e.bits.size(); ++end) {
    std::vector<bool> prefix(e.bits.begin(), e.bits.begin() + end);
    const bool        boundary =
        std::any_of(e.words.begin(), e.words.end(), [&](const auto& w) { return w.end == end; });
    if (boundary)
      EXPECT_NO_THROW(fibonacci_decode(prefix));
    else
      EXPECT_THROW(fibonacci_decode(prefix), std::invalid_argument);
  }
}
TEST(FibonacciTrace, AppendPreservesArbitraryPrefix) {
  std::vector<bool> out{false, true, false};
  const auto        before = out;
  EXPECT_THROW(fibonacci_encode_value(0, out), std::invalid_argument);
  EXPECT_EQ(out, before);
  fibonacci_encode_value(UINT64_MAX, out);
  const auto word = fibonacci_encode({UINT64_MAX});
  EXPECT_EQ(std::vector<bool>(out.begin(), out.begin() + 3), before);
  EXPECT_EQ(std::vector<bool>(out.begin() + 3, out.end()), word);
  fibonacci_encode_value(10, out);
  const auto second = fibonacci_encode({10});
  EXPECT_EQ(std::vector<bool>(out.end() - second.size(), out.end()), second);
}
TEST(FibonacciTrace, ActualSpansAndOwnership) {
  const auto e = fibonacci_encode_trace({1, 2, 10}, 17);
  ASSERT_EQ(e.words.size(), 3);
  EXPECT_EQ(e.words[0].begin, 0);
  EXPECT_EQ(e.words[0].end, 2);
  EXPECT_EQ(e.words[1].begin, 2);
  EXPECT_EQ(e.words[1].end, 5);
  EXPECT_EQ(e.words[2].begin, 5);
  EXPECT_EQ(e.words[2].delimiter_begin, 9);
  EXPECT_EQ(e.words[2].end, 11);
  EXPECT_EQ(e.work.payload_bits, 8);
  EXPECT_EQ(e.work.terminator_bits, 3);
  EXPECT_EQ(e.captured_entries, 17);
  auto       edited   = e;
  const auto copy     = edited;
  edited.bits[0]      = false;
  edited.values[0]    = 99;
  edited.words[0].end = 99;
  EXPECT_TRUE(copy.bits[0]);
  EXPECT_EQ(copy.values[0], 1);
  EXPECT_EQ(copy.words[0].end, 2);
}
TEST(FibonacciTrace, CaptureQuotasAndOrdinaryMillionBits) {
  EXPECT_THROW(fibonacci_encode_trace({1, 2}, 8), std::length_error);
  EXPECT_THROW(fibonacci_decode_trace(bits("11011"), 8), std::length_error);
  EXPECT_THROW(fibonacci_decode_trace(bits("11011"), 4), std::length_error);
  EXPECT_NO_THROW(fibonacci_encode_trace({1, 2}, 9));
  EXPECT_NO_THROW(fibonacci_decode_trace(bits("11011"), 9));
  EXPECT_NO_THROW(fibonacci_encode_trace({1, 2}, 0));
  EXPECT_NO_THROW(fibonacci_decode_trace(bits("11011"), 0));
  EXPECT_EQ(fibonacci_encode_trace({}, 1).captured_entries, 0);
  EXPECT_EQ(fibonacci_decode_trace({}, 1).captured_entries, 0);
  const std::vector<std::uint64_t> large(500001, 1);
  const auto                       b = fibonacci_encode(large);
  EXPECT_EQ(b.size(), 1000002);
  EXPECT_EQ(fibonacci_decode(b), large);
  EXPECT_THROW(fibonacci_encode_trace(large), std::length_error);
  EXPECT_THROW(fibonacci_decode_trace(b), std::length_error);
}
TEST(FibonacciTrace, CorruptionIsNotErrorCorrection) {
  EXPECT_EQ(fibonacci_decode(bits("0011")), std::vector<std::uint64_t>{3});
  EXPECT_EQ(fibonacci_decode(bits("1011")), std::vector<std::uint64_t>{4});
  for (std::size_t n : {4, 16, 64}) {
    auto b = fibonacci_encode(std::vector<std::uint64_t>(n, 1));
    b[2]   = false;
    EXPECT_THROW(fibonacci_decode(b), std::invalid_argument);
  }
  auto b   = fibonacci_encode({10});
  b.back() = false;
  EXPECT_THROW(fibonacci_decode(b), std::invalid_argument);
}
TEST(FibonacciTrace, DomainAndCallableSignatures) {
  EXPECT_THROW(fibonacci_code_length(0), std::invalid_argument);
  EXPECT_THROW(fibonacci_encode({1, 0, 2}), std::invalid_argument);
  EXPECT_THROW(fibonacci_encode_work({0}), std::invalid_argument);
  EXPECT_THROW(fibonacci_encode_trace({0}, 1), std::invalid_argument);
  EXPECT_THROW(fibonacci_coding_weight(92), std::out_of_range);
  EXPECT_THROW(fibonacci_coding_weight(std::numeric_limits<std::size_t>::max()), std::out_of_range);
  void (*append)(std::uint64_t, std::vector<bool>&) = &fibonacci_encode_value;
  std::vector<bool> out;
  append(1, out);
  EXPECT_EQ(text(out), "11");
}
} // namespace
