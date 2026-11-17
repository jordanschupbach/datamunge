#include <gtest/gtest.h>

#include <algorithm>
#include <datamunge/algorithms/kmp.hpp>
#include <random>
using namespace datamunge::algorithms;
namespace {
std::vector<std::size_t> naive(const std::string& t, const std::string& p) {
  std::vector<std::size_t> out;
  if (p.empty())
    return out;
  for (std::size_t i = 0; i <= t.size(); ++i)
    if (t.size() - i >= p.size() && t.compare(i, p.size(), p) == 0)
      out.push_back(i);
  return out;
}
std::vector<std::size_t> borders(const std::string& p) {
  std::vector<std::size_t> out(p.size());
  for (std::size_t i = 0; i < p.size(); ++i)
    for (std::size_t len = 1; len <= i; ++len)
      if (p.substr(0, len) == p.substr(i + 1 - len, len))
        out[i] = len;
  return out;
}
std::string binary(int word, int n) {
  std::string s;
  for (int i = 0; i < n; ++i)
    s += char('a' + ((word >> i) & 1));
  return s;
}
void check(const std::string& t, const std::string& p) {
  const auto trace = kmp_search_trace(t, p);
  EXPECT_EQ(trace.matches, naive(t, p));
  EXPECT_EQ(trace.matches, kmp_search(t, p));
  EXPECT_EQ(kmp_prefix_function(p), borders(p));
  const auto a = kmp_search_analyze(t, p);
  EXPECT_EQ(trace.matches, a.matches);
  std::size_t prefix = 0, text = 0, pf = 0, tf = 0;
  std::size_t last_prefix = 0, last_text = 0;
  for (const auto& e : trace.comparisons) {
    const auto& source = e.phase == KMPPhase::Prefix ? p : t;
    ASSERT_LT(e.index, source.size());
    ASSERT_LT(e.border, p.size());
    EXPECT_EQ(e.source_byte, static_cast<unsigned char>(source[e.index]));
    EXPECT_EQ(e.pattern_byte, static_cast<unsigned char>(p[e.border]));
    EXPECT_EQ(e.equal, e.source_byte == e.pattern_byte);
    if (e.phase == KMPPhase::Prefix) {
      ++prefix;
      EXPECT_GE(e.index, last_prefix);
      last_prefix = e.index;
      pf += (e.site == KMPSite::While && !e.equal);
    } else {
      ++text;
      EXPECT_GE(e.index, last_text);
      last_text = e.index;
      tf += (e.site == KMPSite::While && !e.equal);
    }
    if (e.site == KMPSite::While)
      EXPECT_GT(e.border, 0U);
  }
  EXPECT_EQ(trace.work.prefix_comparisons, prefix);
  EXPECT_EQ(trace.work.text_comparisons, text);
  EXPECT_EQ(trace.work.prefix_fallbacks, pf);
  EXPECT_EQ(trace.work.text_fallbacks, tf);
  EXPECT_EQ(trace.work.match_resets, trace.matches.size());
  EXPECT_EQ(a.work.text_comparisons, text);
  EXPECT_LE(text, 3 * t.size());
  EXPECT_LE(prefix, 3 * p.size());
  EXPECT_EQ(
      trace.captured_entries,
      t.size() + p.size() + trace.prefix.size() + trace.matches.size() + trace.comparisons.size());
  if (p.empty() || p.size() > t.size()) {
    EXPECT_TRUE(trace.comparisons.empty());
    EXPECT_TRUE(trace.prefix.empty());
  } else
    EXPECT_EQ(trace.prefix, borders(p));
}
} // namespace
TEST(KMPTrace, ExhaustiveByteMatchesAndPrefixBorders) {
  for (int n = 0; n <= 8; ++n)
    for (int word = 0; word < (1 << n); ++word)
      for (int m = 0; m <= 5; ++m)
        for (int needle = 0; needle < (1 << m); ++needle)
          check(binary(word, n), binary(needle, m));
}
TEST(KMPTrace, OriginalOverlapAndActualFallbackRecords) {
  check("ababacaababaca", "ababaca");
  check("ababababa", "aba");
  check(std::string(100, 'a') + 'c', std::string(20, 'a') + 'b');
  const auto t = kmp_search_trace("ababacaababaca", "ababaca");
  EXPECT_EQ(t.matches, (std::vector<std::size_t>{0, 7}));
  EXPECT_EQ(t.prefix, (std::vector<std::size_t>{0, 0, 1, 2, 3, 0, 1}));
  EXPECT_GT(t.work.prefix_fallbacks, 0U);
}
TEST(KMPTrace, BinaryNullAndHighBytes) {
  std::mt19937 rng(185);
  for (int r = 0; r < 1000; ++r) {
    std::string t, p;
    for (int i = 0; i < 100; ++i)
      t += char(rng() % 256);
    if (r % 2)
      p = t.substr(r % 90, 1 + r % 10);
    else
      for (int i = 0; i < r % 12; ++i)
        p += char(rng() % 256);
    check(t, p);
  }
  check(std::string("a\0a\0a", 5), std::string("a\0a", 3));
}
TEST(KMPTrace, ExactQuotaEarlyExitAndOwnedCopies) {
  for (auto [t, p] : std::vector<std::pair<std::string, std::string>>{
           {"aa", "a"}, {"", ""}, {"x", "long"}, {"ababaca", "aba"}}) {
    const auto a = kmp_search_trace(t, p);
    const auto n = a.captured_entries;
    EXPECT_NO_THROW((void)kmp_search_trace(t, p, n));
    if (n > 1)
      EXPECT_THROW((void)kmp_search_trace(t, p, n - 1), std::length_error);
    EXPECT_NO_THROW((void)kmp_search_trace(t, p, 0));
  }
  auto t    = kmp_search_trace("ababaca", "aba");
  auto c    = t;
  c.text[0] = 'z';
  c.pattern.clear();
  c.prefix.clear();
  c.matches.clear();
  c.comparisons[0].index = 99;
  EXPECT_EQ(t.text, "ababaca");
  EXPECT_EQ(t.pattern, "aba");
  EXPECT_FALSE(t.matches.empty());
  EXPECT_NE(t.comparisons[0].index, 99U);
}
