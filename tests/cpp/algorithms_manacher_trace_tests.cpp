#include <gtest/gtest.h>

#include <datamunge/algorithms/manacher.hpp>
#include <limits>
#include <random>
using namespace datamunge::algorithms;
namespace {
void check(const std::string& s) {
  auto                     t = manacher_trace(s);
  std::vector<std::size_t> expected(2 * s.size() + 3, 0);
  PalindromeResult         longest;
  for (std::size_t start = 0; start < s.size(); ++start)
    for (std::size_t end = start + 1; end <= s.size(); ++end) {
      bool palindrome = true;
      for (std::size_t k = 0; k < (end - start) / 2; ++k)
        palindrome = palindrome && s[start + k] == s[end - 1 - k];
      if (palindrome) {
        const auto length                = end - start;
        expected[2 * start + length + 1] = std::max(expected[2 * start + length + 1], length);
        if (length > longest.length)
          longest = {start, length};
      }
    }
  EXPECT_EQ(t.analysis.radii, expected);
  EXPECT_EQ(t.analysis.longest.start, longest.start);
  EXPECT_EQ(t.analysis.longest.length, longest.length);
  EXPECT_EQ(longest_palindrome(s).start, longest.start);
  EXPECT_EQ(longest_palindrome(s).length, longest.length);
  EXPECT_EQ(palindrome_radii(s), expected);
  EXPECT_EQ(t.symbols.size(), 2 * s.size() + 3);
  EXPECT_EQ(t.analysis.work.centers, 2 * s.size() + 1);
  EXPECT_EQ(t.centers.size(), t.analysis.work.centers);
  EXPECT_EQ(t.comparisons.size(), t.analysis.work.comparisons);
  EXPECT_LE(t.analysis.work.comparisons, 2 * t.symbols.size());
  std::size_t successes = 0, reuses = 0, updates = 0;
  for (auto& c : t.comparisons) {
    EXPECT_EQ(c.left, c.center - c.radius - 1);
    EXPECT_EQ(c.right, c.center + c.radius + 1);
    EXPECT_LT(c.left, t.symbols.size());
    EXPECT_LT(c.right, t.symbols.size());
    EXPECT_EQ(c.left_symbol, t.symbols[c.left]);
    EXPECT_EQ(c.right_symbol, t.symbols[c.right]);
    EXPECT_EQ(c.equal, c.left_symbol == c.right_symbol);
    successes += c.equal;
  }
  for (auto& c : t.centers) {
    EXPECT_EQ(c.final, expected[c.index]);
    EXPECT_GE(c.right_after, c.right_before);
    EXPECT_EQ(c.reused, c.index < c.right_before);
    if (c.reused) {
      EXPECT_EQ(c.mirror, c.center_before - (c.index - c.center_before));
      EXPECT_EQ(c.initial, std::min(expected[c.mirror], c.right_before - c.index));
    } else
      EXPECT_EQ(c.initial, 0);
    EXPECT_EQ(c.frontier_updated, c.index + c.final > c.right_before);
    reuses += c.reused;
    updates += c.frontier_updated;
  }
  EXPECT_EQ(successes, t.analysis.work.expansions);
  EXPECT_EQ(reuses, t.analysis.work.mirror_reuses);
  EXPECT_EQ(updates, t.analysis.work.frontier_updates);
  EXPECT_EQ(
      t.captured_entries,
      s.size() + 2 * t.symbols.size() + t.centers.size() + t.comparisons.size());
}
} // namespace
TEST(ManacherTrace, ExhaustiveBinaryAndSentinelByteWords) {
  for (int n = 0; n <= 8; ++n)
    for (int word = 0; word < (1 << n); ++word) {
      std::string s(n, 'a');
      for (int i = 0; i < n; ++i)
        s[i] += (word >> i) & 1;
      check(s);
    }
  for (int n = 0, power = 1; n <= 6; ++n, power *= 3)
    for (int word = 0; word < power; ++word) {
      std::string s(n, '^');
      int         value = word;
      for (int i = 0; i < n; ++i) {
        s[i] = "^#$"[value % 3];
        value /= 3;
      }
      check(s);
    }
}
TEST(ManacherTrace, OriginalLeftmostTiesAllBytesAndRandomBinaryData) {
  check("abacaba");
  auto original = manacher_analyze("abacaba");
  EXPECT_EQ(original.best_center, 8);
  EXPECT_EQ(original.longest.length, 7);
  EXPECT_EQ(
      original.radii,
      (std::vector<std::size_t>{0, 0, 1, 0, 3, 0, 1, 0, 7, 0, 1, 0, 3, 0, 1, 0, 0}));
  EXPECT_EQ(longest_palindrome("babad").start, 0);
  EXPECT_EQ(longest_palindrome("babad").length, 3);
  check("abba");
  check("^");
  check("#");
  check("$");
  check(std::string("\0\xff\0", 3));
  std::string all;
  for (int i = 0; i < 256; ++i)
    all.push_back(char(i));
  check(all);
  std::mt19937_64 rng(117);
  for (int run = 0; run < 500; ++run) {
    std::string s(rng() % 25, '\0');
    for (auto& c : s)
      c = char(rng() % 256);
    check(s);
  }
}
TEST(ManacherTrace, QuotaEmptyLabelsAndOwnership) {
  auto t = manacher_trace("abacaba");
  EXPECT_THROW(manacher_trace("abacaba", t.captured_entries - 1), std::length_error);
  EXPECT_EQ(manacher_trace("abacaba", t.captured_entries).captured_entries, t.captured_entries);
  std::string s   = "^#^";
  auto        own = manacher_trace(s, 0);
  s[0]            = 'x';
  EXPECT_EQ(own.input, "^#^");
  EXPECT_EQ(own.analysis.longest.length, 3);
  auto empty = manacher_trace("");
  EXPECT_EQ(empty.analysis.longest.length, 0);
  EXPECT_EQ(empty.analysis.radii.size(), 3);
  EXPECT_EQ(empty.centers.size(), 1);
  EXPECT_EQ(empty.comparisons.size(), 1);
  EXPECT_EQ(manacher_symbol_label(94), "byte0x5E");
  EXPECT_EQ(manacher_symbol_label(256), "separator");
  EXPECT_EQ(manacher_symbol_label(257), "left guard");
  EXPECT_EQ(manacher_symbol_label(258), "right guard");
  EXPECT_THROW(manacher_symbol_label(-1), std::invalid_argument);
  EXPECT_THROW(manacher_symbol_label(259), std::invalid_argument);
}
