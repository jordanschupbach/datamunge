#include <gtest/gtest.h>

#include <algorithm>
#include <datamunge/algorithms/match_rating.hpp>
#include <random>
#include <stdexcept>
using namespace datamunge::algorithms;
namespace {
std::string reference(const std::string& input) {
  std::string letters;
  for (unsigned char raw : input) {
    if (raw >= 'a' && raw <= 'z')
      raw -= 32;
    if (raw >= 'A' && raw <= 'Z')
      letters += raw;
  }
  std::string keep;
  for (std::size_t i = 0; i < letters.size(); ++i)
    if (i == 0 || (std::string("AEIOU").find(letters[i]) == std::string::npos &&
                   letters[i] != letters[i - 1]))
      keep += letters[i];
  if (keep.size() > 6)
    keep.erase(3, keep.size() - 6);
  return keep;
}
MatchRatingResult compare(const std::string& a, const std::string& b) {
  auto              x = reference(a), y = reference(b);
  MatchRatingResult out;
  auto              gap = x.size() > y.size() ? x.size() - y.size() : y.size() - x.size();
  if (gap >= 3)
    return out;
  out.comparable = true;
  constexpr int threshold[]{5, 5, 5, 5, 5, 4, 4, 4, 3, 3, 3, 3, 2};
  out.min_rating = threshold[x.size() + y.size()];
  std::string rx, ry;
  for (std::size_t i = 0; i < std::max(x.size(), y.size()); ++i) {
    char a = i < x.size() ? x[i] : 0, b = i < y.size() ? y[i] : 0;
    if (a != b) {
      if (a)
        rx += a;
      if (b)
        ry += b;
    }
  }
  std::reverse(rx.begin(), rx.end());
  std::reverse(ry.begin(), ry.end());
  int ua = 0, ub = 0;
  for (std::size_t i = 0; i < std::max(rx.size(), ry.size()); ++i) {
    char a = i < rx.size() ? rx[i] : 0, b = i < ry.size() ? ry[i] : 0;
    if (a != b) {
      ua += a != 0;
      ub += b != 0;
    }
  }
  out.similarity = 6 - std::max(ua, ub);
  out.match      = out.similarity >= out.min_rating;
  return out;
}
void equal(MatchRatingResult a, MatchRatingResult b) {
  EXPECT_EQ(a.comparable, b.comparable);
  EXPECT_EQ(a.min_rating, b.min_rating);
  EXPECT_EQ(a.similarity, b.similarity);
  EXPECT_EQ(a.match, b.match);
}
} // namespace
TEST(MatchRatingTrace, AllSmallByteWordsActualCodexPhasesAndQuotaCounts) {
  const std::string alphabet = "BAb1-";
  for (std::size_t n = 0; n <= 6; ++n) {
    std::size_t count = 1;
    for (std::size_t j = 0; j < n; ++j)
      count *= alphabet.size();
    for (std::size_t code = 0; code < count; ++code) {
      std::string input(n, ' ');
      auto        bits = code;
      for (char& c : input) {
        c = alphabet[bits % alphabet.size()];
        bits /= alphabet.size();
      }
      auto t = match_rating_codex_trace(input);
      EXPECT_EQ(t.analysis.codex, reference(input));
      EXPECT_EQ(match_rating_codex(input), reference(input));
      EXPECT_EQ(t.steps.size(), n);
      std::string   normalized, stripped;
      unsigned char previous = 0;
      bool          first    = true;
      for (std::size_t i = 0; i < n; ++i) {
        auto&         s      = t.steps[i];
        unsigned char c      = input[i];
        bool          letter = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
        unsigned char upper  = letter ? (c >= 'a' ? c - 32 : c) : 0;
        EXPECT_EQ(s.index, i);
        EXPECT_EQ(s.raw, c);
        EXPECT_EQ(s.alphabetic, letter);
        EXPECT_EQ(s.previous, previous);
        EXPECT_EQ(s.first, first);
        EXPECT_EQ(s.upper, upper);
        EXPECT_EQ(s.before, stripped.size());
        if (letter) {
          normalized += upper;
          if (s.kept)
            stripped += upper;
          previous = upper;
          first    = false;
        } else
          EXPECT_FALSE(s.kept);
        EXPECT_EQ(s.after, stripped.size());
      }
      EXPECT_EQ(t.analysis.normalized, normalized);
      EXPECT_EQ(t.analysis.stripped, stripped);
      EXPECT_EQ(
          t.captured_entries,
          2 * n + normalized.size() + stripped.size() + t.analysis.codex.size());
    }
  }
  EXPECT_EQ(match_rating_codex("Babab"), "BBB");
  EXPECT_EQ(match_rating_codex("Ashcraft"), "ASHRFT");
  EXPECT_EQ(match_rating_codex_analyze("Ashcraft").stripped, "ASHCRFT");
}
TEST(MatchRatingTrace, RandomPairsIndependentCancellationSymmetryAndActualRecords) {
  std::mt19937 rng(7);
  std::string  letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz -19";
  for (int trial = 0; trial < 5000; ++trial) {
    std::string a(rng() % 30, ' '), b(rng() % 30, ' ');
    for (auto& c : a)
      c = letters[rng() % letters.size()];
    for (auto& c : b)
      c = letters[rng() % letters.size()];
    auto t = match_rating_trace(a, b);
    equal(t.analysis.result, compare(a, b));
    equal(t.analysis.result, match_rating_comparison(a, b));
    equal(t.analysis.result, match_rating_comparison(b, a));
    std::size_t count = t.a.captured_entries + t.b.captured_entries + t.cancellations.size() +
                        t.analysis.residual_a.size() + t.analysis.residual_b.size();
    EXPECT_EQ(t.captured_entries, count);
    std::string ra, rb;
    std::size_t ua = 0, ub = 0;
    for (auto& s : t.cancellations) {
      if (!s.reverse) {
        EXPECT_EQ(s.before_a, ra.size());
        EXPECT_EQ(s.before_b, rb.size());
        unsigned char x = s.index < t.analysis.codex_a.size() ? t.analysis.codex_a[s.index] : 0,
                      y = s.index < t.analysis.codex_b.size() ? t.analysis.codex_b[s.index] : 0;
        EXPECT_EQ(s.a, x);
        EXPECT_EQ(s.b, y);
        if (x != y) {
          if (x)
            ra += x;
          if (y)
            rb += y;
        }
        EXPECT_EQ(s.after_a, ra.size());
        EXPECT_EQ(s.after_b, rb.size());
      } else {
        EXPECT_EQ(s.before_a, ua);
        EXPECT_EQ(s.before_b, ub);
        auto          i = s.index;
        unsigned char x = i < ra.size() ? ra[ra.size() - 1 - i] : 0,
                      y = i < rb.size() ? rb[rb.size() - 1 - i] : 0;
        EXPECT_EQ(s.a, x);
        EXPECT_EQ(s.b, y);
        if (x != y) {
          ua += x != 0;
          ub += y != 0;
        }
        EXPECT_EQ(s.after_a, ua);
        EXPECT_EQ(s.after_b, ub);
      }
      EXPECT_EQ(s.equal, s.a == s.b);
    }
    EXPECT_EQ(t.analysis.residual_a, ra);
    EXPECT_EQ(t.analysis.residual_b, rb);
    EXPECT_EQ(t.analysis.unmatched_a, ua);
    EXPECT_EQ(t.analysis.unmatched_b, ub);
  }
}
TEST(MatchRatingTrace, OriginalPairsThresholdsEmptyNontransitivityQuotasAndOwnership) {
  EXPECT_EQ(match_rating_demo_pairs().size(), 12);
  for (auto& [a, b] : match_rating_demo_pairs())
    equal(match_rating_comparison(a, b), compare(a, b));
  std::string   consonants = "BCDFGH";
  constexpr int thresholds[]{5, 5, 5, 5, 5, 4, 4, 4, 3, 3, 3, 3, 2};
  for (std::size_t n = 0; n <= 6; ++n)
    for (std::size_t m = 0; m <= 6; ++m) {
      auto r   = match_rating_comparison(consonants.substr(0, n), consonants.substr(0, m));
      auto gap = n > m ? n - m : m - n;
      EXPECT_EQ(r.comparable, gap < 3);
      if (r.comparable)
        EXPECT_EQ(r.min_rating, thresholds[n + m]);
    }
  auto empty = match_rating_comparison("123", "---");
  EXPECT_TRUE(empty.match);
  EXPECT_EQ(empty.similarity, 6);
  EXPECT_TRUE(match_rating_comparison("AB", "AC").match);
  EXPECT_TRUE(match_rating_comparison("AC", "DC").match);
  EXPECT_FALSE(match_rating_comparison("AB", "DC").match);
  EXPECT_NO_THROW(match_rating_codex_trace("A", 5));
  EXPECT_THROW(match_rating_codex_trace("A", 4), std::length_error);
  EXPECT_NO_THROW(match_rating_trace("A", "A", 11));
  EXPECT_THROW(match_rating_trace("A", "A", 10), std::length_error);
  EXPECT_NO_THROW(match_rating_trace("A", "A", 0));
  auto name = std::string("A\0B", 3);
  auto t    = match_rating_codex_trace(name);
  name[0]   = 'X';
  EXPECT_EQ(t.input[0], 'A');
  EXPECT_EQ(t.analysis.codex, "AB");
  EXPECT_EQ(t.steps[1].raw, 0);
}
