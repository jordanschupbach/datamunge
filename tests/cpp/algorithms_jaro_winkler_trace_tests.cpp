#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <datamunge/algorithms/jaro_winkler.hpp>
#include <datamunge/algorithms/jaro_winkler_analysis.hpp>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
using namespace datamunge::algorithms;
namespace {
void check(const JaroWinklerTrace& t, double p = .1) {
  const auto&           r     = t.run;
  std::size_t           slots = 0, tests = 0, order = 0, prefix = 0, mismatches = 0, common = 0;
  std::set<std::size_t> left, right;
  std::vector<std::pair<std::size_t, std::size_t>> ordered;
  int                                              phase = 0;
  for (const auto& e : t.events) {
    ASSERT_LT(e.first_index, t.first.size());
    ASSERT_LT(e.second_index, t.second.size());
    if (e.compared) {
      EXPECT_EQ(e.first_byte, static_cast<unsigned char>(t.first[e.first_index]));
      EXPECT_EQ(e.second_byte, static_cast<unsigned char>(t.second[e.second_index]));
      EXPECT_EQ(e.equal, e.first_byte == e.second_byte);
    } else {
      EXPECT_EQ(e.first_byte, 0);
      EXPECT_EQ(e.second_byte, 0);
    }
    if (e.phase == JaroWinklerPhase::Candidate) {
      EXPECT_EQ(phase, 0);
      ++slots;
      EXPECT_EQ(e.occupied, right.contains(e.second_index));
      EXPECT_EQ(e.compared, !e.occupied);
      tests += e.compared;
      EXPECT_LE(
          e.first_index > e.second_index ? e.first_index - e.second_index
                                         : e.second_index - e.first_index,
          r.window);
      if (e.compared && e.equal) {
        EXPECT_TRUE(left.insert(e.first_index).second);
        EXPECT_TRUE(right.insert(e.second_index).second);
      }
    } else if (e.phase == JaroWinklerPhase::Order) {
      EXPECT_LE(phase, 1);
      phase = 1;
      ++order;
      EXPECT_TRUE(e.compared);
      mismatches += !e.equal;
      ordered.push_back({e.first_index, e.second_index});
    } else {
      phase = 2;
      EXPECT_EQ(e.first_index, prefix);
      EXPECT_EQ(e.second_index, prefix);
      EXPECT_TRUE(e.compared);
      ++prefix;
      common += e.equal;
    }
  }
  EXPECT_EQ(slots, r.work.candidate_slots);
  EXPECT_EQ(tests, r.work.candidate_byte_tests);
  EXPECT_EQ(order, r.work.order_byte_tests);
  EXPECT_EQ(prefix, r.work.prefix_byte_tests);
  EXPECT_EQ(left.size(), r.matches);
  EXPECT_EQ(order, r.matches);
  EXPECT_EQ(mismatches, r.order_mismatches);
  EXPECT_EQ(common, r.prefix);
  auto a = left.begin(), b = right.begin();
  for (const auto& [i, j] : ordered) {
    ASSERT_NE(a, left.end());
    ASSERT_NE(b, right.end());
    EXPECT_EQ(i, *a++);
    EXPECT_EQ(j, *b++);
  }
  const double m = r.matches;
  const double expected =
      t.first.empty() && t.second.empty() ? 1
      : m ? ((m / t.first.size() + m / t.second.size() + (m - mismatches / 2.) / m) / 3)
          : 0;
  EXPECT_DOUBLE_EQ(r.jaro, expected);
  EXPECT_DOUBLE_EQ(r.winkler, r.jaro + common * p * (1 - r.jaro));
  EXPECT_GE(r.jaro, 0);
  EXPECT_LE(r.jaro, 1);
  EXPECT_GE(r.winkler, r.jaro);
  EXPECT_LE(r.winkler, 1);
}
} // namespace
TEST(JaroWinklerTrace, CanonicalMarthaActualMatchesAndOrder) {
  const auto t = jaro_winkler_trace("MARTHA", "MARHTA");
  check(t);
  EXPECT_EQ(t.run.window, 2);
  EXPECT_EQ(t.run.matches, 6);
  EXPECT_EQ(t.run.order_mismatches, 2);
  EXPECT_EQ(t.run.prefix, 3);
  EXPECT_NEAR(t.run.jaro, 17. / 18, 1e-15);
  EXPECT_NEAR(t.run.winkler, 173. / 180, 1e-15);
  std::vector<std::pair<std::size_t, std::size_t>> accepted;
  for (const auto& e : t.events)
    if (e.phase == JaroWinklerPhase::Candidate && e.compared && e.equal)
      accepted.push_back({e.first_index, e.second_index});
  EXPECT_EQ(
      accepted,
      (std::vector<std::pair<std::size_t, std::size_t>>{
          {0, 0}, {1, 1}, {2, 2}, {3, 4}, {4, 3}, {5, 5}}));
}
TEST(JaroWinklerTrace, ExhaustiveSmallBinaryByteStringsAndScales) {
  std::vector<std::string> strings{std::string{}};
  for (std::size_t n = 1; n <= 4; ++n) {
    std::size_t count = 1;
    for (std::size_t i = 0; i < n; ++i)
      count *= 3;
    for (std::size_t code = 0; code < count; ++code) {
      auto        state = code;
      std::string s(n, 0);
      for (auto& c : s) {
        c = std::array<char, 3>{0, 'A', char(255)}[state % 3];
        state /= 3;
      }
      strings.push_back(s);
    }
  }
  for (const auto& a : strings)
    for (const auto& b : strings)
      for (double p : {0., .1, .25}) {
        const auto t = jaro_winkler_trace(a, b, p);
        check(t, p);
        EXPECT_DOUBLE_EQ(t.run.jaro, jaro_similarity(a, b));
        EXPECT_DOUBLE_EQ(t.run.winkler, jaro_winkler_similarity(a, b, p));
        EXPECT_NEAR(t.run.jaro, jaro_similarity(b, a), 1e-15);
      }
}
TEST(JaroWinklerTrace, OddOrderDisagreementAndNonmetricControl) {
  const auto t = jaro_winkler_trace("ABCDEF", "BCADEF");
  check(t);
  EXPECT_EQ(t.run.matches, 6);
  EXPECT_EQ(t.run.order_mismatches, 3);
  EXPECT_NEAR(t.run.jaro, 11. / 12, 1e-15);
  EXPECT_GT(
      1 - jaro_similarity("A", "BA"),
      (1 - jaro_similarity("A", "AA")) + (1 - jaro_similarity("AA", "BA")));
  EXPECT_EQ(jaro_similarity("AB", "BA"), 0); // Equal bytes outside the zero window do not match.
}
TEST(JaroWinklerTrace, PrefixDomainUnconditionalLowScoreAndSaturation) {
  const auto low = jaro_winkler_trace("ABCDxxxxxxxxxx", "ABCDyyyyyyyyyy");
  check(low);
  EXPECT_LT(low.run.jaro, .7);
  EXPECT_GT(low.run.winkler, .7);
  EXPECT_EQ(low.run.prefix, 4);
  const auto saturated = jaro_winkler_trace("ABCDx", "ABCDy", .25);
  check(saturated, .25);
  EXPECT_EQ(saturated.run.winkler, 1);
  for (double p :
       {-.1,
        .25000001,
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_THROW(jaro_winkler_similarity("", "", p), std::invalid_argument);
    EXPECT_THROW(jaro_winkler_trace("A", "A", p), std::invalid_argument);
  }
}
TEST(JaroWinklerTrace, Utf8IsBytesEmbeddedNullAndCompleteCaps) {
  const auto utf = jaro_winkler_trace("\xC3\xA9", "\xC3\xAA");
  check(utf);
  EXPECT_EQ(utf.run.matches, 1);
  EXPECT_NEAR(utf.run.winkler, .7, 1e-15);
  const auto nul = jaro_winkler_trace(std::string("A\0B", 3), std::string("A\0C", 3));
  check(nul);
  EXPECT_EQ(nul.run.prefix, 2);
  EXPECT_EQ(nul.first.size(), 3);
  EXPECT_THROW(jaro_winkler_trace("ABC", "ABC", .1, {5, 0}), std::length_error);
  EXPECT_NO_THROW(jaro_winkler_trace("ABC", "ABC", .1, {6, 0}));
  EXPECT_EQ(jaro_winkler_trace("A", "A", .1, {2, 3}).events.size(), 3);
  EXPECT_THROW(jaro_winkler_trace("A", "A", .1, {2, 2}), std::length_error);
  EXPECT_EQ(jaro_winkler_run("A", "A", .1, {2, 1}).winkler, 1);
  EXPECT_NO_THROW(jaro_winkler_trace("ABC", "ABC", .1, {0, 0}));
}
TEST(JaroWinklerTrace, QuadraticActualNoMatchWorkAndOwnedInputs) {
  for (std::size_t n = 2; n <= 256; n *= 2) {
    const auto r = jaro_winkler_run(std::string(n, 'A'), std::string(n, 'B'));
    EXPECT_EQ(r.work.candidate_slots, (3 * n * n - 2 * n) / 4);
    EXPECT_EQ(r.work.candidate_byte_tests, r.work.candidate_slots);
    EXPECT_EQ(r.work.order_byte_tests, 0);
    EXPECT_EQ(r.work.prefix_byte_tests, 1);
  }
  std::string a = "MARTHA", b = "MARHTA";
  const auto  t = jaro_winkler_trace(a, b);
  a.clear();
  b.clear();
  EXPECT_EQ(t.first, "MARTHA");
  EXPECT_EQ(t.second, "MARHTA");
  auto copy            = t;
  copy.first[0]        = 'Z';
  copy.events[0].equal = false;
  EXPECT_NE(copy.first, t.first);
  EXPECT_NE(copy.events[0].equal, t.events[0].equal);
}
TEST(JaroWinklerTrace, NativeDiagramActualEdgesAndAdmission) {
  const auto t = jaro_winkler_trace("MARTHA", "MARHTA");
  EXPECT_FALSE(jaro_winkler_match_diagram(t, 12).to_svg().empty());
  EXPECT_THROW(jaro_winkler_match_diagram(t, 11), std::length_error);
  auto bad                  = t;
  bad.events[0].second_byte = 99;
  EXPECT_THROW(jaro_winkler_match_diagram(bad), std::invalid_argument);
  EXPECT_THROW(jaro_winkler_match_diagram(jaro_winkler_trace("", "A")), std::invalid_argument);
  EXPECT_EQ(jaro_winkler_name_pairs().size(), 5);
}
TEST(JaroWinklerTrace, CompleteNameRankingAndOriginalIndexTies) {
  std::vector<std::string> names{"DIXON", "MARHTA", "MARTHA", "MARTHA"};
  const auto               rows = jaro_winkler_rank(names, "MARTHA", 3, .1, {4, 29});
  ASSERT_EQ(rows.size(), 3);
  EXPECT_EQ(rows[0].index, 2);
  EXPECT_EQ(rows[1].index, 3);
  EXPECT_EQ(rows[2].index, 1);
  for (const auto& row : rows) {
    EXPECT_EQ(row.value, names[row.index]);
    EXPECT_EQ(row.score.winkler, jaro_winkler_similarity("MARTHA", row.value));
  }
  names.clear();
  EXPECT_EQ(rows[0].value, "MARTHA");
  EXPECT_THROW(jaro_winkler_rank({"A", "B"}, "A", 1, .1, {1, 0}), std::length_error);
  EXPECT_THROW(jaro_winkler_rank({"AAA", "BBBB"}, "Q", 1, .1, {2, 7}), std::length_error);
  EXPECT_NO_THROW(jaro_winkler_rank({"AAA", "BBBB"}, "Q", 1, .1, {2, 8}));
  EXPECT_THROW(jaro_winkler_rank({"AAA", "BBBB"}, "Q", 0, .1, {2, 7}), std::length_error);
  EXPECT_TRUE(jaro_winkler_rank({"A"}, "Q", 0).empty());
  EXPECT_TRUE(jaro_winkler_rank({}, "Q").empty());
  EXPECT_THROW(jaro_winkler_rank({}, "", 0, 1), std::invalid_argument);
  EXPECT_NO_THROW(jaro_winkler_rank({"A", "B"}, "A", 9, .1, {0, 0}));
}
