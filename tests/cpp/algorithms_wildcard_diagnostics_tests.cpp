#include <gtest/gtest.h>

#include <datamunge/algorithms/wildcard.hpp>
#include <functional>
#include <random>
#include <string>
#include <vector>
namespace alg = datamunge::algorithms;
namespace {
bool wildcard_diagnostic_reference(const std::string& text, const std::string& pattern) {
  std::vector<std::vector<int>> memo(text.size() + 1, std::vector<int>(pattern.size() + 1, -1));
  std::function<bool(std::size_t, std::size_t)> solve = [&](std::size_t i, std::size_t j) {
    if (memo[i][j] != -1)
      return memo[i][j] != 0;
    bool result;
    if (j == pattern.size())
      result = i == text.size();
    else if (pattern[j] == '*')
      result = solve(i, j + 1) || (i < text.size() && solve(i + 1, j));
    else
      result =
          i < text.size() && (pattern[j] == '?' || pattern[j] == text[i]) && solve(i + 1, j + 1);
    memo[i][j] = result;
    return result;
  };
  return solve(0, 0);
}
} // namespace
TEST(WildcardDiagnostics, PatternStarIsAlwaysWildcardEvenWhenTextContainsStar) {
  EXPECT_TRUE(alg::wildcard_match("*ba", "*a"));
  EXPECT_TRUE(alg::wildcard_match("**anything", "*anything"));
  EXPECT_TRUE(alg::wildcard_match("a*b*c", "a*c"));
  EXPECT_TRUE(alg::wildcard_match(std::string("a\0*b", 4), "a?*b"));
  EXPECT_FALSE(alg::wildcard_match("\xc3\xa9", "?"));
  EXPECT_TRUE(alg::wildcard_match("\xc3\xa9", "??"));
  EXPECT_FALSE(alg::wildcard_match("a", "[ab]"));
  EXPECT_TRUE(alg::wildcard_match("[ab]", "[ab]"));
  EXPECT_TRUE(alg::wildcard_match(".hidden/x", "*"));
  EXPECT_TRUE(alg::wildcard_match("\\x", "\\?"));
  EXPECT_FALSE(alg::wildcard_match("ABC", "abc"));
}
TEST(WildcardDiagnostics, ExhaustiveMetacharacterTextsMatchIndependentRecurrence) {
  std::vector<std::string> words{""};
  std::vector<std::string> level{""};
  for (int length = 1; length <= 5; ++length) {
    std::vector<std::string> next;
    for (const auto& word : level)
      for (char c : std::string("a*?"))
        next.push_back(word + c);
    words.insert(words.end(), next.begin(), next.end());
    level = std::move(next);
  }
  for (const auto& text : words)
    for (const auto& pattern : words)
      ASSERT_EQ(alg::wildcard_match(text, pattern), wildcard_diagnostic_reference(text, pattern))
          << text << " / " << pattern;
}
TEST(WildcardDiagnostics, CountersTraceAndDpAgreeOnRandomBinaryInputs) {
  std::mt19937 engine(2026);
  for (int trial = 0; trial < 1000; ++trial) {
    std::string text, pattern;
    for (unsigned i = 0, n = engine() % 20; i < n; ++i)
      text.push_back(char(engine() % 256));
    for (unsigned i = 0, n = engine() % 10; i < n; ++i) {
      const auto choice = engine() % 4;
      pattern.push_back(choice == 0 ? '*' : choice == 1 ? '?' : char(engine() % 256));
    }
    const bool expected = wildcard_diagnostic_reference(text, pattern);
    const auto stats    = alg::wildcard_match_stats(text, pattern);
    EXPECT_EQ(stats.matched, expected);
    EXPECT_EQ(alg::wildcard_match(text, pattern), expected);
    const auto grid = alg::wildcard_dp_table(text, pattern);
    EXPECT_EQ(grid.double_at("reachable", grid.nrows() - 1), expected);
    const auto trace = alg::wildcard_match_trace(text, pattern);
    ASSERT_GT(trace.nrows(), 0);
    EXPECT_EQ(trace.double_at("literal_tests", trace.nrows() - 1), stats.literal_tests);
    std::size_t retries = 0, stars = 0, questions = 0;
    for (std::size_t i = 0; i < trace.nrows(); ++i) {
      const auto action = trace.string_at("action", i);
      retries += action == "expand-star";
      stars += action == "star-empty" || action == "trailing-star";
      questions += action == "question";
    }
    EXPECT_EQ(retries, stats.retries);
    EXPECT_EQ(stars, stats.star_steps);
    EXPECT_EQ(questions, stats.question_matches);
    const auto terminal = trace.string_at("action", trace.nrows() - 1);
    EXPECT_EQ(trace.nrows(), stats.steps + (terminal == "reject" ? 0 : 1));
    EXPECT_EQ(terminal == "accept", expected);
  }
}
TEST(WildcardDiagnostics, QuadraticFamilyCountsExactLoopWork) {
  for (std::size_t r : {1, 2, 4, 8, 16, 32, 64, 128, 256}) {
    const auto stats =
        alg::wildcard_match_stats(std::string(2 * r, 'a'), "*" + std::string(r, 'a') + "b");
    EXPECT_FALSE(stats.matched);
    EXPECT_EQ(stats.steps, (r + 1) * (r + 1));
    EXPECT_EQ(stats.literal_tests, stats.steps - 1);
    EXPECT_EQ(stats.retries, r);
    EXPECT_EQ(stats.star_steps, 1);
    EXPECT_EQ(stats.question_matches, 0);
  }
}
TEST(WildcardDiagnostics, TableAndTraceBudgetsAreCheckedBeforeUnboundedAllocation) {
  const auto empty = alg::wildcard_dp_table("", "", 1);
  ASSERT_EQ(empty.nrows(), 1);
  EXPECT_EQ(empty.double_at("reachable", 0), 1);
  EXPECT_THROW((void)alg::wildcard_dp_table("", "", 0), std::invalid_argument);
  EXPECT_THROW((void)alg::wildcard_dp_table(std::string(100, 'a'), "*", 100), std::length_error);
  EXPECT_THROW((void)alg::wildcard_match_trace("abc", "a*c", 1), std::length_error);
  EXPECT_THROW((void)alg::wildcard_match_trace("", "", 0), std::invalid_argument);
}
TEST(WildcardDiagnostics, DataFrameRowsAreIndependentAndNullsAreSkipped) {
  datamunge::dstruct::DataFrame data;
  data.add_column("name", std::vector<std::string>{"a*", "ba", "", "a"});
  data.set_null("name", 3);
  data.add_column("id", std::vector<double>{0, 1, 2, 3});
  const auto out = alg::wildcard_summary(data, "name", {"*a", "*", "a?", ""});
  EXPECT_EQ(out.nrows(), 4);
  EXPECT_EQ(out.double_at("matches", 0), 1);
  EXPECT_EQ(out.double_at("matches", 1), 3);
  EXPECT_EQ(out.double_at("matches", 2), 1);
  EXPECT_EQ(out.double_at("matches", 3), 1);
  EXPECT_EQ(out.double_at("skipped_rows", 0), 1);
  EXPECT_EQ(out.double_at("text_bytes", 0), 4);
  const auto kept = alg::wildcard_filter(data, "name", "*");
  EXPECT_EQ(kept.nrows(), 3);
  EXPECT_EQ(kept.get_column("id").values(), (std::vector<double>{0, 1, 2}));
  const auto none = alg::wildcard_filter(data, "name", "never");
  EXPECT_EQ(none.nrows(), 0);
  EXPECT_EQ(none.ncols(), 2);
  EXPECT_EQ(alg::wildcard_summary(data, "name", {}).ncols(), 9);
  EXPECT_THROW((void)alg::wildcard_filter(data, "id", "*"), std::invalid_argument);
  EXPECT_THROW((void)alg::wildcard_filter(data, "missing", "*"), std::out_of_range);
}
