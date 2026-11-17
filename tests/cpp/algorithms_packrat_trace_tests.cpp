#include <gtest/gtest.h>

#include <algorithm>
#include <datamunge/algorithms/packrat_parser.hpp>
#include <random>
#include <set>
#include <stdexcept>
using namespace datamunge::algorithms;
TEST(PackratTrace, OriginalAnBnLanguageIndependentExhaustiveBytesAndActualRuleRecords) {
  PackratParser parser(packrat_anbn_rules(), "S");
  for (std::size_t length = 0; length <= 10; ++length)
    for (std::size_t mask = 0; mask < (std::size_t(1) << length); ++mask) {
      std::string input;
      for (std::size_t i = 0; i < length; ++i)
        input += (mask >> i) & 1 ? 'a' : 'b';
      bool expected = length > 0 && length % 2 == 0 &&
                      input == std::string(length / 2, 'a') + std::string(length / 2, 'b');
      EXPECT_EQ(parser.parse(input), expected);
      EXPECT_TRUE(parser.last_run().completed);
    }
  auto t = parser.parse_trace("aabb");
  EXPECT_TRUE(t.run.accepted);
  EXPECT_EQ(t.run.rule_evaluations, 3);
  EXPECT_EQ(t.run.memo_hits, 0);
  EXPECT_EQ(parser.rule_calls(), 3);
  std::size_t              units = t.input.size() + 3, entered = 0, exited = 0;
  std::vector<std::size_t> expressions;
  for (auto& e : t.events) {
    units += 1 + e.rule.size();
    entered += e.kind == PackratEventKind::rule_enter;
    exited += e.kind == PackratEventKind::rule_exit;
    if (e.kind == PackratEventKind::expression_result)
      expressions.push_back(e.expression);
    if (e.matched)
      EXPECT_LE(e.end, t.input.size());
  }
  for (auto& m : t.memo) {
    units += 1 + m.rule.size();
    EXPECT_LE(m.position, t.input.size());
    if (m.matched)
      EXPECT_LE(m.end, t.input.size());
  }
  EXPECT_EQ(entered, t.run.rule_evaluations);
  EXPECT_EQ(exited, entered);
  EXPECT_EQ(t.memo.size(), entered);
  EXPECT_EQ(expressions.size(), t.run.expression_evaluations);
  std::sort(expressions.begin(), expressions.end());
  for (std::size_t i = 0; i < expressions.size(); ++i)
    EXPECT_EQ(expressions[i], i + 1);
  EXPECT_EQ(t.captured_entries, units);
  PackratOptions exact;
  exact.max_trace_entries = units;
  PackratParser cap(packrat_anbn_rules(), "S", exact);
  EXPECT_NO_THROW(cap.parse_trace("aabb"));
  --exact.max_trace_entries;
  PackratParser short_cap(packrat_anbn_rules(), "S", exact);
  EXPECT_THROW(short_cap.parse_trace("aabb"), std::length_error);
  EXPECT_FALSE(short_cap.last_run().completed);
}
TEST(PackratTrace, ActualSuccessFailureMemoReuseAndOrderedChoiceCommit) {
  PackratParser reuse(packrat_reuse_rules(), "S");
  auto          t = reuse.parse_trace("aaac");
  EXPECT_TRUE(t.run.accepted);
  EXPECT_EQ(t.run.rule_evaluations, 2);
  EXPECT_EQ(t.run.memo_hits, 1);
  std::size_t hits = 0;
  for (auto& e : t.events)
    if (e.kind == PackratEventKind::memo_hit) {
      ++hits;
      EXPECT_EQ(e.rule, "A");
      EXPECT_EQ(e.position, 0);
      EXPECT_TRUE(e.matched);
      EXPECT_EQ(e.end, 3);
    }
  EXPECT_EQ(hits, 1);
  auto failure = reuse.parse_trace("q");
  EXPECT_FALSE(failure.run.accepted);
  EXPECT_EQ(failure.run.memo_hits, 1);
  bool failed_hit = false;
  for (auto& e : failure.events)
    if (e.kind == PackratEventKind::memo_hit)
      failed_hit = !e.matched;
  EXPECT_TRUE(failed_hit);
  PackratParser commit(
      {{"S", peg_seq({peg_choice({peg_lit("a"), peg_lit("ab")}), peg_lit("c")})}}, "S");
  EXPECT_FALSE(commit.parse("abc"));
  PackratParser opposite(
      {{"S", peg_seq({peg_choice({peg_lit("ab"), peg_lit("a")}), peg_lit("c")})}}, "S");
  EXPECT_TRUE(opposite.parse("abc"));
  PackratParser predicate(packrat_negative_lookahead_rules(), "Start");
  EXPECT_TRUE(predicate.parse("aaa"));
  EXPECT_FALSE(predicate.parse("aaac"));
  EXPECT_FALSE(predicate.parse(""));
  PackratParser nullable({{"S", peg_plus(peg_lit(""))}}, "S");
  EXPECT_TRUE(nullable.parse(""));
  std::string   bytes("a\0b", 3);
  PackratParser binary({{"S", peg_lit(bytes)}}, "S");
  EXPECT_TRUE(binary.parse(bytes));
  EXPECT_FALSE(binary.parse("a"));
}
TEST(PackratTrace, CompleteGrammarGuardsLeftRecursionOwnershipPoliciesAndQuadraticWork) {
  EXPECT_THROW((PackratParser({{"S", peg_ref("missing")}}, "S")), std::invalid_argument);
  EXPECT_THROW((PackratParser({{"S", PegExpr{PegOp::Star, "", {}}}}, "S")), std::invalid_argument);
  EXPECT_THROW(
      (PackratParser({{"S", PegExpr{static_cast<PegOp>(99), "", {}}}}, "S")),
      std::invalid_argument);
  PackratParser left({{"S", peg_choice({peg_lit("a"), peg_ref("S")})}}, "S");
  EXPECT_THROW(left.parse("b"), std::invalid_argument);
  EXPECT_FALSE(left.last_run().completed);
  EXPECT_TRUE(left.parse("a"));
  PackratOptions depth;
  depth.max_recursion_depth = 3;
  PackratParser limited(packrat_anbn_rules(), "S", depth);
  EXPECT_THROW(limited.parse("aabb"), std::length_error);
  PackratOptions bytes;
  bytes.max_input_bytes = 2;
  PackratParser input_cap(packrat_anbn_rules(), "S", bytes);
  EXPECT_THROW(input_cap.parse("aabb"), std::length_error);
  EXPECT_TRUE(input_cap.parse("ab"));
  PackratParser owned(packrat_anbn_rules(), "S");
  {
    std::string temporary = "aabb";
    EXPECT_TRUE(owned.parse(temporary));
  }
  EXPECT_EQ(owned.rule_calls(), 3);
  PackratParser copy = owned;
  EXPECT_TRUE(copy.parse("ab"));
  EXPECT_EQ(owned.rule_calls(), 3);
  PackratParser quadratic(packrat_quadratic_rules(), "S");
  EXPECT_TRUE(quadratic.parse(std::string(16, 'a')));
  auto small = quadratic.last_run();
  EXPECT_TRUE(quadratic.parse(std::string(32, 'a')));
  auto large = quadratic.last_run();
  EXPECT_GT(large.expression_evaluations, 2 * small.expression_evaluations);
  EXPECT_LT(large.rule_evaluations, 3 * small.rule_evaluations);
  EXPECT_EQ(packrat_balanced_input(3), "aaabbb");
  EXPECT_THROW(packrat_balanced_input(SIZE_MAX), std::length_error);
}
