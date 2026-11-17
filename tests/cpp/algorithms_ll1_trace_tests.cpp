#include <gtest/gtest.h>

#include <datamunge/algorithms/ll1_parser.hpp>
using namespace datamunge::algorithms;
TEST(LL1Trace, ExhaustiveIndependentAnBnLanguageAndActualStackActions) {
  LL1Parser p({{{"S", {"a", "S", "b"}}, {"S", {}}}, "S"});
  EXPECT_TRUE(p.is_ll1());
  for (int n = 0; n <= 10; ++n)
    for (int bits = 0; bits < (1 << n); ++bits) {
      std::vector<std::string> input(n);
      for (int i = 0; i < n; ++i)
        input[i] = bits >> i & 1 ? "b" : "a";
      bool expected = n % 2 == 0;
      for (int i = 0; i < n; ++i)
        expected = expected && (input[i] == (i < n / 2 ? "a" : "b"));
      auto t = p.trace(input);
      EXPECT_EQ(t.analysis.accepted, expected);
      EXPECT_EQ(p.recognize(input), expected);
      EXPECT_EQ(t.analysis.steps, t.events.size());
      auto        stack  = t.initial_stack;
      std::size_t offset = 0;
      for (auto& e : t.events) {
        EXPECT_EQ(e.before, stack);
        EXPECT_EQ(e.input_offset, offset);
        EXPECT_EQ(e.top, stack.back());
        if (e.action == LL1ParseAction::Expand) {
          stack.pop_back();
          auto& rhs = p.grammar().productions[e.production].rhs;
          for (auto i = rhs.rbegin(); i != rhs.rend(); ++i)
            stack.push_back(*i);
        } else if (e.action == LL1ParseAction::Match) {
          EXPECT_EQ(e.top, input[offset]);
          stack.pop_back();
          ++offset;
        }
        EXPECT_EQ(e.after, stack);
      }
      EXPECT_EQ(t.analysis.consumed, offset);
    }
}
TEST(LL1Trace, OriginalExpressionGrammarAndNullableFixedPoints) {
  LL1Parser p(ll1_expression_grammar());
  EXPECT_TRUE(p.is_ll1());
  EXPECT_EQ(p.first("E"), (std::set<std::string>{"(", "id"}));
  EXPECT_EQ(p.follow("E"), (std::set<std::string>{"$", ")"}));
  for (auto input : std::vector<std::vector<std::string>>{
           {"id", "+", "id", "*", "id"}, {"(", "id", "+", "id", ")", "*", "id"}, {"id"}})
    EXPECT_TRUE(p.recognize(input));
  EXPECT_FALSE(p.recognize({"id", "+"}));
  EXPECT_FALSE(p.recognize({"id", "id"}));
  EXPECT_FALSE(p.recognize({"z"}));
  LL1Parser nullable({{{"S", {"A", "B"}}, {"A", {"a"}}, {"A", {}}, {"B", {"b"}}, {"B", {}}}, "S"});
  EXPECT_EQ(nullable.first("S"), (std::set<std::string>{"", "a", "b"}));
  EXPECT_EQ(nullable.follow("A"), (std::set<std::string>{"$", "b"}));
  EXPECT_TRUE(nullable.recognize({}));
  EXPECT_TRUE(nullable.recognize({"a", "b"}));
}
TEST(LL1Trace, ActualConflictsRefuseParsingAndDoNotHang) {
  LL1Parser p(ll1_expression_grammar(true));
  EXPECT_FALSE(p.is_ll1());
  EXPECT_FALSE(p.conflicts().empty());
  EXPECT_EQ(p.construction_work().conflicts, p.conflicts().size());
  for (auto& e : p.conflicts()) {
    EXPECT_NE(e.retained, e.incoming);
    EXPECT_EQ(p.table().at({e.nonterminal, e.lookahead}), e.retained);
  }
  EXPECT_THROW(p.recognize({"id"}), std::logic_error);
  LL1Parser duplicate({{{"S", {"a"}}, {"S", {"a"}}}, "S"});
  EXPECT_FALSE(duplicate.is_ll1());
  EXPECT_THROW(duplicate.trace({"a"}), std::logic_error);
}
TEST(LL1Trace, CompleteGrammarTokenQuotaAndBudgetGuards) {
  EXPECT_THROW(LL1Parser({{}, "S"}), std::invalid_argument);
  EXPECT_THROW(LL1Parser({{{"S", {"$"}}}, "S"}), std::invalid_argument);
  EXPECT_THROW(LL1Parser({{{"S", {""}}}, "S"}), std::invalid_argument);
  LL1Parser p({{{"S", {"a"}}}, "S"});
  EXPECT_THROW(p.recognize({"a", "$", "extra"}), std::invalid_argument);
  EXPECT_THROW(p.recognize({""}), std::invalid_argument);
  EXPECT_THROW(p.first("unknown"), std::out_of_range);
  auto zero = p.trace({"a"}, 0);
  EXPECT_EQ(zero.analysis.status, LL1ParseStatus::StepLimit);
  EXPECT_EQ(zero.analysis.steps, 0);
  EXPECT_EQ(zero.captured_entries, 3);
  auto t = p.trace({"a"});
  EXPECT_EQ(t.analysis.status, LL1ParseStatus::Accepted);
  EXPECT_EQ(t.analysis.steps, 3);
  EXPECT_THROW(p.trace({"a"}, 100000, t.captured_entries - 1), std::length_error);
  EXPECT_EQ(p.trace({"a"}, 100000, t.captured_entries).captured_entries, t.captured_entries);
  std::vector<std::string> input{"a"};
  auto                     own = p.trace(input, 100000, 0);
  input[0]                     = "b";
  EXPECT_EQ(own.tokens[0], "a");
}
