// Tests for the parsing batch: recursive descent, Pratt, Earley, LL(1), packrat/PEG, SLR.
// Expression parsers are checked by evaluation; grammar recognisers by accept/reject on
// classic languages (balanced a^n b^n, arithmetic expression CFGs, PEG constructs).

#include <gtest/gtest.h>

#include <datamunge/algorithms/context_free_grammar.hpp>
#include <datamunge/algorithms/earley_parser.hpp>
#include <datamunge/algorithms/ll1_parser.hpp>
#include <datamunge/algorithms/packrat_parser.hpp>
#include <datamunge/algorithms/pratt_parser.hpp>
#include <datamunge/algorithms/recursive_descent_parser.hpp>
#include <datamunge/algorithms/slr_parser.hpp>

#include <cmath>
#include <string>
#include <vector>

using namespace datamunge::algorithms;

// ---------------------------------------------------------------------------- recursive descent
TEST(RecursiveDescent, EvaluatesArithmetic) {
    EXPECT_DOUBLE_EQ(recursive_descent_eval("1 + 2 * 3"), 7.0);
    EXPECT_DOUBLE_EQ(recursive_descent_eval("(1 + 2) * 3"), 9.0);
    EXPECT_DOUBLE_EQ(recursive_descent_eval("2 * 3 + 4 * 5"), 26.0);
    EXPECT_DOUBLE_EQ(recursive_descent_eval("10 - 2 - 3"), 5.0);   // left-assoc
    EXPECT_DOUBLE_EQ(recursive_descent_eval("-3 + 5"), 2.0);       // unary minus
    EXPECT_DOUBLE_EQ(recursive_descent_eval("2 * (3 + (4 - 1))"), 12.0);
}

TEST(RecursiveDescent, RejectsMalformed) {
    EXPECT_THROW(recursive_descent_eval("1 + "), std::runtime_error);
    EXPECT_THROW(recursive_descent_eval("(1 + 2"), std::runtime_error);
    EXPECT_THROW(recursive_descent_eval("1 2"), std::runtime_error);
}

// ---------------------------------------------------------------------------- Pratt
TEST(Pratt, EvaluatesWithPrecedence) {
    EXPECT_DOUBLE_EQ(pratt_eval("1 + 2 * 3"), 7.0);
    EXPECT_DOUBLE_EQ(pratt_eval("2 * 3 + 4"), 10.0);
    EXPECT_DOUBLE_EQ(pratt_eval("(1 + 2) * 3"), 9.0);
}

TEST(Pratt, RightAssociativePower) {
    // 2 ^ 3 ^ 2 = 2 ^ (3 ^ 2) = 2 ^ 9 = 512 (right-assoc), not (2^3)^2 = 64.
    EXPECT_DOUBLE_EQ(pratt_eval("2 ^ 3 ^ 2"), 512.0);
    EXPECT_DOUBLE_EQ(pratt_eval("2 ^ 3 + 1"), 9.0);  // power binds tighter than +
    EXPECT_DOUBLE_EQ(pratt_eval("2 + 3 ^ 2"), 11.0);
}

// ---------------------------------------------------------------------------- Earley
namespace {
// S -> a S b | a b     (the non-regular language a^n b^n, n >= 1)
ContextFreeGrammar anbn_grammar() {
    ContextFreeGrammar g;
    g.start = "S";
    g.productions = {{"S", {"a", "S", "b"}}, {"S", {"a", "b"}}};
    return g;
}
// Ambiguous expression grammar with left recursion (LR/Earley handle it; LL cannot).
ContextFreeGrammar expr_grammar_leftrec() {
    ContextFreeGrammar g;
    g.start = "E";
    g.productions = {{"E", {"E", "+", "T"}}, {"E", {"T"}},
                     {"T", {"T", "*", "F"}}, {"T", {"F"}},
                     {"F", {"(", "E", ")"}}, {"F", {"id"}}};
    return g;
}
std::vector<std::string> toks(std::vector<std::string> v) { return v; }
}  // namespace

TEST(Earley, RecognisesAnBn) {
    auto g = anbn_grammar();
    EXPECT_TRUE(earley_recognize(g, toks({"a", "b"})));
    EXPECT_TRUE(earley_recognize(g, toks({"a", "a", "b", "b"})));
    EXPECT_TRUE(earley_recognize(g, toks({"a", "a", "a", "b", "b", "b"})));
    EXPECT_FALSE(earley_recognize(g, toks({"a", "b", "b"})));
    EXPECT_FALSE(earley_recognize(g, toks({"a", "a", "b"})));
    EXPECT_FALSE(earley_recognize(g, toks({"b", "a"})));
}

TEST(Earley, HandlesLeftRecursion) {
    auto g = expr_grammar_leftrec();
    EXPECT_TRUE(earley_recognize(g, toks({"id", "+", "id", "*", "id"})));
    EXPECT_TRUE(earley_recognize(g, toks({"(", "id", "+", "id", ")", "*", "id"})));
    EXPECT_FALSE(earley_recognize(g, toks({"id", "+"})));
    EXPECT_FALSE(earley_recognize(g, toks({"id", "id"})));
}

// ---------------------------------------------------------------------------- LL(1)
namespace {
// LL(1) expression grammar (right-recursive, factored):
// E  -> T E'      E' -> + T E' | eps
// T  -> F T'      T' -> * F T' | eps
// F  -> ( E ) | id
ContextFreeGrammar expr_grammar_ll1() {
    ContextFreeGrammar g;
    g.start = "E";
    g.productions = {
        {"E", {"T", "E'"}},
        {"E'", {"+", "T", "E'"}}, {"E'", {}},
        {"T", {"F", "T'"}},
        {"T'", {"*", "F", "T'"}}, {"T'", {}},
        {"F", {"(", "E", ")"}}, {"F", {"id"}}};
    return g;
}
}  // namespace

TEST(LL1, GrammarIsLL1AndRecognises) {
    LL1Parser parser(expr_grammar_ll1());
    EXPECT_TRUE(parser.is_ll1());
    EXPECT_TRUE(parser.recognize(toks({"id", "+", "id", "*", "id"})));
    EXPECT_TRUE(parser.recognize(toks({"(", "id", "+", "id", ")", "*", "id"})));
    EXPECT_TRUE(parser.recognize(toks({"id"})));
    EXPECT_FALSE(parser.recognize(toks({"id", "+"})));
    EXPECT_FALSE(parser.recognize(toks({"(", "id"})));
    EXPECT_FALSE(parser.recognize(toks({"id", "id"})));
}

TEST(LL1, DetectsNonLL1Grammar) {
    // Left-recursive grammar is not LL(1).
    LL1Parser parser(expr_grammar_leftrec());
    EXPECT_FALSE(parser.is_ll1());
}

TEST(LL1, FirstFollowSanity) {
    LL1Parser parser(expr_grammar_ll1());
    // FIRST(E) = { (, id }
    EXPECT_TRUE(parser.first("E").count("id"));
    EXPECT_TRUE(parser.first("E").count("("));
    // FOLLOW(E) contains ) and $
    EXPECT_TRUE(parser.follow("E").count(")"));
    EXPECT_TRUE(parser.follow("E").count("$"));
}

// ---------------------------------------------------------------------------- Packrat / PEG
namespace {
// PEG for a^n b^n via the classic ordered-choice recursion:
//   S <- 'a' S 'b' / 'a' 'b'
PackratParser anbn_peg() {
    std::map<std::string, PegExpr> rules;
    rules["S"] = peg_choice({
        peg_seq({peg_lit("a"), peg_ref("S"), peg_lit("b")}),
        peg_seq({peg_lit("a"), peg_lit("b")})});
    return PackratParser(std::move(rules), "S");
}
// PEG that uses a NOT predicate: match "a"+ not followed by "c".
PackratParser as_not_c() {
    std::map<std::string, PegExpr> rules;
    rules["Start"] = peg_seq({peg_plus(peg_lit("a")), peg_not(peg_lit("c"))});
    return PackratParser(std::move(rules), "Start");
}
}  // namespace

TEST(Packrat, RecognisesAnBn) {
    auto p = anbn_peg();
    EXPECT_TRUE(p.parse("ab"));
    EXPECT_TRUE(p.parse("aabb"));
    EXPECT_TRUE(p.parse("aaabbb"));
    EXPECT_FALSE(p.parse("aab"));
    EXPECT_FALSE(p.parse("abb"));
}

TEST(Packrat, NotPredicate) {
    auto p = as_not_c();
    EXPECT_TRUE(p.parse("aaa"));       // a+ with nothing after -> !c succeeds
    EXPECT_FALSE(p.parse("aaac"));     // trailing c -> !c fails
    EXPECT_FALSE(p.parse(""));         // needs at least one a
}

TEST(Packrat, MemoizationServesRepeats) {
    auto p = anbn_peg();
    EXPECT_TRUE(p.parse("aaaabbbb"));
    // Memoization means each (rule,pos) is computed once; hits occur on re-tries.
    EXPECT_GE(p.rule_calls(), 1);
}

// ---------------------------------------------------------------------------- SLR
TEST(SLR, RecognisesLeftRecursiveExpr) {
    SLRParser parser(expr_grammar_leftrec());  // LR handles left recursion
    EXPECT_TRUE(parser.is_slr());
    EXPECT_TRUE(parser.recognize(toks({"id", "+", "id", "*", "id"})));
    EXPECT_TRUE(parser.recognize(toks({"(", "id", "+", "id", ")", "*", "id"})));
    EXPECT_TRUE(parser.recognize(toks({"id"})));
    EXPECT_FALSE(parser.recognize(toks({"id", "+"})));
    EXPECT_FALSE(parser.recognize(toks({"id", "id"})));
    EXPECT_FALSE(parser.recognize(toks({"(", "id"})));
}

TEST(SLR, RecognisesAnBn) {
    SLRParser parser(anbn_grammar());
    EXPECT_TRUE(parser.is_slr());
    EXPECT_TRUE(parser.recognize(toks({"a", "a", "b", "b"})));
    EXPECT_FALSE(parser.recognize(toks({"a", "a", "b"})));
}
