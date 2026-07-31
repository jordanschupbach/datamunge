#pragma once

/// \file packrat_parser.hpp
/// \brief A packrat parser for Parsing Expression Grammars (PEGs) with memoization.
///
/// A *Parsing Expression Grammar* (Ford, 2004) looks like a CFG but is fundamentally
/// *recognition-based* and unambiguous: its choice operator is *ordered* (\f$e_1/e_2\f$ tries
/// \f$e_1\f$ first and only falls back to \f$e_2\f$ if it fails), and it adds syntactic
/// predicates \f$\&e\f$ (match without consuming) and \f$!e\f$ (fail if \f$e\f$ matches).
/// PEGs never backtrack across an accepted choice, so they describe exactly what a
/// hand-written recursive-descent parser does -- but greedily and deterministically.
///
/// A naive PEG parser can be exponential because the same rule may be retried at the same
/// position many times. *Packrat* parsing fixes this by *memoizing* every
/// \f$(\text{rule}, \text{position})\f$ result, guaranteeing each is computed once -- giving
/// *linear time* in exchange for linear memory. This implementation interprets a small PEG
/// (literals, ordered choice, sequence, repetition, optional, and the and/not predicates)
/// over a character string, and counts rule invocations so the memoization win is visible.

#include <map>
#include <string>
#include <vector>

namespace datamunge::algorithms {

/// \brief PEG expression node kinds.
enum class PegOp { Literal, Ref, Seq, Choice, Star, Plus, Opt, And, Not };

/// \brief A PEG expression tree node.
struct PegExpr {
    PegOp op;
    std::string text;               // literal text, or referenced rule name
    std::vector<PegExpr> children;  // sub-expressions
};

// --- convenience constructors --------------------------------------------------------
inline PegExpr peg_lit(std::string s) { return {PegOp::Literal, std::move(s), {}}; }
inline PegExpr peg_ref(std::string name) { return {PegOp::Ref, std::move(name), {}}; }
inline PegExpr peg_seq(std::vector<PegExpr> es) { return {PegOp::Seq, "", std::move(es)}; }
inline PegExpr peg_choice(std::vector<PegExpr> es) { return {PegOp::Choice, "", std::move(es)}; }
inline PegExpr peg_star(PegExpr e) { return {PegOp::Star, "", {std::move(e)}}; }
inline PegExpr peg_plus(PegExpr e) { return {PegOp::Plus, "", {std::move(e)}}; }
inline PegExpr peg_opt(PegExpr e) { return {PegOp::Opt, "", {std::move(e)}}; }
inline PegExpr peg_and(PegExpr e) { return {PegOp::And, "", {std::move(e)}}; }
inline PegExpr peg_not(PegExpr e) { return {PegOp::Not, "", {std::move(e)}}; }

/// \brief A packrat parser: named PEG rules parsed with (rule, pos) memoization.
class PackratParser {
   public:
    PackratParser(std::map<std::string, PegExpr> rules, std::string start)
        : rules_(std::move(rules)), start_(std::move(start)) {}

    /// \brief Parse \p input; returns true iff the start rule matches the entire string.
    bool parse(const std::string& input) {
        input_ = &input;
        memo_.clear();
        rule_calls_ = 0;
        memo_hits_ = 0;
        int end = parse_rule(start_, 0);
        return end == static_cast<int>(input.size());
    }

    /// \brief Number of (rule, pos) rule bodies evaluated in the last parse.
    long rule_calls() const { return rule_calls_; }
    /// \brief Number of memo-table hits (invocations served from cache) in the last parse.
    long memo_hits() const { return memo_hits_; }

   private:
    std::map<std::string, PegExpr> rules_;
    std::string start_;
    const std::string* input_ = nullptr;
    std::map<std::pair<std::string, int>, int> memo_;  // (rule, pos) -> end pos or -1 (fail)
    long rule_calls_ = 0;
    long memo_hits_ = 0;

    static constexpr int kFail = -1;

    // Parse a named rule at pos, memoizing the result.
    int parse_rule(const std::string& name, int pos) {
        auto key = std::make_pair(name, pos);
        auto it = memo_.find(key);
        if (it != memo_.end()) { ++memo_hits_; return it->second; }
        ++rule_calls_;
        int r = parse_expr(rules_.at(name), pos);
        memo_[key] = r;
        return r;
    }

    // Parse an expression at pos; return end position or kFail.
    int parse_expr(const PegExpr& e, int pos) {
        const std::string& s = *input_;
        switch (e.op) {
            case PegOp::Literal: {
                if (pos + static_cast<int>(e.text.size()) <= static_cast<int>(s.size()) &&
                    s.compare(pos, e.text.size(), e.text) == 0)
                    return pos + static_cast<int>(e.text.size());
                return kFail;
            }
            case PegOp::Ref:
                return parse_rule(e.text, pos);
            case PegOp::Seq: {
                int p = pos;
                for (const auto& c : e.children) {
                    p = parse_expr(c, p);
                    if (p == kFail) return kFail;
                }
                return p;
            }
            case PegOp::Choice: {
                for (const auto& c : e.children) {
                    int p = parse_expr(c, pos);
                    if (p != kFail) return p;  // ordered choice: first success wins
                }
                return kFail;
            }
            case PegOp::Star: {
                int p = pos;
                for (;;) {
                    int q = parse_expr(e.children[0], p);
                    if (q == kFail || q == p) break;  // stop on failure or no progress
                    p = q;
                }
                return p;
            }
            case PegOp::Plus: {
                int p = parse_expr(e.children[0], pos);
                if (p == kFail) return kFail;
                for (;;) {
                    int q = parse_expr(e.children[0], p);
                    if (q == kFail || q == p) break;
                    p = q;
                }
                return p;
            }
            case PegOp::Opt: {
                int p = parse_expr(e.children[0], pos);
                return p == kFail ? pos : p;
            }
            case PegOp::And:  // &e : succeeds without consuming if e matches
                return parse_expr(e.children[0], pos) == kFail ? kFail : pos;
            case PegOp::Not:  // !e : succeeds without consuming if e fails
                return parse_expr(e.children[0], pos) == kFail ? pos : kFail;
        }
        return kFail;
    }
};

}  // namespace datamunge::algorithms
