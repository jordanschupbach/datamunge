#pragma once

/// \file ll1_parser.hpp
/// \brief A table-driven LL(1) predictive parser with FIRST/FOLLOW construction.
///
/// LL(1) parsing is *top-down* and *predictive*: reading input Left-to-right, producing a
/// Leftmost derivation, using *1* token of lookahead, it decides which production to expand
/// with no backtracking. The decision comes from a parse table \f$M[A][a]\f$ built from two
/// classic sets:
/// - \f$\mathrm{FIRST}(\alpha)\f$: the terminals that can begin a string derived from
///   \f$\alpha\f$ (plus \f$\varepsilon\f$ if \f$\alpha\f$ can vanish);
/// - \f$\mathrm{FOLLOW}(A)\f$: the terminals that can appear immediately after \f$A\f$.
/// For each production \f$A\to\alpha\f$, we set \f$M[A][a]=\alpha\f$ for every
/// \f$a\in\mathrm{FIRST}(\alpha)\f$, and -- if \f$\alpha\f$ is nullable -- for every
/// \f$a\in\mathrm{FOLLOW}(A)\f$. A grammar is *LL(1)* exactly when this assignment has no
/// conflicts. Parsing then runs a stack: expand a nonterminal via the table, or match a
/// terminal against the input, until the stack empties.
///
/// This implementation builds FIRST/FOLLOW and the table, reports whether the grammar is
/// LL(1), and recognises input strings. The empty production is written as an empty rhs.

#include <map>
#include <set>
#include <string>
#include <vector>

#include <datamunge/algorithms/context_free_grammar.hpp>

namespace datamunge::algorithms {

/// \brief An LL(1) predictive parser: FIRST/FOLLOW sets, parse table, and recognition.
class LL1Parser {
   public:
    static constexpr const char* kEpsilon = "";   // empty string marks epsilon in FIRST sets
    static constexpr const char* kEndMarker = "$"; // end-of-input marker

    explicit LL1Parser(ContextFreeGrammar grammar) : g_(std::move(grammar)) {
        compute_first();
        compute_follow();
        build_table();
    }

    /// \brief True if the grammar has no LL(1) table conflicts.
    bool is_ll1() const { return !has_conflict_; }

    /// \brief FIRST set of a single symbol.
    const std::set<std::string>& first(const std::string& sym) const { return first_.at(sym); }
    /// \brief FOLLOW set of a nonterminal.
    const std::set<std::string>& follow(const std::string& nt) const { return follow_.at(nt); }

    /// \brief Recognise whether \p tokens (terminals) is in the language.
    bool recognize(const std::vector<std::string>& tokens) const {
        std::vector<std::string> input = tokens;
        input.push_back(kEndMarker);
        std::vector<std::string> stack = {kEndMarker, g_.start};
        std::size_t ip = 0;
        while (!stack.empty()) {
            std::string top = stack.back();
            const std::string& look = input[ip];
            if (top == kEndMarker) return look == kEndMarker;
            if (!g_.is_nonterminal(top)) {
                // terminal on top: must match the lookahead
                if (top == look) { stack.pop_back(); ++ip; }
                else return false;
            } else {
                auto it = table_.find({top, look});
                if (it == table_.end()) return false;  // no table entry -> reject
                stack.pop_back();
                const auto& rhs = g_.productions[it->second].rhs;
                for (auto r = rhs.rbegin(); r != rhs.rend(); ++r) stack.push_back(*r);
            }
        }
        return input[ip] == kEndMarker;
    }

   private:
    ContextFreeGrammar g_;
    std::map<std::string, std::set<std::string>> first_, follow_;
    std::map<std::pair<std::string, std::string>, int> table_;
    bool has_conflict_ = false;

    bool is_nt(const std::string& s) const { return g_.is_nonterminal(s); }

    // FIRST of a symbol sequence (with epsilon marker "").
    std::set<std::string> first_of_seq(const std::vector<std::string>& seq) const {
        std::set<std::string> result;
        bool all_nullable = true;
        for (const auto& sym : seq) {
            const auto& fs = first_.at(sym);
            for (const auto& t : fs)
                if (t != kEpsilon) result.insert(t);
            if (!fs.count(kEpsilon)) { all_nullable = false; break; }
        }
        if (all_nullable) result.insert(kEpsilon);
        return result;
    }

    void compute_first() {
        // terminals: FIRST = {self}; nonterminals: start empty.
        for (const auto& t : g_.terminals()) first_[t] = {t};
        for (const auto& nt : g_.nonterminals()) first_[nt];  // empty set
        bool changed = true;
        while (changed) {
            changed = false;
            for (const auto& p : g_.productions) {
                std::set<std::string> add;
                if (p.rhs.empty()) {
                    add.insert(kEpsilon);
                } else {
                    bool all_nullable = true;
                    for (const auto& sym : p.rhs) {
                        for (const auto& t : first_[sym])
                            if (t != kEpsilon) add.insert(t);
                        if (!first_[sym].count(kEpsilon)) { all_nullable = false; break; }
                    }
                    if (all_nullable) add.insert(kEpsilon);
                }
                for (const auto& t : add)
                    if (first_[p.lhs].insert(t).second) changed = true;
            }
        }
    }

    void compute_follow() {
        for (const auto& nt : g_.nonterminals()) follow_[nt];
        follow_[g_.start].insert(kEndMarker);
        bool changed = true;
        while (changed) {
            changed = false;
            for (const auto& p : g_.productions) {
                for (std::size_t i = 0; i < p.rhs.size(); ++i) {
                    const std::string& B = p.rhs[i];
                    if (!is_nt(B)) continue;
                    std::vector<std::string> beta(p.rhs.begin() + i + 1, p.rhs.end());
                    auto fb = first_of_seq(beta);
                    for (const auto& t : fb)
                        if (t != kEpsilon)
                            if (follow_[B].insert(t).second) changed = true;
                    if (beta.empty() || fb.count(kEpsilon)) {
                        for (const auto& t : follow_[p.lhs])
                            if (follow_[B].insert(t).second) changed = true;
                    }
                }
            }
        }
    }

    void build_table() {
        for (int i = 0; i < static_cast<int>(g_.productions.size()); ++i) {
            const auto& p = g_.productions[i];
            auto fa = first_of_seq(p.rhs);
            for (const auto& a : fa) {
                if (a == kEpsilon) continue;
                insert_entry(p.lhs, a, i);
            }
            if (fa.count(kEpsilon)) {
                for (const auto& b : follow_[p.lhs]) insert_entry(p.lhs, b, i);
            }
        }
    }

    void insert_entry(const std::string& A, const std::string& a, int prod) {
        auto key = std::make_pair(A, a);
        auto it = table_.find(key);
        if (it != table_.end() && it->second != prod) has_conflict_ = true;
        else table_[key] = prod;
    }
};

}  // namespace datamunge::algorithms
