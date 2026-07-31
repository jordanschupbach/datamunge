#pragma once

/// \file slr_parser.hpp
/// \brief A Simple LR (SLR(1)) bottom-up shift-reduce parser.
///
/// LR parsers work *bottom-up*: instead of predicting productions from the top like LL, they
/// shift input onto a stack and *reduce* a matched right-hand side to its nonterminal once the
/// whole thing is on top. The engine is a DFA over *LR(0) items* -- productions with a dot,
/// \f$A\to\alpha\bullet\beta\f$ -- whose states are built by *closure* (a dot before a
/// nonterminal \f$B\f$ pulls in every \f$B\to\bullet\gamma\f$) and *goto* (advancing the dot
/// over a symbol). From this automaton we build an ACTION/GOTO table:
/// - a dot before a terminal \f$a\f$ with \f$\mathrm{goto}=j\f$ gives *shift j*;
/// - a completed item \f$A\to\alpha\bullet\f$ gives *reduce* \f$A\to\alpha\f$.
/// *SLR* resolves when to reduce with the simplest rule: reduce on lookahead \f$a\f$ only if
/// \f$a\in\mathrm{FOLLOW}(A)\f$. A grammar is *SLR(1)* exactly when the resulting table has no
/// shift/reduce or reduce/reduce conflicts. LR parsers accept a strictly larger class of
/// grammars than LL(1) -- including left-recursive ones, which top-down parsers cannot handle.
///
/// This implementation augments the grammar, builds the LR(0) automaton and SLR table, reports
/// whether the grammar is SLR(1), and recognises input strings.

#include <map>
#include <set>
#include <string>
#include <vector>

#include <datamunge/algorithms/context_free_grammar.hpp>

namespace datamunge::algorithms {

/// \brief A Simple LR (SLR(1)) parser: LR(0) automaton, SLR table, and recognition.
class SLRParser {
   public:
    static constexpr const char* kEpsilon = "";
    static constexpr const char* kEndMarker = "$";
    static constexpr const char* kAugStart = "__START__";

    explicit SLRParser(const ContextFreeGrammar& grammar) {
        // Augmented grammar: production 0 is __START__ -> start.
        aug_start_prod_ = {kAugStart, {grammar.start}};
        prods_.push_back(aug_start_prod_);
        for (const auto& p : grammar.productions) prods_.push_back(p);
        start_ = grammar.start;

        collect_symbols();
        compute_first();
        compute_follow();
        build_automaton();
        build_table();
    }

    /// \brief True if the grammar is SLR(1) (no table conflicts).
    bool is_slr() const { return !has_conflict_; }
    /// \brief Number of LR(0) states in the automaton.
    std::size_t state_count() const { return states_.size(); }

    /// \brief Recognise whether \p tokens (terminals) is in the language.
    bool recognize(const std::vector<std::string>& tokens) const {
        std::vector<std::string> input = tokens;
        input.push_back(kEndMarker);
        std::vector<int> stack = {0};  // stack of state indices
        std::size_t ip = 0;
        for (;;) {
            int state = stack.back();
            const std::string& a = input[ip];
            auto it = action_.find({state, a});
            if (it == action_.end()) return false;  // error
            const Action& act = it->second;
            if (act.type == Action::Accept) return true;
            if (act.type == Action::Shift) {
                stack.push_back(act.value);
                ++ip;
            } else {  // Reduce by production act.value
                const Production& p = prods_[act.value];
                for (std::size_t i = 0; i < p.rhs.size(); ++i) stack.pop_back();
                int top = stack.back();
                auto g = goto_.find({top, p.lhs});
                if (g == goto_.end()) return false;
                stack.push_back(g->second);
            }
        }
    }

    /// \brief The parse stack depth after each shift/reduce step, for an accepted input.
    /// Illustrates how bottom-up parsing grows the stack on shifts and shrinks it on reduces.
    /// Returns an empty vector if the input is rejected.
    std::vector<int> stack_depth_trace(const std::vector<std::string>& tokens) const {
        std::vector<std::string> input = tokens;
        input.push_back(kEndMarker);
        std::vector<int> stack = {0};
        std::vector<int> trace = {1};
        std::size_t ip = 0;
        for (;;) {
            int state = stack.back();
            const std::string& a = input[ip];
            auto it = action_.find({state, a});
            if (it == action_.end()) return {};
            const Action& act = it->second;
            if (act.type == Action::Accept) return trace;
            if (act.type == Action::Shift) {
                stack.push_back(act.value);
                ++ip;
            } else {
                const Production& p = prods_[act.value];
                for (std::size_t i = 0; i < p.rhs.size(); ++i) stack.pop_back();
                int top = stack.back();
                auto g = goto_.find({top, p.lhs});
                if (g == goto_.end()) return {};
                stack.push_back(g->second);
            }
            trace.push_back(static_cast<int>(stack.size()));
        }
    }

   private:
    struct Item {
        int prod;
        int dot;
        bool operator<(const Item& o) const {
            return prod < o.prod || (prod == o.prod && dot < o.dot);
        }
        bool operator==(const Item& o) const { return prod == o.prod && dot == o.dot; }
    };
    struct Action {
        enum Type { Shift, Reduce, Accept } type;
        int value;  // shift: state; reduce: production index
    };

    std::vector<Production> prods_;
    Production aug_start_prod_;
    std::string start_;
    std::set<std::string> terminals_, nonterminals_;
    std::map<std::string, std::set<std::string>> first_, follow_;
    std::vector<std::set<Item>> states_;
    std::map<std::pair<int, std::string>, int> transitions_;  // (state, symbol) -> state
    std::map<std::pair<int, std::string>, Action> action_;    // (state, terminal) -> action
    std::map<std::pair<int, std::string>, int> goto_;         // (state, nonterminal) -> state
    bool has_conflict_ = false;

    bool is_nt(const std::string& s) const { return nonterminals_.count(s) > 0; }

    void collect_symbols() {
        for (const auto& p : prods_) nonterminals_.insert(p.lhs);
        for (const auto& p : prods_)
            for (const auto& s : p.rhs)
                if (!nonterminals_.count(s)) terminals_.insert(s);
        terminals_.insert(kEndMarker);
    }

    void compute_first() {
        for (const auto& t : terminals_) first_[t] = {t};
        for (const auto& nt : nonterminals_) first_[nt];
        bool changed = true;
        while (changed) {
            changed = false;
            for (const auto& p : prods_) {
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
        for (const auto& nt : nonterminals_) follow_[nt];
        follow_[kAugStart].insert(kEndMarker);
        bool changed = true;
        while (changed) {
            changed = false;
            for (const auto& p : prods_) {
                for (std::size_t i = 0; i < p.rhs.size(); ++i) {
                    const std::string& B = p.rhs[i];
                    if (!is_nt(B)) continue;
                    bool all_nullable = true;
                    for (std::size_t j = i + 1; j < p.rhs.size(); ++j) {
                        for (const auto& t : first_[p.rhs[j]])
                            if (t != kEpsilon)
                                if (follow_[B].insert(t).second) changed = true;
                        if (!first_[p.rhs[j]].count(kEpsilon)) { all_nullable = false; break; }
                    }
                    if (all_nullable) {
                        for (const auto& t : follow_[p.lhs])
                            if (follow_[B].insert(t).second) changed = true;
                    }
                }
            }
        }
    }

    std::set<Item> closure(std::set<Item> items) const {
        bool changed = true;
        while (changed) {
            changed = false;
            for (const auto& it : std::set<Item>(items)) {
                const Production& p = prods_[it.prod];
                if (it.dot < static_cast<int>(p.rhs.size())) {
                    const std::string& B = p.rhs[it.dot];
                    if (nonterminals_.count(B)) {
                        for (int i = 0; i < static_cast<int>(prods_.size()); ++i)
                            if (prods_[i].lhs == B)
                                if (items.insert({i, 0}).second) changed = true;
                    }
                }
            }
        }
        return items;
    }

    std::set<Item> goto_set(const std::set<Item>& I, const std::string& X) const {
        std::set<Item> moved;
        for (const auto& it : I) {
            const Production& p = prods_[it.prod];
            if (it.dot < static_cast<int>(p.rhs.size()) && p.rhs[it.dot] == X)
                moved.insert({it.prod, it.dot + 1});
        }
        return closure(moved);
    }

    int find_state(const std::set<Item>& s) const {
        for (int i = 0; i < static_cast<int>(states_.size()); ++i)
            if (states_[i] == s) return i;
        return -1;
    }

    void build_automaton() {
        states_.push_back(closure({{0, 0}}));  // start: __START__ -> . start
        for (std::size_t i = 0; i < states_.size(); ++i) {
            // Gather symbols appearing right of a dot.
            std::set<std::string> syms;
            for (const auto& it : states_[i]) {
                const Production& p = prods_[it.prod];
                if (it.dot < static_cast<int>(p.rhs.size())) syms.insert(p.rhs[it.dot]);
            }
            for (const auto& X : syms) {
                auto j_set = goto_set(states_[i], X);
                if (j_set.empty()) continue;
                int j = find_state(j_set);
                if (j == -1) { states_.push_back(j_set); j = static_cast<int>(states_.size()) - 1; }
                transitions_[{static_cast<int>(i), X}] = j;
            }
        }
    }

    void build_table() {
        for (int i = 0; i < static_cast<int>(states_.size()); ++i) {
            for (const auto& it : states_[i]) {
                const Production& p = prods_[it.prod];
                if (it.dot < static_cast<int>(p.rhs.size())) {
                    const std::string& X = p.rhs[it.dot];
                    auto tr = transitions_.find({i, X});
                    if (tr == transitions_.end()) continue;
                    if (terminals_.count(X)) {
                        set_action(i, X, {Action::Shift, tr->second});
                    } else {
                        goto_[{i, X}] = tr->second;
                    }
                } else {
                    // completed item
                    if (p.lhs == kAugStart) {
                        set_action(i, kEndMarker, {Action::Accept, 0});
                    } else {
                        for (const auto& a : follow_[p.lhs])
                            set_action(i, a, {Action::Reduce, it.prod});
                    }
                }
            }
        }
    }

    void set_action(int state, const std::string& a, Action act) {
        auto key = std::make_pair(state, a);
        auto it = action_.find(key);
        if (it != action_.end()) {
            // Conflict unless it is the identical action.
            if (it->second.type != act.type || it->second.value != act.value)
                has_conflict_ = true;
        } else {
            action_[key] = act;
        }
    }
};

}  // namespace datamunge::algorithms
