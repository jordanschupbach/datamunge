#pragma once

/// \file context_free_grammar.hpp
/// \brief A minimal context-free grammar representation shared by the CFG parsers.
///
/// A production is a left-hand nonterminal and a right-hand sequence of symbols (an empty
/// sequence is the epsilon / empty production). Symbols are plain strings; a symbol is a
/// *nonterminal* exactly when it appears as some production's left-hand side, and a
/// *terminal* otherwise. Used by the Earley, LL(1), and SLR parsers.

#include <set>
#include <string>
#include <vector>

namespace datamunge::algorithms {

/// \brief One grammar rule: \c lhs -> \c rhs (empty \c rhs means the epsilon production).
struct Production {
    std::string lhs;
    std::vector<std::string> rhs;
};

/// \brief A context-free grammar: a list of productions plus a start symbol.
struct ContextFreeGrammar {
    std::vector<Production> productions;
    std::string start;

    /// \brief The set of nonterminal symbols (those appearing on some left-hand side).
    std::set<std::string> nonterminals() const {
        std::set<std::string> nt;
        for (const auto& p : productions) nt.insert(p.lhs);
        return nt;
    }

    /// \brief True if \p sym is a nonterminal (appears as a production's lhs).
    bool is_nonterminal(const std::string& sym) const {
        for (const auto& p : productions)
            if (p.lhs == sym) return true;
        return false;
    }

    /// \brief The set of terminal symbols (rhs symbols that are not nonterminals).
    std::set<std::string> terminals() const {
        auto nt = nonterminals();
        std::set<std::string> t;
        for (const auto& p : productions)
            for (const auto& sym : p.rhs)
                if (!nt.count(sym)) t.insert(sym);
        return t;
    }
};

}  // namespace datamunge::algorithms
