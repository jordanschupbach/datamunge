#pragma once

/// \file earley_parser.hpp
/// \brief The Earley parser: recognises any context-free grammar in \f$O(n^3)\f$.
///
/// Earley's 1970 algorithm parses *arbitrary* context-free grammars -- ambiguous,
/// left-recursive, anything -- without preprocessing, running in \f$O(n^3)\f$ worst case,
/// \f$O(n^2)\f$ for unambiguous grammars, and linear for many practical ones. It builds a
/// *chart*: one state set \f$S_k\f$ per input position \f$k\f$, each holding *Earley items*
/// -- a production with a dot marking progress and an *origin* position where the match began:
/// \f[
///   A \to \alpha \bullet \beta,\ (j)
/// \f]
/// meaning "we are parsing an \f$A\f$ that started at position \f$j\f$, have matched
/// \f$\alpha\f$, and expect \f$\beta\f$." Three operations grow the chart:
/// - *predict*: at a dot before nonterminal \f$B\f$, add \f$B\to\bullet\gamma,(k)\f$;
/// - *scan*: at a dot before terminal \f$a\f$ matching the input, advance the dot into \f$S_{k+1}\f$;
/// - *complete*: when an item \f$B\to\gamma\bullet,(j)\f$ finishes, advance every item in
///   \f$S_j\f$ whose dot sat before \f$B\f$.
/// The input is accepted iff \f$S_n\f$ contains a completed start production spanning
/// \f$[0,n]\f$. This implementation recognises whether a token string is in the language.

#include <string>
#include <vector>

#include <datamunge/algorithms/context_free_grammar.hpp>

namespace datamunge::algorithms {

namespace detail {

struct EarleyItem {
    int prod;    // index into grammar.productions
    int dot;     // position of the dot within the rhs
    int origin;  // state set where this item began
    bool operator==(const EarleyItem& o) const {
        return prod == o.prod && dot == o.dot && origin == o.origin;
    }
};

}  // namespace detail

namespace detail {

/// Build the full Earley chart for a grammar and token string.
inline std::vector<std::vector<EarleyItem>> build_earley_chart(
    const ContextFreeGrammar& grammar, const std::vector<std::string>& tokens) {
    const auto& P = grammar.productions;
    const int n = static_cast<int>(tokens.size());
    std::vector<std::vector<EarleyItem>> chart(n + 1);

    auto add = [](std::vector<EarleyItem>& set, const EarleyItem& it) {
        for (const auto& e : set)
            if (e == it) return;
        set.push_back(it);
    };

    for (int i = 0; i < static_cast<int>(P.size()); ++i)
        if (P[i].lhs == grammar.start) add(chart[0], {i, 0, 0});

    for (int k = 0; k <= n; ++k) {
        for (std::size_t idx = 0; idx < chart[k].size(); ++idx) {
            EarleyItem item = chart[k][idx];
            const Production& prod = P[item.prod];
            if (item.dot < static_cast<int>(prod.rhs.size())) {
                const std::string& sym = prod.rhs[item.dot];
                if (grammar.is_nonterminal(sym)) {
                    for (int i = 0; i < static_cast<int>(P.size()); ++i)  // PREDICT
                        if (P[i].lhs == sym) add(chart[k], {i, 0, k});
                } else if (k < n && tokens[k] == sym) {  // SCAN
                    add(chart[k + 1], {item.prod, item.dot + 1, item.origin});
                }
            } else {  // COMPLETE
                for (const auto& e : chart[item.origin]) {
                    const Production& ep = P[e.prod];
                    if (e.dot < static_cast<int>(ep.rhs.size()) && ep.rhs[e.dot] == prod.lhs)
                        add(chart[k], {e.prod, e.dot + 1, e.origin});
                }
            }
        }
    }
    return chart;
}

}  // namespace detail

/// \brief Recognise whether \p tokens is derivable from \p grammar's start symbol.
/// \param grammar a context-free grammar (may be ambiguous or left-recursive).
/// \param tokens the input as a sequence of terminal symbols.
/// \returns true iff the token string is in the grammar's language.
inline bool earley_recognize(const ContextFreeGrammar& grammar,
                             const std::vector<std::string>& tokens) {
    auto chart = detail::build_earley_chart(grammar, tokens);
    const int n = static_cast<int>(tokens.size());
    for (const auto& e : chart[n]) {
        const Production& ep = grammar.productions[e.prod];
        if (ep.lhs == grammar.start && e.origin == 0 &&
            e.dot == static_cast<int>(ep.rhs.size()))
            return true;
    }
    return false;
}

/// \brief Number of Earley items in each chart position \f$S_0,\dots,S_n\f$.
/// Illustrates the algorithm's \f$O(n^2)\f$ chart growth for unambiguous grammars.
inline std::vector<int> earley_chart_sizes(const ContextFreeGrammar& grammar,
                                           const std::vector<std::string>& tokens) {
    auto chart = detail::build_earley_chart(grammar, tokens);
    std::vector<int> sizes;
    sizes.reserve(chart.size());
    for (const auto& s : chart) sizes.push_back(static_cast<int>(s.size()));
    return sizes;
}

}  // namespace datamunge::algorithms
