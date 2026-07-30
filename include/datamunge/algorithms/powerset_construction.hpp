#pragma once

/// \file powerset_construction.hpp
/// \brief The powerset (subset) construction: convert an NFA to an equivalent DFA.
///
/// A nondeterministic automaton can be in a *set* of states at once. The powerset construction
/// makes that set explicit: each DFA state is a subset of NFA states, the start state is the
/// set of NFA start states, and the transition on a symbol is the union of the NFA moves from
/// every state in the current subset. A DFA subset accepts iff it contains an NFA accepting
/// state. The construction can blow up exponentially (\f$2^n\f$ subsets worst case) but for
/// most automata only a few subsets are reachable. This module builds the reachable-subset
/// DFA, optionally materializing the empty subset as an explicit dead/trap state.

#include <datamunge/algorithms/automaton.hpp>

#include <algorithm>
#include <cstddef>
#include <map>
#include <vector>

namespace datamunge::algorithms {

/// \brief Convert an NFA to a DFA by the subset construction.
///
/// \param nfa       the source automaton (no epsilon transitions).
/// \param complete  if true, the empty subset becomes a real (trap) state and every
///                  transition is defined; if false, missing transitions are \c -1.
/// \return the reachable-subset DFA. DFA state 0 is the start subset.
inline DFA powerset_construction(const NFA& nfa, bool complete = false) {
    std::map<std::vector<int>, int> index;
    std::vector<std::vector<int>>   subsets;

    auto get_index = [&](std::vector<int> s) -> int {
        std::sort(s.begin(), s.end());
        s.erase(std::unique(s.begin(), s.end()), s.end());
        if (s.empty() && !complete) return -1;              // dead state, not materialized
        auto it = index.find(s);
        if (it != index.end()) return it->second;
        int id = static_cast<int>(subsets.size());
        index[s] = id;
        subsets.push_back(std::move(s));
        return id;
    };

    DFA d;
    d.num_symbols = nfa.num_symbols;
    d.start       = get_index(nfa.start_states);

    std::vector<std::vector<int>> delta;
    for (int cur = 0; cur < static_cast<int>(subsets.size()); ++cur) {
        delta.emplace_back(nfa.num_symbols, -1);
        for (int sym = 0; sym < nfa.num_symbols; ++sym) {
            std::vector<int> nxt;
            for (int q : subsets[cur])
                for (int r : nfa.delta[q][sym]) nxt.push_back(r);
            delta[cur][sym] = get_index(std::move(nxt));
        }
    }

    d.num_states = static_cast<int>(subsets.size());
    d.delta      = std::move(delta);
    d.accept.assign(d.num_states, 0);
    for (int i = 0; i < d.num_states; ++i)
        for (int q : subsets[i])
            if (nfa.accept[q]) { d.accept[i] = 1; break; }
    return d;
}

}  // namespace datamunge::algorithms
