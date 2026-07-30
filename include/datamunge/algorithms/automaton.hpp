#pragma once

/// \file automaton.hpp
/// \brief Finite-automaton types (NFA/DFA) shared by the automata algorithms.
///
/// A *deterministic finite automaton* (DFA) reads a string over a finite alphabet, following
/// one transition per symbol from a start state, and accepts if it ends in an accepting
/// state. A *nondeterministic* automaton (NFA) may have several (or no) transitions per
/// symbol. This header gives compact representations used by the powerset construction
/// (NFA to DFA) and the DFA-minimization algorithms (Hopcroft, Moore, Brzozowski). Symbols
/// are indexed \f$0..k-1\f$; a DFA transition of \c -1 denotes a (implicit) dead state.

#include <cstddef>
#include <string>
#include <vector>

namespace datamunge::algorithms {

/// A nondeterministic finite automaton (no epsilon transitions).
struct NFA {
    int                                       num_states;    ///< States 0..num_states-1.
    int                                       num_symbols;   ///< Alphabet symbols 0..num_symbols-1.
    std::vector<std::vector<std::vector<int>>> delta;        ///< delta[state][symbol] = next states.
    std::vector<int>                          start_states;  ///< Initial states.
    std::vector<char>                         accept;        ///< accept[state] != 0 if accepting.
};

/// A deterministic finite automaton.
struct DFA {
    int                           num_states;   ///< States 0..num_states-1.
    int                           num_symbols;  ///< Alphabet symbols 0..num_symbols-1.
    std::vector<std::vector<int>> delta;        ///< delta[state][symbol] = next state (-1 = dead).
    int                           start;        ///< Initial state.
    std::vector<char>             accept;       ///< accept[state] != 0 if accepting.
};

/// \brief Run a DFA on a symbol string; returns true if it accepts.
inline bool dfa_accepts(const DFA& d, const std::vector<int>& input) {
    int s = d.start;
    for (int sym : input) {
        if (s < 0) return false;         // already in the dead state
        s = d.delta[s][sym];
    }
    return s >= 0 && d.accept[s] != 0;
}

}  // namespace datamunge::algorithms
