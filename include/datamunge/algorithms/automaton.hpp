#pragma once
/// @file automaton.hpp
/// @brief Validated finite-automaton aggregates with integer symbols and an implicit rejecting sink.
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>
namespace datamunge::algorithms {
/// @brief Owned epsilon-free nondeterministic finite automaton, possibly with multiple initial states.
/// States/symbols are contiguous integer labels. delta has num_states rows of
/// num_symbols destination lists; accept has one byte per state, nonzero meaning
/// accepting. Duplicate destinations/starts are allowed. No epsilon transitions
/// are represented. Zero states with empty starts accepts no words. Aggregate
/// field order/types are unchanged; default construction now denotes that empty
/// zero-alphabet machine. Containers own their storage; concurrent read-only use
/// is permitted if nobody mutates the aggregate.
/// @par Example
/// @code
/// const dm::algorithms::NFA n{1,1,{{{0}}},{0},{1}};
/// std::print("{}\n",dm::algorithms::dfa_accepts(dm::algorithms::powerset_construction(n,true),{0,0})); // true
/// @endcode
/// @see validate_nfa(), powerset_construction()
struct NFA {
    int num_states{0}; ///< Nonnegative number of states, labelled 0..num_states-1.
    int num_symbols{0}; ///< Nonnegative alphabet size, symbols 0..num_symbols-1.
    std::vector<std::vector<std::vector<int>>> delta; ///< Owned delta[state][symbol] destination lists, without epsilon moves.
    std::vector<int> start_states; ///< Owned initial state labels, possibly empty or duplicated.
    std::vector<char> accept; ///< Owned accepting flags; any nonzero byte means true.
};

/// @brief Owned deterministic finite automaton with optional missing transitions/start.
/// delta has num_states rows of num_symbols entries, and accept has num_states
/// bytes. Each target is -1 or a valid state. -1 denotes an implicit rejecting
/// sink, which loops on all symbols; start=-1 denotes the empty language. A
/// zero-state DFA requires start=-1. Zero symbols permits only the empty word.
/// Aggregate field order/types remain unchanged; default construction is now a
/// valid empty zero-alphabet machine. Containers own their storage. Read-only
/// calls share no mutable state, provided the aggregate is not concurrently changed.
/// @par Example
/// @code
/// const dm::algorithms::DFA d{2,2,{{0,1},{0,1}},0,{0,1}};
/// std::print("{} {}\n",dm::algorithms::dfa_accepts(d,{1}),dm::algorithms::dfa_accepts(d,{1,0})); // true false
/// @endcode
/// @see validate_dfa(), dfa_accepts()
struct DFA {
    int num_states{0}; ///< Nonnegative number of explicitly represented states.
    int num_symbols{0}; ///< Nonnegative alphabet size, indexed from zero.
    std::vector<std::vector<int>> delta; ///< Owned targets; -1 is the implicit rejecting sink.
    int start{-1}; ///< Explicit initial state, or -1 for no initial state/empty language.
    std::vector<char> accept; ///< Owned flags; nonzero is accepting, with the implicit sink always rejecting.
};

/// @brief Check the complete shape and every index of an epsilon-free NFA.
/// Accepts duplicates, nonzero acceptance bytes, empty starts and zero symbols.
/// Requires nonnegative counts, exact row/flag dimensions, and valid explicit
/// start/destination indices. O(states+symbols*states+listed destinations+starts)
/// work, O(1) auxiliary storage, without allocation, repair or mutation.
/// @param nfa Borrowed aggregate to validate; no reference is retained.
/// @throws std::invalid_argument For a negative count, dimension mismatch or invalid label.
/// @par Example
/// @code
/// const dm::algorithms::NFA n{1,1,{{{0,0}}},{0,0},{1}};
/// dm::algorithms::validate_nfa(n);
/// std::print("{}\n",n.start_states.size()); // 2
/// @endcode
/// @see NFA, validate_dfa()
inline void validate_nfa(const NFA& nfa) {
    if (nfa.num_states<0 || nfa.num_symbols<0 || nfa.delta.size()!=static_cast<std::size_t>(nfa.num_states)
        || nfa.accept.size()!=static_cast<std::size_t>(nfa.num_states)) throw std::invalid_argument("NFA: invalid counts or dimensions");
    for (const auto& row:nfa.delta) {
        if (row.size()!=static_cast<std::size_t>(nfa.num_symbols)) throw std::invalid_argument("NFA: invalid row width");
        for (const auto& targets:row) for (int target:targets)
            if (target<0 || target>=nfa.num_states) throw std::invalid_argument("NFA: invalid target");
    }
    for (int start:nfa.start_states)
        if (start<0 || start>=nfa.num_states) throw std::invalid_argument("NFA: invalid start");
}

/// @brief Check the complete shape and every index of a possibly partial DFA.
/// Requires nonnegative counts, exact row/flag dimensions and targets/start in
/// {-1,0,...,num_states-1}. For zero states the only valid start is -1. Nonzero
/// acceptance bytes are accepted. O(states*(symbols+1)) work and O(1) storage,
/// with no allocation, repair or input/shared-state mutation.
/// @param dfa Borrowed machine to validate; no reference is retained.
/// @throws std::invalid_argument For invalid counts, dimensions, start or targets.
/// @par Example
/// @code
/// const dm::algorithms::DFA d{1,1,{{-1}},0,{1}};
/// dm::algorithms::validate_dfa(d);
/// std::print("{}\n",dm::algorithms::dfa_accepts(d,{})); // true
/// @endcode
/// @see DFA, validate_nfa(), dfa_accepts()
inline void validate_dfa(const DFA& dfa) {
    if (dfa.num_states<0 || dfa.num_symbols<0 || dfa.delta.size()!=static_cast<std::size_t>(dfa.num_states)
        || dfa.accept.size()!=static_cast<std::size_t>(dfa.num_states)) throw std::invalid_argument("DFA: invalid counts or dimensions");
    if (dfa.start < -1 || dfa.start>=dfa.num_states) throw std::invalid_argument("DFA: invalid start");
    for (const auto& row:dfa.delta) {
        if (row.size()!=static_cast<std::size_t>(dfa.num_symbols)) throw std::invalid_argument("DFA: invalid row width");
        for (int target:row) if (target < -1 || target>=dfa.num_states) throw std::invalid_argument("DFA: invalid target");
    }
}

/// @brief Run a validated DFA on integer symbols, treating -1 as a rejecting sink.
/// Validates the entire machine and entire word before simulation, including
/// symbols after an eventual missing move. Empty input tests the explicit start's
/// acceptance, or rejects for start=-1. O(states*(symbols+1)+word length) work,
/// O(1) auxiliary storage, with no allocation, borrowed result or mutation.
/// @param d Borrowed possibly partial DFA.
/// @param input Borrowed sequence of symbols in [0,d.num_symbols).
/// @return Whether the word ends in an accepting explicit state.
/// @throws std::invalid_argument If the machine or any input symbol is invalid.
/// @par Example
/// @code
/// const dm::algorithms::DFA d{1,1,{{-1}},0,{1}};
/// std::print("{} {}\n",dm::algorithms::dfa_accepts(d,{}),dm::algorithms::dfa_accepts(d,{0})); // true false
/// @endcode
/// @par References
/// - M. O. Rabin and D. Scott (1959), Finite Automata and Their Decision Problems,
///   doi:10.1147/rd.32.0114, finite-word acceptance model.
/// @see validate_dfa(), powerset_construction()
inline bool dfa_accepts(const DFA& d,const std::vector<int>& input) {
    validate_dfa(d);
    for (int symbol:input) if (symbol<0 || symbol>=d.num_symbols) throw std::invalid_argument("DFA: invalid input symbol");
    int state=d.start;
    for (int symbol:input) {
        if (state<0) return false;
        state=d.delta[state][symbol];
    }
    return state>=0 && d.accept[state]!=0;
}
} // namespace datamunge::algorithms
