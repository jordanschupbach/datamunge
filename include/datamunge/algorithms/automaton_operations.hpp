#pragma once
/// @file automaton_operations.hpp
/// @brief Validated reversal, completion, reachability and exact DFA language comparison.
#include <datamunge/algorithms/automaton.hpp>
#include <algorithm>
#include <limits>
#include <map>
#include <utility>
#include <vector>
namespace datamunge::algorithms {
/// @brief Reverse every explicit DFA edge and swap initial/accepting roles into an NFA.
/// Multiple accepting states become starts in ascending state order; the old
/// explicit start is the sole accepting state. Missing moves are omitted, and
/// start=-1 yields no accepting NFA state. Represents exactly the reversed word
/// language, without epsilon transitions. O(n*(k+1)) work/storage for n states
/// and k symbols. Validates before allocation, borrows only during the call and
/// returns owned containers, without input/shared-state mutation.
/// @param d Borrowed possibly partial DFA, including a valid zero-state machine.
/// @return Owned epsilon-free NFA accepting the reversed language.
/// @throws std::invalid_argument If the source machine is malformed; allocation failures propagate.
/// @par Example
/// @code
/// const dm::algorithms::DFA d{2,1,{{1},{1}},0,{0,1}};
/// const auto reversed=dm::algorithms::reverse_dfa(d);
/// std::print("{} {}\n",reversed.start_states,reversed.delta[1][0]); // [1] [0, 1]
/// @endcode
/// @par References
/// - J. A. Brzozowski, Canonical regular expressions and minimal state graphs for
///   definite events, Mathematical Theory of Automata (1963), 529-561, Theorem 13.
/// @see complete_dfa(), powerset_construction(), brzozowski_minimize()
inline NFA reverse_dfa(const DFA& d) {
    validate_dfa(d);
    NFA reversed;reversed.num_states=d.num_states;reversed.num_symbols=d.num_symbols;
    reversed.delta.assign(d.num_states,std::vector<std::vector<int>>(d.num_symbols));
    for (int state=0;state<d.num_states;++state) for (int symbol=0;symbol<d.num_symbols;++symbol) {
        const auto target=d.delta[state][symbol];
        if (target>=0) reversed.delta[target][symbol].push_back(state);
    }
    for (int state=0;state<d.num_states;++state) if (d.accept[state]) reversed.start_states.push_back(state);
    reversed.accept.assign(d.num_states,0);
    if (d.start>=0) reversed.accept[d.start]=1;
    return reversed;
}

/// @brief Materialise an implicit rejecting sink when any move or the start is -1.
/// Retains all original states, acceptance bytes and numbering, including
/// unreachable states; it does not minimise. Adds one fresh rejecting self-loop
/// state only when needed and redirects missing entries/start there. A valid
/// zero-state machine therefore becomes one rejecting state. O(n*(k+1)) work
/// and owned output storage, without input/shared-state mutation.
/// @param d Borrowed validated-or-validatable possibly partial DFA.
/// @return Owned complete DFA with unchanged language and alphabet.
/// @throws std::invalid_argument If d is malformed.
/// @throws std::overflow_error If adding a sink would exceed int state counts.
/// Allocation failures propagate.
/// @par Example
/// @code
/// const auto complete=dm::algorithms::complete_dfa(dm::algorithms::DFA{1,1,{{-1}},0,{1}});
/// std::print("{} {}\n",complete.num_states,complete.delta[0][0]); // 2 1
/// @endcode
/// @see reverse_dfa(), validate_dfa()
inline DFA complete_dfa(const DFA& d) {
    validate_dfa(d);
    bool needed=d.start<0;
    for (const auto& row:d.delta) for (int target:row) needed=needed || target<0;
    if (!needed) return d;
    if (d.num_states==std::numeric_limits<int>::max()) throw std::overflow_error("DFA sink exceeds int state count");
    DFA complete=d;
    const int dead=complete.num_states++;
    for (auto& row:complete.delta) for (int& target:row) if (target<0) target=dead;
    complete.delta.emplace_back(complete.num_symbols,dead);complete.accept.push_back(0);
    if (complete.start<0) complete.start=dead;
    return complete;
}

/// @brief List reachable explicit DFA states in BFS order, scanning symbols ascending.
/// Missing targets and the implicit sink are not returned. start=-1 yields an
/// empty list. O(n*(k+1)) work including validation and O(n) auxiliary/output
/// storage. Inputs are borrowed only for the call and never modified.
/// @param d Borrowed possibly partial DFA.
/// @return Owned reachable original state labels, with the start first when present.
/// @throws std::invalid_argument If d is malformed; allocation failures propagate.
/// @par Example
/// @code
/// const dm::algorithms::DFA d{3,1,{{1},{1},{2}},0,{0,1,1}};
/// std::print("{}\n",dm::algorithms::dfa_reachable_states(d)); // [0, 1]
/// @endcode
/// @see validate_dfa(), brzozowski_minimize_trace()
inline std::vector<int> dfa_reachable_states(const DFA& d) {
    validate_dfa(d);
    if (d.start<0) return {};
    std::vector<char> seen(d.num_states,0);
    std::vector<int> order{d.start};seen[d.start]=1;
    for (std::size_t head=0;head<order.size();++head) for (int target:d.delta[order[head]])
        if (target>=0 && !seen[target]) {seen[target]=1;order.push_back(target);}
    return order;
}

/// @brief Exact whole-language comparison result, optionally with a shortest counterexample.
/// equivalent=true means all reachable product pairs were exhausted, not a
/// bounded word test. If false, witness is a shortest distinguishing word,
/// lexicographically first among that length; an empty witness then denotes
/// epsilon. No witness is provided when equivalent=true. Counters include the
/// implicit rejecting sinks when reached. Containers own their storage.
/// @par Example
/// @code
/// const auto result=dm::algorithms::dfa_equivalence(
///     dm::algorithms::DFA{1,1,{{0}},0,{1}},dm::algorithms::DFA{1,1,{{0}},0,{0}});
/// std::print("{} {}\n",result.equivalent,result.witness.size()); // false 0
/// @endcode
/// @see dfa_equivalence()
struct DFAEquivalenceResult {
    bool equivalent{true}; ///< True iff reachable-product exhaustion proves equality of all words.
    std::vector<int> witness; ///< Shortest differing word when false; empty may be epsilon.
    std::size_t pairs_examined{0}; ///< Product pairs dequeued, including a distinguishing pair if found.
    std::size_t pairs_reached{0}; ///< Distinct product pairs discovered before returning.
};

/// @brief Private sparse product-state bookkeeping.
namespace automaton_operations_detail {
/// @brief One owned reachable pair and the predecessor that first discovered it.
struct ProductNode {
    int left{0}; ///< Left explicit state or its num_states sink sentinel.
    int right{0}; ///< Right explicit state or its num_states sink sentinel.
    std::size_t parent{0}; ///< Predecessor node index; root points to itself.
    int symbol{-1}; ///< Discovering symbol, or -1 at the root.
};
/// @brief Read acceptance, with the implicit sink rejecting.
/// @param d Borrowed validated DFA.
/// @param state Explicit state or num_states sentinel.
/// @return Nonzero explicit acceptance, or false at the sink.
inline bool accepts(const DFA& d,int state) {return state<d.num_states && d.accept[state]!=0;}
/// @brief Follow one symbol while retaining the virtual rejecting sink.
/// @param d Borrowed validated DFA.
/// @param state Explicit state or num_states sentinel.
/// @param symbol Valid symbol index.
/// @return Explicit target or num_states sentinel for a missing/sink move.
inline int next(const DFA& d,int state,int symbol) {
    if (state==d.num_states) return state;
    const int target=d.delta[state][symbol];return target<0?d.num_states:target;
}
} // namespace automaton_operations_detail

/// @brief Decide exact DFA language equivalence with sparse BFS over reachable state pairs.
/// Requires the same indexed alphabet size. Each missing move/start behaves as
/// an implicit rejecting sink. A mismatch yields a shortest differing word, with
/// ascending-symbol lexicographic ties; epsilon can distinguish the starts.
/// Otherwise exhausting the finite product proves equivalence for every word.
/// For P reached pairs and alphabet size k, O(input sizes+k*P*log(P+1)) work and
/// O(P) auxiliary storage, rather than preallocating the full Cartesian product.
/// Inputs are borrowed only during the call; no shared/input data are changed.
/// A positive pair budget throws before discovering an excess pair, never
/// returning a truncated search as equivalent. Zero (default) is unlimited.
/// @param left Borrowed possibly partial DFA.
/// @param right Borrowed DFA over the same indexed alphabet.
/// @param max_pairs Maximum discovered product pairs, or zero for no caller limit.
/// @return Exact equality flag, actual product counts and shortest counterexample when unequal.
/// @throws std::invalid_argument For malformed machines or unequal alphabet sizes.
/// @throws std::runtime_error If a positive product-pair budget is exhausted.
/// @throws std::overflow_error If a reached-pair index cannot fit size_t; allocation failures propagate.
/// @par Example
/// @code
/// const dm::algorithms::DFA ends_in_one{2,2,{{0,1},{0,1}},0,{0,1}};
/// const auto result=dm::algorithms::dfa_equivalence(ends_in_one,dm::algorithms::DFA{1,2,{{0,0}},0,{0}});
/// std::print("{} {}\n",result.equivalent,result.witness); // false [1]
/// @endcode
/// @par References
/// - M. O. Rabin and D. Scott (1959), Finite Automata and Their Decision Problems,
///   doi:10.1147/rd.32.0114, language-decision background; this implementation
///   explores the reachable product by BFS and retains a shortest witness.
/// @see DFAEquivalenceResult, dfa_accepts()
inline DFAEquivalenceResult dfa_equivalence(const DFA& left,const DFA& right,std::size_t max_pairs=0) {
    validate_dfa(left);validate_dfa(right);
    if (left.num_symbols!=right.num_symbols) throw std::invalid_argument("DFA comparison: unequal alphabet sizes");
    using automaton_operations_detail::ProductNode;
    std::vector<ProductNode> nodes{{left.start<0?left.num_states:left.start,right.start<0?right.num_states:right.start,0,-1}};
    std::map<std::pair<int,int>,std::size_t> index{{{nodes[0].left,nodes[0].right},0}};
    DFAEquivalenceResult result;
    for (std::size_t head=0;head<nodes.size();++head) {
        const auto current=nodes[head];++result.pairs_examined;
        if (automaton_operations_detail::accepts(left,current.left)!=automaton_operations_detail::accepts(right,current.right)) {
            result.equivalent=false;
            for (auto at=head;nodes[at].symbol>=0;at=nodes[at].parent) result.witness.push_back(nodes[at].symbol);
            std::reverse(result.witness.begin(),result.witness.end());result.pairs_reached=nodes.size();return result;
        }
        for (int symbol=0;symbol<left.num_symbols;++symbol) {
            const auto target=std::make_pair(automaton_operations_detail::next(left,current.left,symbol),
                                             automaton_operations_detail::next(right,current.right,symbol));
            if (index.find(target)!=index.end()) continue;
            if (max_pairs && nodes.size()>=max_pairs) throw std::runtime_error("DFA comparison pair budget exhausted");
            if (nodes.size()==std::numeric_limits<std::size_t>::max()) throw std::overflow_error("DFA product index exceeds size_t");
            const auto id=nodes.size();index.emplace(target,id);nodes.push_back({target.first,target.second,head,symbol});
        }
    }
    result.pairs_reached=nodes.size();return result;
}
} // namespace datamunge::algorithms
