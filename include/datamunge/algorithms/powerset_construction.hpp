#pragma once
/// @file powerset_construction.hpp
/// @brief Validated reachable-subset determinisation, optional completion and actual subset provenance.
#include <datamunge/algorithms/automaton.hpp>
#include <algorithm>
#include <limits>
#include <map>
#include <utility>
#include <vector>
namespace datamunge::algorithms {
/// @brief Owned determinised DFA and the actual NFA subset represented by each state.
/// Subsets are sorted/unique and indexed exactly like dfa states; numbering is
/// discovery order, scanning symbols ascending. Only reachable subsets are
/// created. complete records the requested policy, not whether a partial-policy
/// result happens to have every move. The empty initial subset is omitted in
/// partial mode, yielding zero states/start=-1, or retained in complete mode.
/// @par Example
/// @code
/// const dm::algorithms::NFA n{2,1,{{{0,1}},{{}}},{0},{0,1}};
/// const auto trace=dm::algorithms::powerset_construction_trace(n,true);
/// std::print("{}\n",trace.subsets); // [[0], [0, 1]]
/// @endcode
/// @see powerset_construction_trace()
struct PowersetTrace {
    DFA dfa; ///< Owned reachable-subset deterministic machine, with start zero when a subset exists.
    std::vector<std::vector<int>> subsets; ///< Actual sorted source-NFA state ids, one vector per output state.
    std::size_t transition_unions{0}; ///< Output-state/symbol unions processed, including missing/empty targets.
    std::size_t nfa_arcs_examined{0}; ///< Source destination entries scanned during those unions, including duplicates.
    std::size_t max_subset_size{0}; ///< Largest created subset cardinality, zero when none is created.
    bool complete{false}; ///< Requested explicit-empty-trap policy.
};

/// @brief Private checked subset interning and the single determinisation kernel.
namespace powerset_detail {
/// @brief Intern a sorted/unique actual subset, or retain a partial missing move.
/// @param subset Candidate destination set, owned for sorting/deduplication.
/// @param index Mutable map from canonical subsets to discovery ids.
/// @param trace Mutable owned DFA/provenance result.
/// @param max_states Optional cap on created output subsets, zero meaning unlimited.
/// @return Existing/new nonnegative subset id, or -1 for a partial empty move.
/// @throws std::runtime_error Before creating a subset beyond a positive cap.
/// @throws std::overflow_error If the output count would exceed int; allocation failures propagate.
inline int intern(std::vector<int> subset,std::map<std::vector<int>,int>& index,
                  PowersetTrace& trace,std::size_t max_states) {
    std::sort(subset.begin(),subset.end());subset.erase(std::unique(subset.begin(),subset.end()),subset.end());
    if (subset.empty() && !trace.complete) return -1;
    const auto found=index.find(subset);if (found!=index.end()) return found->second;
    if (max_states && trace.subsets.size()>=max_states) throw std::runtime_error("powerset state budget exhausted");
    if (trace.subsets.size()>=static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::overflow_error("powerset state count exceeds int");
    const auto id=static_cast<int>(trace.subsets.size());
    trace.max_subset_size=std::max(trace.max_subset_size,subset.size());
    index.emplace(subset,id);trace.subsets.push_back(std::move(subset));return id;
}
/// @brief Count one actual union or scanned NFA arc without wrapping size_t.
/// @param value Counter to increment.
/// @throws std::overflow_error If another increment cannot fit.
inline void increment(std::size_t& value) {
    if (value==std::numeric_limits<std::size_t>::max()) throw std::overflow_error("powerset work count exceeds size_t");
    ++value;
}
/// @brief Build reachable canonical subsets using the ordinary construction's discovery order.
/// @param nfa Borrowed epsilon-free machine, validated before output allocation.
/// @param complete Whether to retain the empty subset as an explicit trap.
/// @param max_states Optional output-state cap, zero meaning unlimited.
/// @return Owned output and actual provenance/work counters.
/// @throws std::invalid_argument If nfa is malformed.
/// @throws std::runtime_error If a positive state cap is exceeded.
/// @throws std::overflow_error Before state/counter overflow; allocation failures propagate.
inline PowersetTrace construct(const NFA& nfa,bool complete,std::size_t max_states) {
    validate_nfa(nfa);
    PowersetTrace trace;trace.complete=complete;trace.dfa.num_symbols=nfa.num_symbols;
    std::map<std::vector<int>,int> index;
    trace.dfa.start=intern(nfa.start_states,index,trace,max_states);
    for (std::size_t current=0;current<trace.subsets.size();++current) {
        trace.dfa.delta.emplace_back(nfa.num_symbols,-1);
        for (int symbol=0;symbol<nfa.num_symbols;++symbol) {
            increment(trace.transition_unions);
            std::vector<int> next;
            for (int state:trace.subsets[current]) for (int target:nfa.delta[state][symbol]) {
                increment(trace.nfa_arcs_examined);next.push_back(target);
            }
            trace.dfa.delta[current][symbol]=intern(std::move(next),index,trace,max_states);
        }
    }
    trace.dfa.num_states=static_cast<int>(trace.subsets.size());
    trace.dfa.accept.assign(trace.subsets.size(),0);
    for (std::size_t i=0;i<trace.subsets.size();++i) for (int state:trace.subsets[i])
        if (nfa.accept[state]) {trace.dfa.accept[i]=1;break;}
    return trace;
}
} // namespace powerset_detail

/// @brief Determinise an epsilon-free NFA by constructing only reachable source subsets.
/// Retains the original signature, discovery numbering and default partial policy.
/// Each move unions source destinations; a subset accepts iff any source state
/// accepts. Duplicates are collapsed. With complete=true the empty subset is a
/// rejecting trap; otherwise empty moves are -1. No initial states in partial
/// mode yields a valid zero-state DFA with start=-1. Validates every source row,
/// start and target before allocation. The vector/map implementation can require
/// exponentially many subsets and stores their explicit members while building;
/// costs depend on generated subsets, union sizes, sorting and map comparisons,
/// not just the number of source states. Results own storage; inputs/shared data
/// are unchanged. There are no epsilon-closure operations.
/// @param nfa Borrowed valid-or-validatable epsilon-free NFA.
/// @param complete Retain an explicit empty trap (default false: use -1 missing moves).
/// @return Owned reachable-subset DFA, start zero when its initial subset is retained.
/// @throws std::invalid_argument If nfa is malformed.
/// @throws std::overflow_error Before state/work-counter overflow; allocation failures propagate.
/// @par Example
/// @code
/// const dm::algorithms::NFA n{2,1,{{{1}},{{}}},{0},{0,1}};
/// const auto d=dm::algorithms::powerset_construction(n,true);
/// std::print("{} {}\n",d.num_states,dm::algorithms::dfa_accepts(d,{0})); // 3 true
/// @endcode
/// @par References
/// - M. O. Rabin and D. Scott (1959), Finite Automata and Their Decision Problems,
///   IBM Journal of Research and Development 3(2), 114-125, doi:10.1147/rd.32.0114.
/// @see powerset_construction_trace(), validate_nfa()
inline DFA powerset_construction(const NFA& nfa,bool complete=false) {
    return powerset_detail::construct(nfa,complete,0).dfa;
}

/// @brief Retain the ordinary construction's actual source subsets and logical work.
/// Shares the ordinary kernel; adds no second determinisation or provenance replay.
/// Retaining subsets extends their lifetime into the owned result. A positive
/// max_states throws before adding an excess subset, with no partial returned DFA;
/// zero is unlimited. This cap is not a byte limit and cannot prevent large union
/// temporaries or validation of the source. Numbering and completion match the
/// ordinary overload, including zero-state partial empty-language output.
/// @param nfa Borrowed epsilon-free NFA.
/// @param complete Requested explicit-empty-trap policy, default false.
/// @param max_states Maximum created DFA subsets, or zero for no caller limit.
/// @return Owned DFA, actual subset provenance and completed-work counts.
/// @throws std::invalid_argument If nfa is malformed.
/// @throws std::runtime_error If a positive output-state cap is exhausted.
/// @throws std::overflow_error Before state/counter overflow; allocation failures propagate.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::powerset_construction_trace(dm::algorithms::NFA{0,1,{}, {},{}},true,1);
/// std::print("{} {}\n",trace.dfa.num_states,trace.subsets[0].empty()); // 1 true
/// @endcode
/// @par References
/// - M. O. Rabin and D. Scott (1959), doi:10.1147/rd.32.0114; actual reachable-subset records.
/// @see PowersetTrace, powerset_construction()
inline PowersetTrace powerset_construction_trace(const NFA& nfa,bool complete=false,std::size_t max_states=0) {
    return powerset_detail::construct(nfa,complete,max_states);
}
} // namespace datamunge::algorithms
