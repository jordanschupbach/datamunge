#pragma once
/// @file brzozowski.hpp
/// @brief Complete DFA minimisation by two actual reversal/determinisation stages.
#include <datamunge/algorithms/automaton_operations.hpp>
#include <datamunge/algorithms/powerset_construction.hpp>
#include <limits>
#include <utility>
#include <vector>
namespace datamunge::algorithms {
/// @brief Separate optional output-state caps for the two determinisation stages.
/// Zero is unlimited. Each positive cap is enforced before creating an excess
/// subset; exceeding either throws without returning a partial minimisation.
/// Caps count states, not bytes; source validation and large union temporaries
/// can still cost substantial time/storage.
/// @par Example
/// @code
/// const dm::algorithms::BrzozowskiOptions options{128,128};
/// std::print("{}\n",options.max_first_states); // 128
/// @endcode
/// @see brzozowski_minimize_trace()
struct BrzozowskiOptions {
    std::size_t max_first_states{0}; ///< Maximum created first-stage subsets, or zero for unlimited.
    std::size_t max_second_states{0}; ///< Maximum created second-stage subsets, or zero for unlimited.
};

/// @brief Owned actual double-reversal stages, with final minimal DFA in second.dfa.
/// first.subsets contains original input-state ids; second.subsets contains
/// first.dfa state ids. Reversed NFAs are the actual sources passed to each
/// construction. Reachable input labels exclude the implicit sink. The first
/// DFA is reachable but need not be minimal if the original has unreachable
/// explicit states. The second DFA is reachable, complete and language-minimal.
/// All snapshots own their storage and can be exponentially large.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::brzozowski_minimize_trace(dm::algorithms::DFA{2,1,{{0},{0}},0,{0,1}});
/// std::print("{} {}\n",trace.first.dfa.num_states,trace.second.dfa.num_states); // 2 1
/// @endcode
/// @see brzozowski_minimize_trace(), PowersetTrace
struct BrzozowskiTrace {
    std::vector<int> reachable_input_states; ///< Actual BFS-reachable explicit input ids, in ascending-symbol scan order.
    NFA first_reversal; ///< Actual reverse(input), with original accepting states as starts.
    PowersetTrace first; ///< First reachable complete determinisation, accepting the reversed language.
    NFA second_reversal; ///< Actual reverse(first.dfa), with first-stage accepting states as starts.
    PowersetTrace second; ///< Second complete determinisation; second.dfa is the final minimal machine.
};

/// @brief Private shared two-stage pipeline; ordinary calls discard diagnostic snapshots.
namespace brzozowski_detail {
/// @brief Run exactly two reversals and reachable complete subset constructions.
/// @tparam Record Whether to retain reversal/subset/reachability snapshots.
/// @param input Borrowed DFA, validated by reversal before output allocation.
/// @param options Independent stage caps, zero meaning unlimited.
/// @return Owned stages; only their DFA fields survive when Record is false.
/// @throws std::invalid_argument If input is malformed.
/// @throws std::runtime_error If a positive subset cap is exceeded.
/// @throws std::overflow_error Before state/work overflow; allocation failures propagate.
template<bool Record>
inline BrzozowskiTrace minimize(const DFA& input,BrzozowskiOptions options) {
    BrzozowskiTrace trace;
    if constexpr (Record) trace.reachable_input_states=dfa_reachable_states(input);
    {
        auto reversed=reverse_dfa(input);
        trace.first=powerset_construction_trace(reversed,true,options.max_first_states);
        if constexpr (Record) trace.first_reversal=std::move(reversed);
    }
    if constexpr (!Record) std::vector<std::vector<int>>{}.swap(trace.first.subsets);
    {
        auto reversed=reverse_dfa(trace.first.dfa);
        trace.second=powerset_construction_trace(reversed,true,options.max_second_states);
        if constexpr (Record) trace.second_reversal=std::move(reversed);
    }
    if constexpr (!Record) std::vector<std::vector<int>>{}.swap(trace.second.subsets);
    return trace;
}
} // namespace brzozowski_detail

/// @brief Minimise a possibly partial DFA by reverse/determinise/reverse/determinise.
/// Retains the original inline signature, complete output and deterministic
/// subset-discovery numbering. Validates all input shapes/indices before
/// allocation. The final DFA accepts exactly the original language, has start
/// zero and is reachable and minimal among complete DFAs. Empty/no-start input
/// produces a one-state rejecting machine; zero symbols are allowed. The first
/// pass need not be minimal for an input containing unreachable explicit states,
/// but its reachable deterministic result makes the second pass minimal.
/// The map/vector subset constructions can grow exponentially; explicit subsets,
/// union sorting and map comparisons determine work/storage. Ordinary calls
/// discard diagnostic provenance/reversal snapshots once their stages finish.
/// Inputs are borrowed only during the call; results own storage with no mutation.
/// @param d Borrowed valid-or-validatable DFA, with -1 missing transitions/start allowed.
/// @return Owned reachable complete minimal DFA for the same indexed-alphabet language.
/// @throws std::invalid_argument If d is malformed.
/// @throws std::overflow_error Before subset/state/work overflow; allocation failures propagate.
/// @par Example
/// @code
/// const auto result=dm::algorithms::brzozowski_minimize(dm::algorithms::DFA{2,2,{{0,1},{0,1}},0,{0,1}});
/// std::print("{}\n",result.num_states); // 2
/// @endcode
/// @par References
/// - J. A. Brzozowski, Canonical regular expressions and minimal state graphs for
///   definite events, Mathematical Theory of Automata (1963), 529-561, Theorem 13;
///   the algorithm was discovered in 1962, as clarified by the author in EATCS 37 (1989), 130.
/// - J. Berstel et al., Minimization of Automata, section 3,
///   https://arxiv.org/abs/1010.5318 (reachability/minimality and reversal examples).
/// @see brzozowski_minimize_trace(), dfa_equivalence()
inline DFA brzozowski_minimize(const DFA& d) {
    return brzozowski_detail::minimize<false>(d,{}).second.dfa;
}

/// @brief Retain the ordinary minimiser's actual reversals, subsets and state tables.
/// Uses the same two-stage kernel, ordering and complete final machine. Adds
/// owned reversal/provenance snapshots and a reachable-input list; neither
/// stage is replayed. A positive state cap throws without a partial result.
/// Snapshots and union temporaries can be exponential; caps are not memory limits.
/// @param d Borrowed possibly partial DFA, validated before stage allocation.
/// @param options Independent first/second subset caps, both unlimited by default.
/// @return Owned actual stages, with the final minimal DFA at second.dfa.
/// @throws std::invalid_argument If d is malformed.
/// @throws std::runtime_error If either positive stage cap is exhausted.
/// @throws std::overflow_error Before state/work overflow; allocation failures propagate.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::brzozowski_minimize_trace(dm::algorithms::DFA{1,1,{{-1}},0,{1}},{4,4});
/// std::print("{}\n",trace.second.dfa.num_states); // 2
/// @endcode
/// @par References
/// - J. A. Brzozowski (1963), Theorem 13; J. Berstel et al., section 3,
///   https://arxiv.org/abs/1010.5318; actual shared double-reversal pipeline.
/// @see BrzozowskiTrace, BrzozowskiOptions, brzozowski_minimize()
inline BrzozowskiTrace brzozowski_minimize_trace(const DFA& d,BrzozowskiOptions options={}) {
    return brzozowski_detail::minimize<true>(d,options);
}

/// @brief Build a binary DFA accepting iff the one-based position-th symbol exists and is one.
/// All later symbols are ignored. The position+2 states consist of a prefix
/// counter and accepting/rejecting sinks; every state is reachable. Its reversed
/// language tests that position from the end and needs 2^position complete DFA
/// states, making a deterministic reversal-growth fixture. O(position) work and
/// owned storage, without randomness or shared/input mutation.
/// @param position Positive one-based binary word position.
/// @return Owned complete binary DFA with start zero and position+2 states.
/// @throws std::invalid_argument If position is nonpositive.
/// @throws std::overflow_error If position+2 exceeds int, before allocation.
/// Allocation/capacity failures propagate for large valid positions.
/// @par Example
/// @code
/// const auto d=dm::algorithms::binary_kth_symbol_dfa(2);
/// std::print("{} {}\n",dm::algorithms::dfa_accepts(d,{0,1}),dm::algorithms::dfa_accepts(d,{1,0})); // true false
/// @endcode
/// @par References
/// - J. Berstel et al., Minimization of Automata, section 3, Figure 1,
///   https://arxiv.org/abs/1010.5318; this fixture includes an explicit rejecting sink.
/// @see brzozowski_minimize_trace()
inline DFA binary_kth_symbol_dfa(int position) {
    if (position<=0) throw std::invalid_argument("binary symbol position must be positive");
    if (position>std::numeric_limits<int>::max()-2) throw std::overflow_error("binary position DFA exceeds int state count");
    DFA d;d.num_states=position+2;d.num_symbols=2;d.start=0;
    d.delta.assign(d.num_states,std::vector<int>(2));d.accept.assign(d.num_states,0);
    for (int state=0;state<position-1;++state) d.delta[state]={state+1,state+1};
    d.delta[position-1]={position+1,position};d.delta[position]={position,position};
    d.delta[position+1]={position+1,position+1};d.accept[position]=1;return d;
}
} // namespace datamunge::algorithms
