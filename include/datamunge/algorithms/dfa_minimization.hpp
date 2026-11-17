#pragma once

/// \file dfa_minimization.hpp
/// \brief DFA minimization by the Moore, Hopcroft, and Brzozowski algorithms.
///
/// Every regular language has a minimal reachable complete DFA unique up to state
/// renaming. Three classic
/// algorithms compute it:
///
/// - **Moore** (partition refinement): start by separating accepting from non-accepting
///   states, then repeatedly split any block whose members disagree on which block a symbol
///   leads to, until stable. Ordered signature maps add key-comparison overhead.
/// - **Hopcroft** (worklist refinement): set-based splitter/predecessor refinement.
///   This implementation scans entire partitions; it does not promise the optimized
///   textbook time bound. Actual selected work and histories are available.
/// - **Brzozowski**: reverse the automaton, determinize, reverse, determinize. Two rounds of
///   reverse-then-subset-construct yield the minimal DFA directly. Simple but worst-case
///   exponential.
///
/// All three preserve the language and minimal reachable state count; internal
/// numbering can differ. This module implements them on the
/// shared automaton types.

#include <algorithm>
#include <cstddef>
#include <datamunge/algorithms/automaton.hpp>
#include <datamunge/algorithms/automaton_operations.hpp>
#include <datamunge/algorithms/brzozowski.hpp>
#include <datamunge/algorithms/powerset_construction.hpp>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <vector>

namespace datamunge::algorithms {

/// @brief Internal stable-partition quotient construction.
namespace detail {
/// @brief Build the reachable quotient of a complete DFA and stable nonempty blocks.
/// Internal precondition: every state has a block ID in 0..numblocks-1, every
/// block is nonempty, acceptance and transitions agree within blocks, start valid.
/// Uses first-state representatives and ascending-symbol BFS; no input mutation.
/// @param d Borrowed complete validated DFA with at least one state.
/// @param block Complete state-to-block labels satisfying the stable partition invariant.
/// @param numblocks Complete positive internal block count.
/// @return Owned reachable complete quotient, starting at 0.
/// @throws std::bad_alloc On allocation failure.
/// @throws std::length_error On container representation limits.
inline DFA build_from_blocks(const DFA& d, const std::vector<int>& block, int numblocks) {
  std::vector<int> rep(numblocks, -1);
  for (int s = 0; s < d.num_states; ++s)
    if (rep[block[s]] < 0)
      rep[block[s]] = s;

  std::vector<std::vector<int>> bdelta(numblocks, std::vector<int>(d.num_symbols));
  std::vector<char>             bacc(numblocks, 0);
  for (int b = 0; b < numblocks; ++b) {
    bacc[b] = d.accept[rep[b]];
    for (int c = 0; c < d.num_symbols; ++c)
      bdelta[b][c] = block[d.delta[rep[b]][c]];
  }

  int              bstart = block[d.start];
  std::vector<int> newid(numblocks, -1), order;
  std::queue<int>  q;
  q.push(bstart);
  newid[bstart] = 0;
  order.push_back(bstart);
  int cnt = 1;
  while (!q.empty()) {
    int b = q.front();
    q.pop();
    for (int c = 0; c < d.num_symbols; ++c) {
      int nb = bdelta[b][c];
      if (newid[nb] < 0) {
        newid[nb] = cnt++;
        order.push_back(nb);
        q.push(nb);
      }
    }
  }

  DFA m;
  m.num_symbols = d.num_symbols;
  m.num_states  = cnt;
  m.start       = 0;
  m.delta.assign(cnt, std::vector<int>(d.num_symbols));
  m.accept.assign(cnt, 0);
  for (int i = 0; i < cnt; ++i) {
    int b       = order[i];
    m.accept[i] = bacc[b];
    for (int c = 0; c < d.num_symbols; ++c)
      m.delta[i][c] = newid[bdelta[b][c]];
  }
  return m;
}
} // namespace detail

/// @brief Complete projected-input and actual history policies for Moore refinement.
/// States/transitions include unreachable states and any materialized rejecting
/// sink. Zero disables each independent cap. Ordinary moore_minimize() disables
/// policies, preserving its legacy uncapped entry point.
/// @par Example
/// @code
/// dm::algorithms::MooreOptions o;std::print("{}\n",o.max_states); // 100000
/// @endcode
/// @see moore_minimize_trace()
struct MooreOptions {
  std::size_t max_states{100000};       ///< Complete projected-state cap before completion copies.
  std::size_t max_transitions{1000000}; ///< Complete projected states times alphabet cap.
  std::size_t max_trace_entries{1000000}; ///< Online whole-round signature/label capture cap.
};
/// @brief Actual Moore rounds, state/signature work and ordered-map comparisons.
/// Signature components count examined integer pairs, not individual less-than
/// instructions. Map counts depend on the standard-library tree implementation.
/// Completion, quotient construction, allocations and copying are excluded.
/// @par Example
/// @code
/// auto r=dm::algorithms::moore_minimize_run(dm::algorithms::moore_demo_dfa());
/// std::print("{} {}\n",r.work.rounds,r.work.transition_reads); // 2 24
/// @endcode
/// @see MooreRun
struct MooreWork {
  std::size_t rounds{0}; ///< Actual completed signature refinements, including final stable pass.
  std::size_t state_visits{0};     ///< Actual states for which a signature is constructed.
  std::size_t transition_reads{0}; ///< Actual symbol transitions inspected for those signatures.
  std::size_t map_comparisons{0};  ///< Actual ordered-map lexicographic comparator calls.
  std::size_t signature_components{0}; ///< Actual component pairs examined in map comparisons.
};
/// @brief Owned completed graph, final partition and reachable minimal quotient.
/// Completed graph/partition include unreachable states; result removes unreachable
/// quotient blocks in ascending-symbol BFS order. Acceptance bytes retain the
/// first state's original nonzero representation, rather than normalizing to one.
/// @par Example
/// @code
/// auto r=dm::algorithms::moore_minimize_run(dm::algorithms::moore_demo_dfa());
/// std::print("{} {}\n",r.completed.num_states,r.result.num_states); // 6 3
/// @endcode
/// @see moore_minimize_run()
struct MooreRun {
  DFA              completed; ///< Actual completed input, including materialized sink when needed.
  DFA              result;    ///< Actual reachable quotient, start0, complete transitions.
  std::vector<int> partition; ///< Actual final completed-state block IDs.
  std::vector<std::vector<int>>
            blocks; ///< Actual complete final classes, sorted member IDs, in partition-ID order.
  MooreWork work;   ///< Actual selected signature/refinement counts.
};
/// @brief Actual before/after labels and per-state signatures in one synchronous round.
/// Each signature begins with its state's old block, then destination old-block
/// IDs in symbol order. New IDs follow first state encounter, not map key order.
/// Stable means block count unchanged; numeric labels can still be renamed.
/// @par Example
/// @code
/// auto t=dm::algorithms::moore_minimize_trace(dm::algorithms::moore_demo_dfa());
/// std::print("{}\n",t.rounds.front().blocks_after); // 3
/// @endcode
/// @see MooreTrace
struct MooreRound {
  std::vector<int>              before;     ///< Actual previous state-to-block labels.
  std::vector<std::vector<int>> signatures; ///< Actual produced signature by ascending state ID.
  std::vector<int>              after;      ///< Actual new state-to-block labels.
  int                           blocks_before{0}; ///< Actual distinct block count at entry.
  int                           blocks_after{0};  ///< Actual map signature count after all states.
  bool                          stable{false};    ///< Actual equal-count termination decision.
};
/// @brief Owned actual initial labels, complete round histories and producing run.
/// Initial capture units count one label per completed state. Each full round
/// adds one record, before/after labels, signature row records and their integers:
/// 1+3n+n*(k+1). Working maps and ordinary run output are excluded from this
/// logical capture policy. Admission precedes complete history copies; no replay.
/// @par Example
/// @code
/// auto t=dm::algorithms::moore_minimize_trace(dm::algorithms::DFA{});
/// std::print("{}\n",t.captured_entries); // 6
/// @endcode
/// @see moore_minimize_trace()
struct MooreTrace {
  MooreRun run; ///< Owned actual completed graph, partition, quotient and selected work.
  std::vector<int>
      initial_partition; ///< Actual acceptance-based labels0/1, even if only label1 exists.
  std::vector<MooreRound> rounds; ///< Actual synchronous passes, including terminal stable pass.
  std::size_t             captured_entries{0}; ///< Actual logical retained history units.
};
/// @brief Checked counts, comparison instrumentation and canonical Moore producer.
namespace moore_detail {
/// @brief Add represented work or capture counts without wrapping.
/// @param a Existing count.
/// @param b Actual additional units.
/// @return Checked sum.
/// @throws std::length_error If addition exceeds size_t.
inline std::size_t add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::length_error("Moore: count overflow");
  return a + b;
}
/// @brief Multiply complete represented geometry counts without wrapping.
/// @param a First nonnegative count.
/// @param b Second nonnegative count.
/// @return Checked product.
/// @throws std::length_error If product exceeds size_t.
inline std::size_t product(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b)
    throw std::length_error("Moore: geometry overflow");
  return a * b;
}
/// @brief Enforce one optional complete count cap.
/// @param amount Complete required units.
/// @param cap Inclusive optional limit, zero disables.
/// @throws std::length_error If a nonzero cap is exceeded.
inline void limit(std::size_t amount, std::size_t cap) {
  if (cap && amount > cap)
    throw std::length_error("Moore: policy limit exceeded");
}
/// @brief Admit actual complete history units before copying a round.
/// @param out Optional actual-history sink.
/// @param amount Actual retained units.
/// @param options History quota policy.
/// @throws std::length_error If count/quota is exceeded.
inline void capture(MooreTrace* out, std::size_t amount, MooreOptions options) {
  if (out) {
    auto next = add(out->captured_entries, amount);
    limit(next, options.max_trace_entries);
    out->captured_entries = next;
  }
}
/// @brief Lexicographic signature comparison with actual selected work observations.
struct Compare {
  MooreWork* work; ///< Call-local actual work sink, never null.

  /// @brief Compare complete signature keys in the same order as vector lexicographic less.
  /// @param a Actual first map key.
  /// @param b Actual second map key.
  /// @return True exactly when a is lexicographically smaller.
  /// @throws std::length_error If work counters overflow.
  bool operator()(const std::vector<int>& a, const std::vector<int>& b) const {
    work->map_comparisons = add(work->map_comparisons, 1);
    const auto count      = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < count; ++i) {
      work->signature_components = add(work->signature_components, 1);
      if (a[i] != b[i])
        return a[i] < b[i];
    }
    return a.size() < b.size();
  }
};
/// @brief Execute the preserved synchronous signature loop and optional actual capture.
/// @param input Borrowed possibly partial validated-in-body DFA.
/// @param options Complete projected graph and actual history policies.
/// @param trace Optional actual history sink.
/// @return Owned actual complete graph, partition, reachable quotient and work.
/// @throws std::invalid_argument If input is malformed before policy checks.
/// @throws std::overflow_error If a needed sink exceeds int state counts.
/// @throws std::length_error If geometry/count/policies exceed limits.
inline MooreRun execute(const DFA& input, MooreOptions options, MooreTrace* trace) {
  validate_dfa(input);
  bool sink = input.start < 0;
  for (auto& row : input.delta)
    for (auto target : row)
      sink = sink || target < 0;
  if (sink && input.num_states == std::numeric_limits<int>::max())
    throw std::overflow_error("Moore: sink state count overflow");
  const auto projected = add(std::size_t(input.num_states), sink ? 1 : 0),
             alphabet  = std::size_t(input.num_symbols);
  limit(projected, options.max_states);
  limit(product(projected, alphabet), options.max_transitions);
  DFA              d = complete_dfa(input);
  const int        n = d.num_states;
  MooreRun         run;
  std::vector<int> block(n);
  for (int i = 0; i < n; ++i)
    block[i] = d.accept[i] ? 1 : 0;
  int current =
      (std::count(block.begin(), block.end(), 1) && std::count(block.begin(), block.end(), 0)) ? 2
                                                                                               : 1;
  if (trace) {
    capture(trace, projected, options);
    trace->initial_partition = block;
  }
  while (true) {
    MooreRound record;
    if (trace) {
      auto units = add(1, add(product(3, projected), product(projected, add(alphabet, 1))));
      capture(trace, units, options);
      record.before        = block;
      record.blocks_before = current;
      record.signatures.reserve(projected);
    }
    std::map<std::vector<int>, int, Compare> signatures(Compare{&run.work});
    std::vector<int>                         next(n);
    for (int state = 0; state < n; ++state) {
      run.work.state_visits = add(run.work.state_visits, 1);
      std::vector<int> key{block[state]};
      for (int symbol = 0; symbol < d.num_symbols; ++symbol) {
        run.work.transition_reads = add(run.work.transition_reads, 1);
        key.push_back(block[d.delta[state][symbol]]);
      }
      if (trace)
        record.signatures.push_back(key);
      auto found = signatures.find(key);
      if (found == signatures.end()) {
        int id          = int(signatures.size());
        signatures[key] = id;
        next[state]     = id;
      } else
        next[state] = found->second;
    }
    const int  count  = int(signatures.size());
    const bool stable = count == current;
    run.work.rounds   = add(run.work.rounds, 1);
    if (trace) {
      record.after        = next;
      record.blocks_after = count;
      record.stable       = stable;
      trace->rounds.push_back(std::move(record));
    }
    block = std::move(next);
    if (stable)
      break;
    current = count;
  }
  run.result = detail::build_from_blocks(d, block, current);
  run.blocks.resize(current);
  for (int state = 0; state < n; ++state)
    run.blocks[block[state]].push_back(state);
  run.partition = std::move(block);
  run.completed = std::move(d);
  return run;
}
} // namespace moore_detail
/// @brief Minimize a possibly partial DFA by actual ordered-map Moore signature refinement.
/// Complete rejecting moves/start with a sink, refine all states synchronously,
/// then return reachable quotient blocks in symbol-order BFS numbering. No
/// input mutation/references retained. Empty-language zero-state input yields
/// one rejecting state; zero alphabet allows epsilon only. Ordinary call disables
/// optional policies. At most n rounds; vector-key maps give a conservative
/// O((k+1)*n^2*log(n+1)) work bound and O(n*(k+1)) driver/output storage.
/// @param din Borrowed valid possibly partial DFA, including−1 implicit rejecting targets/start.
/// @return Owned reachable minimal complete quotient, start0.
/// @throws std::invalid_argument If input is malformed.
/// @throws std::overflow_error If sink state creation exceeds int.
/// @throws std::length_error If represented counts/container geometry overflow.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const dm::algorithms::DFA d{2,1,{{1},{1}},0,{0,0}};
/// std::print("{}\n",dm::algorithms::moore_minimize(d).num_states); // 1
/// @endcode
/// @par References
/// - E. F. Moore (1956), Gedanken-experiments on sequential machines, Automata Studies,129–153. Ordered vector-key map implementation here.
/// @see moore_minimize_run(), moore_minimize_trace(), dfa_equivalence()
inline DFA moore_minimize(const DFA& din) {
  return moore_detail::execute(din, {0, 0, 0}, nullptr).result;
}
/// @brief Return actual completed graph, final partition and selected Moore work.
/// Same producer/ordinary numbering; complete projected caps precede graph copies.
/// Capture cap is ignored. No input/shared-state mutation. Ordinary complexity
/// applies; completed and quotient outputs are owned independently.
/// @param din Borrowed valid possibly partial DFA.
/// @param options Complete projected caps, defaults100k states/1m transitions; zero disables each.
/// @return Owned actual run, with no reconstructed history.
/// @throws std::invalid_argument If input is malformed.
/// @throws std::overflow_error If sink count exceeds int.
/// @throws std::length_error If geometry/work/projected caps exceed limits.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// auto r=dm::algorithms::moore_minimize_run(dm::algorithms::DFA{});std::print("{}\n",r.result.num_states); // 1
/// @endcode
/// @see MooreRun
inline MooreRun moore_minimize_run(const DFA& din, MooreOptions options = {}) {
  return moore_detail::execute(din, options, nullptr);
}
/// @brief Capture actual old/new partitions and every produced signature per round.
/// Same producer and complete caps as moore_minimize_run(). Admit initial labels
/// then each whole known round before history copies; no replay/truncation or
/// partial result. Storage adds the actual retained units described by MooreTrace.
/// @param din Borrowed original possibly partial DFA.
/// @param options Projected graph and online capture policies, zero disables each cap.
/// @return Owned actual initial partition, rounds, completed graph, quotient and work.
/// @throws std::invalid_argument If input is malformed before policy checks.
/// @throws std::overflow_error If completion exceeds int state counts.
/// @throws std::length_error If geometry/work/capture/projected limits are exceeded.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// auto t=dm::algorithms::moore_minimize_trace(dm::algorithms::moore_demo_dfa());
/// std::print("{}\n",t.rounds.size()); // 2
/// @endcode
/// @see MooreTrace, MooreOptions
inline MooreTrace moore_minimize_trace(const DFA& din, MooreOptions options = {}) {
  MooreTrace t;
  t.run = moore_detail::execute(din, options, &t);
  return t;
}
/// @brief Return the original reachable six-state, two-symbol demonstration DFA.
/// Rows(1,2),(0,3),(4,5),(4,5),(4,5),(5,5); start0, accepting2,3,4.
/// @return Owned original DFA, whose minimal reachable quotient has three states.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::moore_demo_dfa().num_states); // 6
/// @endcode
/// @see moore_minimize_trace()
inline DFA moore_demo_dfa() {
  return {6, 2, {{1, 2}, {0, 3}, {4, 5}, {4, 5}, {4, 5}, {5, 5}}, 0, {0, 0, 1, 1, 1, 0}};
}
/// @brief Construct a unary acceptance-distance chain for refinement scaling.
/// State i moves to min(i+1,n−1); only the final state accepts. All states are
/// reachable from0 and pairwise distinguishable by distance to acceptance.
/// @param states Positive count, at most100000.
/// @return Owned complete unary DFA, with minimal state count equal to states.
/// @throws std::invalid_argument If states is below one.
/// @throws std::length_error If states exceeds100000.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::moore_chain_dfa(3).num_states); // 3
/// @endcode
/// @see moore_minimize_run()
inline DFA moore_chain_dfa(int states) {
  if (states < 1)
    throw std::invalid_argument("Moore chain: positive state count required");
  if (states > 100000)
    throw std::length_error("Moore chain: state cap exceeded");
  DFA d;
  d.num_states  = states;
  d.num_symbols = 1;
  d.start       = 0;
  d.delta.resize(states);
  d.accept.assign(states, 0);
  d.accept.back() = 1;
  for (int i = 0; i < states; ++i)
    d.delta[i] = {std::min(i + 1, states - 1)};
  return d;
}

/// @brief Complete projected graph and online logical history policies, each zero disabling its cap.
/// Counts include a materialized sink and unreachable states. Trace units count
/// records and retained state IDs, not bytes/capacity/peak memory. Input validation
/// precedes caps and completed graph/predecessor allocations.
/// @par Example
/// @code
/// const dm::algorithms::HopcroftOptions o;std::print("{}\n",o.max_trace_entries); // 1000000
/// @endcode
struct HopcroftOptions {
  std::size_t max_states{100000};       ///< Complete projected state cap, including any added sink.
  std::size_t max_transitions{1000000}; ///< Complete projected states times alphabet-size cap.
  std::size_t max_trace_entries{
      1000000}; ///< Online retained record/state-ID cap, ignored without tracing.
};
/// @brief Selected producing visits of the ordered-set worklist implementation.
/// These exclude validation, completion, set insertion/comparison internals,
/// copying, allocation, quotient construction and history; not timing or FLOPs.
/// @par Example
/// @code
/// const dm::algorithms::HopcroftWork w;std::print("{}\n",w.splits); // 0
/// @endcode
struct HopcroftWork {
  std::size_t prepared_transitions{
      0};                       ///< Actual completed transitions inserted into inverse lists.
  std::size_t splitter_pops{0}; ///< Actual last-in-first-out splitter removals.
  std::size_t symbol_passes{
      0}; ///< Actual symbols examined for popped splitters, including empty preimages.
  std::size_t predecessor_targets{
      0};                           ///< Actual splitter states visited while reading inverse lists.
  std::size_t predecessor_edges{0}; ///< Actual inverse-list edges visited, before set insertion.
  std::size_t partition_blocks{0};  ///< Actual blocks examined for nonempty preimages.
  std::size_t partition_states{0};  ///< Actual state-membership tests across those blocks.
  std::size_t worklist_candidates{
      0};                ///< Actual pending-set equality candidates examined after splits.
  std::size_t splits{0}; ///< Actual proper partition splits.
};
/// @brief Owned reachable quotient, completed-state partition labels and selected producing work.
/// blocks index completed states, including any sink and unreachable input states.
/// Their IDs describe the final internal partition, not the quotient's reachable
/// BFS numbering. result starts at0 and retains only reachable quotient blocks.
/// @par Example
/// @code
/// const dm::algorithms::HopcroftRun r;std::print("{}\n",r.blocks.size()); // 0
/// @endcode
struct HopcroftRun {
  DFA result; ///< Actual reachable complete minimized DFA, numbered by symbol-order BFS.
  std::vector<int> blocks;              ///< Final internal block ID for every completed state.
  std::size_t      completed_states{0}; ///< Input states plus a rejecting sink exactly when needed.
  HopcroftWork     work;                ///< Selected actual producing visits.
};
/// @brief Actual worklist action for a proper split of one current block.
/// @par Example
/// @code
/// std::print("{}\n",int(dm::algorithms::HopcroftSchedule::Both)); // 0
/// @endcode
/// @see HopcroftSplit
enum class HopcroftSchedule {
  Both,   ///< Pending parent slot replaced by inside, with outside appended.
  Inside, ///< Nonpending parent: append inside, the smaller half or the tied first half.
  Outside ///< Nonpending parent: append strictly smaller outside half.
};
/// @brief Owned actual block split and its producing worklist branch.
/// Every vector is ascending in original completed-state labels. inside and
/// outside partition parent exactly. Both means an existing pending parent was
/// replaced; it does not mean two additional pushes.
/// @par Example
/// @code
/// const dm::algorithms::HopcroftSplit s;std::print("{}\n",s.parent.size()); // 0
/// @endcode
/// @see HopcroftPass, HopcroftSchedule
struct HopcroftSplit {
  std::vector<int> parent;  ///< Actual current partition block before this split.
  std::vector<int> inside;  ///< Actual parent intersection with current preimage.
  std::vector<int> outside; ///< Actual parent minus current preimage.
  HopcroftSchedule schedule{HopcroftSchedule::Both}; ///< Actual pending-or-smaller-half branch.
};
/// @brief Owned actual preimage and splits from one producing splitter/symbol pass.
/// Empty preimages retain an empty split list; a zero-symbol machine has no
/// passes despite actual splitter pops. Splitter labels refer to the copied set
/// popped at this pop index, even if later passes refine its original block.
/// @par Example
/// @code
/// const dm::algorithms::HopcroftPass p;std::print("{}\n",p.splits.size()); // 0
/// @endcode
/// @see HopcroftTrace
struct HopcroftPass {
  std::size_t pop_index{0};  ///< One-based actual splitter pop index shared by its symbol passes.
  int         symbol{0};     ///< Actual symbol processed in ascending order.
  std::vector<int> splitter; ///< Actual copied popped state set, ascending.
  std::vector<int> predecessors; ///< Actual union of inverse transitions into that splitter.
  std::vector<HopcroftSplit>
      splits; ///< Actual proper splits, in current partition traversal order.
};
/// @brief Owned actual initial partition, chronological producing passes and final run.
/// Initial units are one per block plus one per retained state ID. Each pass
/// adds 1+|splitter|+|predecessors|; each proper split adds 1+|parent|+|inside|+|outside|.
/// Counts are admitted before history copies, after their triggering sets are
/// known. Working sets and ordinary output are outside this capture policy.
/// No truncation, replay, partial result or retained input references.
/// @par Example
/// @code
/// const dm::algorithms::HopcroftTrace t;std::print("{}\n",t.captured_entries); // 0
/// @endcode
/// @see HopcroftOptions, hopcroft_minimize_trace()
struct HopcroftTrace {
  HopcroftRun run; ///< Actual producing quotient, final partition and work.
  std::vector<std::vector<int>>
      initial_blocks;               ///< Accepting block first, rejecting block second when present.
  std::vector<HopcroftPass> passes; ///< Actual pop/symbol order; no post-hoc reconstruction.
  std::size_t captured_entries{0};  ///< Retained logical record/state-ID units, not bytes.
};

/// @brief Internal checked counts, projected-input admission and shared producing refinement.
namespace hopcroft_detail {
/// @brief Add selected logical counts without wraparound.
/// @param a Existing units.
/// @param b Additional units.
/// @return Exact sum.
/// @throws std::length_error On unrepresentable sum.
inline std::size_t add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::length_error("Hopcroft count overflow");
  return a + b;
}
/// @brief Admit a complete projected or retained logical count.
/// @param value Complete required units.
/// @param cap Maximum units, zero disabling policy.
/// @throws std::length_error On a positive cap excess.
inline void admit(std::size_t value, std::size_t cap) {
  if (cap && value > cap)
    throw std::length_error("Hopcroft policy limit");
}
/// @brief Admit additional actual history units before their copies.
/// @param trace Owned history, or null for no capture.
/// @param amount Complete additional record/state-ID units.
/// @param options History cap.
/// @throws std::length_error On count/capture-cap excess.
inline void capture(HopcroftTrace* trace, std::size_t amount, HopcroftOptions options) {
  if (trace) {
    const auto next = add(trace->captured_entries, amount);
    admit(next, options.max_trace_entries);
    trace->captured_entries = next;
  }
}
/// @brief Execute preserved set-based refinement and optional actual capture without replay.
/// @param input Borrowed possibly partial DFA, never modified or retained.
/// @param options Complete projected graph and online history policies.
/// @param trace Optional owned capture destination.
/// @return Actual reachable quotient, completed partition and selected work.
/// @throws std::invalid_argument On malformed input before policy checks.
/// @throws std::overflow_error If completion would exceed int state counts.
/// @throws std::length_error On graph/history/container/count limits.
/// @throws std::bad_alloc On allocation failure.
inline HopcroftRun execute(const DFA& input, HopcroftOptions options, HopcroftTrace* trace) {
  validate_dfa(input);
  bool sink = input.start < 0;
  for (const auto& row : input.delta)
    for (int target : row)
      sink = sink || target < 0;
  if (sink && input.num_states == std::numeric_limits<int>::max())
    throw std::overflow_error("Hopcroft sink count");
  const auto projected = add(static_cast<std::size_t>(input.num_states), sink ? 1 : 0);
  const auto alphabet  = static_cast<std::size_t>(input.num_symbols);
  if (alphabet && projected > std::numeric_limits<std::size_t>::max() / alphabet)
    throw std::length_error("Hopcroft transition count");
  admit(projected, options.max_states);
  admit(projected * alphabet, options.max_transitions);
  DFA         d = complete_dfa(input);
  const int   n = d.num_states, K = d.num_symbols;
  HopcroftRun run;
  run.completed_states = projected;
  std::vector<std::vector<std::vector<int>>> pred(K, std::vector<std::vector<int>>(n));
  for (int s = 0; s < n; ++s)
    for (int c = 0; c < K; ++c) {
      pred[c][d.delta[s][c]].push_back(s);
      run.work.prepared_transitions = add(run.work.prepared_transitions, 1);
    }
  std::set<int> F, NF;
  for (int s = 0; s < n; ++s)
    (d.accept[s] ? F : NF).insert(s);
  std::vector<std::set<int>> P, W;
  if (!F.empty()) {
    P.push_back(F);
    W.push_back(F);
  }
  if (!NF.empty()) {
    P.push_back(NF);
    W.push_back(NF);
  }
  if (trace) {
    capture(trace, add(projected, P.size()), options);
    for (const auto& block : P)
      trace->initial_blocks.emplace_back(block.begin(), block.end());
  }
  while (!W.empty()) {
    const std::set<int> A = W.back();
    W.pop_back();
    run.work.splitter_pops = add(run.work.splitter_pops, 1);
    for (int c = 0; c < K; ++c) {
      run.work.symbol_passes = add(run.work.symbol_passes, 1);
      std::set<int> X;
      for (int t : A) {
        run.work.predecessor_targets = add(run.work.predecessor_targets, 1);
        for (int s : pred[c][t]) {
          X.insert(s);
          run.work.predecessor_edges = add(run.work.predecessor_edges, 1);
        }
      }
      HopcroftPass pass;
      if (trace) {
        capture(trace, add(1, add(A.size(), X.size())), options);
        pass.pop_index = run.work.splitter_pops;
        pass.symbol    = c;
        pass.splitter.assign(A.begin(), A.end());
        pass.predecessors.assign(X.begin(), X.end());
      }
      if (!X.empty()) {
        std::vector<std::set<int>> newP;
        for (const auto& Y : P) {
          run.work.partition_blocks = add(run.work.partition_blocks, 1);
          std::set<int> inter, diff;
          for (int y : Y) {
            (X.count(y) ? inter : diff).insert(y);
            run.work.partition_states = add(run.work.partition_states, 1);
          }
          if (!inter.empty() && !diff.empty()) {
            run.work.splits = add(run.work.splits, 1);
            newP.push_back(inter);
            newP.push_back(diff);
            auto wit = W.begin();
            for (; wit != W.end(); ++wit) {
              run.work.worklist_candidates = add(run.work.worklist_candidates, 1);
              if (*wit == Y)
                break;
            }
            const auto schedule = wit != W.end()
                                      ? HopcroftSchedule::Both
                                      : (inter.size() <= diff.size() ? HopcroftSchedule::Inside
                                                                     : HopcroftSchedule::Outside);
            if (trace) {
              capture(trace, add(1, add(Y.size(), add(inter.size(), diff.size()))), options);
              pass.splits.push_back(
                  {{Y.begin(), Y.end()},
                   {inter.begin(), inter.end()},
                   {diff.begin(), diff.end()},
                   schedule});
            }
            if (wit != W.end()) {
              *wit = inter;
              W.push_back(diff);
            } else
              W.push_back(inter.size() <= diff.size() ? inter : diff);
          } else
            newP.push_back(Y);
        }
        P = std::move(newP);
      }
      if (trace)
        trace->passes.push_back(std::move(pass));
    }
  }
  run.blocks.resize(n);
  for (int b = 0; b < static_cast<int>(P.size()); ++b)
    for (int s : P[b])
      run.blocks[s] = b;
  run.result = detail::build_from_blocks(d, run.blocks, static_cast<int>(P.size()));
  return run;
}
} // namespace hopcroft_detail

/// @brief Minimize a possibly partial DFA using preserved ordered-set worklist refinement.
/// Validates the entire input, materializes an implicit rejecting sink if needed,
/// then refines accepting/rejecting blocks. Splitters use LIFO order, symbols
/// ascend, and nonpending ties append the intersection half. Pending parents
/// become both halves. The reachable quotient uses symbol-order BFS numbering
/// and retains representative acceptance bytes. Empty language yields one rejecting
/// state, including zero input states; zero symbols permits only epsilon.
/// The set-based implementation scans every partition state for every nonempty
/// preimage and linearly searches pending sets. It does not claim the optimized
/// textbook O(k*n*log(n)) implementation bound. A conservative bound is
/// O((k+1)*n*n*log(n+1)) time and O(n*(k+1)) ordinary working/output storage for
/// completed n and alphabet k, excluding borrowed input and optional history.
/// Legacy has no optional graph/capture policies; all inputs/outputs own their
/// storage and no references/shared mutable state are retained.
/// @param din Borrowed possibly partial DFA; -1 means implicit rejecting sink.
/// @return Owned reachable minimal complete DFA, preserving legacy quotient numbering.
/// @throws std::invalid_argument On invalid complete input shape/indices.
/// @throws std::overflow_error If completion would exceed int state counts.
/// @throws std::length_error On count/container representation limits.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const dm::algorithms::DFA d{2,1,{{1},{1}},0,{0,0}};
/// std::print("{}\n",dm::algorithms::hopcroft_minimize(d).num_states); // 1
/// @endcode
/// @par References
/// - J. Hopcroft (1971), An n log n Algorithm for Minimizing States in a Finite Automaton,
///   doi:10.1016/B978-0-12-417750-5.50022-1 . This set-scanning implementation
///   preserves the worklist rule but not the optimized representation/time bound.
/// @see hopcroft_minimize_run(), hopcroft_minimize_trace(), dfa_equivalence()
inline DFA hopcroft_minimize(const DFA& din) {
  return hopcroft_detail::execute(din, {0, 0, 0}, nullptr).result;
}
/// @brief Execute the shared refinement with complete projected-input caps and selected work.
/// Same algorithm/order/domains as hopcroft_minimize(). Caps include unreachable
/// states and any added sink before graph copies/predecessor allocation. Capture
/// cap ignored. Selected counters exclude set internals, copying and quotient work.
/// @param din Borrowed possibly partial DFA.
/// @param options Complete projected states/transitions caps, defaults 100k/1m; zero disables each.
/// @return Owned quotient, final completed-state partition IDs and selected work.
/// @throws std::invalid_argument On malformed input before policy checks.
/// @throws std::overflow_error If sink creation exceeds int state counts.
/// @throws std::length_error On projected graph or count/container limits.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto r=dm::algorithms::hopcroft_minimize_run(dm::algorithms::DFA{});
/// std::print("{} {}\n",r.completed_states,r.result.num_states); // 1 1
/// @endcode
/// @see HopcroftRun, HopcroftOptions
inline HopcroftRun hopcroft_minimize_run(const DFA& din, HopcroftOptions options = {}) {
  return hopcroft_detail::execute(din, options, nullptr);
}
/// @brief Capture actual initial blocks, preimages and proper splits during producing refinement.
/// Same contracts as hopcroft_minimize_run(). Whole known capture records are
/// admitted before history copies after computing their triggering sets. No
/// truncation/replay/partial history. Capture units count records/state IDs,
/// excluding ordinary working sets, output, metadata and capacity; not peak RAM.
/// @param din Borrowed possibly partial DFA.
/// @param options Projected graph and online history caps; zero disables each policy.
/// @return Owned complete producing trace and actual final run.
/// @throws std::invalid_argument On malformed input before policy checks.
/// @throws std::overflow_error If sink creation exceeds int state counts.
/// @throws std::length_error On graph/history/count/container limits.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::hopcroft_minimize_trace(dm::algorithms::DFA{});
/// std::print("{} {}\n",t.initial_blocks.size(),t.passes.size()); // 1 0
/// @endcode
/// @see HopcroftTrace, HopcroftOptions
inline HopcroftTrace hopcroft_minimize_trace(const DFA& din, HopcroftOptions options = {}) {
  HopcroftTrace t;
  t.run = hopcroft_detail::execute(din, options, &t);
  return t;
}

} // namespace datamunge::algorithms
