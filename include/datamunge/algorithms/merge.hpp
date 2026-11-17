#pragma once
/// @file merge.hpp
/// @brief Checked sorted-run merges with actual heap-order work and owned push/pop evidence.
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {
/// @brief Actual heap operation recorded by the shared sorted-run producer.
/// @par Example
/// @code
/// const auto t=dm::algorithms::k_way_merge_trace({{1},{2}});
/// std::print("{}\n",t.steps[0].kind==dm::algorithms::MergeHeapKind::Push); // true
/// @endcode
/// @see MergeHeapStep
enum class MergeHeapKind {
  Push, ///< Actual front insertion, including initial nonempty runs.
  Pop   ///< Actual minimum-front removal, emitted or duplicate-discarded.
};
/// @brief Owned actual heap operation, original value identity and before/after occupancy.
/// Original identity is(list,index), preserving duplicate multiplicity. Push has
/// emitted=false; Pop records whether its value entered output. This is not a
/// copy of the private heap array or a replayed heap-comparison history.
/// @par Example
/// @code
/// const auto t=dm::algorithms::k_way_merge_trace({{1}});
/// std::print("{} {}\n",t.steps.back().value,t.steps.back().emitted); // 1 true
/// @endcode
/// @see MergeTrace
struct MergeHeapStep {
  MergeHeapKind kind{MergeHeapKind::Push}; ///< Actual push/pop operation.
  int           value{0};                  ///< Actual original run value at the operation.
  std::size_t   list{0};                   ///< Original outer-list identity.
  std::size_t   index{0};                  ///< Original within-list identity.
  std::size_t   heap_before{0};            ///< Actual queue size immediately before operation.
  std::size_t   heap_after{0};  ///< Actual queue size after operation, before any next operation.
  bool          emitted{false}; ///< Actual output append on Pop, false on Push or dedup discard.
};
/// @brief Selected actual heap operations, tuple-comparator calls and output decisions.
/// node_comparisons counts actual calls ordering whole(value,list,index) tuples,
/// not individual integer comparisons, memory reads or elapsed time. Excludes
/// validation, sort checks, copies, allocation and output writes.
/// @par Example
/// @code
/// const auto a=dm::algorithms::k_way_merge_analyze({{1,3},{2}});
/// std::print("{} {} {}\n",a.work.input_values,a.work.pushes,a.work.pops); // 3 3 3
/// @endcode
/// @see k_way_merge_analyze()
struct MergeWork {
  std::size_t input_values{0};     ///< Complete original N values, including duplicate identities.
  std::size_t pushes{0};           ///< Actual front insertions, exactly N after complete run.
  std::size_t pops{0};             ///< Actual minimum-front removals, exactly N after complete run.
  std::size_t node_comparisons{0}; ///< Actual whole-tuple heap comparator invocations.
  std::size_t emitted{0};          ///< Actual output values appended.
  std::size_t discarded_duplicates{0}; ///< Actual dedup Pop decisions, zero for ordinary merge.
  std::size_t peak_heap{0};            ///< Actual maximum simultaneous nonempty-run fronts.
};
/// @brief Owned sorted output and actual same-execution selected work, without input/history capture.
/// @par Example
/// @code
/// const auto a=dm::algorithms::k_way_merge_analyze({{1,1},{1,2}},true);
/// std::print("{} {}\n",a.values.size(),a.work.discarded_duplicates); // 2 2
/// @endcode
/// @see MergeTrace
struct MergeAnalysis {
  std::vector<int> values; ///< Owned actual sorted output, possibly deduplicated.
  MergeWork        work;   ///< Actual shared-kernel selected work.
};
/// @brief Owned complete original runs and actual chronological heap operations/output.
/// captured_entries counts K outer-row records + N original value records + actual
/// push/pop records. Excludes output values, tuple fields, runtime heap/copies,
/// capacities and bytes/RAM. Validate all sorted rows first, then admit complete
/// K+N before input copies and each event before its heap operation. Default
/// million, zero off, no partial trace escapes failure. No merge/heap replay.
/// @par Example
/// @code
/// const auto t=dm::algorithms::k_way_merge_trace({{1}});
/// std::print("{} {}\n",t.steps.size(),t.captured_entries); // 2 4
/// @endcode
/// @see k_way_merge_trace()
struct MergeTrace {
  std::vector<std::vector<int>> lists;  ///< Owned complete sorted original runs and empty rows.
  std::vector<int>              values; ///< Actual owned output values.
  MergeWork work; ///< Actual producing selected work, including comparator calls.
  std::vector<MergeHeapStep>
              steps; ///< Actual chronological initial pushes and subsequent pops/pushes.
  std::size_t captured_entries{0}; ///< Complete K+N plus actual operation record count.
};
/// @brief Stably merge two nondecreasing int vectors in one linear pass.
/// Validate both complete inputs before allocation. Equal values take a first;
/// multiplicity retained, empty vectors allowed, integer extremes unchanged.
/// O(a.size()+b.size()) work/output, O(1) cursor state, no mutation/shared state.
/// @param a Borrowed complete nondecreasing first run.
/// @param b Borrowed complete nondecreasing second run.
/// @return Owned sorted union with duplicate multiplicity.
/// @throws std::invalid_argument On either unsorted input.
/// @throws std::length_error On combined/output size overflow.
/// @throws std::bad_alloc On output allocation failure.
/// @par Example
/// @code
/// const auto x=dm::algorithms::simple_merge({1,3},{1,2});
/// std::print("{} {}\n",x.size(),x[2]); // 4 2
/// @endcode
/// @par References
/// - Knuth, The Art of Computer Programming, Volume3: Sorting and Searching, second edition(1998), chapter5(Sorting).
/// @see k_way_merge(), union_merge()
std::vector<int> simple_merge(const std::vector<int>& a, const std::vector<int>& b);
/// @brief Merge all nondecreasing int runs with a minimum-front heap, preserving every identity.
/// Validate complete sorted rows and total size before heap/output allocation.
/// Tuple ordering(value,list,index) breaks equal-value ties by original list then
/// within-list position. Empty runs ignored, outer empty allowed. Each original
/// identity is pushed/popped once. O(K+N log(R+1)) work for K rows,N values and R
/// nonempty runs, O(R) heap and O(N) output. No caller mutation/input copies/history
/// or shared mutable state. Comparator calls depend on the standard-library heap.
/// @param lists Borrowed complete nondecreasing runs, possibly empty/duplicated.
/// @return Owned sorted merged values with original multiplicity.
/// @throws std::invalid_argument On any unsorted row, including an otherwise unused row.
/// @throws std::length_error On combined/output/count limits.
/// @throws std::bad_alloc On heap/output allocation failure.
/// @par Example
/// @code
/// const auto x=dm::algorithms::k_way_merge({{1,4},{2,3}});
/// std::print("{} {}\n",x.size(),x.back()); // 4 4
/// @endcode
/// @par References
/// - Knuth, The Art of Computer Programming, Volume3: Sorting and Searching, second edition(1998), chapter5(Sorting).
/// @see k_way_merge_analyze(), k_way_merge_trace(), union_merge()
std::vector<int> k_way_merge(const std::vector<std::vector<int>>& lists);
/// @brief Produce sorted distinct values through the same minimum-front merge kernel.
/// Same validation/tie/order/complexity as k_way_merge(), but skip a popped value
/// equal to the last emitted one. Every original identity still enters/leaves heap.
/// Output uses O(number of distinct values) storage; no optional capture quota.
/// @param lists Borrowed complete nondecreasing runs, duplicates/empty allowed.
/// @return Owned sorted set union, each int once.
/// @throws std::invalid_argument On any unsorted row.
/// @throws std::length_error On combined/output/count limits.
/// @throws std::bad_alloc On heap/output allocation failure.
/// @par Example
/// @code
/// const auto x=dm::algorithms::union_merge({{1,1},{1,2}});
/// std::print("{} {}\n",x.size(),x.back()); // 2 2
/// @endcode
/// @par References
/// - Knuth, The Art of Computer Programming, Volume3: Sorting and Searching, second edition(1998), chapter5(Sorting).
/// @see k_way_merge(), k_way_merge_analyze()
std::vector<int> union_merge(const std::vector<std::vector<int>>& lists);
/// @brief Return actual heap-merge values and selected work without retaining input/history.
/// Shared producer and contracts with k_way_merge()/union_merge(), not estimated
/// comparison formulas or a second merge. Actual tuple comparator invocations
/// are counted, including list/index tie ordering inside each invocation.
/// @param lists Borrowed complete nondecreasing runs.
/// @param unique Deduplicate output if true, default false preserves all values.
/// @return Owned actual output and actual selected work.
/// @throws std::invalid_argument On any unsorted row.
/// @throws std::length_error On combined/output/work-count limits.
/// @throws std::bad_alloc On native heap/output allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::k_way_merge_analyze({{}, {1}}).work.peak_heap); // 1
/// @endcode
/// @par References
/// - Knuth, The Art of Computer Programming, Volume3: Sorting and Searching, second edition(1998), chapter5(Sorting).
/// @see MergeWork, k_way_merge_trace()
MergeAnalysis k_way_merge_analyze(const std::vector<std::vector<int>>& lists, bool unique = false);
/// @brief Capture complete original runs and every actual producing heap push/pop once.
/// Shared kernel/contracts with k_way_merge_analyze(); no second merge or synthetic
/// heap history. Complete input and online operation admission follow MergeTrace.
/// Adds owned O(K+N+events) original/history capture to native heap/output storage.
/// @param lists Borrowed complete nondecreasing original runs.
/// @param unique Deduplicate output on Pop if true, default false.
/// @param max_entries Complete K+N plus operation ceiling, default1000000, zero off.
/// @return Owned complete original inputs, actual operations/work and values.
/// @throws std::invalid_argument On any unsorted row before copying.
/// @throws std::length_error On enabled quota or count/container limits.
/// @throws std::bad_alloc On native/capture allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::k_way_merge_trace({{1}},false,4);
/// std::print("{}\n",t.captured_entries); // 4
/// @endcode
/// @par References
/// - Knuth, The Art of Computer Programming, Volume3: Sorting and Searching, second edition(1998), chapter5(Sorting).
/// @see MergeTrace, k_way_merge_analyze()
MergeTrace k_way_merge_trace(
    const std::vector<std::vector<int>>& lists,
    bool                                 unique      = false,
    std::size_t                          max_entries = 1000000);
} // namespace datamunge::algorithms
