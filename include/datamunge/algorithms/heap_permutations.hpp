#pragma once
/// @file heap_permutations.hpp
/// @brief Checked materialized Heap permutations with actual recursive swap histories.
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {
/// @brief Checked exact logical size of a complete Heap enumeration, without generation.
/// Counts include one empty permutation for n=0. Copied values exclude the
/// initial working copy and snapshots; recursive calls include the initial call.
/// This is a count model, not bytes, peak RAM, timing or an allocation guarantee.
/// @par Example
/// @code
/// const auto s=dm::algorithms::heap_permutation_size(4);
/// std::print("{} {}\n",s.permutations,s.copied_values); // 24 96
/// @endcode
/// @see heap_permutation_size(), HeapPermutationOptions
struct HeapPermutationSize {
  std::size_t permutations{1};    ///< Complete n! rows, including one empty row at n=0.
  std::size_t copied_values{0};   ///< Complete n*n! output integer values.
  std::size_t recursive_calls{1}; ///< Complete producing calls, C0=C1=1, Cn=1+n*C(n-1).
};
/// @brief Complete output/history policy caps, each zero disabling that policy.
/// Output values count integers across all rows, not bytes or peak memory.
/// Trace entries count swap records only; ordinary output/fields/capacity and
/// allocation overhead are excluded. All known requirements precede copies and swaps.
/// @par Example
/// @code
/// const dm::algorithms::HeapPermutationOptions o;
/// std::print("{}\n",o.max_values); // 1000000
/// @endcode
/// @see heap_permutations_trace(), HeapPermutationSize
struct HeapPermutationOptions {
  std::size_t max_permutations{1000000};  ///< Complete row cap; zero disables policy.
  std::size_t max_values{1000000};        ///< Complete output-integer cap; zero disables policy.
  std::size_t max_trace_entries{1000000}; ///< Complete swap-record cap; ignored without capture.
};
/// @brief Selected actual producing recursive calls, emissions, swaps and output value copies.
/// Excludes validation, reserve/allocation, initial working copy and trace writes.
/// Equal-valued swaps still count as actual swaps, although values do not change.
/// Counts are not FLOPs, byte traffic or elapsed time.
/// @par Example
/// @code
/// const auto r=dm::algorithms::heap_permutations_run({1,2,3});
/// std::print("{} {}\n",r.work.swaps,r.work.copied_values); // 5 18
/// @endcode
/// @see HeapPermutationRun
struct HeapPermutationWork {
  std::size_t recursive_calls{0}; ///< Actual invocations including base calls.
  std::size_t emissions{0};       ///< Actual complete row insertions.
  std::size_t swaps{0};           ///< Actual std::swap calls between emissions.
  std::size_t even_swaps{0};      ///< Swaps at an even-sized recursive prefix.
  std::size_t odd_swaps{0};       ///< Swaps at an odd-sized recursive prefix.
  std::size_t copied_values{0};   ///< Actual integer values copied into output rows.
};
/// @brief Actual swap event preparing the next emitted row, with zero-based indices.
/// before values come directly from the producing array immediately before the
/// swap. No pre/post state replay or copied snapshots; ordinary output owns rows.
/// @par Example
/// @code
/// const auto t=dm::algorithms::heap_permutations_trace({1,2,3});
/// std::print("{} {}\n",t.swaps[0].left,t.swaps[0].right); // 0 1
/// @endcode
/// @see HeapPermutationTrace
struct HeapPermutationSwap {
  std::size_t output_index{0}; ///< Zero-based next output row, at least1.
  std::size_t prefix_size{0};  ///< Producing recursive prefix k, at least2.
  std::size_t block{0};        ///< Zero-based completed (k-1)-prefix block index i.
  std::size_t left{0};         ///< Actual swap partner: i for even k, otherwise0.
  std::size_t right{0};        ///< Actual other swap index k-1.
  int         left_value{0};   ///< Actual value at left before swap.
  int         right_value{0};  ///< Actual value at right before swap.
};
/// @brief Owned complete rows in legacy Heap order with selected producing work.
/// Arbitrary integers and repeated values are allowed. Positions are permuted,
/// so equal values produce repeated rows; no sorting or deduplication occurs.
/// Input and output rows own separate storage, including the single empty row.
/// @par Example
/// @code
/// const auto r=dm::algorithms::heap_permutations_run({1,1,2});
/// std::print("{}\n",r.permutations.size()); // 6
/// @endcode
/// @see heap_permutations_run()
struct HeapPermutationRun {
  std::vector<std::vector<int>> permutations; ///< Owned all n! rows, duplicates retained.
  HeapPermutationWork           work;         ///< Selected actual producing work.
};
/// @brief Owned complete rows and actual swap records captured in their producing execution.
/// Complete n!-1 records are preflighted before any working copy, allocation or
/// swap. No replay, truncation or partial result; copying is deep. Ordinary
/// output has O(n*n!) values plus O(n!) row objects; capture adds O(n!) records.
/// @par Example
/// @code
/// const auto t=dm::algorithms::heap_permutations_trace({1,2,3});
/// std::print("{} {}\n",t.run.permutations.size(),t.swaps.size()); // 6 5
/// @endcode
/// @see heap_permutations_trace(), HeapPermutationOptions
struct HeapPermutationTrace {
  HeapPermutationRun               run;   ///< Actual producing rows and work.
  std::vector<HeapPermutationSwap> swaps; ///< Actual chronological swap events.
};
/// @brief Compute checked n!, n*n! and recursive-call requirements without allocation.
/// C0=C1=1 and Cn=1+n*C(n-1). Large n fails at the first overflowing recurrence;
/// no loop over an unrepresentably large n completes. No container/policy checks
/// or generation are performed; representable counts do not guarantee allocability.
/// @param n Number of input positions.
/// @return Owned exact size_t counts.
/// @throws std::length_error On any count overflow.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::heap_permutation_size(3).recursive_calls); // 10
/// @endcode
/// @see HeapPermutationSize, heap_permutations_run()
HeapPermutationSize heap_permutation_size(std::size_t n);
/// @brief Materialize every positional permutation in the preserved recursive Heap order.
/// Even prefix k swaps i with k-1 after block i; odd k swaps0 with k-1.
/// One swap separates outputs, n!-1 total. With distinct values the rows are
/// unique; repeated values can produce identical consecutive rows. Empty input
/// yields one empty row. No sorting, deduplication, input mutation or references.
/// Checks exact counts/container sizes before copies. No optional policy caps;
/// allocation failure remains possible. Total time/storage O(n*n!+n!) including
/// row materialization, with O(n) working array and recursion for n>=1. Independent
/// calls share no mutable state. This is not the heapsort algorithm.
/// @param items Borrowed arbitrary integer values, interpreted as distinct positions.
/// @return Owned n! rows in legacy order, duplicates preserved.
/// @throws std::length_error On count/container representability.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::heap_permutations({1,2,3,4}).size()); // 24
/// @endcode
/// @par References
/// - B. R. Heap (1963), Permutations by Interchanges, The Computer Journal6,293–298,
///   https://doi.org/10.1093/comjnl/6.3.293 .
/// @see heap_permutations_run(), heap_permutations_trace()
std::vector<std::vector<int>> heap_permutations(const std::vector<int>& items);
/// @brief Execute the shared Heap kernel with complete output caps and actual work counts.
/// Same positional order and domains as heap_permutations(). Complete checked
/// output limits precede working/output allocation. History cap is ignored here.
/// @param items Borrowed arbitrary integers, repeated values allowed.
/// @param options Complete row/value caps, default1m each; zero disables either policy.
/// @return Owned actual complete rows and selected producing counts.
/// @throws std::length_error On complete output policy/count/container limits.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto r=dm::algorithms::heap_permutations_run({}, {1,0,0});
/// std::print("{} {}\n",r.permutations.size(),r.work.swaps); // 1 0
/// @endcode
/// @see HeapPermutationRun, heap_permutations_trace()
HeapPermutationRun heap_permutations_run(
    const std::vector<int>& items,
    HeapPermutationOptions  options = {});
/// @brief Capture actual recursive swap events without a second execution or snapshots.
/// Same output contracts as heap_permutations_run(), plus complete n!-1 history
/// admission before any allocation/copy/swap. Failure returns no partial result.
/// A history entry is one record, excluding ordinary output/fields/capacity/bytes.
/// @param items Borrowed arbitrary integers, repeated values allowed.
/// @param options Complete row/value/history caps, default1m each; zero disables either policy.
/// @return Owned complete producing run and actual chronological events.
/// @throws std::length_error On complete policy/count/container limits.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::heap_permutations_trace({1,2,3},{6,18,5});
/// std::print("{}\n",t.swaps.size()); // 5
/// @endcode
/// @see HeapPermutationTrace, heap_permutations_run()
HeapPermutationTrace heap_permutations_trace(
    const std::vector<int>& items,
    HeapPermutationOptions  options = {});
} // namespace datamunge::algorithms
