#pragma once
/// @file insertion_sort.hpp
/// @brief Stable integer insertion sorting with actual shifts, comparisons and owned pass states.
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {
/// @brief Inclusive complete element/history policies, zero disables each optional cap.
/// Each captured insertion reserves 1+2*n logical entries, one record plus n
/// values and n original identities. Complete (n-1)*(1+2*n) admission precedes
/// copying/sorting. Excludes original/working/final arrays, scalar metadata,
/// vector capacity and byte/peak RAM; untraced calls ignore the history cap.
/// @par Example
/// @code
/// const dm::algorithms::InsertionSortOptions o;
/// std::print("{}\n",o.max_elements); // 1000000
/// @endcode
/// @see insertion_sort_trace()
struct InsertionSortOptions {
  std::size_t max_elements{1000000}; ///< Complete input count; zero disables.
  std::size_t max_history_entries{
      1000000}; ///< Complete insertion snapshot logical-entry cap; zero disables.
};
/// @brief Selected actual key tests and array-cell assignments, not elapsed time.
/// Key comparisons include the final false key test when j>0; index-bound tests
/// are excluded. Shifts count successful > tests and rightward cell writes.
/// Placements include unchanged destinations. Input copies, identity bookkeeping,
/// key temporary reads, allocation and observations are excluded.
/// @par Example
/// @code
/// const auto r=dm::algorithms::insertion_sort_run({2,1});
/// std::print("{} {} {}\n",r.work.comparisons,r.work.shifts,r.work.placements); // 1 1 1
/// @endcode
/// @see insertion_sort_run()
struct InsertionSortWork {
  std::size_t comparisons{0}; ///< Actual strict larger-than key comparisons.
  std::size_t shifts{0};      ///< Actual predecessor-to-right value writes.
  std::size_t placements{0};  ///< Actual final key writes, one per insertion.
};
/// @brief Owned stable sorted values, corresponding original indices and producing work.
/// Identities distinguish equal records; equal keys preserve original index order.
/// No input mutation, retained references or shared state. Value-only legacy calls
/// omit identity/history storage while sharing the same executing loop.
/// @par Example
/// @code
/// const auto r=dm::algorithms::insertion_sort_run({2,1,2});
/// std::print("{}\n",r.identities); // [1, 0, 2]
/// @endcode
/// @see insertion_sort_run()
struct InsertionSortRun {
  std::vector<int>         values;     ///< Nondecreasing owned values.
  std::vector<std::size_t> identities; ///< Corresponding original zero-based input positions.
  InsertionSortWork        work;       ///< Actual selected producing counts.
};
/// @brief Actual completed insertion and its owned working-array snapshot.
/// Each step follows exactly one insertion, including zero-shift insertions;
/// no repeated-prefix sorting is used to reconstruct a state.
/// @par Example
/// @code
/// const auto t=dm::algorithms::insertion_sort_trace({2,1});
/// std::print("{} {}\n",t.steps[0].index,t.steps[0].destination); // 1 0
/// @endcode
/// @see insertion_sort_trace()
struct InsertionSortStep {
  std::size_t      index{0};        ///< Actual outer insertion position, from one to n-1.
  int              key{0};          ///< Actual key held during this insertion.
  std::size_t      key_identity{0}; ///< Original identity of the held key.
  std::size_t      destination{0};  ///< Actual final key destination.
  std::size_t      comparisons{0};  ///< Actual key tests for this insertion.
  std::size_t      shifts{0};       ///< Actual shifts for this insertion, index-destination.
  std::vector<int> values;          ///< Actual complete working values after placement.
  std::vector<std::size_t>
      identities; ///< Actual complete working original identities after placement.
};
/// @brief Owned original input and actual insertion states from one execution.
/// Empty/singleton inputs have no insertion records. The original input supplies
/// the initial state; no synthetic initial insertion is added.
/// @par Example
/// @code
/// const auto t=dm::algorithms::insertion_sort_trace({3,1,2});
/// std::print("{} {}\n",t.original,t.run.values); // [3, 1, 2] [1, 2, 3]
/// @endcode
/// @see insertion_sort_trace()
struct InsertionSortTrace {
  InsertionSortRun               run;      ///< Final state from this same producing execution.
  std::vector<int>               original; ///< Owned complete unchanged input.
  std::vector<InsertionSortStep> steps;    ///< Actual completed insertion snapshots in order.
};
/// @brief Return a stable nondecreasing integer copy by insertion sorting.
/// Original signature retained. The by-value API copies lvalue inputs: only the
/// internal working vector sorts in place. Strict > shifts preserve ties. Empty
/// and singleton inputs return unchanged. O(n+I) body work for I strict inversions,
/// worst O(n²); O(1) body workspace excluding the owned input/output. Shares the
/// actual loop with run/trace, but omits identities/history and optional caps.
/// @param a Owned input, copied by the call for lvalues and moved when supplied as rvalue.
/// @return Owned sorted values with multiplicities preserved.
/// @throws std::length_error On counter/capacity exhaustion.
/// @throws std::bad_alloc On input allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::insertion_sort({3,1,2})); // [1, 2, 3]
/// @endcode
/// @par References
/// - NIST Dictionary of Algorithms and Data Structures, insertion sort,
///   https://xlinux.nist.gov/dads/HTML/insertionSort.html.
/// @see insertion_sort_run(), insertion_sort_trace()
std::vector<int> insertion_sort(std::vector<int> a);
/// @brief Sort once with stable original identities and actual selected work.
/// Same strict-shift loop as insertion_sort(); O(n+I) body time, O(n) owned
/// working identity/value arrays. Complete input cap precedes copying; history
/// cap ignored. No mutation, retained references or shared state.
/// @param a Borrowed complete integer input, empty allowed.
/// @param options Complete policies, default million entries each; zero disables caps.
/// @return Owned stable sorted values/identities and actual work.
/// @throws std::length_error On element cap, count overflow or capacity exhaustion.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto r=dm::algorithms::insertion_sort_run({3,1,2});
/// std::print("{} {}\n",r.work.comparisons,r.work.shifts); // 3 2
/// @endcode
/// @par References
/// - NIST, insertion sort, https://xlinux.nist.gov/dads/HTML/insertionSort.html.
/// @see insertion_sort(), insertion_sort_trace()
InsertionSortRun insertion_sort_run(const std::vector<int>& a, InsertionSortOptions options = {});
/// @brief Capture actual complete insertion snapshots from the shared producing loop.
/// Same result/work as insertion_sort_run(), with O(n²) additional snapshot
/// storage/time. Complete trace admission before copies/sorting uses n-1 records
/// for n>=1, zero for empty. No partial result, input mutation, retained references
/// or shared state. History budget is logical entries, not bytes or peak memory.
/// @param a Borrowed complete integer input, empty allowed.
/// @param options Complete element/history policies, default million each; zero disables caps.
/// @return Owned original input, actual steps and final stable result/work.
/// @throws std::length_error On complete element/history cap, count overflow or capacity exhaustion.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::insertion_sort_trace({3,1,2}).steps.size()); // 2
/// @endcode
/// @par References
/// - NIST, insertion sort, https://xlinux.nist.gov/dads/HTML/insertionSort.html.
/// @see insertion_sort(), insertion_sort_run(), InsertionSortOptions
InsertionSortTrace insertion_sort_trace(
    const std::vector<int>& a,
    InsertionSortOptions    options = {});
} // namespace datamunge::algorithms
