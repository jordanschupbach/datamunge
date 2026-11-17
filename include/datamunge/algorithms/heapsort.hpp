#pragma once
/// @file heapsort.hpp
/// @brief In-place binary heapsort and owned actual swap/heap-boundary traces.
#include <cstddef>
#include <cstdint>
#include <vector>
namespace datamunge::algorithms {
/// @brief Identify an input snapshot, bottom-up build or extraction phase.
/// @par Example
/// @code
/// std::print("{}\n",int(dm::algorithms::HeapSortPhase::Build)); // 1
/// @endcode
/// @see HeapSortSnapshot
enum class HeapSortPhase {
  Input,  ///< Initial unchanged working array.
  Build,  ///< Bottom-up heap construction.
  Extract ///< Root parking and reduced-prefix repair.
};
/// @brief Selected actual sift visits, strict key comparisons and swaps.
/// Excludes child-index tests, allocation, snapshots, validation and byte traffic.
/// A sift visit includes the terminal leaf/order check; root swaps count the
/// extraction exchange separately. These are not elapsed time or total FLOPs.
/// @par Example
/// @code
/// const dm::algorithms::HeapSortWork w;
/// std::print("{}\n",w.root_swaps); // 0
/// @endcode
/// @see HeapSortRun
struct HeapSortWork {
  std::size_t sift_calls{0};      ///< Actual sift_down invocations.
  std::size_t sift_visits{0};     ///< Actual loop entries, including terminal checks.
  std::size_t key_comparisons{0}; ///< Actual existing-child > candidate comparisons.
  std::size_t sift_swaps{0};      ///< Actual exchanges while descending a sift path.
  std::size_t root_swaps{0};      ///< Actual root-to-tail extraction exchanges.
};
/// @brief Complete input cap and complete-snapshot plus online-swap capture policy.
/// Zero disables either optional policy, not count/container representability.
/// Capture units are one per swap record plus1+n per full snapshot (one record
/// and n integer values). Fields, capacity, allocation overhead and ordinary
/// output are excluded; this is not bytes or peak RAM.
/// @par Example
/// @code
/// const dm::algorithms::HeapSortOptions o;
/// std::print("{}\n",o.max_values); // 1000000
/// @endcode
/// @see heapsort_trace()
struct HeapSortOptions {
  std::size_t max_values{1000000};        ///< Complete input integer cap; zero disables.
  std::size_t max_trace_entries{1000000}; ///< Logical capture cap; ignored without tracing.
};
/// @brief Actual exchange recorded immediately before the producing swap.
/// Indices are zero-based; before-values are read from the producing array.
/// Root parking records the old heap size; sift swaps record the reduced active
/// heap size. A swap can exchange equal values and remains an actual exchange.
/// @par Example
/// @code
/// const auto t=dm::algorithms::heapsort_trace({2,2});
/// std::print("{} {}\n",t.swaps[0].left,t.swaps[0].right); // 0 1
/// @endcode
/// @see HeapSortTrace
struct HeapSortSwap {
  HeapSortPhase phase{HeapSortPhase::Build}; ///< Producing build/extraction phase.
  std::size_t   heap_size{0};                ///< Active prefix length at the exchange.
  std::size_t   left{0};                     ///< Actual first exchanged index.
  std::size_t   right{0};                    ///< Actual second exchanged index.
  std::int64_t  left_before{0};              ///< Actual first value before exchange.
  std::int64_t  right_before{0};             ///< Actual second value before exchange.
  bool          root_parking{false};         ///< True only for root-to-tail extraction exchange.
};
/// @brief Owned actual state at input, completed build or completed extraction repair.
/// Build/extraction snapshots have a valid heap in[0,heap_size), and an ascending
/// suffix at least as large as every heap value. Input snapshots need not be heaps.
/// swaps_done ties the snapshot to a prefix of actual swap events, without replay.
/// @par Example
/// @code
/// const auto t=dm::algorithms::heapsort_trace({1,3,2});
/// std::print("{}\n",t.states[1].heap_size); // 3
/// @endcode
/// @see HeapSortTrace
struct HeapSortSnapshot {
  HeapSortPhase phase{HeapSortPhase::Input}; ///< Actual checkpoint phase.
  std::size_t   heap_size{0};                ///< Prefix length; suffix starts at this index.
  std::size_t   swaps_done{0};    ///< Number of producing exchanges preceding this snapshot.
  std::vector<std::int64_t> data; ///< Owned complete actual working values.
};
/// @brief Owned sorted output and actual selected build/extraction work from one execution.
/// Input values are copied once into working/output storage. Arbitrary int64 values
/// and ties are supported, no references retained or shared mutable state.
/// @par Example
/// @code
/// const auto r=dm::algorithms::heapsort_run({3,1,2});
/// std::print("{} {}\n",r.data[0],r.extraction.root_swaps); // 1 2
/// @endcode
/// @see heapsort_run()
struct HeapSortRun {
  std::vector<std::int64_t> data;       ///< Owned actual ascending output.
  HeapSortWork              build;      ///< Actual bottom-up phase work.
  HeapSortWork              extraction; ///< Actual root parking and reduced-prefix repair work.
};
/// @brief Owned actual checkpoints/exchanges and producing result, without a second sort.
/// Always captures input and completed build, including empty/singleton inputs,
/// then n-1 post-repair states for n>=1. Known snapshot units are preflighted
/// before allocation/copy; online whole swap records are admitted before swaps.
/// No truncation or partial result. Copies are deep. Snapshot storage is O(n²)
/// values and events O(n log n), in addition to ordinary O(n) output, unlike the
/// O(1) extra space of the legacy in-place algorithm.
/// @par Example
/// @code
/// const auto t=dm::algorithms::heapsort_trace({2,1});
/// std::print("{} {}\n",t.states.size(),t.swaps.size()); // 3 1
/// @endcode
/// @see HeapSortOptions, heapsort_trace()
struct HeapSortTrace {
  HeapSortRun                   run;    ///< Actual producing sorted result/work.
  std::vector<HeapSortSnapshot> states; ///< Actual chronological complete boundary snapshots.
  std::vector<HeapSortSwap> swaps; ///< Actual chronological exchanges, including equal-valued ones.
  std::size_t               captured_entries{0}; ///< Retained logical units, not bytes or peak RAM.
};
/// @brief Build a binary max-heap in place using bottom-up iterative sift-down.
/// Strict comparisons preserve the old left-before-right candidate ties. Empty
/// and singleton inputs are unchanged. O(n) worst-case work/O(1) extra storage,
/// no allocation, references or shared mutable state. No sorting of the full array
/// is promised. Input supports all int64 values without key arithmetic.
/// @param data Borrowed mutable integer vector, rearranged in place.
/// @par Example
/// @code
/// std::vector<std::int64_t> x{3,7,1,9,2,8};dm::algorithms::build_max_heap(x);
/// std::print("{} {}\n",x[0],dm::algorithms::is_max_heap(x)); // 9 true
/// @endcode
/// @par References
/// - R. W. Floyd (1964), Algorithm245: Treesort3, https://doi.org/10.1145/355588.365103 .
/// @see heapsort(), heapsort_trace()
void build_max_heap(std::vector<std::int64_t>& data);
/// @brief Check every existing parent-child max-heap relation without mutating input.
/// Empty and singleton inputs are heaps. O(n) worst-case/O(1) storage; stops at
/// first violation. No allocation, retained references or shared mutable state.
/// @param data Borrowed arbitrary int64 vector.
/// @return True iff no child exceeds its parent.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::is_max_heap({1,2,3})); // false
/// @endcode
/// @see build_max_heap()
[[nodiscard]] bool is_max_heap(const std::vector<std::int64_t>& data);
/// @brief Sort integers ascending in place using bottom-up heap build then root extraction.
/// Preserves original strict candidate order and swaps, including equal-valued
/// root exchanges. Not stable for labelled equal keys. O(n log n) worst-case
/// time/O(1) extra space; no claim all input families take Theta(n log n), no
/// allocation or optional policy caps. Empty/singleton unchanged. All int64 values
/// supported; no input references retained or shared mutable state.
/// @param data Borrowed mutable integer vector, sorted in place.
/// @par Example
/// @code
/// std::vector<std::int64_t> x{3,1,2};dm::algorithms::heapsort(x);
/// std::print("{} {} {}\n",x[0],x[1],x[2]); // 1 2 3
/// @endcode
/// @par References
/// - J. W. J. Williams (1964), Algorithm232: Heapsort, https://doi.org/10.1145/512274.512284 .
/// - Floyd (1964), https://doi.org/10.1145/355588.365103 .
/// @see build_max_heap(), heapsort_run()
void heapsort(std::vector<std::int64_t>& data);
/// @brief Execute the same heapsort kernel on an owned copy with selected phase work.
/// Complete input cap precedes copying/allocation. No full history, history cap
/// ignored. Same ordering, comparisons and domains as heapsort(). O(n) owned
/// working/output space in addition to constant algorithmic auxiliary storage.
/// @param data Borrowed arbitrary int64 values.
/// @param options Complete integer cap, default1m; zero disables policy.
/// @return Owned actual sorted values and separate build/extraction work.
/// @throws std::length_error On input/count limits.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::heapsort_run({3,1,2}).data[2]); // 3
/// @endcode
/// @see HeapSortRun, heapsort_trace()
HeapSortRun heapsort_run(const std::vector<std::int64_t>& data, HeapSortOptions options = {});
/// @brief Capture actual heapsort boundary states and exchange decisions in their producing pass.
/// Complete known snapshot storage precedes copying; each unknown swap record
/// is admitted before exchange. Same scalar kernel, no replay. Capture failure
/// returns no partial trace and never mutates the caller's input.
/// @param data Borrowed arbitrary int64 values.
/// @param options Complete input and snapshot/online-event caps, default1m each; zero disables.
/// @return Owned actual checkpoints, events, result/work and logical capture count.
/// @throws std::length_error On input/known snapshot/online event/count/container limits.
/// @throws std::bad_alloc On capture/output allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::heapsort_trace({2,1},{2,10});
/// std::print("{}\n",t.captured_entries); // 10
/// @endcode
/// @see HeapSortTrace, HeapSortOptions
HeapSortTrace heapsort_trace(const std::vector<std::int64_t>& data, HeapSortOptions options = {});
} // namespace datamunge::algorithms
