#pragma once
/// @file
/// @brief Write-minimal integer cycle sorting, actual logical work and owned placement traces.
#include <cstddef>
#include <vector>

namespace datamunge::algorithms {
/// @brief Owned sorted values and actual operations from the canonical cycle-sort kernel.
/// array_writes counts assignments to working value-array cells only. It excludes
/// argument/output copying, the held item, counters, trace identities, allocation
/// and physical device writes. Rank comparisons count strict less-than tests;
/// equality comparisons include the final false duplicate-skip test. Loop/index
/// tests are excluded. Empty/singleton inputs have zero counts. Values and counts
/// are owned; no input references or shared mutable state survive.
/// @par Example
/// @code
/// const auto w=dm::algorithms::cycle_sort_work({3,1,2});
/// std::print("{} {} {} {}\n",w.output,w.array_writes,w.rank_comparisons,w.equality_comparisons); // [1, 2, 3] 3 7 3
/// @endcode
/// @see cycle_sort_work(), CycleSortTrace
struct CycleSortWork {
  std::vector<int> output; ///< Owned nondecreasing values, preserving multiplicities.
  std::size_t array_writes{0}; ///< Actual working value-cell assignments; at most n.
  std::size_t rank_comparisons{0}; ///< Actual suffix key<held-item evaluations.
  std::size_t equality_comparisons{0}; ///< Actual candidate key==held-item evaluations, true or false.
  std::size_t duplicate_skips{0}; ///< True equality tests that advance a destination.
  std::size_t rank_scans{0}; ///< Complete suffix scans, including repeated scans within cycles.
  std::size_t outer_starts{0}; ///< n-1 examined starts for n>=2, otherwise zero.
  std::size_t cycle_count{0}; ///< Opened nontrivial rotations; correct starts open no cycle.
};

/// @brief One actual final-cell placement in a cycle rotation.
/// Each target occurs once in the complete trace. A placement stores the held
/// value and its displaced value/identity before the value-cell write. The
/// displaced item becomes the next held item until target==cycle_start closes
/// the cycle; that final displaced duplicate of the original start is discarded.
/// rank_begin is the first destination for this key in the remaining suffix;
/// equal values may advance target. No full array snapshot is stored.
/// @par Example
/// @code
/// const auto t=dm::algorithms::cycle_sort_trace({2,2,1});
/// std::print("{} {} {}\n",t.placements[0].rank_begin,t.placements[0].target,t.placements[0].duplicate_skips); // 1 2 1
/// @endcode
/// @see CycleSortTrace, cycle_sort_trace()
struct CycleSortPlacement {
  std::size_t cycle_start{0}; ///< Outer start whose active rotation owns this write.
  std::size_t rank_begin{0}; ///< start plus actual number of smaller suffix values.
  std::size_t target{0}; ///< Destination after skipping already-correct equal values.
  int value{0}; ///< Held value assigned to this target.
  int displaced_value{0}; ///< Value in that cell immediately before this write.
  std::size_t original_index{0}; ///< Input identity of the placed held item.
  std::size_t displaced_original_index{0}; ///< Input identity becoming held after this write.
  std::size_t duplicate_skips{0}; ///< Equal candidates skipped during this placement's search.
  std::size_t rank_comparisons{0}; ///< Suffix comparisons in this placement's actual rank scan.
  bool closes_cycle{false}; ///< True precisely when target equals cycle_start.
};

/// @brief Owned input, actual aggregate work and complete ordered placement records.
/// Original identities travel with the same moves as values, exposing instability
/// on equal keys. Final original_indices is output-to-input; intermediate array
/// states must be interpreted with the held item because a cycle's start remains
/// unchanged until closure. Empty/singleton traces have no placements and identity
/// final indices. At most n records and O(n) extra storage, with no snapshots.
/// @par Example
/// @code
/// const auto t=dm::algorithms::cycle_sort_trace({2,2,1});
/// std::print("{} {} {}\n",t.work.output,t.original_indices,t.placements.size()); // [1, 2, 2] [2, 1, 0] 2
/// @endcode
/// @see cycle_sort_trace(), CycleSortPlacement
struct CycleSortTrace {
  std::vector<int> input; ///< Owned original values, unchanged throughout sorting.
  CycleSortWork work; ///< Actual canonical sorted output and operation counts.
  std::vector<std::size_t> original_indices; ///< Final input identities, possibly unstable within equal keys.
  std::vector<CycleSortPlacement> placements; ///< Complete actual writes in execution order.
};

/// @brief Return a sorted integer vector using write-minimal rank-scan cycle rotations.
/// Retains the legacy by-value signature. The working vector is sorted in place
/// with O(1) auxiliary storage; an lvalue argument is copied by the call. Every
/// changed value-cell is written once, already-correct value-cells remain intact,
/// and writes equal the Hamming distance to the unique nondecreasing key sequence.
/// Equal-key records need not be stable. Signed extremes are valid; keys are
/// compared without subtraction. For n>=2, even sorted/all-equal inputs require
/// Theta(n^2) rank comparisons. No random/shared state or input size cap. The
/// cycle-placement idea is related to Haddon, but repeated rank scans here are
/// quadratic, rather than the paper's conditional linear-time variants.
/// @param a Owned/by-value integer vector; empty/singleton is returned unchanged.
/// @return Owned nondecreasing values, preserving every multiplicity.
/// @throws std::bad_alloc By-value copy allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::cycle_sort({5,3,8,1,9,2,7})); // [1, 2, 3, 5, 7, 8, 9]
/// @endcode
/// @par References
/// - B. K. Haddon (1990), Cycle-Sort: A Linear Sorting Method, The Computer Journal 33(4), 365–367, doi:10.1093/comjnl/33.4.365; this API uses repeated comparisons to find ranks.
/// @see cycle_sort_work(), cycle_sort_trace(), cycle_sort_write_lower_bound()
std::vector<int> cycle_sort(std::vector<int> a);

/// @brief Sort and count the actual canonical cycle-sort operations without replay.
/// Same placements and by-value ownership as cycle_sort(), with O(1) auxiliary
/// storage beyond the working vector. No sorted reference, instrumentation copy
/// of the algorithm or trace snapshots are used. Counts distinguish suffix
/// less-than tests from duplicate equality tests and logical value-cell writes.
/// A held-item swap counts one array write. All aggregate increments are checked.
/// @param a By-value integer values; all int values and repetitions are valid.
/// @return Owned sorted output and checked actual work, with zero work for n<=1.
/// @throws std::length_error Any aggregate operation count would overflow size_t.
/// @throws std::bad_alloc By-value copy allocation failure.
/// @par Example
/// @code
/// const auto w=dm::algorithms::cycle_sort_work({5,3,8,1,9,2,7});
/// std::print("{} {} {}\n",w.array_writes,w.rank_comparisons,w.cycle_count); // 7 39 3
/// @endcode
/// @par References
/// - B. K. Haddon (1990), doi:10.1093/comjnl/33.4.365; comparison-based rank-scan variant.
/// @see CycleSortWork, cycle_sort(), cycle_sort_trace()
CycleSortWork cycle_sort_work(std::vector<int> a);

/// @brief Capture the complete actual canonical cycle sort without replay or snapshots.
/// Same integer/key/tie semantics as cycle_sort(). Owns two value vectors, final
/// identities and at most n placement records: O(n) storage and Theta(n^2) time.
/// No references survive. A complete element cap is checked before any copy,
/// identity allocation or sorting; it bounds both history and quadratic work.
/// Default 4096; zero disables only this optional cap. Even a sorted oversized
/// input rejects. Results never silently truncate. Identity bookkeeping and
/// result copies are deliberately excluded from the logical array-write count.
/// @param values Read-only integer keys; empty/singleton is valid.
/// @param max_elements Complete input cap, default 4096; zero disables.
/// @return Owned input, sorted result, actual identity order and every placement.
/// @throws std::length_error Optional complete cap, container capacity or aggregate count exceeded.
/// @throws std::bad_alloc Input/output/identity/history allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::cycle_sort_trace({3,1,2},3);
/// std::print("{} {}\n",t.placements[0].target,t.placements.back().closes_cycle); // 2 true
/// @endcode
/// @par References
/// - B. K. Haddon (1990), doi:10.1093/comjnl/33.4.365; rank-scan variant with actual placement capture.
/// @see CycleSortTrace, CycleSortPlacement, cycle_sort_work()
CycleSortTrace cycle_sort_trace(const std::vector<int>& values, std::size_t max_elements=4096);

/// @brief Compute the minimum logical value-cell writes needed for nondecreasing keys.
/// Makes a private std::sort reference, then counts positions whose input key
/// differs from the sorted key. Every such cell must change; cycle_sort attains
/// this bound, including duplicate keys, by leaving correct equal-value cells
/// alone. The bound does not impose stable record identities and excludes
/// argument copies, temporaries, allocation and device erase/write granularity.
/// O(n log(n+1)) time and O(n) owned temporary storage; input read-only, no shared
/// state. This independent reference computation is not part of the sorting
/// kernel's measured work. Empty/singleton/already-sorted inputs return zero.
/// @param values Original integer keys of any length representable by a vector.
/// @return Hamming distance from values to its nondecreasing key sequence, at most n.
/// @throws std::bad_alloc Reference copy allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::cycle_sort_write_lower_bound({2,2,1})); // 2
/// @endcode
/// @see cycle_sort(), cycle_sort_work()
std::size_t cycle_sort_write_lower_bound(const std::vector<int>& values);
} // namespace datamunge::algorithms
