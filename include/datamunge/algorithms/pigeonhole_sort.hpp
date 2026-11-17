#pragma once
/// @file pigeonhole_sort.hpp
/// @brief Checked dense-range integer pigeonholes and actual placement/scan counts.
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {
/// @brief Dense working-range and retained-history policies, independently disabled by zero.
/// @par Example
/// @code
/// const dm::algorithms::PigeonholeOptions o;std::print("{}\n",o.max_holes); // 1000000
/// @endcode
struct PigeonholeOptions {
  std::size_t max_holes{
      1000000}; ///< Maximum allocated dense value buckets, including empty buckets.
  std::size_t max_trace_entries{
      1000000}; ///< Placement records plus retained bucket counts; excludes ordinary output.
};
/// @brief Actual append of one incoming integer to its dense value bucket.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pigeonhole_sort_trace({2,1}).placements[0].hole); // 1
/// @endcode
struct PigeonholePlacement {
  std::size_t input_index; ///< Original zero-based input position.
  int         value;       ///< Actual incoming integer.
  std::size_t hole;        ///< Exact widened value-minus-minimum index.
  std::size_t count_after; ///< Actual bucket size immediately after append.
};
/// @brief Owned sorted output and selected actual producer counts.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pigeonhole_sort_run({2,1}).emitted); // 2
/// @endcode
struct PigeonholeRun {
  std::vector<int> sorted;           ///< Actual ascending emission, preserving multiplicities.
  bool             has_range{false}; ///< False for original empty/singleton early return.
  int              minimum{0};       ///< Actual minimum only when has_range.
  int              maximum{0};       ///< Actual maximum only when has_range.
  std::size_t      holes{0};         ///< Allocated bucket count, including empty buckets.
  std::size_t      placements{0};    ///< Actual input-to-bucket appends; excludes bound searches.
  std::size_t scanned_holes{0}; ///< Actual buckets scanned during emission, including empty ones.
  std::size_t emitted{0};       ///< Actual bucket values emitted; zero for early-return path.
};
/// @brief Owned run and actual incoming placements/final bucket occupancies.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pigeonhole_sort_trace({2,1,2}).occupancies); // [1, 2]
/// @endcode
struct PigeonholeTrace {
  PigeonholeRun                    run;        ///< Ordinary output/counts from the producing run.
  std::vector<PigeonholePlacement> placements; ///< Actual append events, in original input order.
  std::vector<std::size_t> occupancies; ///< Actual final bucket sizes in increasing value order.
};
/// @brief Sort integers by dense buckets and ascending bucket emission.
/// Empty/singleton uses the historical immediate return with no buckets. For
/// larger inputs, R=max-min+1 is computed in widened arithmetic; every value
/// offset is also widened before subtraction. O(n+R) work/storage, including
/// allocating and scanning empty buckets; comparison-based bounds searches are
/// excluded from selected counters. Input is owned by value; lvalues are copied.
/// @param a Integers, including duplicates and signed extremes.
/// @return Owned nondecreasing values; default maximum dense range is one million.
/// @throws std::length_error If default working-range policy or storage limits are exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pigeonhole_sort({3,1,2})); // [1, 2, 3]
/// @endcode
/// @par References
/// - NIST DADS, pigeonhole sort, https://xlinux.nist.gov/dads/HTML/pigeonholeSort.html
std::vector<int> pigeonhole_sort(std::vector<int> a);
/// @brief Run the shared dense-bucket producer with selected actual counts.
/// No historical placements or occupancy copy is retained; ordinary memory O(n+R).
/// @param a Owned integer input.
/// @param options Dense bucket and optional-history policies; history cap is unused here.
/// @return Owned sorted output and actual selected counts; early return has no range metadata.
/// @throws std::length_error If dense working-range policy or storage limits are exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pigeonhole_sort_run({3,1,2}).scanned_holes); // 3
/// @endcode
PigeonholeRun pigeonhole_sort_run(std::vector<int> a, PigeonholeOptions options = {});
/// @brief Capture actual bucket appends and final occupancies without replay.
/// Retains n placement records and R bucket counts beyond ordinary output.
/// @param a Owned integer input, unchanged for lvalue callers.
/// @param options Dense-range and retained-entry policies, admitted before bucket allocation.
/// @return Owned run, actual placements and actual bucket sizes.
/// @throws std::length_error If working/history policies or storage limits are exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pigeonhole_sort_trace({3,1,3}).occupancies); // [1, 0, 2]
/// @endcode
PigeonholeTrace pigeonhole_sort_trace(std::vector<int> a, PigeonholeOptions options = {});
} // namespace datamunge::algorithms
