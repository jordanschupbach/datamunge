#pragma once
/// @file counting_sort.hpp
/// @brief Stable bounded-integer sorting, owned permutations and actual placement traces.
#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::algorithms {

/// @brief Inclusive maximum bucket count retained by the legacy sorting overloads.
/// One billion size_t counters alone occupy about 8 GB on a 64-bit platform;
/// this compatibility ceiling is not a promise that allocation is practical.
/// Diagnostic calls additionally default to a one-million-bucket budget.
inline constexpr std::int64_t kCountingSortMaxRange = 1'000'000'000;

/// @brief Logical operations actually performed by the counting-sort kernel.
/// Excludes validation, allocation/zeroing, copies, cursor decrements, loop tests
/// and hardware timings. Range discovery examines n items only in automatic
/// calls; count/prefix/place work is n,k,n for nonempty accepted input.
/// @par Example
/// @code
/// const auto a=dm::algorithms::counting_sort_analyze({2,0,2});
/// std::print("{} {} {}\n",a.work.count_increments,a.work.prefix_steps,a.work.placements); // 3 3 3
/// @endcode
/// @see CountingSortAnalysis
struct CountingSortWork {
  std::size_t range_items=0; ///< Items visited by automatic range discovery; zero for explicit ranges.
  std::size_t count_increments=0; ///< Actual histogram increments, one per input item.
  std::size_t prefix_steps=0; ///< Actual bucket-prefix iterations, including empty buckets.
  std::size_t placements=0; ///< Actual reverse-scan output assignments, one per input item.
};

/// @brief Owned sorted values, stable original identities and bucket boundaries.
/// For nonempty input, bucket b denotes min_value+b. Its output interval is
/// [starts[b],ends[b]); counts[b] is its length. original_indices[j] identifies
/// the input item copied to output[j], and increases within each equal-key block.
/// Empty calls allocate no buckets: automatic bounds are both zero, explicit
/// bounds retain the supplied values. No references or shared mutable state survive.
/// @par Example
/// @code
/// const auto a=dm::algorithms::counting_sort_analyze({2,0,2});
/// std::print("{} {}\n",a.output,a.original_indices); // [0, 2, 2] [1, 0, 2]
/// @endcode
/// @par References
/// - Pat Morin, Open Data Structures (C++), Section 11.2: https://opendatastructures.org/ods-cpp/11_2_Counting_Sort_Radix_So.html
/// @see counting_sort_analyze(), CountingSortTrace
struct CountingSortAnalysis {
  std::int64_t min_value=0; ///< Inclusive lower key bound; zero for automatic empty input.
  std::int64_t max_value=0; ///< Inclusive upper key bound; zero for automatic empty input.
  std::vector<std::int64_t> output; ///< Owned ascending keys.
  std::vector<std::size_t> original_indices; ///< Output-to-input permutation preserving equal-key order.
  std::vector<std::size_t> counts; ///< Owned histogram before prefix summation.
  std::vector<std::size_t> starts; ///< Owned zero-based first output slots, including zero-width blocks.
  std::vector<std::size_t> ends; ///< Owned initial one-past-end cursors, before placement decrements.
  CountingSortWork work; ///< Actual logical work for this call.
};

/// @brief One actual right-to-left assignment using a predecremented end cursor.
/// Events appear in execution order, with input_index descending. The chosen
/// output slot is cursor_before-1. No snapshot or report-side replay is needed.
/// @par Example
/// @code
/// const auto t=dm::algorithms::counting_sort_trace({2,0,2},0,2);
/// const auto& p=t.placements.front();
/// std::print("{} {} {}\n",p.input_index,p.cursor_before,p.output_index); // 2 3 2
/// @endcode
/// @par References
/// - Pat Morin, Open Data Structures (C++), Section 11.2: https://opendatastructures.org/ods-cpp/11_2_Counting_Sort_Radix_So.html
/// @see CountingSortTrace
struct CountingSortPlacement {
  std::size_t input_index=0; ///< Original input position being copied.
  std::int64_t key=0; ///< Actual input key, without packed arrival indices.
  std::size_t bucket=0; ///< Unsigned key offset from the inclusive lower bound.
  std::size_t cursor_before=0; ///< One-past-last-free slot immediately before decrement.
  std::size_t output_index=0; ///< Actual assigned output slot after decrement.
};

/// @brief Complete owned counting-sort analysis and actual placement events.
/// The same kernel drives legacy sorting, analysis and tracing. Storage and
/// time are O(n+k), including n events; no partial trace is returned on failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::counting_sort_trace({1,0,1},0,1);
/// std::print("{} {}\n",t.analysis.original_indices,t.placements.size()); // [1, 0, 2] 3
/// @endcode
/// @par References
/// - Pat Morin, Open Data Structures (C++), Section 11.2: https://opendatastructures.org/ods-cpp/11_2_Counting_Sort_Radix_So.html
/// @see counting_sort_trace(), CountingSortPlacement
struct CountingSortTrace {
  CountingSortAnalysis analysis; ///< Owned output, identities, bucket boundaries and work.
  std::vector<CountingSortPlacement> placements; ///< Exactly n events in actual reverse-scan order.
};

/// @brief Stably sort signed 64-bit keys ascending, discovering their range.
/// Replaces data only after successful completion; empty input is unchanged.
/// O(n+k) time/space, k=max-min+1, with the legacy billion-bucket ceiling.
/// Equal-key placement is shared with the diagnostic permutation kernel.
/// Calls on independent inputs are thread-safe; no state is retained.
/// @param data Mutable keys, replaced by owned sorted storage on success.
/// @throws std::length_error If nonempty range exceeds kCountingSortMaxRange or container limits.
/// @throws std::bad_alloc If temporary storage cannot be allocated.
/// @par Example
/// @code
/// std::vector<std::int64_t> a{2,-1,2};
/// dm::algorithms::counting_sort(a);
/// std::print("{}\n",a); // [-1, 2, 2]
/// @endcode
/// @par References
/// - Pat Morin, Open Data Structures (C++), Section 11.2: https://opendatastructures.org/ods-cpp/11_2_Counting_Sort_Radix_So.html
/// @see counting_sort_analyze(), counting_sort_trace()
void counting_sort(std::vector<std::int64_t>& data);

/// @brief Stably sort signed 64-bit keys within a supplied inclusive range.
/// Same kernel, complexity and strong exception guarantee as automatic sorting.
/// Inverted bounds always fail; valid empty input skips span checks/allocation.
/// Nonempty keys are fully validated before bucket allocation. Offsets use
/// unsigned subtraction, so narrow ranges near either int64 endpoint are safe.
/// @param data Mutable keys, replaced only after successful sorting.
/// @param min_value Inclusive lower key bound.
/// @param max_value Inclusive upper key bound.
/// @throws std::invalid_argument If bounds invert or any key lies outside them.
/// @throws std::length_error If nonempty range exceeds kCountingSortMaxRange or container limits.
/// @throws std::bad_alloc If temporary storage cannot be allocated.
/// @par Example
/// @code
/// std::vector<std::int64_t> a{1,0,1};
/// dm::algorithms::counting_sort(a,0,2);
/// std::print("{}\n",a); // [0, 1, 1]
/// @endcode
/// @par References
/// - Morin, Open Data Structures, Section 11.2 (see automatic overload).
/// @see counting_sort_analyze(), counting_sort_trace()
void counting_sort(std::vector<std::int64_t>& data, std::int64_t min_value, std::int64_t max_value);

/// @brief Sort keys with owned stable identities and an automatically discovered histogram.
/// Leaves input untouched. Uses the same reverse-placement kernel as counting_sort.
/// O(n+k) time/space; range discovery visits n items. Empty input returns zero
/// bounds and empty arrays. Thread-safe for independent calls; all storage is owned.
/// @param data Read-only signed integer keys.
/// @param max_buckets Complete bucket budget, default 1000000; zero disables only this optional cap.
/// @return Owned sorted keys, output-to-input permutation and actual bucket/work information.
/// @throws std::length_error If nonempty range exceeds the optional budget, hard ceiling or container limits.
/// @throws std::bad_alloc If owned storage cannot be allocated.
/// @par Example
/// @code
/// const auto a=dm::algorithms::counting_sort_analyze({2,-1,2});
/// std::print("{} {}\n",a.output,a.original_indices); // [-1, 2, 2] [1, 0, 2]
/// @endcode
/// @par References
/// - Pat Morin, Open Data Structures (C++), Section 11.2: https://opendatastructures.org/ods-cpp/11_2_Counting_Sort_Radix_So.html
/// @see CountingSortAnalysis, counting_sort_trace()
CountingSortAnalysis counting_sort_analyze(const std::vector<std::int64_t>& data, std::size_t max_buckets=1'000'000);

/// @brief Analyze stable sorting with caller-supplied inclusive key bounds.
/// Same ownership and O(n+k) contract as automatic analysis, without range
/// discovery. Validates all keys before allocation and optional budget checks.
/// Valid empty input retains bounds but has no buckets; inverted bounds fail.
/// @param data Read-only signed integer keys.
/// @param min_value Inclusive lower bound.
/// @param max_value Inclusive upper bound.
/// @param max_buckets Complete bucket budget, default 1000000; zero disables only this optional cap.
/// @return Owned ascending keys, stable original identities and actual bucket/work information.
/// @throws std::invalid_argument If bounds invert or any key lies outside them.
/// @throws std::length_error If nonempty range exceeds the optional budget, hard ceiling or container limits.
/// @throws std::bad_alloc If owned storage cannot be allocated.
/// @par Example
/// @code
/// const auto a=dm::algorithms::counting_sort_analyze({2,0,2},0,3);
/// std::print("{} {}\n",a.counts,a.ends); // [1, 0, 2, 0] [1, 1, 3, 3]
/// @endcode
/// @par References
/// - Morin, Open Data Structures, Section 11.2 (see automatic overload).
/// @see CountingSortAnalysis, counting_sort_trace()
CountingSortAnalysis counting_sort_analyze(const std::vector<std::int64_t>& data, std::int64_t min_value, std::int64_t max_value, std::size_t max_buckets=1'000'000);

/// @brief Capture complete actual reverse placements without replay or packed keys.
/// Explicit-range analysis plus n owned events, O(n+k) time/space. Input remains
/// unchanged and the result retains no references. Full bounds/key validation
/// precedes budget checks and allocation. Independent calls are thread-safe.
/// @param data Read-only keys, visited right to left during placement.
/// @param min_value Inclusive lower bound.
/// @param max_value Inclusive upper bound.
/// @param max_buckets Complete bucket budget, default 1000000; zero disables this optional cap.
/// @param max_placements Complete event budget, default 100000; zero disables this optional cap.
/// @return Owned analysis and exactly data.size() actual events; empty input has none.
/// @throws std::invalid_argument If bounds invert or any key lies outside them, even when over budget.
/// @throws std::length_error If a complete budget, hard bucket ceiling or container limit is exceeded.
/// @throws std::bad_alloc If owned storage cannot be allocated.
/// @par Example
/// @code
/// const auto t=dm::algorithms::counting_sort_trace({2,0,2},0,2,3,3);
/// std::print("{} {}\n",t.analysis.original_indices,t.placements.back().output_index); // [1, 0, 2] 1
/// @endcode
/// @par References
/// - Morin, Open Data Structures, Section 11.2 (see counting_sort_analyze()).
/// @see CountingSortTrace, CountingSortPlacement
CountingSortTrace counting_sort_trace(const std::vector<std::int64_t>& data, std::int64_t min_value, std::int64_t max_value, std::size_t max_buckets=1'000'000, std::size_t max_placements=100'000);

} // namespace datamunge::algorithms
