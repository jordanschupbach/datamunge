#pragma once
/// @file bucket_sort.hpp
/// @brief Checked integer distribution sorting with actual bucket and insertion-work records.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::algorithms {

/// @brief Owned scatter order, sorted contents and actual work for one bucket.
/// Equal integers retain insertion order; no satellite payload API is exposed.
/// Comparisons count evaluations of predecessor>key, shifts count successful
/// predecessor moves, and placements count final key stores (one per insertion
/// after the first item). Loop-bound tests, scattering and gathering are excluded.
/// @par Example
/// @code
/// const auto t=dm::algorithms::bucket_sort_trace({0,3,2,1,24});
/// std::print("{} {}\n",t.buckets[0].input,t.buckets[0].shifts); // [0, 3, 2, 1] 3
/// @endcode
/// @see BucketSortTrace
struct BucketSortBucket {
  std::vector<int> input; ///< This bucket's values in original encounter order.
  std::vector<int> output; ///< The same values after stable insertion sort.
  std::size_t key_comparisons{0}; ///< Evaluated predecessor>key tests, including false tests.
  std::size_t shifts{0}; ///< Successful predecessor moves; equals strict inversions in input.
  std::size_t placements{0}; ///< Final key stores, excluding the first bucket element.
};

/// @brief Owned snapshots and checked counters from one canonical bucket-sort run.
/// Empty, singleton and all-equal inputs take the historical early exit, with
/// shortcut=true, no allocated buckets/assignments and zero insertion counters.
/// Otherwise bucket_count=input.size(), buckets includes empty buckets, and
/// bucket_of_input records actual scatter destinations. Snapshots use O(n)
/// extra storage; counters throw on size_t overflow rather than wrap. All values
/// are owned; calls share no mutable state. Occupancy alone is an upper-bound
/// predictor: actual shifts also depend on order within each bucket.
/// @par Example
/// @code
/// const auto t=dm::algorithms::bucket_sort_trace({0,3,2,1,24});
/// std::print("{} {} {}\n",t.output,t.max_occupancy,t.shifts); // [0, 1, 2, 3, 24] 4 3
/// @endcode
/// @see bucket_sort_trace(), BucketSortBucket
struct BucketSortTrace {
  std::vector<int> input; ///< Unmodified original sequence.
  std::vector<int> output; ///< Ascending result.
  int minimum{0}; ///< Smallest value, or zero for empty input.
  int maximum{0}; ///< Largest value, or zero for empty input.
  std::uint64_t span{0}; ///< Inclusive maximum-minimum+1; zero for empty input.
  bool shortcut{true}; ///< Whether no scattering/insertion/gathering was needed.
  std::size_t bucket_count{0}; ///< Actually allocated bucket count; zero on shortcut.
  std::vector<std::size_t> bucket_of_input; ///< Actual scatter index for each input, or empty on shortcut.
  std::vector<BucketSortBucket> buckets; ///< Per-bucket owned records in increasing index order.
  std::size_t nonempty_buckets{0}; ///< Number of nonempty allocated buckets.
  std::size_t max_occupancy{0}; ///< Largest allocated bucket size; zero on shortcut.
  std::size_t occupancy_square_sum{0}; ///< Sum of squared allocated bucket sizes.
  std::size_t key_comparisons{0}; ///< Sum of per-bucket predecessor>key tests.
  std::size_t shifts{0}; ///< Sum of per-bucket successful predecessor moves.
  std::size_t placements{0}; ///< Sum of per-bucket final key stores.
};

/// @brief Return the exact integer scatter index without allocating any buckets.
/// Computes floor((value-minimum)*bucket_count/(maximum-minimum+1)), widening
/// before subtraction and using quotient/remainder decomposition to avoid product
/// overflow. Supports the complete 32-bit int domain and size_t bucket counts
/// through SIZE_MAX. The inclusive span keeps the maximum strictly below the
/// bucket count; some buckets may be unreachable when the span is small.
/// @param value Value in [minimum,maximum].
/// @param minimum Inclusive lower endpoint.
/// @param maximum Inclusive upper endpoint, at least minimum.
/// @param bucket_count Positive number of ordered buckets.
/// @return Index in [0,bucket_count-1].
/// @throws std::invalid_argument If endpoints, value or bucket_count are invalid.
/// @par Example
/// @code
/// std::print("{} {}\n",dm::algorithms::bucket_sort_index(5,1,9,7),
///     dm::algorithms::bucket_sort_index(9,1,9,7)); // 3 6
/// @endcode
/// @par References
/// - T. H. Cormen et al. (2022), Introduction to Algorithms, 4th ed., section 8.4,
///   https://mitpress.mit.edu/9780262046305/introduction-to-algorithms/.
///   This implementation rescales an observed inclusive integer range.
/// @see bucket_sort(), bucket_sort_trace()
std::size_t bucket_sort_index(int value,int minimum,int maximum,std::size_t bucket_count);

/// @brief Return an ascending integer copy using range buckets and insertion sort.
/// Uses n buckets for n>1 nonconstant inputs; empty, singleton and constant inputs
/// return immediately. Scattering is monotone in value, preserves encounter order,
/// and handles INT_MIN/INT_MAX without signed overflow. Insertion moves only
/// strictly greater predecessors. O(n+sum(bucket_size^2)) worst-case work and
/// O(n) auxiliary storage; expected linear work requires suitably balanced random
/// bucket assignments. All-equal inputs take O(n) despite maximal concentration.
/// The function does not allocate trace records or alter caller-owned input.
/// @param a Input values, owned by value and possibly empty.
/// @return Sorted nondecreasing sequence with all duplicates preserved.
/// @throws std::bad_alloc If output or bucket storage cannot be allocated.
/// @throws std::length_error If the required vector size exceeds its capacity.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::bucket_sort({5,3,8,1,9,2,7})); // [1, 2, 3, 5, 7, 8, 9]
/// @endcode
/// @par References
/// - T. H. Cormen et al. (2022), Introduction to Algorithms, 4th ed., section 8.4,
///   https://mitpress.mit.edu/9780262046305/introduction-to-algorithms/.
///   Integer-range scaling replaces the textbook fixed [0,1) domain.
/// @see bucket_sort_trace(), bucket_sort_index()
std::vector<int> bucket_sort(std::vector<int> a);

/// @brief Sort once and record the actual scatter buckets and insertion work.
/// Uses the same canonical kernel as bucket_sort(); no insertion sort is replayed
/// to generate counters. Output and each bucket's before/after values are owned.
/// Shortcut behavior and exact counter conventions are defined by BucketSortTrace.
/// Time is O(n+sum(bucket_size^2)); trace storage is O(n). No shared mutable state.
/// @param a Input sequence, owned by value.
/// @return Complete trace, including the sorted output.
/// @throws std::overflow_error If any trace count cannot fit size_t.
/// @throws std::bad_alloc If trace, bucket or output storage cannot be allocated.
/// @throws std::length_error If a required vector size exceeds its capacity.
/// @par Example
/// @code
/// const auto t=dm::algorithms::bucket_sort_trace({0,3,2,1,24});
/// std::print("{} {}\n",t.bucket_of_input,t.key_comparisons); // [0, 0, 0, 0, 4] 6
/// @endcode
/// @par References
/// - T. H. Cormen et al. (2022), Introduction to Algorithms, 4th ed., section 8.4,
///   https://mitpress.mit.edu/9780262046305/introduction-to-algorithms/.
/// @see bucket_sort(), BucketSortTrace
BucketSortTrace bucket_sort_trace(std::vector<int> a);

/// @brief Paired synthetic samples for comparing bucket occupancy and actual work.
/// Both vectors contain count integers in [0,999]. Uniform values use an integer
/// uniform draw; clustered values use a normal draw with mean 500 and deviation 4,
/// truncated toward zero and clamped. For count>=2 the first two clustered values
/// are replaced by 0 and 999 to stretch the observed range. A local mt19937 engine
/// consumes alternating uniform/normal draws. A seed repeats on a given standard
/// library; distribution mappings are not promised identical across toolchains.
/// @par Example
/// @code
/// const auto samples=dm::algorithms::bucket_sort_workloads(100,1);
/// std::print("{} {} {}\n",samples.uniform.size(),samples.clustered[0],samples.clustered[1]); // 100 0 999
/// @endcode
/// @see bucket_sort_workloads()
struct BucketSortWorkloads {
  std::vector<int> uniform; ///< Integer-uniform draws in [0,999].
  std::vector<int> clustered; ///< Narrow normal draws with two range-stretching outliers when possible.
};

/// @brief Generate reproducible-toolchain uniform and clustered comparison inputs.
/// See BucketSortWorkloads for sampling details, endpoints and seed portability.
/// Empty/singleton counts are supported; generation needs O(count) time/storage.
/// @param count Number of items in each vector, possibly zero.
/// @param seed Local mt19937 seed; defaults to one.
/// @return Two owned samples; no global random state is used.
/// @throws std::bad_alloc If sample storage cannot be allocated.
/// @throws std::length_error If count exceeds vector capacity.
/// @par Example
/// @code
/// const auto samples=dm::algorithms::bucket_sort_workloads(2);
/// std::print("{}\n",samples.clustered); // [0, 999]
/// @endcode
/// @par References
/// - I. O. Bercea and G. Even (2020), Upper Tail Analysis of Bucket Sort and Random
///   Tries, https://arxiv.org/abs/2002.10499 (uniform input analysis, not this generator).
/// @see BucketSortWorkloads, bucket_sort_trace()
BucketSortWorkloads bucket_sort_workloads(std::size_t count,std::uint32_t seed=1);

/// @brief Construct matched occupancies with controlled insertion-order difficulty.
/// Returns [0,1,...,m,(m+2)(m+1)] for ascending order, or [0,m,...,1,(m+2)(m+1)]
/// for descending order. Here m=cluster_size, n=m+2 buckets, and the inclusive span
/// sends 0 and all m interior distinct values to bucket zero, with the maximum
/// in bucket n-1. Ascending uses m key comparisons and no shifts; descending uses
/// m(m+1)/2 comparisons and m(m-1)/2 shifts. Zero gives [0,2]. O(m) generation.
/// @param cluster_size Number of distinct interior values, possibly zero.
/// @param descending Whether to reverse their encounter order; defaults to true.
/// @return Owned collision-family input with exactly two occupied buckets.
/// @throws std::length_error If the endpoint cannot fit int or the vector cannot fit.
/// @throws std::bad_alloc If input storage cannot be allocated.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::bucket_sort_collision_input(3)); // [0, 3, 2, 1, 20]
/// @endcode
/// @see bucket_sort_trace()
std::vector<int> bucket_sort_collision_input(std::size_t cluster_size,bool descending=true);

} // namespace datamunge::algorithms
