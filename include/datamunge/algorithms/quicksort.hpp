#pragma once
/// @file quicksort.hpp
/// @brief Median-of-three Lomuto quicksort with actual comparison and range histories.

#include <cstddef>
#include <cstdint>
#include <vector>

/// @brief Canonical sorting algorithms and owned histories.
namespace datamunge::algorithms {

/// @brief Selected-work and retained-history limits; zero disables that limit.
/// @par Example
/// @code
/// const dm::algorithms::QuicksortOptions x{};
/// std::print("{}\n",x.max_operations); // 20000000
/// @endcode
/// @see quicksort_run()
struct QuicksortOptions {
  std::size_t max_operations{20000000};   ///< Comparisons, array swaps and insertion assignments.
  std::size_t max_trace_entries{1000000}; ///< Retained range scalars plus record headers.
};
/// @brief One actual completed partition or insertion-sort range.
/// @par Example
/// @code
/// const dm::algorithms::QuicksortStep x{};
/// std::print("{}\n",x.insertion); // false
/// @endcode
/// @see quicksort_run()
struct QuicksortStep {
  std::size_t lo{};          ///< Inclusive range beginning.
  std::size_t hi{};          ///< Exclusive range end.
  std::size_t pivot_index{}; ///< Absolute final pivot position; hi for insertion records.
  std::size_t depth{};       ///< Live smaller-side recursion depth, starting at one.
  bool        insertion{};   ///< Whether the actual sixteen-key cutoff selected insertion sort.
  std::vector<std::int64_t> values; ///< Actual range immediately after the selected operation.
};
/// @brief Owned final keys and counters from the shared native quicksort producer.
/// @par Example
/// @code
/// const dm::algorithms::QuicksortResult x{};
/// std::print("{}\n",x.partitions); // 0
/// @endcode
/// @see quicksort_run()
struct QuicksortResult {
  std::vector<std::int64_t>  values; ///< Ascending keys.
  std::vector<QuicksortStep> steps;  ///< Actual ranges, populated only by quicksort_trace().
  std::uint64_t comparisons{};       ///< Key comparisons, preserving the historical overload count.
  std::size_t   swaps{};             ///< Actual array swap calls, including self-swaps.
  std::size_t   writes{};            ///< Actual insertion assignments, excluding swap internals.
  std::size_t   operations{};        ///< Comparisons plus array swaps plus insertion assignments.
  std::size_t   partitions{};        ///< Actual Lomuto partitions.
  std::size_t   max_depth{}; ///< Peak smaller-side recursion depth; zero for fewer than two keys.
};
/// @brief Sort owned keys with the canonical sixteen-key-cutoff quicksort producer.
/// Same median-of-three, Lomuto partition, smaller-side recursion and comparison-count sequence as the historical overload. Expected O(n log n) on suitable random inputs, worst-case O(n^2), including equal keys; live recursion O(log n). Result owns an O(n) input copy. Selected counters are events, not all machine instructions. Input is unchanged.
/// @param data Keys taken by value.
/// @param options Selected-work policy; zero disables.
/// @return Sorted keys and actual cumulative selected work.
/// @throws std::length_error Work budget or signed indexing capacity exceeded.
/// @par Example
/// @code
/// auto r = dm::algorithms::quicksort_run({3,1,2});
/// std::print("{}\n",r.values[0]); // 1
/// @endcode
/// @see quicksort_trace(), quicksort()
QuicksortResult quicksort_run(std::vector<std::int64_t> data, const QuicksortOptions& options = {});
/// @brief Capture actual partition and cutoff ranges from the same quicksort producer.
/// Each retained record costs range length plus one scalar/header entries, admitted before copying. Worst-case retained history can be quadratic; policy stops it. All other semantics match quicksort_run().
/// @param data Owned input copy.
/// @param options Work and retained-history policy; zero disables a cap.
/// @return Sorted keys, actual counters and owned real range history.
/// @throws std::length_error Work, history or signed indexing bound exceeded.
/// @par Example
/// @code
/// auto r = dm::algorithms::quicksort_trace({3,1,2});
/// std::print("{}\n",r.steps[0].insertion); // true
/// @endcode
/// @see quicksort_run()
QuicksortResult quicksort_trace(
    std::vector<std::int64_t> data,
    const QuicksortOptions&   options = {});

/// @brief Sorts @p data ascending, in place, using Quicksort (Hoare 1961) -- the classic
///        divide-and-conquer sort. A *pivot* is chosen and the range is *partitioned* so that every
///        element strictly less than the pivot precedes it and everything else follows; the pivot
///        thereby lands in its final sorted position and the two sides are sorted recursively. This
///        implementation makes three refinements that turn textbook Quicksort into a robust one.
///        First, *median-of-three* pivoting: the pivot is the median of the first, middle, and last
///        elements of the range, which balances already-sorted and reverse-sorted distinct inputs while remaining fully
///        deterministic. It offers no worst-case guarantee. Second, *tail-call elimination*: after each
///        partition the smaller side is sorted by recursion and the larger side by looping, which
///        bounds the recursion depth to O(log n) and so prevents stack overflow on adversarial
///        inputs. Third, a *small-subarray cutoff*: ranges of at most sixteen elements are finished
///        with insertion sort, which is faster than recursion on tiny inputs. The average-case cost
///        is O(n log n) comparisons; the worst case is O(n^2) including all-equal inputs and adversarial permutations.
///        Quicksort is *in place* (O(log n) auxiliary stack, no heap allocation) but *not stable*:
///        equal elements may be reordered relative to one another. Empty and single-element inputs
///        are returned unchanged.
///
/// @par Example
/// @code
/// std::vector<std::int64_t> v{2,1};
/// dm::algorithms::quicksort(v);
/// std::print("{}\n",v[0]); // 1
/// @endcode
/// @see quicksort_run()
/// @param data the array to sort in place (ascending).
void quicksort(std::vector<std::int64_t>& data);

/// @brief Identical to @ref quicksort(std::vector<std::int64_t>&) but additionally reports, in
///        @p comparisons, the total number of element-to-element comparisons performed (including
///        those made while selecting median-of-three pivots and during the insertion-sort cutoff).
///        This instrumentation exposes the algorithm's O(n log n) average behaviour for study and
///        plotting; the sort itself is unchanged. @p comparisons is overwritten (not accumulated).
///
/// @param data the array to sort in place (ascending).
/// @par Example
/// @code
/// std::vector<std::int64_t> v{2,1};
/// std::uint64_t comparisons=99;
/// dm::algorithms::quicksort(v,comparisons);
/// std::print("{}\n",comparisons); // 1
/// @endcode
/// @see quicksort_trace()
/// @param comparisons out-parameter receiving the count of element comparisons performed.
void quicksort(std::vector<std::int64_t>& data, std::uint64_t& comparisons);

/// @brief One actual canonical sort measurement on the original sequential seeded workload.
/// @par Example
/// @code
/// const dm::algorithms::QuicksortScaling x{};
/// std::print("{}\n",x.size); // 0
/// @endcode
/// @see quicksort_run()
struct QuicksortScaling {
  std::size_t   size{};        ///< Actual generated key count.
  std::uint64_t comparisons{}; ///< Actual canonical key comparisons.
  double        reference{};   ///< Mathematical n*log2(n) reference, not measured operations.
};
/// @brief Measure canonical quicksort on one sequential native seeded workload.
/// Creates one mt19937_64(seed) stream and consumes raw words modulo1000000 for each requested array in order, matching the original report. Executes the canonical counted producer, never a report mirror. Validates all cumulative key sizes before generation; empty arrays are supported. O(total sort work), O(max size) temporary keys plus result rows. The n log2 n reference is a mathematical scale, not a complexity guarantee for every input.
/// @param sizes Array sizes in stream-consumption order.
/// @param seed MT19937-64 seed, default20240927.
/// @param max_keys Cumulative generated-key bound, default1000000; zero disables.
/// @return Actual comparison count and explicit theoretical reference for each size.
/// @throws std::length_error Total key/count capacity or native selected-work policy exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::quicksort_scaling({0,1});
/// std::print("{} {}\n",r[0].comparisons,r[1].comparisons); // 0 0
/// @endcode
/// @see quicksort_run()
std::vector<QuicksortScaling> quicksort_scaling(
    const std::vector<std::size_t>& sizes,
    std::uint64_t                   seed     = 20240927,
    std::size_t                     max_keys = 1000000);
} // namespace datamunge::algorithms
