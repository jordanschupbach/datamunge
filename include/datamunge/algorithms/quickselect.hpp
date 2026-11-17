#pragma once
/// @file quickselect.hpp
/// @brief Duplicate-safe three-way quickselect and median-of-medians selection.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

/// @brief Selection algorithms and owned partition records.
namespace datamunge::algorithms {

/// @brief Selected-work and history bounds; zero disables a size/work cap.
/// @par Example
/// @code
/// const dm::algorithms::QuickselectOptions x{};
/// std::print("{}\n",x.median_of_medians); // false
/// @endcode
/// @see quickselect_run()
struct QuickselectOptions {
  bool median_of_medians{false}; ///< Choose guaranteed linear BFPRT instead of median-of-three.
  std::size_t max_operations{10000000};   ///< Comparisons, swaps and insertion writes combined.
  std::size_t max_trace_entries{1000000}; ///< Recorded range scalars plus step headers.
};
/// @brief One actual three-way partition, including recursive BFPRT pivot searches.
/// @par Example
/// @code
/// const dm::algorithms::QuickselectStep x{};
/// std::print("{}\n",x.depth); // 0
/// @endcode
/// @see quickselect_run()
struct QuickselectStep {
  std::size_t               lo{};          ///< Inclusive range beginning.
  std::size_t               hi{};          ///< Exclusive range end.
  std::size_t               rank{};        ///< Absolute target rank of this selection call.
  std::size_t               equal_begin{}; ///< First index equal to the pivot after partitioning.
  std::size_t               equal_end{};   ///< First index greater than the pivot.
  std::size_t               depth{}; ///< BFPRT nested selection depth; outer selection has zero.
  std::int64_t              pivot{}; ///< Actual selected pivot value.
  std::vector<std::int64_t> values;  ///< Actual range after the completed partition.
};
/// @brief Owned answer and selected operations from the actual selection producer.
/// @par Example
/// @code
/// const dm::algorithms::QuickselectResult x{};
/// std::print("{}\n",x.partitions); // 0
/// @endcode
/// @see quickselect_run()
struct QuickselectResult {
  std::int64_t                 value{}; ///< Value at the requested rank.
  std::vector<std::int64_t>    values;  ///< Complete final rearranged input copy.
  std::vector<QuickselectStep> steps;   ///< Actual records, empty for noncapturing execution.
  std::size_t comparisons{}; ///< All key comparisons, including pivot selection/group sorting.
  std::size_t swaps{};       ///< Actual swap calls, including self-swaps.
  std::size_t writes{};      ///< Insertion-sort assignments, excluding swap internals.
  std::size_t operations{};  ///< Sum of comparisons, swap calls and insertion assignments.
  std::size_t partitions{};  ///< Actual three-way partitions, including BFPRT subsearches.
};
/// @brief Select an order statistic using the shared duplicate-safe producer.
/// Median-of-three is deterministic, expected O(n) on suitable random inputs and worst-case O(n^2). BFPRT with groups of five and three-way partitioning is worst-case O(n), including duplicates. Uses an owned O(n) input copy and O(log n) BFPRT recursion. Counters cover all nested pivot searches; operations is a selected event count, not CPU instructions. Input is unchanged.
/// @param data Owned integer keys.
/// @param k Zero-based rank, less than data.size().
/// @param options Pivot strategy and selected-work policy; history cap used only by the trace.
/// @return Answer, actual rearrangement and cumulative selected work.
/// @throws std::invalid_argument Empty input or invalid rank.
/// @throws std::length_error Selected work limit exceeded.
/// @par Example
/// @code
/// auto r = dm::algorithms::quickselect_run({3,1,1,2},2);
/// std::print("{}\n",r.value); // 2
/// @endcode
/// @see quickselect_trace(), quickselect_mom()
QuickselectResult quickselect_run(
    std::vector<std::int64_t> data,
    std::size_t               k,
    const QuickselectOptions& options = {});
/// @brief Capture actual partition bands from the same selection producer.
/// Includes nested BFPRT pivot-selection partitions, distinguished by depth; histories cost range length plus one per record and are admitted before copying. Zero cap disables that policy. Runtime/storage otherwise match quickselect_run().
/// @param data Owned input copy.
/// @param k Zero-based rank.
/// @param options Pivot, selected-work and retained scalar policies.
/// @return Answer, complete rearrangement, work and actual partition records.
/// @throws std::invalid_argument Empty input or invalid rank.
/// @throws std::length_error Work or retained history policy exceeded.
/// @par Example
/// @code
/// auto r = dm::algorithms::quickselect_trace({2,2,2},1);
/// std::print("{} {}\n",r.partitions,r.steps[0].equal_end); // 1 3
/// @endcode
/// @see quickselect_run()
QuickselectResult quickselect_trace(
    std::vector<std::int64_t> data,
    std::size_t               k,
    const QuickselectOptions& options = {});

/// @brief Returns the @p k-th *smallest* element of @p data (0-indexed: k=0 is the minimum, k=1
///        the second smallest, ..., k=n-1 the maximum) -- the k-th *order statistic* -- without
///        fully sorting the array. This is Hoare's Quickselect (1961): pick a pivot, *partition*
///        the array so that everything smaller precedes it and everything larger follows it, note
///        the pivot's final rank p, and then continue into *only one* strict side, stopping whenever the target rank is in
///        the entire equal-pivot band. Because each step discards one partition instead of
///        recursing into both (as quicksort does), the expected work follows the recurrence
///        T(n) = T(n/2) + O(n) = O(n) on random input, versus O(n log n) to sort the whole array
///        just to read one element. This overload takes @p data *by value* precisely because it
///        reorders it internally; the caller's array is untouched. The pivot is chosen by the
///        *median-of-three* rule (median of the first, middle, and last elements), a cheap
///        deterministic pivot heuristic, though
///        a truly adversarial permutation can still force the O(n^2) worst case -- for a guaranteed
///        linear bound use @ref quickselect_mom. Duplicate values stop in one equal band. Delegates to the shared three-way producer without work/history caps.
///
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::quickselect({3,1,2},1)); // 2
/// @endcode
/// @see quickselect_trace()
/// @param data the values to select from (taken by value; reordered internally, caller's copy is
///        unaffected).
/// @param k the 0-indexed rank to retrieve (0 = minimum, data.size()-1 = maximum).
/// @return the k-th smallest element of @p data.
/// @throws std::invalid_argument if @p data is empty or if k >= data.size().
[[nodiscard]] std::int64_t quickselect(std::vector<std::int64_t> data, std::size_t k);

/// @brief The median-of-medians (BFPRT) variant of @ref quickselect: identical k-semantics and
///        identical return value, but with a *guaranteed worst-case O(n)* running time. Blum,
///        Floyd, Pratt, Rivest & Tarjan (1973) removed Quickselect's quadratic worst case by
///        replacing the pivot heuristic with a provably good pivot: split the range into groups of
///        five, find each group's median (a constant-time sort of at most five elements), then
///        recursively take the *median of those medians* as the pivot. That pivot is guaranteed to
///        be greater than at least (roughly) 3n/10 elements and less than at least 3n/10, so every
///        partition discards a constant fraction of the array. The resulting recurrence
///        T(n) <= T(n/5) + T(7n/10) + O(n) solves to O(n) because 1/5 + 7/10 < 1. The constant is
///        larger than plain Quickselect's, so this variant trades average-case speed for an
///        adversary-proof worst case; both functions always agree on the answer.
///
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::quickselect_mom({7,7,7},1)); // 7
/// @endcode
/// @see quickselect_run()
/// @param data the values to select from (taken by value; reordered internally, caller's copy is
///        unaffected).
/// @param k the 0-indexed rank to retrieve (0 = minimum, data.size()-1 = maximum).
/// @return the k-th smallest element of @p data (equal to what @ref quickselect returns).
/// @throws std::invalid_argument if @p data is empty or if k >= data.size().
[[nodiscard]] std::int64_t quickselect_mom(std::vector<std::int64_t> data, std::size_t k);

/// @brief Actual cumulative work for both pivot strategies on identical generated arrays.
/// @par Example
/// @code
/// const dm::algorithms::QuickselectScaling x{};
/// std::print("{}\n",x.size); // 0
/// @endcode
/// @see quickselect_run()
struct QuickselectScaling {
  std::size_t size{};              ///< Input keys per trial.
  std::size_t trials{};            ///< Actual independent generated arrays per size.
  std::size_t comparisons{};       ///< Actual median-of-three comparisons over all trials.
  std::size_t bfprt_comparisons{}; ///< Actual BFPRT comparisons over the same arrays.
};
/// @brief Measure both canonical selection strategies on the original seeded uniform workload.
/// Generates keys with one sequential mt19937_64(seed) stream and uniform_int_distribution<int64_t>(0,1000000), traversing sizes then trials. Both canonical producers select rank n/2 on identical copies; no report-local surrogate is used. Every size and total key count is validated before generation. Work/storage scales with actual trials/keys plus native selection; one array pair per trial. Returns actual comparison sums, not timing or a universal expected-time estimate.
/// @param sizes Nonempty positive input sizes, in stream-consumption order.
/// @param trials Positive generated arrays per size.
/// @param seed MT19937-64 seed, default20240727.
/// @param max_keys Cumulative generated keys, default1000000; zero disables.
/// @return One selected-work record per size.
/// @throws std::invalid_argument Zero trials or a zero input size.
/// @throws std::length_error Cumulative key/count capacity or native work policy exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::quickselect_scaling({10},2);
/// std::print("{} {}\n",r[0].size,r[0].trials); // 10 2
/// @endcode
/// @see quickselect_run()
std::vector<QuickselectScaling> quickselect_scaling(
    const std::vector<std::size_t>& sizes,
    std::size_t                     trials,
    std::uint64_t                   seed     = 20240727,
    std::size_t                     max_keys = 1000000);
} // namespace datamunge::algorithms
