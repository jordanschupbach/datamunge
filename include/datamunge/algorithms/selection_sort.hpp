#pragma once
/// @file selection_sort.hpp
/// @brief Checked same-execution minimum scans and selected swaps.
#include <cstddef>
#include <vector>
/// @brief Canonical checked native producers and owned algorithm values.
namespace datamunge::algorithms {
/// @brief Complete key, actual comparison and optional capture limits; zero disables a size cap.
/// @par Example
/// @code
/// const dm::algorithms::SelectionSortOptions value{};
/// std::print("{}\n",value.max_comparisons); // 20000000
/// @endcode
struct SelectionSortOptions {
  std::size_t max_keys{1000000};          ///< Complete input key count.
  std::size_t max_comparisons{20000000};  ///< Actual strict value comparisons.
  std::size_t max_trace_entries{1000000}; ///< Actual pass records and full after-pass key copies.
};
/// @brief One actual completed minimum-selection pass.
/// @par Example
/// @code
/// const dm::algorithms::SelectionSortStep value{};
/// std::print("{}\n",value.swapped); // false
/// @endcode
struct SelectionSortStep {
  std::size_t      position{};      ///< Actual prefix position filled.
  std::size_t      minimum_index{}; ///< Actual first minimum address in the pre-swap suffix.
  int              minimum{};       ///< Actual selected minimum value.
  bool             swapped{};       ///< Actual minimum address differed from the prefix address.
  std::size_t      comparisons{};   ///< Actual strict suffix comparisons in this pass.
  std::vector<int> after;           ///< Actual complete array immediately after this pass.
};
/// @brief Owned sorted values and actual minimum scan/swap work.
/// @par Example
/// @code
/// const dm::algorithms::SelectionSortRun value{};
/// std::print("{}\n",value.swaps); // 0
/// @endcode
struct SelectionSortRun {
  std::vector<int> values;        ///< Actual final nondecreasing values; sorting is unstable.
  std::size_t      comparisons{}; ///< Actual strict value comparisons.
  std::size_t      swaps{};       ///< Actual non-self swaps, at most n-1; zero when already sorted.
  std::vector<SelectionSortStep> steps; ///< Optional actual completed-pass history.
};
/// @brief Sort integers by actual suffix-minimum selection.
/// Each nonfinal position scans the remaining suffix with strict less-than and exchanges only a non-self minimum. Ties select the first encountered minimum; distant swaps can reorder equal keys. Empty/singleton inputs perform no passes. Exactly n*(n-1)/2 comparisons and at most n-1 actual swaps; sorted inputs make zero swaps. This is not a universal minimum-write claim, especially with duplicate values or other sorting algorithms. Input passed by value, leaving caller storage unchanged. Intrinsic work O(n^2), storage O(n); optional full-state histories O(n^2). Complete comparison/capture geometry is checked before sorting; each counter is still incremented only by actual executed work. Zero disables a policy, never arithmetic capacity checks. Outputs own storage; no shared state.
/// @param values Owned complete working integer sequence.
/// @param options Input, actual comparison and optional captured-state bounds.
/// @return Owned sorted values, actual work and optional pass states.
/// @throws std::length_error Input, complete work, history or counter/container capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::selection_sort_run({3,1,2});
/// std::print("{} {} {}\n",r.values,r.comparisons,r.swaps); // [1, 2, 3] 3 2
/// @endcode
/// @see selection_sort()
SelectionSortRun selection_sort_run(
    std::vector<int>            values,
    const SelectionSortOptions& options = {});
/// @brief Sort integers by actual suffix-minimum selection with native completed-pass states.
/// Each nonfinal position scans the remaining suffix with strict less-than and exchanges only a non-self minimum. Ties select the first encountered minimum; distant swaps can reorder equal keys. Empty/singleton inputs perform no passes. Exactly n*(n-1)/2 comparisons and at most n-1 actual swaps; sorted inputs make zero swaps. This is not a universal minimum-write claim, especially with duplicate values or other sorting algorithms. Input passed by value, leaving caller storage unchanged. Intrinsic work O(n^2), storage O(n); optional full-state histories O(n^2). Complete comparison/capture geometry is checked before sorting; each counter is still incremented only by actual executed work. Zero disables a policy, never arithmetic capacity checks. Outputs own storage; no shared state.
/// @param values Owned complete working integer sequence.
/// @param options Input, actual comparison and optional captured-state bounds.
/// @return Owned sorted values, actual work and optional pass states.
/// @throws std::length_error Input, complete work, history or counter/container capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::selection_sort_trace({3,1,2});
/// std::print("{} {} {}\n",r.values,r.comparisons,r.swaps); // [1, 2, 3] 3 2
/// @endcode
/// @see selection_sort()
SelectionSortRun selection_sort_trace(
    std::vector<int>            values,
    const SelectionSortOptions& options = {});
/// @brief Return sorted integers through the preserved checked selection-sort signature.
/// Uses the same actual producer, default bounds and no optional retained pass history.
/// @param a Owned complete input sequence.
/// @return Owned sorted sequence.
/// @throws std::length_error Default complete input/work policy or capacity exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::selection_sort({3,1,2})); // [1, 2, 3]
/// @endcode
/// @see selection_sort_run()
std::vector<int> selection_sort(std::vector<int> a);
/// @brief Generate the complete original shuffled-size selection-sort study.
/// Uses one local mt19937 seeded 2 and shuffles ascending integer keys at n=5,10,...,200 sequentially; preserves the original engine draw order.
/// @return Forty owned original controlled arrays.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::selection_sort_example_arrays().back().size()); // 200
/// @endcode
/// @see selection_sort_trace()
std::vector<std::vector<int>> selection_sort_example_arrays();
} // namespace datamunge::algorithms
