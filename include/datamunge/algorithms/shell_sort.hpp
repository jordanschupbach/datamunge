#pragma once
/// @file shell_sort.hpp
/// @brief Actual halving-gap insertion passes and work.
#include <cstddef>
#include <vector>
/// @brief Canonical checked native producers and owned algorithm values.
namespace datamunge::algorithms {
/// @brief ShellSortOptions stores owned actual values or explicit invocation policies; field units and defaults are described below.
/// @par Example
/// @code
/// const dm::algorithms::ShellSortOptions value{};
/// std::print("{}\n",value.max_keys); // 1000000
/// @endcode
struct ShellSortOptions {
  std::size_t max_keys{1000000};           ///< Input size policy; zero disables.
  std::size_t max_comparisons{20000000};   ///< Executed key comparisons; zero disables.
  std::size_t max_history_values{1000000}; ///< Retained keys plus pass records; zero disables.
};
/// @brief ShellSortPass stores owned actual values or explicit invocation policies; field units and defaults are described below.
/// @par Example
/// @code
/// const dm::algorithms::ShellSortPass value{};
/// std::print("{}\n",value.gap); // 0
/// @endcode
struct ShellSortPass {
  std::size_t gap{};         ///< Actual gap used.
  std::size_t comparisons{}; ///< Actual key comparisons including final unsuccessful comparisons.
  std::size_t shifts{};      ///< Actual displaced key assignments.
  std::vector<int> after;    ///< Actual state after this gap.
};
/// @brief ShellSortRun stores owned actual values or explicit invocation policies; field units and defaults are described below.
/// @par Example
/// @code
/// const dm::algorithms::ShellSortRun value{};
/// std::print("{}\n",value.shifts); // 0
/// @endcode
struct ShellSortRun {
  std::vector<int> values;        ///< Sorted owned keys.
  std::size_t      comparisons{}; ///< Actual key comparisons only; excludes index tests.
  std::size_t      shifts{}; ///< Actual displaced key assignments; excludes final key placements.
  std::vector<ShellSortPass> passes; ///< Optional actual pass histories.
};
/// @brief Sort by the original n/2,n/4,...,1 gap sequence.
/// Strict greater-than shifts. Failed key comparisons count, boundary index tests do not. Owned copy leaves callers unchanged. Empty and singleton arrays have no passes. Shell halving gaps have O(n^2) worst-case time and O(n) working storage; captured full passes add O(n log n) values. Ordinary insertion has one gap-1 pass when n>1. No shared state. Policies throw before violating a bound; zero disables that policy, not capacity checks.
/// @param values Owned complete integer keys.
/// @param options Complete input, executed work and optional capture policies.
/// @return Owned sorted values and actual work/history.
/// @throws std::length_error Input, comparison, history policy or counter capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::shell_sort_run({3,1,2});
/// std::print("{} {}\n",r.values,r.shifts); // [1, 2, 3] 2
/// @endcode
/// @par References
/// - D. L. Shell, A High-Speed Sorting Procedure, CACM 2(7), 1959, DOI 10.1145/368370.368387.
/// @see shell_sort()
ShellSortRun shell_sort_run(std::vector<int> values, const ShellSortOptions& options = {});
/// @brief Retain actual halving-gap pass states.
/// Strict greater-than shifts. Failed key comparisons count, boundary index tests do not. Owned copy leaves callers unchanged. Empty and singleton arrays have no passes. Shell halving gaps have O(n^2) worst-case time and O(n) working storage; captured full passes add O(n log n) values. Ordinary insertion has one gap-1 pass when n>1. No shared state. Policies throw before violating a bound; zero disables that policy, not capacity checks.
/// @param values Owned complete integer keys.
/// @param options Complete input, executed work and optional capture policies.
/// @return Owned sorted values and actual work/history.
/// @throws std::length_error Input, comparison, history policy or counter capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::shell_sort_trace({3,1,2});
/// std::print("{}\n",r.passes.front().gap); // 1
/// @endcode
/// @par References
/// - D. L. Shell, A High-Speed Sorting Procedure, CACM 2(7), 1959, DOI 10.1145/368370.368387.
/// @see shell_sort()
ShellSortRun shell_sort_trace(std::vector<int> values, const ShellSortOptions& options = {});
/// @brief Run ordinary insertion sorting with identical comparison counting.
/// Strict greater-than shifts. Failed key comparisons count, boundary index tests do not. Owned copy leaves callers unchanged. Empty and singleton arrays have no passes. Shell halving gaps have O(n^2) worst-case time and O(n) working storage; captured full passes add O(n log n) values. Ordinary insertion has one gap-1 pass when n>1. No shared state. Policies throw before violating a bound; zero disables that policy, not capacity checks.
/// @param values Owned complete integer keys.
/// @param options Complete input, executed work and optional capture policies.
/// @return Owned sorted values and actual work/history.
/// @throws std::length_error Input, comparison, history policy or counter capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::insertion_sort_work({3,1,2});
/// std::print("{}\n",r.comparisons); // 3
/// @endcode
/// @par References
/// - D. L. Shell, A High-Speed Sorting Procedure, CACM 2(7), 1959, DOI 10.1145/368370.368387.
/// @see shell_sort()
ShellSortRun insertion_sort_work(std::vector<int> values, const ShellSortOptions& options = {});
/// @brief Return sorted keys using the preserved Shell signature.
/// Uses the same checked producer with default policies.
/// @param a Owned complete integer keys.
/// @return Owned sorted sequence.
/// @throws std::length_error Default size/work policy exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::shell_sort({3,1,2})); // [1, 2, 3]
/// @endcode
/// @see shell_sort_run()
std::vector<int> shell_sort(std::vector<int> a);
/// @brief Generate the original forty shuffled Shell comparison fixtures.
/// One local mt19937 seed3, n=10,20,...,400, ascending keys shuffled sequentially. Owned outputs and no shared state.
/// @return Forty owned arrays.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::shell_sort_example_arrays().back().size()); // 400
/// @endcode
/// @see shell_sort_run()
std::vector<std::vector<int>> shell_sort_example_arrays();
} // namespace datamunge::algorithms
