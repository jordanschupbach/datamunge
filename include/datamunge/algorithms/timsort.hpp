#pragma once
/// @file timsort.hpp
/// @brief Generic actual natural-run sorting with checked comparisons and repaired full merge-stack conditions.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
namespace datamunge::algorithms {
/// @brief Own TimsortOptions values for one independent native execution.
/// Fields describe actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record alone performs no algorithm work and retains no caller storage or shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::TimsortOptions r{};
/// (void)r;
/// @endcode
struct TimsortOptions {
  std::size_t max_keys{
      1000000}; ///< Complete key count; zero disables policy, int index capacity remains.
  std::size_t max_comparisons{
      20000000}; ///< Actual natural-run, insertion and merge key comparisons; zero disables.
  std::size_t max_assignments{
      20000000}; ///< Actual insertion/merge writes, excluding buffer copies/reversal swaps; zero disables.
  std::size_t max_history_values{
      1000000}; ///< Actual record plus stack-length and complete key copies; zero disables.
};
/// @brief Own TimsortStep values for one independent native execution.
/// Fields describe actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record alone performs no algorithm work and retains no caller storage or shared mutable state.
/// @tparam T Copyable strict-weak-order key type; example uses int.
/// @par Example
/// @code
/// dm::algorithms::TimsortStep<int> r{};
/// (void)r;
/// @endcode
template<class T>
struct TimsortStep {
  std::string action;         ///< Actual natural, extend, push or merge event.
  std::size_t base{};         ///< Actual involved left run start.
  std::size_t left_length{};  ///< Actual natural/extended/left-merge run length at this event.
  std::size_t right_length{}; ///< Actual right-merge run length; zero for other events.
  std::vector<std::size_t> stack_lengths; ///< Actual run-stack lengths after this event.
  std::vector<T> after; ///< Actual complete current keys, including unsorted unvisited suffix.
};
/// @brief Own TimsortRun values for one independent native execution.
/// Fields describe actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record alone performs no algorithm work and retains no caller storage or shared mutable state.
/// @tparam T Copyable strict-weak-order key type; example uses int.
/// @par Example
/// @code
/// dm::algorithms::TimsortRun<int> r{};
/// (void)r;
/// @endcode
template<class T>
struct TimsortRun {
  std::vector<T> values;           ///< Owned actual stable ascending result.
  std::size_t    minrun{};         ///< Actual minrun, n itself below64; zero on empty input.
  std::size_t    natural_runs{};   ///< Actual detected natural runs before extension.
  std::size_t    reversed_swaps{}; ///< Actual pair swaps reversing strictly descending runs.
  std::size_t    comparisons{};    ///< Actual strict operator< invocations.
  std::size_t assignments{}; ///< Actual insertion/merge key writes, excluding buffer copies/swaps.
  std::size_t buffer_copies{};       ///< Actual left-run values copied into merge scratch buffers.
  std::size_t merges{};              ///< Actual adjacent stable merges.
  std::vector<TimsortStep<T>> steps; ///< Optional actual sorting and run-stack events.
};
/// @brief Own TimsortInputProfile values for one independent native execution.
/// Fields describe actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record alone performs no algorithm work and retains no caller storage or shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::TimsortInputProfile r{};
/// (void)r;
/// @endcode
struct TimsortInputProfile {
  std::vector<std::size_t>
      lengths; ///< Actual maximal input run lengths in left-to-right scan order, before minrun extension.
  std::vector<bool>
      descending; ///< Actual strict-descent branch per raw run, false for ascending/singleton runs.
  std::size_t
      comparisons{}; ///< Actual raw input profile strict comparisons, separate from real sorting comparisons.
};
/// @brief Invocation-owned generic natural-run sort kernels.
namespace timsort_detail {
/// @brief Detect one actual maximal nondecreasing/strict-descending run without modifying keys.
/// @tparam T Borrowed strict-weak-order key type.
/// @tparam Less Actual comparison observer.
/// @param values Complete borrowed key vector.
/// @param lo Valid initial index.
/// @param less Borrowed actual strict comparator.
/// @return Actual one-past-end and strict-descent flag.
template<class T, class Less>
std::pair<std::size_t, bool> natural_end(const std::vector<T>& values, std::size_t lo, Less& less) {
  auto end        = lo + 1;
  bool descending = false;
  if (end < values.size()) {
    descending = less(values[end++], values[lo]);
    if (descending) {
      while (end < values.size() && less(values[end], values[end - 1]))
        ++end;
    } else
      while (end < values.size() && !less(values[end], values[end - 1]))
        ++end;
  }
  return {end, descending};
}
/// @brief Produce actual stable natural-run sorting with correct four-run merge-collapse checks.
/// @tparam T Copyable strict-weak-order keys; user copy/comparison exceptions propagate.
/// @param values Owned complete keys.
/// @param o Complete and selected work/capture policies.
/// @param capture Retain actual stack/key events.
/// @return Owned actual result/work/states.
template<class T>
TimsortRun<T> execute(std::vector<T> values, const TimsortOptions& o, bool capture) {
  auto n = values.size();
  if (n > std::size_t(std::numeric_limits<int>::max()) || (o.max_keys && n > o.max_keys))
    throw std::length_error("Timsort keys/index capacity");
  if constexpr (std::is_floating_point_v<T>)
    for (T x : values)
      if (!std::isfinite(x))
        throw std::invalid_argument("Timsort finite keys");
  TimsortRun<T> r;
  r.values          = std::move(values);
  std::size_t shift = n, carry = 0;
  while (shift >= 64) {
    carry |= shift & 1;
    shift >>= 1;
  }
  r.minrun = shift + carry;
  if (n < 2)
    return r;
  /// @brief Count an actual selected event before exceeding its work policy.
  auto count = [&](std::size_t& counter, std::size_t limit) {
    if (counter == std::numeric_limits<std::size_t>::max() || (limit && counter >= limit))
      throw std::length_error("Timsort work");
    ++counter;
  };
  /// @brief Execute one actual checked strict key comparison.
  auto less = [&](const T& x, const T& y) {
    count(r.comparisons, o.max_comparisons);
    return x < y;
  };
  /// @brief One actual adjacent sorted run on the producing stack.
  struct Run {
    std::size_t base;   ///< Actual run start.
    std::size_t length; ///< Actual run length.
  };
  std::vector<Run> stack;
  std::vector<T>   tmp;
  std::size_t      retained = 0;
  /// @brief Capture actual current keys and stack lengths after a real sorting event.
  auto record = [&](const char* action, std::size_t base, std::size_t left, std::size_t right = 0) {
    if (!capture)
      return;
    auto cap = std::numeric_limits<std::size_t>::max();
    if (stack.size() > cap - n - 1)
      throw std::length_error("Timsort history capacity");
    auto added = 1 + n + stack.size();
    if (retained > cap - added ||
        (o.max_history_values &&
         (retained > o.max_history_values || added > o.max_history_values - retained)))
      throw std::length_error("Timsort history");
    retained += added;
    std::vector<std::size_t> lengths;
    for (const auto& run : stack)
      lengths.push_back(run.length);
    r.steps.push_back({action, base, left, right, std::move(lengths), r.values});
  };
  /// @brief Execute an actual adjacent stable merge and remove its consumed stack entry.
  auto merge = [&](std::size_t at) {
    auto base = stack[at].base, left = stack[at].length, right = stack[at + 1].length,
         mid = base + left, hi = mid + right;
    tmp.assign(r.values.begin() + base, r.values.begin() + mid);
    if (r.buffer_copies > std::numeric_limits<std::size_t>::max() - left)
      throw std::length_error("Timsort scratch count");
    r.buffer_copies += left;
    std::size_t i = 0, j = mid, k = base;
    while (i < left && j < hi) {
      count(r.assignments, o.max_assignments);
      if (!less(r.values[j], tmp[i]))
        r.values[k++] = tmp[i++];
      else
        r.values[k++] = r.values[j++];
    }
    while (i < left) {
      count(r.assignments, o.max_assignments);
      r.values[k++] = tmp[i++];
    }
    stack[at].length += right;
    stack.erase(stack.begin() + at + 1);
    ++r.merges;
    record("merge", base, left, right);
  };
  /// @brief Restore both top-three and next-deeper run inequalities before consuming further input.
  auto collapse = [&]() {
    while (stack.size() > 1) {
      auto at = stack.size() - 2;
      if ((at > 0 && stack[at - 1].length <= stack[at].length + stack[at + 1].length) ||
          (at > 1 && stack[at - 2].length <= stack[at - 1].length + stack[at].length)) {
        if (at > 0 && stack[at - 1].length < stack[at + 1].length)
          --at;
        merge(at);
      } else if (stack[at].length <= stack[at + 1].length)
        merge(at);
      else
        break;
    }
  };
  std::size_t lo = 0;
  while (lo < n) {
    auto [end, descending] = natural_end(r.values, lo, less);
    if (descending)
      for (std::size_t a = lo, b = end; a < --b; ++a) {
        std::swap(r.values[a], r.values[b]);
        ++r.reversed_swaps;
      }
    auto length = end - lo;
    ++r.natural_runs;
    record("natural", lo, length);
    if (length < r.minrun) {
      auto force = std::min(r.minrun, n - lo);
      for (std::size_t start = lo + length; start < lo + force; ++start) {
        const T     pivot = r.values[start];
        std::size_t left = lo, right = start;
        while (left < right) {
          auto mid = left + (right - left) / 2;
          if (less(pivot, r.values[mid]))
            right = mid;
          else
            left = mid + 1;
        }
        for (std::size_t i = start; i > left; --i) {
          count(r.assignments, o.max_assignments);
          r.values[i] = r.values[i - 1];
        }
        count(r.assignments, o.max_assignments);
        r.values[left] = pivot;
      }
      length = force;
      record("extend", lo, length);
    }
    stack.push_back({lo, length});
    record("push", lo, length);
    collapse();
    lo += length;
  }
  while (stack.size() > 1)
    merge(stack.size() - 2);
  return r;
}
} // namespace timsort_detail
/// @brief Sort through the actual checked stable natural-run producer.
/// Copyable keys require strict weak operator<; built-in floating NaN/infinity reject. Detect nondecreasing or strictly descending runs; reverse only strict descent to preserve equal-key order. Extend short runs by stable upper-bound binary insertion and merge adjacent runs preferring the left on ties. Implements both top-three and next-deeper collapse checks; this repairs the old three-run-only stack policy. Galloping is omitted. Standard adaptive stable O(n log n) worst-case comparison behavior and linear scan on already sorted input, with linear scratch/output storage; actual work/capture policies can stop earlier. Owned copies, no shared state; compatibility in-place sort replaces caller vector only on success.
/// @tparam T Copyable strict-weak-order key type.
/// @param values Owned complete key vector.
/// @param options Complete key and actual selected-work/capture policies; zero disables budgets.
/// @return Owned actual stable sorted keys, executed work and optional stack/key events.
/// @throws std::invalid_argument Nonfinite built-in floating keys.
/// @throws std::length_error Int capacity, complete keys or actual work/history/counter capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::timsort_run(std::vector<int>{3,1,2});
/// std::print("{}\n",r.values); // [1, 2, 3]
/// @endcode
/// @par References
/// - Peters, CPython listsort design; corrected merge-collapse checks include the next-deeper triple.
/// @see timsort()
template<class T>
TimsortRun<T> timsort_run(std::vector<T> values, const TimsortOptions& options = {}) {
  return timsort_detail::execute(std::move(values), options, false);
}
/// @brief Sort through the actual checked stable natural-run producer.
/// Copyable keys require strict weak operator<; built-in floating NaN/infinity reject. Detect nondecreasing or strictly descending runs; reverse only strict descent to preserve equal-key order. Extend short runs by stable upper-bound binary insertion and merge adjacent runs preferring the left on ties. Implements both top-three and next-deeper collapse checks; this repairs the old three-run-only stack policy. Galloping is omitted. Standard adaptive stable O(n log n) worst-case comparison behavior and linear scan on already sorted input, with linear scratch/output storage; actual work/capture policies can stop earlier. Owned copies, no shared state; compatibility in-place sort replaces caller vector only on success.
/// @tparam T Copyable strict-weak-order key type.
/// @param values Owned complete key vector.
/// @param options Complete key and actual selected-work/capture policies; zero disables budgets.
/// @return Owned actual stable sorted keys, executed work and optional stack/key events.
/// @throws std::invalid_argument Nonfinite built-in floating keys.
/// @throws std::length_error Int capacity, complete keys or actual work/history/counter capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::timsort_trace(std::vector<int>{3,1,2});
/// std::print("{}\n",r.values); // [1, 2, 3]
/// @endcode
/// @par References
/// - Peters, CPython listsort design; corrected merge-collapse checks include the next-deeper triple.
/// @see timsort()
template<class T>
TimsortRun<T> timsort_trace(std::vector<T> values, const TimsortOptions& options = {}) {
  return timsort_detail::execute(std::move(values), options, true);
}
/// @brief Preserve actual stable in-place Timsort through an owned checked run.
/// Copyable keys require strict weak operator<; built-in floating NaN/infinity reject. Detect nondecreasing or strictly descending runs; reverse only strict descent to preserve equal-key order. Extend short runs by stable upper-bound binary insertion and merge adjacent runs preferring the left on ties. Implements both top-three and next-deeper collapse checks; this repairs the old three-run-only stack policy. Galloping is omitted. Standard adaptive stable O(n log n) worst-case comparison behavior and linear scan on already sorted input, with linear scratch/output storage; actual work/capture policies can stop earlier. Owned copies, no shared state; compatibility in-place sort replaces caller vector only on success.
/// @tparam T Copyable strict-weak-order key type.
/// @param a Mutable complete caller keys, replaced only after successful sort.
/// @throws std::invalid_argument Nonfinite floating keys.
/// @throws std::length_error Default complete/work or representational capacity exceeded.
/// @par Example
/// @code
/// std::vector<int> a{3,1,2};dm::algorithms::timsort(a);
/// std::print("{}\n",a); // [1, 2, 3]
/// @endcode
/// @see timsort_run()
template<class T>
void timsort(std::vector<T>& a) {
  a = timsort_run(a).values;
}
/// @brief Measure actual maximal raw runs before any Timsort minrun extension.
/// Uses the same native run detector as sorting, without reversing or modifying keys. This input profile is distinct from the runs the sorting producer consumes after extending short runs. Finite floating values, int index capacity and complete key/comparison policies checked; unrelated assignment/history controls unused. O(n) scan/output and no shared state.
/// @tparam T Borrowed strict-weak-order key type.
/// @param values Borrowed complete strict-weak-order keys.
/// @param options Complete key and actual comparison budgets.
/// @return Owned actual raw run lengths/directions and profile comparison count.
/// @throws std::invalid_argument Nonfinite floating key.
/// @throws std::length_error Int/key/comparison capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::timsort_input_profile(std::vector<int>{3,2,1,4,5});
/// std::print("{}\n",r.lengths); // [3, 2]
/// @endcode
/// @see timsort_run()
template<class T>
TimsortInputProfile timsort_input_profile(
    const std::vector<T>& values,
    const TimsortOptions& options = {}) {
  if (values.size() > std::size_t(std::numeric_limits<int>::max()) ||
      (options.max_keys && values.size() > options.max_keys))
    throw std::length_error("Timsort profile keys");
  if constexpr (std::is_floating_point_v<T>)
    for (T x : values)
      if (!std::isfinite(x))
        throw std::invalid_argument("Timsort profile finite keys");
  TimsortInputProfile r;
  /// @brief Count and execute one actual profile key comparison.
  auto less = [&](const T& a, const T& b) {
    if (r.comparisons == std::numeric_limits<std::size_t>::max() ||
        (options.max_comparisons && r.comparisons >= options.max_comparisons))
      throw std::length_error("Timsort profile comparisons");
    ++r.comparisons;
    return a < b;
  };
  for (std::size_t lo = 0; lo < values.size();) {
    auto [end, descending] = timsort_detail::natural_end(values, lo, less);
    r.lengths.push_back(end - lo);
    r.descending.push_back(descending);
    lo = end;
  }
  return r;
}
/// @brief Generate the retained eleven1000-key prefix-shuffle fixtures.
/// Fresh ascending0..999 base for each0,10,...,100percent prefix shuffle. One local mt19937_64 seed9 consumed sequentially, preserving original cohort stream. Owned fixture arrays and no shared state.
/// @return Eleven complete1000-key arrays in percentage order.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::timsort_example_arrays().size()); // 11
/// @endcode
/// @see timsort_input_profile()
std::vector<std::vector<int>> timsort_example_arrays();
} // namespace datamunge::algorithms
