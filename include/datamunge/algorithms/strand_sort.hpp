#pragma once
/// @file strand_sort.hpp
/// @brief Generic actual greedy strand extraction and stable merge states.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
namespace datamunge::algorithms {

/// @brief Own the StrandSortOptions contract for one independent native call.
/// Field descriptions specify actual producer values or policy units. Defaults are in-class initializers. Containers own their values; no borrowed storage or shared mutable state. Creating the record alone performs no algorithm execution.
/// @par Example
/// @code
/// dm::algorithms::StrandSortOptions r{};
/// (void)r;
/// @endcode
/// @see strand_sort_run()
struct StrandSortOptions {
  std::size_t max_keys{1000000}; ///< Complete input key count; zero disables.
  std::size_t max_comparisons{
      20000000}; ///< Actual extraction plus merge key comparisons; zero disables.
  std::size_t max_history_values{
      1000000}; ///< Actual pass records plus copied strand/remaining/result keys; zero disables.
};
/// @brief Own the StrandSortStep contract for one independent native call.
/// Field descriptions specify actual producer values or policy units. Defaults are in-class initializers. Containers own their values; no borrowed storage or shared mutable state. Creating the record alone performs no algorithm execution.
/// @tparam T Copyable strict-weak-order key; example instantiates int.
/// @par Example
/// @code
/// dm::algorithms::StrandSortStep<int> r{};
/// (void)r;
/// @endcode
/// @see strand_sort_run()
template<class T>
struct StrandSortStep {
  std::vector<T> strand;    ///< Actual greedily extracted nondecreasing subsequence.
  std::vector<T> remaining; ///< Actual unselected input, preserving original relative order.
  std::vector<T> merged;    ///< Actual result after stable merge with this strand.
};
/// @brief Own the StrandSortRun contract for one independent native call.
/// Field descriptions specify actual producer values or policy units. Defaults are in-class initializers. Containers own their values; no borrowed storage or shared mutable state. Creating the record alone performs no algorithm execution.
/// @tparam T Copyable strict-weak-order key; example instantiates int.
/// @par Example
/// @code
/// dm::algorithms::StrandSortRun<int> r{};
/// (void)r;
/// @endcode
/// @see strand_sort_run()
template<class T>
struct StrandSortRun {
  std::vector<T> values;    ///< Owned actual nondecreasing stable result.
  std::size_t    strands{}; ///< Actual extraction/merge passes, zero for empty input.
  std::size_t    extraction_comparisons{}; ///< Actual less-than tests against current strand tail.
  std::size_t    merge_comparisons{};   ///< Actual less-than tests while both merge inputs remain.
  std::vector<StrandSortStep<T>> steps; ///< Optional actual completed-pass states.
};
/// @brief Invocation-local generic strand extraction and stable-merge kernels.
namespace strand_detail {
/// @brief Produce actual extracted strands and merges.
/// @tparam T Copyable strict-weak-order key type.
/// @param input Owned remaining keys.
/// @param o Policies.
/// @param capture Retain actual pass states.
/// @return Actual owned result.
template<class T>
StrandSortRun<T> run(std::vector<T> input, const StrandSortOptions& o, bool capture) {
  if (o.max_keys && input.size() > o.max_keys)
    throw std::length_error("strand: keys");
  if constexpr (std::is_floating_point_v<T>)
    for (T x : input)
      if (!std::isfinite(x))
        throw std::invalid_argument("strand: finite floating keys");
  StrandSortRun<T> r;
  std::size_t      work = 0, retained = 0;
  const auto       limit = std::numeric_limits<std::size_t>::max();
  auto             count = [&](std::size_t& counter) {
    if (work == limit || (o.max_comparisons && work >= o.max_comparisons))
      throw std::length_error("strand: comparisons");
    ++work;
    ++counter;
  };
  while (!input.empty()) {
    std::vector<T> strand{input.front()}, remaining, merged;
    for (std::size_t i = 1; i < input.size(); ++i) {
      count(r.extraction_comparisons);
      if (!(input[i] < strand.back()))
        strand.push_back(input[i]);
      else
        remaining.push_back(input[i]);
    }
    merged.reserve(r.values.size() + strand.size());
    std::size_t i = 0, j = 0;
    while (i < r.values.size() && j < strand.size()) {
      count(r.merge_comparisons);
      if (strand[j] < r.values[i])
        merged.push_back(strand[j++]);
      else
        merged.push_back(r.values[i++]);
    }
    while (i < r.values.size())
      merged.push_back(r.values[i++]);
    while (j < strand.size())
      merged.push_back(strand[j++]);
    ++r.strands;
    if (capture) {
      for (const auto n : {std::size_t{1}, strand.size(), remaining.size(), merged.size()}) {
        if (n > limit - retained)
          throw std::length_error("strand: history capacity");
        retained += n;
        if (o.max_history_values && retained > o.max_history_values)
          throw std::length_error("strand: history");
      }
      r.steps.push_back({strand, remaining, merged});
    }
    r.values = std::move(merged);
    input    = std::move(remaining);
  }
  return r;
}
} // namespace strand_detail
/// @brief Sort owned keys by actual greedy strands and stable merges.
/// Take the first remaining key, then append every later key not less than the current strand tail; unselected keys retain order. Stable-merge into the result, preferring earlier result keys on equal comparison. At least one key leaves the remaining input per pass, so the process terminates. Every strand and the accumulated result are nondecreasing, proving the final sort. Equal-key order is preserved. Strict weak operator< is required; nonfinite built-in floats are rejected. Worst-case time O(n^2), linear working storage without optional quadratic full-state captures. Already nondecreasing input needs one strand and n-1 extraction comparisons. User key operation exceptions propagate; owned results and no shared state.
/// @tparam T Copyable strict-weak-order key type.
/// @param values Owned complete key vector.
/// @param options Complete input and actual comparison/captured-value bounds; zero disables policies.
/// @return Owned actual sorted values, strand/comparison counts and optional states.
/// @throws std::invalid_argument Nonfinite built-in floating key.
/// @throws std::length_error Input/work/history/counter/container capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::strand_sort_run(std::vector<int>{3,1,2});
/// std::print("{} {}\n",r.values,r.strands); // [1, 2, 3] 2
/// @endcode
/// @see strand_sort()
template<class T>
StrandSortRun<T> strand_sort_run(std::vector<T> values, const StrandSortOptions& options = {}) {
  return strand_detail::run(std::move(values), options, false);
}
/// @brief Sort owned keys by actual greedy strands and stable merges with captured passes.
/// Take the first remaining key, then append every later key not less than the current strand tail; unselected keys retain order. Stable-merge into the result, preferring earlier result keys on equal comparison. At least one key leaves the remaining input per pass, so the process terminates. Every strand and the accumulated result are nondecreasing, proving the final sort. Equal-key order is preserved. Strict weak operator< is required; nonfinite built-in floats are rejected. Worst-case time O(n^2), linear working storage without optional quadratic full-state captures. Already nondecreasing input needs one strand and n-1 extraction comparisons. User key operation exceptions propagate; owned results and no shared state.
/// @tparam T Copyable strict-weak-order key type.
/// @param values Owned complete key vector.
/// @param options Complete input and actual comparison/captured-value bounds; zero disables policies.
/// @return Owned actual sorted values, strand/comparison counts and optional states.
/// @throws std::invalid_argument Nonfinite built-in floating key.
/// @throws std::length_error Input/work/history/counter/container capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::strand_sort_trace(std::vector<int>{3,1,2});
/// std::print("{} {}\n",r.values,r.strands); // [1, 2, 3] 2
/// @endcode
/// @see strand_sort()
template<class T>
StrandSortRun<T> strand_sort_trace(std::vector<T> values, const StrandSortOptions& options = {}) {
  return strand_detail::run(std::move(values), options, true);
}
/// @brief Preserve the generic in-place stable strand-sort signature.
/// Sorts an owned copy with default policies and replaces caller vector only on success. User key copy/comparison exceptions propagate.
/// @tparam T Copyable strict-weak-order key type.
/// @param a Mutable complete caller keys.
/// @throws std::invalid_argument Nonfinite floating key.
/// @throws std::length_error Default input/work/capacity bound exceeded.
/// @par Example
/// @code
/// std::vector<int> a{3,1,2};dm::algorithms::strand_sort(a);
/// std::print("{}\n",a); // [1, 2, 3]
/// @endcode
/// @see strand_sort_run()
template<class T>
void strand_sort(std::vector<T>& a) {
  a = strand_sort_run(a).values;
}
/// @brief Generate the original eleven partial-prefix-shuffle fixtures.
/// Starts with ascending0..399. One mt19937_64 seed7 shuffles a fresh copy prefix at0,10,...,100 percent sequentially, preserving original engine draw order. Owned deterministic arrays, no shared state.
/// @return Eleven complete400-key arrays in percentage order.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::strand_sort_example_arrays().size()); // 11
/// @endcode
/// @see strand_sort_run()
std::vector<std::vector<int>> strand_sort_example_arrays();
} // namespace datamunge::algorithms
