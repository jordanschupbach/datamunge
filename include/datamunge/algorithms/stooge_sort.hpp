#pragma once
/// @file stooge_sort.hpp
/// @brief Generic checked actual overlapping-recursion sorting work.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
namespace datamunge::algorithms {

/// @brief Own the StoogeSortOptions contract for one independent native call.
/// Field descriptions specify actual producer values or policy units. Defaults are in-class initializers. Containers own their values; no borrowed storage or shared mutable state. Creating the record alone performs no algorithm execution.
/// @par Example
/// @code
/// dm::algorithms::StoogeSortOptions r{};
/// (void)r;
/// @endcode
/// @see stooge_sort_run()
struct StoogeSortOptions {
  std::size_t max_keys{1000000}; ///< Complete key count; zero disables.
  std::size_t max_comparisons{
      20000000}; ///< Actual end-key comparisons/recursive calls; zero disables.
  std::size_t max_history_values{
      1000000}; ///< Actual post-call records plus copied complete key states; zero disables.
};
/// @brief Own the StoogeSortStep contract for one independent native call.
/// Field descriptions specify actual producer values or policy units. Defaults are in-class initializers. Containers own their values; no borrowed storage or shared mutable state. Creating the record alone performs no algorithm execution.
/// @tparam T Copyable strict-weak-order key; example instantiates int.
/// @par Example
/// @code
/// dm::algorithms::StoogeSortStep<int> r{};
/// (void)r;
/// @endcode
/// @see stooge_sort_run()
template<class T>
struct StoogeSortStep {
  std::size_t first{};   ///< Actual inclusive recursion start.
  std::size_t last{};    ///< Actual inclusive recursion end.
  std::size_t depth{};   ///< Actual recursion depth, root0.
  bool        swapped{}; ///< Actual endpoint exchange in this invocation before recursive children.
  std::vector<T> after;  ///< Actual complete keys after this invocation and all its children.
};
/// @brief Own the StoogeSortRun contract for one independent native call.
/// Field descriptions specify actual producer values or policy units. Defaults are in-class initializers. Containers own their values; no borrowed storage or shared mutable state. Creating the record alone performs no algorithm execution.
/// @tparam T Copyable strict-weak-order key; example instantiates int.
/// @par Example
/// @code
/// dm::algorithms::StoogeSortRun<int> r{};
/// (void)r;
/// @endcode
/// @see stooge_sort_run()
template<class T>
struct StoogeSortRun {
  std::vector<T> values; ///< Owned actual sorted keys; unstable.
  std::size_t
      comparisons{}; ///< Actual strict endpoint comparisons, one per executed recursive call.
  std::size_t                    swaps{}; ///< Actual end-key exchanges.
  std::vector<StoogeSortStep<T>> steps;   ///< Optional actual completed-call states in postorder.
};
/// @brief Invocation-local generic overlapping-recursion sort kernels.
namespace stooge_detail {
/// @brief Execute actual range comparisons and overlapping recursion.
/// @tparam T Copyable keys with strict weak ordering by operator<.
/// @param r Mutable local producing state.
/// @param o Policies.
/// @param first Inclusive range start.
/// @param last Inclusive range end.
/// @param depth Actual root-relative depth.
/// @param capture Retain actual post-call keys.
/// @param retained Actual retained value units.
template<class T>
void visit(
    StoogeSortRun<T>&        r,
    const StoogeSortOptions& o,
    std::size_t              first,
    std::size_t              last,
    std::size_t              depth,
    bool                     capture,
    std::size_t&             retained) {
  const auto limit = std::numeric_limits<std::size_t>::max();
  if (r.comparisons == limit || (o.max_comparisons && r.comparisons >= o.max_comparisons))
    throw std::length_error("stooge: comparisons");
  ++r.comparisons;
  const bool swapped = r.values[last] < r.values[first];
  if (swapped) {
    std::swap(r.values[first], r.values[last]);
    ++r.swaps;
  }
  const auto length = last - first + 1;
  if (length > 2) {
    const auto third = length / 3;
    visit(r, o, first, last - third, depth + 1, capture, retained);
    visit(r, o, first + third, last, depth + 1, capture, retained);
    visit(r, o, first, last - third, depth + 1, capture, retained);
  }
  if (capture) {
    const auto n = r.values.size();
    if (n == limit || retained > limit - n - 1)
      throw std::length_error("stooge: history capacity");
    retained += n + 1;
    if (o.max_history_values && retained > o.max_history_values)
      throw std::length_error("stooge: history");
    r.steps.push_back({first, last, depth, swapped, r.values});
  }
}
/// @brief Validate and run one owned generic sort.
/// @tparam T Copyable strict-weak-order key type.
/// @param values Owned complete input.
/// @param o Policies.
/// @param capture Retain actual postorder history.
/// @return Actual owned sorted result/work.
template<class T>
StoogeSortRun<T> run(std::vector<T> values, const StoogeSortOptions& o, bool capture) {
  if (o.max_keys && values.size() > o.max_keys)
    throw std::length_error("stooge: keys");
  if constexpr (std::is_floating_point_v<T>)
    for (T x : values)
      if (!std::isfinite(x))
        throw std::invalid_argument("stooge: finite floating keys");
  StoogeSortRun<T> r;
  r.values             = std::move(values);
  std::size_t retained = 0;
  if (r.values.size() > 1)
    visit(r, o, 0, r.values.size() - 1, 0, capture, retained);
  return r;
}
} // namespace stooge_detail
/// @brief Sort owned keys through actual overlapping stooge recursion.
/// Compare/exchange range ends, then for length>2 visit first two-thirds, last two-thirds and first two-thirds. Each recursive child strictly shrinks; overlapping coverage preserves the smallest prefix and largest suffix and produces a sorted interval. Unstable. Empty/singleton inputs execute no calls. Strict weak ordering required; nonfinite built-in floating keys are rejected. Worst-case comparisons Theta(n^(log(3)/log(3/2))) with rounding plateaus; one actual comparison per invocation, not a report recurrence replay. Owned input copy leaves caller keys unchanged. User key copy/comparison exceptions propagate; no shared state. Optional complete post-call states can dominate storage.
/// @tparam T Copyable keys ordered by a strict weak operator<.
/// @param values Owned complete key vector.
/// @param options Complete input and actual comparison/captured-value policies; zero disables policies.
/// @return Owned actual sorted values, comparisons, swaps and optional completed-call history.
/// @throws std::invalid_argument Nonfinite built-in floating key.
/// @throws std::length_error Input/work/history/counter/container capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::stooge_sort_run(std::vector<int>{3,1,2});
/// std::print("{} {}\n",r.values,r.comparisons); // [1, 2, 3] 4
/// @endcode
/// @par References
/// - The standard overlapping two-thirds stooge recurrence; analysis follows the Master theorem with ceiling-aware bounds.
/// @see stooge_sort()
template<class T>
StoogeSortRun<T> stooge_sort_run(std::vector<T> values, const StoogeSortOptions& options = {}) {
  return stooge_detail::run(std::move(values), options, false);
}
/// @brief Sort owned keys through actual overlapping stooge recursion with postorder states.
/// Compare/exchange range ends, then for length>2 visit first two-thirds, last two-thirds and first two-thirds. Each recursive child strictly shrinks; overlapping coverage preserves the smallest prefix and largest suffix and produces a sorted interval. Unstable. Empty/singleton inputs execute no calls. Strict weak ordering required; nonfinite built-in floating keys are rejected. Worst-case comparisons Theta(n^(log(3)/log(3/2))) with rounding plateaus; one actual comparison per invocation, not a report recurrence replay. Owned input copy leaves caller keys unchanged. User key copy/comparison exceptions propagate; no shared state. Optional complete post-call states can dominate storage.
/// @tparam T Copyable keys ordered by a strict weak operator<.
/// @param values Owned complete key vector.
/// @param options Complete input and actual comparison/captured-value policies; zero disables policies.
/// @return Owned actual sorted values, comparisons, swaps and optional completed-call history.
/// @throws std::invalid_argument Nonfinite built-in floating key.
/// @throws std::length_error Input/work/history/counter/container capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::stooge_sort_trace(std::vector<int>{3,1,2});
/// std::print("{} {}\n",r.values,r.comparisons); // [1, 2, 3] 4
/// @endcode
/// @par References
/// - The standard overlapping two-thirds stooge recurrence; analysis follows the Master theorem with ceiling-aware bounds.
/// @see stooge_sort()
template<class T>
StoogeSortRun<T> stooge_sort_trace(std::vector<T> values, const StoogeSortOptions& options = {}) {
  return stooge_detail::run(std::move(values), options, true);
}
/// @brief Preserve the generic in-place stooge signature with checked actual work.
/// Sorts an owned copy and only replaces caller vector on success. Uses default policies; empty/singleton inputs stay unchanged. Strict weak operator< required; user operations may throw.
/// @tparam T Copyable strict-weak-order key type.
/// @param a Mutable complete caller keys.
/// @throws std::invalid_argument Nonfinite floating key.
/// @throws std::length_error Default input/work/capacity bound exceeded.
/// @par Example
/// @code
/// std::vector<int> a{3,1,2};dm::algorithms::stooge_sort(a);
/// std::print("{}\n",a); // [1, 2, 3]
/// @endcode
/// @see stooge_sort_run()
template<class T>
void stooge_sort(std::vector<T>& a) {
  a = stooge_sort_run(a).values;
}
} // namespace datamunge::algorithms
