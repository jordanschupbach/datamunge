/// @file selection_sort.cpp
/// @brief Single checked real suffix-minimum producer and original seeded study.
#include <algorithm>
#include <datamunge/algorithms/selection_sort.hpp>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
/// @brief Canonical checked native producers and owned algorithm values.
namespace datamunge::algorithms {
/// @brief Private invocation-local checked helpers.
namespace {
/// @brief Check geometry before actual minimum scanning and optional actual state retention.
/// @param values Owned complete working keys.
/// @param o Complete bounds.
/// @param capture Retain actual full states.
/// @return Owned actual result/work/history.
SelectionSortRun sort(std::vector<int> values, const SelectionSortOptions& o, bool capture) {
  const auto n = values.size(), max = std::numeric_limits<std::size_t>::max(),
             passes = n ? n - 1 : 0;
  if (o.max_keys && n > o.max_keys)
    throw std::length_error("selection: keys");
  auto first = n, second = passes;
  if (first % 2 == 0)
    first /= 2;
  else
    second /= 2;
  if (second && first > max / second)
    throw std::length_error("selection: work capacity");
  const auto comparisons = first * second;
  if (o.max_comparisons && comparisons > o.max_comparisons)
    throw std::length_error("selection: complete comparison policy");
  if (capture && (n == max || (passes && n + 1 > max / passes) ||
                  (o.max_trace_entries && passes * (n + 1) > o.max_trace_entries)))
    throw std::length_error("selection: complete history policy");
  SelectionSortRun out;
  out.values = std::move(values);
  for (std::size_t i = 0; i < passes; ++i) {
    std::size_t minimum = i, count = 0;
    for (std::size_t j = i + 1; j < n; ++j) {
      ++out.comparisons;
      ++count;
      if (out.values[j] < out.values[minimum])
        minimum = j;
    }
    const int  value   = out.values[minimum];
    const bool swapped = minimum != i;
    if (swapped) {
      std::swap(out.values[i], out.values[minimum]);
      ++out.swaps;
    }
    if (capture)
      out.steps.push_back({i, minimum, value, swapped, count, out.values});
  }
  return out;
}
} // namespace
SelectionSortRun selection_sort_run(std::vector<int> v, const SelectionSortOptions& o) {
  return sort(std::move(v), o, false);
}
SelectionSortRun selection_sort_trace(std::vector<int> v, const SelectionSortOptions& o) {
  return sort(std::move(v), o, true);
}
std::vector<int> selection_sort(std::vector<int> v) {
  return selection_sort_run(std::move(v)).values;
}
std::vector<std::vector<int>> selection_sort_example_arrays() {
  std::mt19937                  rng(2);
  std::vector<std::vector<int>> out;
  for (int n = 5; n <= 200; n += 5) {
    std::vector<int> v(n);
    std::iota(v.begin(), v.end(), 0);
    std::shuffle(v.begin(), v.end(), rng);
    out.push_back(std::move(v));
  }
  return out;
}
} // namespace datamunge::algorithms
