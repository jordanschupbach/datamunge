/// @file shell_sort.cpp
/// @brief Same-execution Shell and insertion work with bounded histories.
#include <algorithm>
#include <datamunge/algorithms/shell_sort.hpp>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
/// @brief Canonical checked native producers and owned algorithm values.
namespace datamunge::algorithms {
/// @brief Private invocation-local checked helpers.
namespace {
/// @brief Run actual gapped insertion, optionally recording completed passes.
/// @param a Owned keys.
/// @param o Policy.
/// @param trace Capture passes.
/// @param insertion Use only gap one.
/// @return Actual sorted result.
ShellSortRun run(std::vector<int> a, const ShellSortOptions& o, bool trace, bool insertion) {
  if (o.max_keys && a.size() > o.max_keys)
    throw std::length_error("shell: keys");
  ShellSortRun r;
  r.values             = std::move(a);
  const auto  n        = r.values.size();
  std::size_t retained = 0;
  for (std::size_t gap = insertion ? (n > 1 ? 1 : 0) : n / 2; gap; gap = insertion ? 0 : gap / 2) {
    const auto before = r.comparisons, shifted = r.shifts;
    for (std::size_t i = gap; i < n; ++i) {
      const int key = r.values[i];
      auto      j   = i;
      while (j >= gap) {
        if (r.comparisons == std::numeric_limits<std::size_t>::max() ||
            (o.max_comparisons && r.comparisons >= o.max_comparisons))
          throw std::length_error("shell: comparisons");
        ++r.comparisons;
        if (r.values[j - gap] <= key)
          break;
        r.values[j] = r.values[j - gap];
        j -= gap;
        if (r.shifts == std::numeric_limits<std::size_t>::max())
          throw std::length_error("shell: shifts");
        ++r.shifts;
      }
      r.values[j] = key;
    }
    if (trace) {
      if (n == std::numeric_limits<std::size_t>::max() ||
          retained > std::numeric_limits<std::size_t>::max() - n - 1)
        throw std::length_error("shell: history capacity");
      retained += n + 1;
      if (o.max_history_values && retained > o.max_history_values)
        throw std::length_error("shell: history");
      r.passes.push_back({gap, r.comparisons - before, r.shifts - shifted, r.values});
    }
  }
  return r;
}
} // namespace
ShellSortRun shell_sort_run(std::vector<int> a, const ShellSortOptions& o) {
  return run(std::move(a), o, false, false);
}
ShellSortRun shell_sort_trace(std::vector<int> a, const ShellSortOptions& o) {
  return run(std::move(a), o, true, false);
}
ShellSortRun insertion_sort_work(std::vector<int> a, const ShellSortOptions& o) {
  return run(std::move(a), o, false, true);
}
std::vector<int> shell_sort(std::vector<int> a) {
  return shell_sort_run(std::move(a)).values;
}
std::vector<std::vector<int>> shell_sort_example_arrays() {
  std::mt19937                  rng(3);
  std::vector<std::vector<int>> out;
  for (int n = 10; n <= 400; n += 10) {
    std::vector<int> v(n);
    std::iota(v.begin(), v.end(), 0);
    std::shuffle(v.begin(), v.end(), rng);
    out.push_back(std::move(v));
  }
  return out;
}
} // namespace datamunge::algorithms
