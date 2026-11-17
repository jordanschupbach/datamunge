/// @file insertion_sort.cpp
/// @brief Single strict-shift insertion kernel with optional stable identities and actual snapshots.
#include <datamunge/algorithms/insertion_sort.hpp>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
namespace datamunge::algorithms {
/// @brief Private complete admission and shared execution implementation.
namespace {
/// @brief Increment one selected actual count without unsigned wrap.
/// @param x Mutable count.
/// @throws std::length_error On exhausted size_t.
void increment(std::size_t& x) {
  if (x == std::numeric_limits<std::size_t>::max())
    throw std::length_error("insertion count");
  ++x;
}
/// @brief Admit complete borrowed input and optional known full snapshot geometry.
/// @param n Complete input length.
/// @param o Optional policies.
/// @param trace Whether full history admission is required.
/// @throws std::length_error On complete cap or geometry overflow.
void admit(std::size_t n, InsertionSortOptions o, bool trace) {
  if (o.max_elements && n > o.max_elements)
    throw std::length_error("insertion complete input cap");
  if (!trace || n < 2)
    return;
  const auto max = std::numeric_limits<std::size_t>::max();
  if (n > (max - 1) / 2)
    throw std::length_error("insertion snapshot width");
  const auto width = 1 + 2 * n;
  if (n - 1 > max / width || (o.max_history_entries && (n - 1) * width > o.max_history_entries))
    throw std::length_error("insertion complete history cap");
}
/// @brief Execute the original strict larger-than insertion loop with optional records.
/// @tparam Identities Retain original identities alongside actual value movement.
/// @tparam Trace Retain actual complete post-insertion snapshots.
/// @param a Owned admitted working input.
/// @return Actual final values/work and optional original/step histories.
/// @throws std::length_error On selected count/capacity overflow.
/// @throws std::bad_alloc On output/working/history allocation failure.
template<bool Identities, bool Trace>
InsertionSortTrace execute(std::vector<int> a) {
  InsertionSortTrace out;
  if constexpr (Trace)
    out.original = a;
  if constexpr (Identities) {
    out.run.identities.resize(a.size());
    std::iota(out.run.identities.begin(), out.run.identities.end(), 0);
  }
  for (std::size_t i = 1; i < a.size(); ++i) {
    const int   key   = a[i];
    std::size_t keyid = 0;
    if constexpr (Identities)
      keyid = out.run.identities[i];
    std::size_t j = i, comparisons = 0, shifts = 0;
    while (j > 0) {
      increment(out.run.work.comparisons);
      increment(comparisons);
      if (!(a[j - 1] > key))
        break;
      a[j] = a[j - 1];
      if constexpr (Identities)
        out.run.identities[j] = out.run.identities[j - 1];
      --j;
      increment(out.run.work.shifts);
      increment(shifts);
    }
    a[j] = key;
    if constexpr (Identities)
      out.run.identities[j] = keyid;
    increment(out.run.work.placements);
    if constexpr (Trace)
      out.steps.push_back({i, key, keyid, j, comparisons, shifts, a, out.run.identities});
  }
  out.run.values = std::move(a);
  return out;
}
} // namespace
std::vector<int> insertion_sort(std::vector<int> a) {
  return execute<false, false>(std::move(a)).run.values;
}
InsertionSortRun insertion_sort_run(const std::vector<int>& a, InsertionSortOptions options) {
  admit(a.size(), options, false);
  return execute<true, false>(a).run;
}
InsertionSortTrace insertion_sort_trace(const std::vector<int>& a, InsertionSortOptions options) {
  admit(a.size(), options, true);
  return execute<true, true>(a);
}
} // namespace datamunge::algorithms
