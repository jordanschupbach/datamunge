/// @file merge_sort.cpp
/// @brief Canonical merge sort API, producing histories and checked native contracts.

#include <algorithm>
#include <bit>
#include <datamunge/algorithms/merge_sort.hpp>
#include <limits>
#include <stdexcept>

namespace datamunge::algorithms {
/// @brief Implementation-only helpers for canonical merge sort production and capture.
namespace {
/// @brief Advance one actual count or logical capture count without wrap.
/// @param value Existing count.
/// @param amount Count to add.
/// @return Exact advanced count.
/// @throws std::length_error On size_t overflow.
std::size_t advance(std::size_t value, std::size_t amount = 1) {
  if (amount > SIZE_MAX - value)
    throw std::length_error("merge-sort count geometry");
  return value + amount;
}
/// @brief Admit the next complete actual history record before the producing operation.
/// @param trace Optional owned capture target.
/// @param limit Optional complete logical cap, zero off.
/// @throws std::length_error On actual admission or count overflow.
void admit(MergeSortTrace* trace, std::size_t limit) {
  if (!trace)
    return;
  auto next = advance(trace->captured_entries);
  if (limit && next > limit)
    throw std::length_error("merge-sort complete capture");
  trace->captured_entries = next;
}
/// @brief Map an integer value to its sorting key.
/// @param value Integer value.
/// @return Unchanged value.
std::int64_t key(std::int64_t value) {
  return value;
}
/// @brief Map a keyed pair to its first field only.
/// @param value Original keyed pair.
/// @return Widened first field.
std::int64_t key(const std::pair<int, int>& value) {
  return value.first;
}
/// @brief Map an owned capture item to its key only.
/// @param value Owned key/tag item.
/// @return Unchanged key.
std::int64_t key(const MergeSortItem& value) {
  return value.key;
}
/// @brief Recover an owned capture item from its same-run source value.
/// @param value Actual source item.
/// @return Exact key/tag copy.
MergeSortItem item(const MergeSortItem& value) {
  return value;
}
/// @brief Noncapturing integer source adapter, never called with a nonnull trace.
/// @param value Actual source value.
/// @return Key and unused zero tag.
MergeSortItem item(std::int64_t value) {
  return {value, 0};
}
/// @brief Noncapturing pair source adapter, preserving payload exactly.
/// @param value Actual source pair.
/// @return Widened key and payload.
MergeSortItem item(const std::pair<int, int>& value) {
  return {value.first, value.second};
}
/// @brief One canonical stable recursive sorting body for all public entry points.
/// @tparam T Int64 value, keyed pair or owned trace item.
/// @param data Actual mutable current runs.
/// @param buffer Reusable scratch storage.
/// @param lo Inclusive range start.
/// @param hi Exclusive range end.
/// @param depth Actual current active call depth.
/// @param work Actual operation counters.
/// @param trace Optional producing capture.
/// @param limit Optional complete logical cap, zero off.
/// @throws std::length_error On actual counter or capture admission overflow.
template<class T>
void sort(
    std::vector<T>& data,
    std::vector<T>& buffer,
    std::size_t     lo,
    std::size_t     hi,
    std::size_t     depth,
    MergeSortWork&  work,
    MergeSortTrace* trace,
    std::size_t     limit) {
  work.max_depth = std::max(work.max_depth, depth);
  if (hi - lo <= 1)
    return;
  auto mid = lo + (hi - lo) / 2;
  sort(data, buffer, lo, mid, depth + 1, work, trace, limit);
  sort(data, buffer, mid, hi, depth + 1, work, trace, limit);
  auto        first = trace ? trace->writes.size() : 0;
  std::size_t i = lo, j = mid, k = lo;
  /// @brief Perform and optionally capture the actual next scratch write.
  /// @param source Actual current data source position.
  /// @throws std::length_error On actual operation/capture admission overflow.
  auto write = [&](std::size_t source) {
    admit(trace, limit);
    buffer[k]           = data[source];
    work.scratch_writes = advance(work.scratch_writes);
    if (trace)
      trace->writes.push_back({false, source, k, item(data[source])});
    ++k;
  };
  while (i < mid && j < hi) {
    admit(trace, limit);
    bool right       = key(data[j]) < key(data[i]);
    work.comparisons = advance(work.comparisons);
    if (trace)
      trace->comparisons.push_back({lo, mid, hi, i, j, key(data[i]), key(data[j]), right});
    if (right)
      write(j++);
    else
      write(i++);
  }
  while (i < mid)
    write(i++);
  while (j < hi)
    write(j++);
  for (std::size_t t = lo; t < hi; ++t) {
    admit(trace, limit);
    data[t]          = buffer[t];
    work.copy_writes = advance(work.copy_writes);
    if (trace)
      trace->writes.push_back({true, t, t, item(buffer[t])});
  }
  admit(trace, limit);
  work.merges = advance(work.merges);
  if (trace)
    trace->merges.push_back({lo, mid, hi, first, trace->writes.size()});
}
/// @brief Allocate the single buffer and run the canonical templated producer.
/// @tparam T Supported actual value type.
/// @param data Mutable actual input/output.
/// @param trace Optional producing capture.
/// @param limit Optional complete capture limit.
/// @return Actual work, with root depth one even on an empty input.
/// @throws std::length_error On count/capture geometry overflow.
template<class T>
MergeSortWork
produce(std::vector<T>& data, MergeSortTrace* trace = nullptr, std::size_t limit = 0) {
  MergeSortWork  work;
  std::vector<T> buffer(data.size() > 1 ? data.size() : 0);
  sort(data, buffer, 0, data.size(), 1, work, trace, limit);
  return work;
}
/// @brief Admit complete owned endpoints before their allocation/copy.
/// @param n Endpoint element count.
/// @param limit Optional complete logical cap.
/// @return Initialized empty capture with exact endpoint count.
/// @throws std::length_error On endpoint/capture geometry overflow.
MergeSortTrace capture(std::size_t n, std::size_t limit) {
  if (n > SIZE_MAX / 2)
    throw std::length_error("merge-sort endpoint geometry");
  MergeSortTrace out;
  out.captured_entries = 2 * n;
  if (limit && out.captured_entries > limit)
    throw std::length_error("merge-sort endpoint capture");
  out.input.reserve(n);
  return out;
}
} // namespace
void merge_sort(std::vector<std::int64_t>& data) {
  produce(data);
}
void merge_sort_pairs(std::vector<std::pair<int, int>>& data) {
  produce(data);
}
MergeSortAnalysis merge_sort_analyze(const std::vector<std::int64_t>& data) {
  MergeSortAnalysis out;
  out.values = data;
  out.work   = produce(out.values);
  return out;
}
MergeSortTrace merge_sort_trace(const std::vector<std::int64_t>& data, std::size_t limit) {
  if (data.size() > std::uint64_t(INT64_MAX))
    throw std::length_error("merge-sort original tag geometry");
  auto out = capture(data.size(), limit);
  for (std::size_t i = 0; i < data.size(); ++i)
    out.input.push_back({data[i], static_cast<std::int64_t>(i)});
  out.output = out.input;
  out.work   = produce(out.output, &out, limit);
  return out;
}
MergeSortTrace merge_sort_pairs_trace(
    const std::vector<std::pair<int, int>>& data,
    std::size_t                             limit) {
  auto out = capture(data.size(), limit);
  for (auto value : data)
    out.input.push_back({value.first, value.second});
  out.output = out.input;
  out.work   = produce(out.output, &out, limit);
  return out;
}
std::size_t merge_sort_comparison_bound(std::size_t n) {
  if (n <= 1)
    return 0;
  auto              height = std::bit_width(n - 1);
  unsigned __int128 bound  = static_cast<unsigned __int128>(n) * height -
                            (static_cast<unsigned __int128>(1) << height) + 1;
  if (bound > SIZE_MAX)
    throw std::overflow_error("merge-sort comparison bound");
  return static_cast<std::size_t>(bound);
}
std::vector<std::int64_t> merge_sort_demo_values() {
  return {4, 1, 3, 2, 6, 5, 2, 6};
}
std::vector<std::pair<int, int>> merge_sort_demo_pairs() {
  return {{3, 0}, {1, 1}, {3, 2}, {2, 3}, {1, 4}, {3, 5}, {2, 6}};
}
} // namespace datamunge::algorithms
