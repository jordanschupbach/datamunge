/// @file heapsort.cpp
/// @brief Shared iterative heapsort with optional actual work and owned boundary capture.
#include <datamunge/algorithms/heapsort.hpp>
#include <limits>
#include <stdexcept>
#include <utility>
namespace datamunge::algorithms {
/// @brief Private producing kernels and logical capture admission.
namespace heapsort_detail {
/// @brief Checked count addition.
/// @param a Current count.
/// @param b Increment.
/// @return Exact sum.
/// @throws std::length_error On count overflow.
std::size_t add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::length_error("heapsort count overflow");
  return a + b;
}
/// @brief Checked count product.
/// @param a First count.
/// @param b Second count.
/// @return Exact product.
/// @throws std::length_error On count overflow.
std::size_t multiply(std::size_t a, std::size_t b) {
  if (a && b > std::numeric_limits<std::size_t>::max() / a)
    throw std::length_error("heapsort count product overflow");
  return a * b;
}
/// @brief Capture admitted exchange before applying the producing std::swap.
/// @param data Mutable working vector with valid indices.
/// @param left First index.
/// @param right Second index.
/// @param size Actual active prefix length.
/// @param phase Producing build/extraction phase.
/// @param parking Root-to-tail extraction indicator.
/// @param trace Optional borrowed capture sink with snapshot units already reserved.
/// @param cap Logical cap, zero disables policy.
/// @throws std::length_error On online capture/count/container limit, before exchange.
/// @throws std::bad_alloc On event allocation failure, before exchange.
void exchange(
    std::vector<std::int64_t>& data,
    std::size_t                left,
    std::size_t                right,
    std::size_t                size,
    HeapSortPhase              phase,
    bool                       parking,
    HeapSortTrace*             trace,
    std::size_t                cap) {
  if (trace) {
    const auto units = add(trace->captured_entries, 1);
    if ((cap && units > cap) || trace->swaps.size() == trace->swaps.max_size())
      throw std::length_error("heapsort online event limit");
    trace->swaps.push_back({phase, size, left, right, data[left], data[right], parking});
    trace->captured_entries = units;
  }
  std::swap(data[left], data[right]);
}
/// @brief Restore a prefix subtree with original left-before-right strict comparisons.
/// @param data Borrowed mutable values.
/// @param root Valid subtree root when size>0.
/// @param size Active prefix length.
/// @param phase Actual producing phase.
/// @param work Optional borrowed producing counters.
/// @param trace Optional borrowed actual history sink.
/// @param cap Logical history policy, zero disables.
/// @throws std::length_error On instrumented work/capture count limits.
/// @throws std::bad_alloc On event capture allocation.
void sift_down(
    std::vector<std::int64_t>& data,
    std::size_t                root,
    std::size_t                size,
    HeapSortPhase              phase,
    HeapSortWork*              work,
    HeapSortTrace*             trace,
    std::size_t                cap) {
  if (work)
    work->sift_calls = add(work->sift_calls, 1);
  for (;;) {
    if (work)
      work->sift_visits = add(work->sift_visits, 1);
    if (root >= size / 2)
      return;
    const auto left = 2 * root + 1, right = left + 1;
    auto       largest = root;
    if (work)
      work->key_comparisons = add(work->key_comparisons, 1);
    if (data[left] > data[largest])
      largest = left;
    if (right < size) {
      if (work)
        work->key_comparisons = add(work->key_comparisons, 1);
      if (data[right] > data[largest])
        largest = right;
    }
    if (largest == root)
      return;
    exchange(data, root, largest, size, phase, false, trace, cap);
    if (work)
      work->sift_swaps = add(work->sift_swaps, 1);
    root = largest;
  }
}
/// @brief Capture an actual already-admitted full working-state checkpoint.
/// @param data Borrowed actual working values.
/// @param size Actual heap prefix length.
/// @param phase Actual checkpoint phase.
/// @param trace Optional borrowed capture sink.
/// @throws std::bad_alloc On snapshot allocation.
void snapshot(
    const std::vector<std::int64_t>& data,
    std::size_t                      size,
    HeapSortPhase                    phase,
    HeapSortTrace*                   trace) {
  if (trace)
    trace->states.push_back({phase, size, trace->swaps.size(), data});
}
/// @brief Build the max heap through the shared sift kernel.
/// @param data Mutable values.
/// @param work Optional selected counters.
/// @param trace Optional captured events.
/// @param cap Optional logical cap.
/// @throws std::length_error On instrumented count/capture limits.
/// @throws std::bad_alloc On event capture allocation.
void build(
    std::vector<std::int64_t>& data,
    HeapSortWork*              work,
    HeapSortTrace*             trace,
    std::size_t                cap) {
  for (std::size_t i = data.size() / 2; i-- > 0;)
    sift_down(data, i, data.size(), HeapSortPhase::Build, work, trace, cap);
}
/// @brief Execute one complete sort with optional actual phase counters/capture.
/// @param data Borrowed mutable working values.
/// @param run Optional selected-work sink; its data member is the working vector when nonnull.
/// @param trace Optional complete snapshot/event sink.
/// @param cap Optional logical capture cap.
/// @throws std::length_error On instrumented count/capture limits.
/// @throws std::bad_alloc On capture allocation failure.
void execute(
    std::vector<std::int64_t>& data,
    HeapSortRun*               run,
    HeapSortTrace*             trace,
    std::size_t                cap) {
  const auto n = data.size();
  snapshot(data, n, HeapSortPhase::Input, trace);
  build(data, run ? &run->build : nullptr, trace, cap);
  snapshot(data, n, HeapSortPhase::Build, trace);
  for (std::size_t end = n; end > 1;) {
    --end;
    exchange(data, 0, end, end + 1, HeapSortPhase::Extract, true, trace, cap);
    if (run)
      run->extraction.root_swaps = add(run->extraction.root_swaps, 1);
    sift_down(data, 0, end, HeapSortPhase::Extract, run ? &run->extraction : nullptr, trace, cap);
    snapshot(data, end, HeapSortPhase::Extract, trace);
  }
}
/// @brief Check input and known snapshot geometry before any working allocation/copy.
/// @param n Complete input size.
/// @param o Optional caps.
/// @param trace Optional initially empty trace sink.
/// @throws std::length_error On complete input/snapshot/count/container limits.
void admit(std::size_t n, HeapSortOptions o, HeapSortTrace* trace) {
  if (o.max_values && n > o.max_values)
    throw std::length_error("heapsort complete input limit");
  if (trace) {
    const auto states = add(2, n ? n - 1 : 0);
    const auto units  = multiply(states, add(n, 1));
    if ((o.max_trace_entries && units > o.max_trace_entries) || states > trace->states.max_size())
      throw std::length_error("heapsort complete snapshot limit");
    trace->captured_entries = units;
    trace->states.reserve(states);
  }
}
} // namespace heapsort_detail
void build_max_heap(std::vector<std::int64_t>& data) {
  heapsort_detail::build(data, nullptr, nullptr, 0);
}
bool is_max_heap(const std::vector<std::int64_t>& data) {
  for (std::size_t i = 0; i < data.size() / 2; ++i) {
    const auto left = 2 * i + 1, right = left + 1;
    if (data[left] > data[i] || (right < data.size() && data[right] > data[i]))
      return false;
  }
  return true;
}
void heapsort(std::vector<std::int64_t>& data) {
  heapsort_detail::execute(data, nullptr, nullptr, 0);
}
HeapSortRun heapsort_run(const std::vector<std::int64_t>& data, HeapSortOptions options) {
  heapsort_detail::admit(data.size(), options, nullptr);
  HeapSortRun r;
  r.data = data;
  heapsort_detail::execute(r.data, &r, nullptr, 0);
  return r;
}
HeapSortTrace heapsort_trace(const std::vector<std::int64_t>& data, HeapSortOptions options) {
  HeapSortTrace t;
  heapsort_detail::admit(data.size(), options, &t);
  t.run.data = data;
  heapsort_detail::execute(t.run.data, &t.run, &t, options.max_trace_entries);
  return t;
}
} // namespace datamunge::algorithms
