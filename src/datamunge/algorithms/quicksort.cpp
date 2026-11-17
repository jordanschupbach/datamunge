/// @file quicksort.cpp
/// @brief Canonical quicksort producer with actual cutoff, pivot and range records.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/quicksort.hpp>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
/// @brief Canonical sorting implementation.
namespace datamunge::algorithms {
/// @brief Private quicksort implementation.
namespace {
/// @brief Borrowed keys and actual work/history accumulator.
struct Sorter {
  std::vector<std::int64_t>& a;         ///< Keys mutated by the shared producer.
  QuicksortOptions           o;         ///< Work and history policies.
  bool                       capture;   ///< Retain actual range snapshots.
  QuicksortResult            result;    ///< Actual selected work and optional history.
  std::size_t                history{}; ///< Scalar/header history entries retained.

                                        /// @brief Admit one actual selected event.
  void tick() {
    if (result.operations == std::numeric_limits<std::size_t>::max() ||
        (o.max_operations && result.operations >= o.max_operations))
      throw std::length_error("quicksort: work policy");
    ++result.operations;
  }
  /// @brief Compare two keys and count one actual comparison.
  /// @param x Left key.
  /// @param y Right key.
  /// @return Whether x is strictly less than y.
  bool less(std::int64_t x, std::int64_t y) {
    tick();
    ++result.comparisons;
    return x < y;
  }
  /// @brief Swap two actual array entries, including self-swaps.
  /// @param i First index.
  /// @param j Second index.
  void swap(std::ptrdiff_t i, std::ptrdiff_t j) {
    tick();
    ++result.swaps;
    std::swap(a[i], a[j]);
  }
  /// @brief Assign one actual insertion-sort key.
  /// @param i Destination.
  /// @param x Value to assign.
  void write(std::ptrdiff_t i, std::int64_t x) {
    tick();
    ++result.writes;
    a[i] = x;
  }
  /// @brief Retain a completed nonempty real range after policy admission.
  /// @param lo Inclusive beginning.
  /// @param hi Inclusive end.
  /// @param p Final pivot, or hi+1 for insertion.
  /// @param depth Live recursion depth.
  /// @param insertion Actual cutoff path.
  void record(
      std::ptrdiff_t lo,
      std::ptrdiff_t hi,
      std::ptrdiff_t p,
      std::size_t    depth,
      bool           insertion) {
    if (!capture || hi < lo)
      return;
    const auto cost = static_cast<std::size_t>(hi - lo) + 2;
    if (cost > std::numeric_limits<std::size_t>::max() - history ||
        (o.max_trace_entries &&
         (history > o.max_trace_entries || cost > o.max_trace_entries - history)))
      throw std::length_error("quicksort: history policy");
    history += cost;
    result.steps.push_back(
        {static_cast<std::size_t>(lo),
         static_cast<std::size_t>(hi + 1),
         static_cast<std::size_t>(p),
         depth,
         insertion,
         {a.begin() + lo, a.begin() + hi + 1}});
  }
  /// @brief Execute historical median-of-three Lomuto partitioning.
  /// @param lo Inclusive beginning.
  /// @param hi Inclusive end.
  /// @return Final pivot position.
  std::ptrdiff_t partition(std::ptrdiff_t lo, std::ptrdiff_t hi) {
    const auto mid = lo + (hi - lo) / 2;
    if (less(a[mid], a[lo]))
      swap(mid, lo);
    if (less(a[hi], a[lo]))
      swap(hi, lo);
    if (less(a[hi], a[mid]))
      swap(hi, mid);
    swap(mid, hi);
    const auto pivot = a[hi];
    auto       store = lo;
    for (auto j = lo; j < hi; ++j)
      if (less(a[j], pivot)) {
        swap(store, j);
        ++store;
      }
    swap(store, hi);
    ++result.partitions;
    return store;
  }
  /// @brief Sort one range, recursing only on its smaller partition.
  /// @param lo Inclusive beginning.
  /// @param hi Inclusive end.
  /// @param depth Live smaller-side recursion depth.
  void range(std::ptrdiff_t lo, std::ptrdiff_t hi, std::size_t depth) {
    result.max_depth = std::max(result.max_depth, depth);
    while (hi - lo + 1 > 16) {
      const auto p = partition(lo, hi);
      record(lo, hi, p, depth, false);
      if (p - lo < hi - p) {
        range(lo, p - 1, depth + 1);
        lo = p + 1;
      } else {
        range(p + 1, hi, depth + 1);
        hi = p - 1;
      }
    }
    for (auto i = lo + 1; i <= hi; ++i) {
      const auto key = a[i];
      auto       j   = i - 1;
      while (j >= lo) {
        if (!less(key, a[j]))
          break;
        write(j + 1, a[j]);
        --j;
      }
      write(j + 1, key);
    }
    record(lo, hi, hi + 1, depth, true);
  }
};
/// @brief Run the shared producer on borrowed keys.
/// @param a Mutable input.
/// @param o Policies.
/// @param capture Retain actual snapshots.
/// @return Counters and optional history; sorted values remain in a.
QuicksortResult solve(std::vector<std::int64_t>& a, const QuicksortOptions& o, bool capture) {
  if (a.size() > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()))
    throw std::length_error("quicksort: signed indexing capacity");
  Sorter s{a, o, capture, {}, 0};
  if (a.size() > 1)
    s.range(0, static_cast<std::ptrdiff_t>(a.size()) - 1, 1);
  return std::move(s.result);
}
} // namespace
QuicksortResult quicksort_run(std::vector<std::int64_t> a, const QuicksortOptions& o) {
  auto r   = solve(a, o, false);
  r.values = std::move(a);
  return r;
}
QuicksortResult quicksort_trace(std::vector<std::int64_t> a, const QuicksortOptions& o) {
  auto r   = solve(a, o, true);
  r.values = std::move(a);
  return r;
}
void quicksort(std::vector<std::int64_t>& a, std::uint64_t& comparisons) {
  comparisons = 0;
  comparisons = solve(a, QuicksortOptions{0, 0}, false).comparisons;
}
void quicksort(std::vector<std::int64_t>& a) {
  std::uint64_t ignored;
  quicksort(a, ignored);
}
std::vector<QuicksortScaling>
quicksort_scaling(const std::vector<std::size_t>& sizes, std::uint64_t seed, std::size_t max_keys) {
  std::size_t total = 0;
  for (auto n : sizes) {
    if (n > std::numeric_limits<std::size_t>::max() - total)
      throw std::length_error("quicksort_scaling: key capacity");
    total += n;
    if (max_keys && total > max_keys)
      throw std::length_error("quicksort_scaling: key policy");
  }
  std::mt19937_64               rng(seed);
  std::vector<QuicksortScaling> out;
  for (auto n : sizes) {
    std::vector<std::int64_t> values(n);
    for (auto& x : values)
      x = static_cast<std::int64_t>(rng() % 1000000);
    const auto r = quicksort_run(std::move(values));
    out.push_back(
        {n, r.comparisons, n ? static_cast<double>(n) * std::log2(static_cast<double>(n)) : 0});
  }
  return out;
}
} // namespace datamunge::algorithms
