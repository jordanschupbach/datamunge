/// @file quickselect.cpp
/// @brief Shared three-way selection with actual nested BFPRT instrumentation.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/quickselect.hpp>
#include <random>
#include <stdexcept>
#include <utility>
/// @brief Canonical selection implementation.
namespace datamunge::algorithms {
/// @brief Private three-way selection producer.
namespace {
/// @brief Owns keys, policy, optional histories and all actual selected-work counters.
struct Selector {
  std::vector<std::int64_t> a;         ///< Owned keys being partitioned.
  QuickselectOptions        o;         ///< Caller-selected policies.
  bool                      capture;   ///< Whether to retain actual ranges.
  QuickselectResult         result;    ///< Accumulated output and selected work.
  std::size_t               history{}; ///< Retained scalar/header entries.

                                       /// @brief Admit one operation before executing it.
  /// @param count Counter to increment in addition to total work.
  void tick(std::size_t& count) {
    if (result.operations == std::numeric_limits<std::size_t>::max() ||
        (o.max_operations && result.operations >= o.max_operations))
      throw std::length_error("quickselect: work policy");
    ++result.operations;
    ++count;
  }
  /// @brief Compare keys and record exactly one key comparison.
  /// @param x Left key.
  /// @param y Right key.
  /// @return Whether x is strictly less than y.
  bool less(std::int64_t x, std::int64_t y) {
    tick(result.comparisons);
    return x < y;
  }
  /// @brief Swap two positions and count the actual call.
  /// @param i First position.
  /// @param j Second position.
  void swap(std::size_t i, std::size_t j) {
    tick(result.swaps);
    std::swap(a[i], a[j]);
  }
  /// @brief Assign one insertion-sort key and count its write.
  /// @param i Destination position.
  /// @param value Value to write.
  void write(std::size_t i, std::int64_t value) {
    tick(result.writes);
    a[i] = value;
  }
  /// @brief Select one rank from a half-open range with the chosen pivot rule.
  /// @param lo Range start.
  /// @param hi Exclusive range end.
  /// @param k Absolute rank in that range.
  /// @param depth Nested BFPRT pivot-selection level.
  /// @return Key at the requested rank.
  std::int64_t select(std::size_t lo, std::size_t hi, std::size_t k, std::size_t depth) {
    while (hi - lo > 1) {
      std::int64_t pivot;
      if (o.median_of_medians) {
        std::size_t groups = 0;
        for (std::size_t g = lo; g < hi;) {
          const auto end = g + std::min<std::size_t>(5, hi - g);
          for (std::size_t i = g + 1; i < end; ++i) {
            const auto key = a[i];
            auto       j   = i;
            while (j > g && less(key, a[j - 1])) {
              write(j, a[j - 1]);
              --j;
            }
            write(j, key);
          }
          swap(lo + groups, g + (end - g - 1) / 2);
          ++groups;
          g = end;
        }
        pivot = groups == 1 ? a[lo] : select(lo, lo + groups, lo + (groups - 1) / 2, depth + 1);
      } else {
        const auto mid = lo + (hi - lo - 1) / 2;
        auto       x = a[lo], y = a[mid], z = a[hi - 1];
        if (less(y, x))
          std::swap(x, y);
        if (less(z, x))
          std::swap(x, z);
        if (less(z, y))
          std::swap(y, z);
        pivot = y;
      }
      auto lt = lo, i = lo, gt = hi;
      while (i < gt) {
        if (less(a[i], pivot)) {
          swap(lt, i);
          ++lt;
          ++i;
        } else if (less(pivot, a[i])) {
          --gt;
          swap(i, gt);
        } else
          ++i;
      }
      ++result.partitions;
      if (capture) {
        const auto cost = hi - lo + 1;
        if (cost > std::numeric_limits<std::size_t>::max() - history ||
            (o.max_trace_entries &&
             (history > o.max_trace_entries || cost > o.max_trace_entries - history)))
          throw std::length_error("quickselect: history policy");
        history += cost;
        result.steps.push_back({lo, hi, k, lt, gt, depth, pivot, {a.begin() + lo, a.begin() + hi}});
      }
      if (k < lt)
        hi = lt;
      else if (k >= gt)
        lo = gt;
      else
        return pivot;
    }
    return a[lo];
  }
};
/// @brief Validate input and execute the shared producer.
/// @param data Owned keys.
/// @param k Zero-based rank.
/// @param o Policies and pivot strategy.
/// @param capture Retain actual partitions.
/// @return Owned result and actual selected work.
QuickselectResult
solve(std::vector<std::int64_t> data, std::size_t k, const QuickselectOptions& o, bool capture) {
  if (data.empty() || k >= data.size())
    throw std::invalid_argument("quickselect: empty input or invalid rank");
  Selector s{std::move(data), o, capture, {}, 0};
  s.result.value  = s.select(0, s.a.size(), k, 0);
  s.result.values = std::move(s.a);
  return std::move(s.result);
}
} // namespace
QuickselectResult
quickselect_run(std::vector<std::int64_t> data, std::size_t k, const QuickselectOptions& o) {
  return solve(std::move(data), k, o, false);
}
QuickselectResult
quickselect_trace(std::vector<std::int64_t> data, std::size_t k, const QuickselectOptions& o) {
  return solve(std::move(data), k, o, true);
}
std::int64_t quickselect(std::vector<std::int64_t> data, std::size_t k) {
  return quickselect_run(std::move(data), k, QuickselectOptions{false, 0, 0}).value;
}
std::int64_t quickselect_mom(std::vector<std::int64_t> data, std::size_t k) {
  return quickselect_run(std::move(data), k, QuickselectOptions{true, 0, 0}).value;
}
std::vector<QuickselectScaling> quickselect_scaling(
    const std::vector<std::size_t>& sizes,
    std::size_t                     trials,
    std::uint64_t                   seed,
    std::size_t                     max_keys) {
  if (!trials)
    throw std::invalid_argument("quickselect_scaling: zero trials");
  std::size_t total = 0;
  for (auto n : sizes) {
    if (!n)
      throw std::invalid_argument("quickselect_scaling: zero size");
    if (n > std::numeric_limits<std::size_t>::max() / trials ||
        n * trials > std::numeric_limits<std::size_t>::max() - total)
      throw std::length_error("quickselect_scaling: key capacity");
    total += n * trials;
    if (max_keys && total > max_keys)
      throw std::length_error("quickselect_scaling: key policy");
  }
  std::mt19937_64                             rng(seed);
  std::uniform_int_distribution<std::int64_t> values(0, 1000000);
  std::vector<QuickselectScaling>             out;
  for (auto n : sizes) {
    QuickselectScaling row{n, trials, 0, 0};
    for (std::size_t t = 0; t < trials; ++t) {
      std::vector<std::int64_t> a(n);
      for (auto& x : a)
        x = values(rng);
      auto               plain = quickselect_run(a, n / 2);
      QuickselectOptions o;
      o.median_of_medians = true;
      auto bfprt          = quickselect_run(std::move(a), n / 2, o);
      if (plain.value != bfprt.value)
        throw std::runtime_error("quickselect_scaling: producer disagreement");
      if (plain.comparisons > std::numeric_limits<std::size_t>::max() - row.comparisons ||
          bfprt.comparisons > std::numeric_limits<std::size_t>::max() - row.bfprt_comparisons)
        throw std::length_error("quickselect_scaling: comparison capacity");
      row.comparisons += plain.comparisons;
      row.bfprt_comparisons += bfprt.comparisons;
    }
    out.push_back(row);
  }
  return out;
}
} // namespace datamunge::algorithms
