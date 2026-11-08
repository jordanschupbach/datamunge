/// @file sorted_search.cpp
/// @brief Sorted-array search implementations and checked shared Eytzinger construction/query
/// kernels.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <datamunge/algorithms/sorted_search.hpp>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace datamunge::algorithms {

SearchResult jump_search(const std::vector<int>& a, int key) {
  SearchResult      r;
  const std::size_t n = a.size();
  if (n == 0)
    return r;

  const std::size_t step =
      std::max<std::size_t>(1, static_cast<std::size_t>(std::sqrt(static_cast<double>(n))));
  // Advance block by block until the block's last element is >= key (or we run off the end).
  std::size_t block_end = step; // one past the current block's last index (exclusive upper)
  std::size_t prev      = 0;
  while (prev < n) {
    const std::size_t last = std::min(block_end, n) - 1;
    ++r.probes;
    if (a[last] >= key)
      break; // key, if present, is in [prev, last]
    prev = block_end;
    block_end += step;
    if (prev >= n)
      return r; // key is larger than every element
  }
  // Linear scan within the identified block.
  const std::size_t hi = std::min(block_end, n);
  for (std::size_t i = prev; i < hi; ++i) {
    ++r.probes;
    if (a[i] == key) {
      r.found = true;
      r.index = static_cast<std::ptrdiff_t>(i);
      return r;
    }
    if (a[i] > key)
      break; // passed where key would be
  }
  return r;
}

SearchResult interpolation_search(const std::vector<int>& a, int key) {
  SearchResult   r;
  std::ptrdiff_t lo = 0;
  std::ptrdiff_t hi = static_cast<std::ptrdiff_t>(a.size()) - 1;
  while (lo <= hi && key >= a[static_cast<std::size_t>(lo)] &&
         key <= a[static_cast<std::size_t>(hi)]) {
    const int lo_val = a[static_cast<std::size_t>(lo)];
    const int hi_val = a[static_cast<std::size_t>(hi)];
    if (lo_val == hi_val) { // flat range: the guard above guarantees key == lo_val
      ++r.probes;
      r.found = true;
      r.index = lo;
      return r;
    }
    // Linear interpolation of the key's likely position, clamped into [lo, hi].
    const long double frac =
        static_cast<long double>(key - lo_val) / static_cast<long double>(hi_val - lo_val);
    std::ptrdiff_t pos = lo + static_cast<std::ptrdiff_t>(frac * static_cast<long double>(hi - lo));
    pos                = std::min(hi, std::max(lo, pos));
    ++r.probes;
    if (a[static_cast<std::size_t>(pos)] == key) {
      r.found = true;
      r.index = pos;
      return r;
    }
    if (a[static_cast<std::size_t>(pos)] < key)
      lo = pos + 1;
    else
      hi = pos - 1;
  }
  return r;
}

/// @brief Checked Fibonacci candidate arithmetic and one shared search kernel.
namespace fibonacci_search_detail {
/// @brief Clamp offset+small without unsigned sentinel wrapping or unchecked addition.
/// @param n Positive input length.
/// @param offset -1 or a previously inspected index below n.
/// @param small Positive smaller Fibonacci predecessor in the main loop.
/// @return Actual index min(offset+small,n-1), represented without wrapping.
std::size_t candidate(std::size_t n, std::ptrdiff_t offset, std::size_t small) {
  if (offset < 0)
    return std::min(small - 1, n - 1);
  const auto base = static_cast<std::size_t>(offset), distance = n - 1 - base;
  return small >= distance ? n - 1 : base + small;
}
/// @brief Run actual initial growth, branch comparisons and optional fallback/capture.
/// @param a Borrowed nondecreasing input, unchecked sortedness.
/// @param key Any int key.
/// @param trace Optional history being constructed; null uses constant storage.
/// @param max_entries Optional whole-record quota; zero disables.
/// @return Actual legacy answer with separate work counts.
/// @throws std::length_error Before signed positions, history size or quota can exceed limits.
/// @throws std::bad_alloc If optional history allocation fails.
FibonacciSearchOutcome
run(const std::vector<int>& a, int key, FibonacciSearchTrace* trace, std::size_t max_entries) {
  FibonacciSearchOutcome out;
  out.schedule = fibonacci_search_schedule(a.size());
  if (a.empty())
    return out;
  auto small = out.schedule.small, middle = out.schedule.middle, large = out.schedule.large;
  std::ptrdiff_t offset = -1;
  /// @brief Admit history before the next read, inspect once and record the actual branch.
  /// @param index Actual checked in-bounds input position.
  /// @param final True for equality-only fallback; false for a main-loop transition.
  /// @return True on actual equality, with result and work updated.
  /// @throws std::length_error On optional capture quota or vector maximum size.
  /// @throws std::bad_alloc On optional record allocation.
  const auto inspect = [&](std::size_t index, bool final) {
    if (trace && ((max_entries && trace->probes.size() >= max_entries) ||
                  trace->probes.size() == trace->probes.max_size()))
      throw std::length_error("Fibonacci search capture limit exceeded");
    FibonacciSearchProbe row;
    row.small_before  = small;
    row.middle_before = middle;
    row.large_before  = large;
    row.offset_before = offset;
    row.index         = index;
    row.value         = a[index];
    row.final_check   = final;
    ++out.result.probes;
    bool match = false;
    if (final) {
      row.comparisons = 1;
      match           = row.value == key;
      row.decision    = match ? 0 : 2;
    } else if (row.value < key) {
      row.comparisons = 1;
      row.decision    = 1;
      large           = middle;
      middle          = small;
      small           = large - middle;
      offset          = static_cast<std::ptrdiff_t>(index);
      ++out.transitions;
    } else if (row.value > key) {
      row.comparisons = 2;
      row.decision    = -1;
      large           = small;
      middle          = middle - small;
      small           = large - middle;
      ++out.transitions;
    } else {
      row.comparisons = 2;
      row.decision    = 0;
      match           = true;
    }
    out.comparisons += row.comparisons;
    if (match) {
      out.result.found = true;
      out.result.index = static_cast<std::ptrdiff_t>(index);
    }
    row.small_after  = small;
    row.middle_after = middle;
    row.large_after  = large;
    row.offset_after = offset;
    if (trace) {
      trace->probes.push_back(row);
      ++trace->captured_entries;
    }
    return match;
  };
  while (large > 1)
    if (inspect(candidate(a.size(), offset, small), false))
      return out;
  if (middle == 1 && offset + 1 < static_cast<std::ptrdiff_t>(a.size()))
    inspect(static_cast<std::size_t>(offset + 1), true);
  return out;
}
} // namespace fibonacci_search_detail
FibonacciSearchSchedule fibonacci_search_schedule(std::size_t n) {
  if (n > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()))
    throw std::length_error("Fibonacci search positions exceed signed index domain");
  FibonacciSearchSchedule s;
  s.length = n;
  if (!n)
    return s;
  s.middle = 1;
  s.large  = 1;
  while (s.large < n) {
    s.small  = s.middle;
    s.middle = s.large;
    if (s.small > std::numeric_limits<std::size_t>::max() - s.middle)
      throw std::length_error("Fibonacci search schedule overflow");
    s.large = s.middle + s.small;
    ++s.growth_steps;
  }
  return s;
}
SearchResult fibonacci_search(const std::vector<int>& a, int key) {
  return fibonacci_search_work(a, key).result;
}
FibonacciSearchOutcome fibonacci_search_work(const std::vector<int>& a, int key) {
  return fibonacci_search_detail::run(a, key, nullptr, 0);
}
FibonacciSearchTrace
fibonacci_search_trace(const std::vector<int>& a, int key, std::size_t max_entries) {
  FibonacciSearchTrace trace;
  trace.key     = key;
  trace.outcome = fibonacci_search_detail::run(a, key, &trace, max_entries);
  return trace;
}

/// @brief Checked complete-tree indexing and shared actual layout/search execution.
namespace eytzinger_detail {
/// @brief Return an existing zero-based child without multiplying a missing-child index.
/// @param n Number of stored nodes.
/// @param i Existing zero-based parent index.
/// @param right True for right child, false for left.
/// @return Child position, or nullopt when that child is not stored.
std::optional<std::size_t> child(std::size_t n, std::size_t i, bool right) {
  if (right) {
    if (n < 2 || i >= (n - 1) / 2)
      return std::nullopt;
    return 2 * i + 2;
  }
  if (i >= n / 2)
    return std::nullopt;
  return 2 * i + 1;
}
/// @brief Reject lengths whose actual positions cannot use the signed search-result index.
/// @param n Borrowed input length.
/// @throws std::length_error If n exceeds the representable signed-index bound.
void positions(std::size_t n) {
  if (n > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()))
    throw std::length_error("Eytzinger positions exceed signed index range");
}
/// @brief Admit whole optional records with checked vector and configured quota arithmetic.
/// @param have Number already retained.
/// @param add Number about to be retained.
/// @param limit Configured whole-record quota; zero disables quota only.
/// @param maximum Container maximum size.
/// @throws std::length_error If addition, vector capacity or the configured quota is exceeded.
void admit(std::size_t have, std::size_t add, std::size_t limit, std::size_t maximum) {
  if (have > maximum || add > maximum - have || (limit && (have > limit || add > limit - have)))
    throw std::length_error("Eytzinger optional capture exceeds record limit");
}
/// @brief Assign the actual next sorted value at each complete-tree in-order visit.
/// @param sorted Borrowed input values.
/// @param out Owned destination under construction.
/// @param i Existing destination node.
/// @param rank Next source rank, advanced once per actual assignment.
/// @param records Optional pre-admitted placement storage; null for ordinary layout.
void fill(
    const std::vector<int>&          sorted,
    std::vector<int>&                out,
    std::size_t                      i,
    std::size_t&                     rank,
    std::vector<EytzingerPlacement>* records) {
  if (const auto left = child(out.size(), i, false))
    fill(sorted, out, *left, rank, records);
  out[i] = sorted[rank];
  if (records)
    records->push_back({rank, i, out[i]});
  ++rank;
  if (const auto right = child(out.size(), i, true))
    fill(sorted, out, *right, rank, records);
}
/// @brief Build the canonical output once, optionally recording the actual assignments.
/// @param sorted Borrowed nondecreasing input; precondition is unchecked.
/// @param records Optional owned chronological placement vector, initially empty.
/// @param limit Whole optional placement-record quota; zero disables quota only.
/// @return Owned BFS complete-tree permutation.
/// @throws std::length_error If positions, vector capacity or optional quota are exceeded.
/// @throws std::bad_alloc If output/history storage cannot be allocated.
std::vector<int>
build(const std::vector<int>& sorted, std::vector<EytzingerPlacement>* records, std::size_t limit) {
  positions(sorted.size());
  if (records) {
    admit(0, sorted.size(), limit, records->max_size());
    records->reserve(sorted.size());
  }
  std::vector<int> out(sorted.size());
  std::size_t      rank = 0;
  if (!out.empty())
    fill(sorted, out, 0, rank, records);
  return out;
}
/// @brief Execute the equality-first exact-match search, optionally retaining each real node read.
/// @param layout Borrowed complete-tree BFS values; tree order is unchecked.
/// @param key Integer target.
/// @param records Optional initially empty owned probe vector.
/// @param limit Whole optional probe-record quota; zero disables quota only.
/// @return Exact legacy match, layout position and actual element-inspection count.
/// @throws std::length_error If positions, vector capacity or optional quota are exceeded.
/// @throws std::bad_alloc If optional history allocation fails.
SearchResult search(
    const std::vector<int>&      layout,
    int                          key,
    std::vector<EytzingerProbe>* records,
    std::size_t                  limit) {
  positions(layout.size());
  SearchResult result;
  if (layout.empty())
    return result;
  std::size_t i = 0;
  for (;;) {
    if (records)
      admit(records->size(), 1, limit, records->max_size());
    const int value = layout[i];
    ++result.probes;
    const int  direction = value == key ? 0 : (key < value ? -1 : 1);
    const auto next      = direction ? child(layout.size(), i, direction > 0) : std::nullopt;
    if (records)
      records->push_back({i, value, direction, next ? static_cast<std::ptrdiff_t>(*next) : -1});
    if (!direction) {
      result.found = true;
      result.index = static_cast<std::ptrdiff_t>(i);
      return result;
    }
    if (!next)
      return result;
    i = *next;
  }
}
} // namespace eytzinger_detail

/// @brief Forward the original compiled layout interface through the shared complete-tree builder.
/// @param sorted Borrowed nondecreasing input.
/// @return Owned BFS permutation; see public declaration for full contract/errors.
std::vector<int> eytzinger_layout(const std::vector<int>& sorted) {
  return eytzinger_detail::build(sorted, nullptr, 0);
}
/// @brief Forward the original compiled exact-match interface through the shared actual search.
/// @param layout Borrowed complete-tree BFS input.
/// @param key Integer target.
/// @return Exact legacy match/layout position/probe count; public declaration gives full contract.
SearchResult eytzinger_search(const std::vector<int>& layout, int key) {
  return eytzinger_detail::search(layout, key, nullptr, 0);
}
/// @brief Capture the actual assignments during one shared complete-tree construction.
/// @param sorted Borrowed nondecreasing input.
/// @param max_entries Whole placement-record cap; zero disables quota only.
/// @return Owned layout and actual placements; public declaration gives full contract/errors.
EytzingerLayoutTrace eytzinger_layout_trace(
    const std::vector<int>& sorted,
    std::size_t             max_entries) {
  EytzingerLayoutTrace trace;
  trace.layout           = eytzinger_detail::build(sorted, &trace.placements, max_entries);
  trace.captured_entries = trace.placements.size();
  return trace;
}
/// @brief Capture the actual element reads and transitions during one shared exact-match search.
/// @param layout Borrowed complete-tree BFS input.
/// @param key Integer query.
/// @param max_entries Whole probe-record cap; zero disables quota only.
/// @return Owned actual snapshots and result; public declaration gives full contract/errors.
EytzingerSearchTrace
eytzinger_search_trace(const std::vector<int>& layout, int key, std::size_t max_entries) {
  EytzingerSearchTrace trace;
  trace.key              = key;
  trace.layout_size      = layout.size();
  trace.result           = eytzinger_detail::search(layout, key, &trace.probes, max_entries);
  trace.captured_entries = trace.probes.size();
  return trace;
}

} // namespace datamunge::algorithms
