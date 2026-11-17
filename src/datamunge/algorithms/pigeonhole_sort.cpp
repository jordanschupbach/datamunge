/// @file pigeonhole_sort.cpp
/// @brief Original dense value buckets with widened offsets and actual-site observation.
#include <algorithm>
#include <cstdint>
#include <datamunge/algorithms/pigeonhole_sort.hpp>
#include <limits>
#include <stdexcept>
#include <utility>
namespace datamunge::algorithms {
/// @brief Shared ordinary and observed dense-range producer.
namespace {
/// @brief Allocate checked value buckets, append once and emit in ascending order.
PigeonholeTrace run(std::vector<int> a, PigeonholeOptions options, bool capture) {
  PigeonholeTrace t;
  auto&           r = t.run;
  const auto      n = a.size();
  if (n <= 1) {
    r.sorted = std::move(a);
    return t;
  }
  r.has_range = true;
  r.minimum   = *std::min_element(a.begin(), a.end());
  r.maximum   = *std::max_element(a.begin(), a.end());
  const auto range =
      static_cast<std::uint64_t>(static_cast<std::int64_t>(r.maximum) - r.minimum + 1);
  if (range > std::numeric_limits<std::size_t>::max() ||
      (options.max_holes && range > options.max_holes))
    throw std::length_error("pigeonhole dense range policy");
  r.holes = static_cast<std::size_t>(range);
  if (capture && (r.holes > std::numeric_limits<std::size_t>::max() - n ||
                  (options.max_trace_entries && r.holes + n > options.max_trace_entries)))
    throw std::length_error("pigeonhole history policy");
  std::vector<std::vector<int>> holes(r.holes);
  for (std::size_t i = 0; i < n; ++i) {
    const auto h = static_cast<std::size_t>(static_cast<std::int64_t>(a[i]) - r.minimum);
    holes[h].push_back(a[i]);
    ++r.placements;
    if (capture)
      t.placements.push_back({i, a[i], h, holes[h].size()});
  }
  r.sorted.reserve(n);
  for (const auto& h : holes) {
    ++r.scanned_holes;
    if (capture)
      t.occupancies.push_back(h.size());
    for (int value : h) {
      r.sorted.push_back(value);
      ++r.emitted;
    }
  }
  return t;
}
} // namespace
std::vector<int> pigeonhole_sort(std::vector<int> a) {
  return run(std::move(a), {}, false).run.sorted;
}
PigeonholeRun pigeonhole_sort_run(std::vector<int> a, PigeonholeOptions options) {
  return run(std::move(a), options, false).run;
}
PigeonholeTrace pigeonhole_sort_trace(std::vector<int> a, PigeonholeOptions options) {
  return run(std::move(a), options, true);
}
} // namespace datamunge::algorithms
