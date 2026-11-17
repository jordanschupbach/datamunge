/// @file jump_search.cpp
/// @brief Division-safe schedule and one actual block/scan kernel for sorted jump search.
#include <algorithm>
#include <datamunge/algorithms/jump_search.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
JumpSearchSchedule jump_search_schedule(std::size_t n) {
  if (n > std::size_t(std::numeric_limits<std::ptrdiff_t>::max()))
    throw std::length_error("jump search signed length");
  JumpSearchSchedule s;
  s.length = n;
  if (!n)
    return s;
  std::size_t lo = 1, hi = n;
  while (lo < hi) {
    const auto middle = lo + (hi - lo) / 2 + 1;
    if (middle <= n / middle)
      lo = middle;
    else
      hi = middle - 1;
  }
  s.block_size  = lo;
  s.block_count = n / lo + (n % lo != 0);
  if (s.block_count > std::numeric_limits<std::size_t>::max() - lo)
    throw std::length_error("jump search bound");
  s.max_probes = s.block_count + lo;
  return s;
}
/// @brief Shared checked producing search, optionally capturing read snapshots.
namespace jump_search_detail {
/// @brief Execute block selection and linear scan exactly once.
/// @param a Borrowed nondecreasing input, sortedness unchecked.
/// @param key Any int key.
/// @param history Optional output history, null for constant-storage calls.
/// @param max_entries Complete actual-record ceiling, zero off.
/// @return Actual answer, exact schedule and logical operation counts.
/// @throws std::length_error On length/schedule/history representability or enabled quota.
/// @throws std::bad_alloc On optional history allocation failure.
JumpSearchAnalysis run(
    const std::vector<int>&       a,
    int                           key,
    std::vector<JumpSearchProbe>* history,
    std::size_t                   max_entries) {
  JumpSearchAnalysis out;
  out.schedule = jump_search_schedule(a.size());
  const auto n = a.size();
  if (!n)
    return out;
  /// @brief Admit an optional complete record before the actual next array read.
  const auto admit = [&]() {
    if (history &&
        ((max_entries && history->size() >= max_entries) || history->size() == history->max_size()))
      throw std::length_error("jump search actual-record ceiling");
  };
  std::size_t begin = 0, end = std::min(n, out.schedule.block_size);
  while (begin < n) {
    admit();
    const auto index = end - 1;
    const int  value = a[index];
    ++out.result.probes;
    ++out.work.block_reads;
    const bool select = value >= key;
    if (history)
      history->push_back(
          {begin,
           end,
           index,
           value,
           select ? JumpSearchAction::Select : JumpSearchAction::Advance});
    if (select)
      break;
    begin = end;
    if (begin == n)
      return out;
    end = begin + std::min(out.schedule.block_size, n - begin);
  }
  for (std::size_t index = begin; index < end; ++index) {
    admit();
    const int value = a[index];
    ++out.result.probes;
    ++out.work.scan_reads;
    JumpSearchAction action;
    if (value == key) {
      out.result.found = true;
      out.result.index = static_cast<std::ptrdiff_t>(index);
      action           = JumpSearchAction::Found;
    } else {
      ++out.work.greater_tests;
      action = value > key ? JumpSearchAction::Passed : JumpSearchAction::Continue;
    }
    if (history)
      history->push_back({begin, end, index, value, action});
    if (action == JumpSearchAction::Found || action == JumpSearchAction::Passed)
      break;
  }
  return out;
}
} // namespace jump_search_detail
SearchResult jump_search(const std::vector<int>& a, int key) {
  return jump_search_detail::run(a, key, nullptr, 0).result;
}
JumpSearchAnalysis jump_search_analyze(const std::vector<int>& a, int key) {
  return jump_search_detail::run(a, key, nullptr, 0);
}
JumpSearchTrace jump_search_trace(const std::vector<int>& a, int key, std::size_t cap) {
  JumpSearchTrace t;
  t.analysis = jump_search_detail::run(a, key, &t.probes, cap);
  return t;
}
} // namespace datamunge::algorithms
