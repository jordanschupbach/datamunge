/// @file kadane.cpp
/// @brief One checked exact-integer Kadane loop with optional actual state capture.
#include <datamunge/algorithms/kadane.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Shared recurrence implementation details.
namespace kadane_detail {
/// @brief Run the actual recurrence once, preserving strict ties and optional owned state.
/// @param a Borrowed original int values.
/// @param steps Optional actual-state output, null for untraced calls.
/// @param cap Complete record ceiling, zero disables optional cap.
/// @return Actual best subarray, integer sum rounded once to double.
/// @throws std::length_error On enabled ceiling/container overflow.
/// @throws std::overflow_error On produced int64 sum overflow.
/// @throws std::bad_alloc On optional history allocation failure.
MaxSubarray run(const std::vector<int>& a, std::vector<KadaneStep>* steps, std::size_t cap) {
  if (steps) {
    if ((cap && a.size() > cap) || a.size() > steps->max_size())
      throw std::length_error("Kadane complete record ceiling");
    steps->reserve(a.size());
  }
  if (a.empty())
    return {};
  std::int64_t current = a[0], best = current;
  std::size_t  begin = 0, bb = 0, be = 1;
  if (steps)
    steps->push_back({0, a[0], KadaneAction::Seed, 0, current, 0, best, bb, be, true});
  for (std::size_t i = 1; i < a.size(); ++i) {
    const auto before  = current;
    const bool restart = current < 0;
    if (restart) {
      current = a[i];
      begin   = i;
    } else {
      if (a[i] > 0 && current > std::numeric_limits<std::int64_t>::max() - a[i])
        throw std::overflow_error("Kadane exact running sum");
      current += a[i];
    }
    const bool changed = current > best;
    if (changed) {
      best = current;
      bb   = begin;
      be   = i + 1;
    }
    if (steps)
      steps->push_back(
          {i,
           a[i],
           restart ? KadaneAction::Restart : KadaneAction::Extend,
           before,
           current,
           begin,
           best,
           bb,
           be,
           changed});
  }
  return {static_cast<double>(best), bb, be};
}
} // namespace kadane_detail
MaxSubarray kadane(const std::vector<int>& a) {
  return kadane_detail::run(a, nullptr, 0);
}
KadaneTrace kadane_trace(const std::vector<int>& a, std::size_t cap) {
  KadaneTrace t;
  t.result = kadane_detail::run(a, &t.steps, cap);
  return t;
}
} // namespace datamunge::algorithms
