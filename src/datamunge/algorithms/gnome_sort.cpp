/// @file gnome_sort.cpp
/// @brief One gnome loop for legacy sorting, checked work and online actual state records.
#include <datamunge/algorithms/gnome_sort.hpp>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace datamunge::algorithms {
/// @brief Implementation-only checked counts, conditional state capture and shared pointer loop.
namespace {
/// @brief Increment one selected operation count without wrapping.
/// @param value Diagnostic count updated in place.
/// @throws std::overflow_error The next count cannot fit size_t.
void increment(std::size_t& value) {
  if (value == std::numeric_limits<std::size_t>::max())
    throw std::overflow_error("gnome sort: diagnostic count overflow");
  ++value;
}
/// @brief Append a complete actual-state record, including metadata, when capture is enabled.
/// @param events Owned target history, or null for ordinary execution.
/// @param max_records Optional complete-record quota; zero disables it.
/// @param event Actual state to append, without sorting or replay.
/// @throws std::length_error The next complete record exceeds quota/container capacity.
/// @throws std::bad_alloc History allocation fails.
void append(
    std::vector<GnomeSortEvent>* events,
    std::size_t                  max_records,
    const GnomeSortEvent&        event) {
  if (!events)
    return;
  if ((max_records && events->size() >= max_records) || events->size() == events->max_size())
    throw std::length_error("gnome sort: complete record quota/capacity exceeded");
  events->push_back(event);
}
/// @brief Execute the historical short-circuit gnome loop once, optionally observing actual work.
/// @tparam Observe Initialize/carry identities, check counters and optionally emit records.
/// @param a Owned values mutated by actual adjacent swaps.
/// @param events Optional owned target history, unused for unobserved scalar sorting.
/// @param max_records Optional complete-record quota; zero disables it.
/// @return Owned sorted output plus actual identities/work when Observe is true.
/// @throws std::length_error Owned storage or the next complete record cannot fit its
/// capacity/quota.
/// @throws std::overflow_error An observed counter cannot fit size_t.
/// @throws std::bad_alloc Owned storage allocation fails.
template<bool Observe>
GnomeSortRun
sort(std::vector<int> a, std::vector<GnomeSortEvent>* events, std::size_t max_records) {
  GnomeSortRun result;
  if constexpr (Observe) {
    result.original_indices.resize(a.size());
    std::iota(result.original_indices.begin(), result.original_indices.end(), std::size_t{0});
    append(events, max_records, {});
  }
  const std::size_t n = a.size();
  std::size_t       i = 0;
  while (i < n) {
    GnomeSortEvent event;
    if constexpr (Observe) {
      increment(result.work.visits);
      event.kind         = GnomeSortEventKind::step;
      event.ordinal      = result.work.visits - 1;
      event.index_before = i;
      if (i == 0)
        increment(result.work.boundary_visits);
      else {
        increment(result.work.key_comparisons);
        event.compared             = true;
        event.left                 = a[i - 1];
        event.right                = a[i];
        event.left_original_index  = result.original_indices[i - 1];
        event.right_original_index = result.original_indices[i];
      }
    }
    if (i == 0 || a[i - 1] <= a[i]) {
      ++i;
      if constexpr (Observe)
        increment(result.work.forward_steps);
    } else {
      std::swap(a[i - 1], a[i]);
      if constexpr (Observe) {
        std::swap(result.original_indices[i - 1], result.original_indices[i]);
        increment(result.work.swaps);
        increment(result.work.backward_steps);
        event.swapped = true;
      }
      --i;
    }
    if constexpr (Observe) {
      event.index_after = i;
      append(events, max_records, event);
    }
  }
  if constexpr (Observe) {
    GnomeSortEvent event;
    event.kind         = GnomeSortEventKind::stop;
    event.index_before = event.index_after = n;
    append(events, max_records, event);
  }
  result.output = std::move(a);
  return result;
}
} // namespace

std::vector<int> gnome_sort(std::vector<int> a) {
  return sort<false>(std::move(a), nullptr, 0).output;
}
GnomeSortRun gnome_sort_run(std::vector<int> a) {
  return sort<true>(std::move(a), nullptr, 0);
}
GnomeSortTrace gnome_sort_trace(std::vector<int> a, std::size_t max_records) {
  GnomeSortTrace result;
  result.run = sort<true>(std::move(a), &result.events, max_records);
  return result;
}
AdjacentSortInputs adjacent_sort_inputs(std::size_t n, std::size_t max_values) {
  AdjacentSortInputs result;
  if ((n && std::cmp_greater(n - 1, std::numeric_limits<int>::max())) ||
      n > std::numeric_limits<std::size_t>::max() / 4 || n > result.ascending.max_size())
    throw std::length_error("adjacent sort inputs: unrepresentable count");
  const auto total = 4 * n;
  if (max_values && total > max_values)
    throw std::length_error("adjacent sort inputs: whole four-array quota exceeded");
  result.ascending.resize(n);
  result.descending.resize(n);
  result.adjacent_inversion.resize(n);
  result.last_minimum.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    result.ascending[i] = result.adjacent_inversion[i] = static_cast<int>(i);
    result.descending[i]                               = static_cast<int>(n - 1 - i);
    result.last_minimum[i]                             = i + 1 == n ? 0 : static_cast<int>(i + 1);
  }
  if (n >= 2)
    std::swap(result.adjacent_inversion[n / 2 - 1], result.adjacent_inversion[n / 2]);
  return result;
}
} // namespace datamunge::algorithms
