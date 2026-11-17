/// @file odd_even_sort.cpp
/// @brief Shared original serial odd-even sorting loop and actual histories.
#include <algorithm>
#include <datamunge/algorithms/odd_even_sort.hpp>
#include <limits>
#include <numeric>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private checked accounting and the one production sorting path.
namespace {
/// @brief Add nonnegative geometry/counter values without wraparound.
/// @param a First count.
/// @param b Added count.
/// @return Checked sum.
/// @throws std::length_error If addition overflows.
std::size_t add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::length_error("odd-even count geometry");
  return a + b;
}
/// @brief Sort actual values with optional carried identities and owned actual history.
/// @param a Owned actual working values.
/// @param identities Whether final IDs are requested.
/// @param out Optional history sink, untouched caller inputs.
/// @param cap Online logical history cap, zero disables.
/// @return Actual sorted result and selected counters.
/// @throws std::length_error If checked counts or online history quota exceed capacity.
OddEvenSortRun run(std::vector<int> a, bool identities, OddEvenSortTrace* out, std::size_t cap) {
  const std::size_t        n = a.size();
  OddEvenSortRun           result;
  std::vector<std::size_t> ids;
  /// @brief Admit actual retained history elements before any corresponding copy/append.
  /// @param units Actual complete next record contribution.
  /// @throws std::length_error If count geometry or selected quota is exceeded.
  auto charge = [&](std::size_t units) {
    if (!out)
      return;
    auto next = add(out->captured_entries, units);
    if (cap && next > cap)
      throw std::length_error("odd-even capture quota");
    out->captured_entries = next;
  };
  if (out) {
    charge(add(n, add(n, n)));
    out->input = a;
  }
  if (identities) {
    ids.resize(n);
    std::iota(ids.begin(), ids.end(), 0);
  }
  bool sorted = false;
  while (!sorted) {
    sorted             = true;
    result.work.rounds = add(result.work.rounds, 1);
    for (bool odd : {true, false}) {
      OddEvenSortPhase phase;
      phase.ordinal      = result.work.phases;
      phase.odd          = odd;
      result.work.phases = add(result.work.phases, 1);
      for (std::size_t i = odd ? 1 : 0; i < n && n - i > 1; i += 2) {
        OddEvenSortComparison c;
        c.phase       = phase.ordinal;
        c.left        = i;
        c.left_value  = a[i];
        c.right_value = a[i + 1];
        if (identities) {
          c.left_id  = ids[i];
          c.right_id = ids[i + 1];
        }
        c.swapped               = a[i] > a[i + 1];
        result.work.comparisons = add(result.work.comparisons, 1);
        phase.comparisons       = add(phase.comparisons, 1);
        if (out) {
          charge(1);
          out->comparisons.push_back(c);
        }
        if (c.swapped) {
          std::swap(a[i], a[i + 1]);
          if (identities)
            std::swap(ids[i], ids[i + 1]);
          sorted            = false;
          result.work.swaps = add(result.work.swaps, 1);
          phase.swaps       = add(phase.swaps, 1);
        }
      }
      if (out) {
        charge(add(1, add(n, n)));
        phase.values           = a;
        phase.original_indices = ids;
        out->phases.push_back(std::move(phase));
      }
    }
  }
  result.output           = std::move(a);
  result.original_indices = std::move(ids);
  return result;
}
} // namespace
std::vector<int> odd_even_sort(std::vector<int> a) {
  return run(std::move(a), false, nullptr, 0).output;
}
OddEvenSortRun odd_even_sort_run(std::vector<int> a) {
  return run(std::move(a), true, nullptr, 0);
}
OddEvenSortTrace odd_even_sort_trace(std::vector<int> a, std::size_t max_entries) {
  OddEvenSortTrace t;
  t.run = run(std::move(a), true, &t, max_entries);
  return t;
}
std::vector<int> odd_even_sort_demo_input() {
  return {5, 3, 8, 1, 9, 2, 7};
}
} // namespace datamunge::algorithms
