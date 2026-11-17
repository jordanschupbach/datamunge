#pragma once
/// @file patience_sort.hpp
/// @brief Leftmost-greater-or-equal pile dealing and deterministic min-heap extraction.
#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
namespace datamunge::algorithms {
/// @brief Actual integer deal decision and post-deal pile tops.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::patience_sort_trace({2,1}).deals[1].pile); // 0
/// @endcode
struct PatienceDeal {
  int              value;       ///< Incoming value.
  std::size_t      pile;        ///< Leftmost pile with top >= value, or newly appended pile.
  std::size_t      comparisons; ///< Actual binary-search top comparisons.
  std::vector<int> tops;        ///< Actual tops immediately after placing value.
};
/// @brief Owned integer producer histories and sorted output.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::patience_sort_trace({2,1}).sorted); // [1, 2]
/// @endcode
struct PatienceTrace {
  std::vector<int>          sorted; ///< Nondecreasing output from actual heap extractions.
  std::vector<PatienceDeal> deals;  ///< Actual incoming-value decisions.
  std::vector<std::vector<int>>
      piles; ///< Complete piles after dealing, before extraction; bottom to top.
  std::vector<std::size_t>
      extracted_piles; ///< Actual min-heap pop pile ids; ties prefer smaller id.
};
/// @brief Internal shared generic patience-sort kernel.
namespace patience_detail {
/// @brief Deal, optionally observe integer states, and heap-merge without replay.
/// @tparam T Copyable ordered value supporting >= and pair ordering.
/// @param a Input/output values; changed only after a complete merge.
/// @param trace Optional integer observer, valid only for T=int.
/// @param cap Retained scalar cap, zero disables policy.
template<class T>
void kernel(std::vector<T>& a, PatienceTrace* trace, std::size_t cap) {
  std::size_t retained = 0;
  /// @brief Admit retained history before its allocation.
  auto admit = [&](std::size_t n) {
    if (n > std::numeric_limits<std::size_t>::max() - retained ||
        (cap && (retained > cap || n > cap - retained)))
      throw std::length_error("patience history cap");
    retained += n;
  };
  std::vector<std::vector<T>> piles;
  for (const T& x : a) {
    std::size_t lo = 0, hi = piles.size(), comparisons = 0;
    while (lo < hi) {
      const auto mid = lo + (hi - lo) / 2;
      ++comparisons;
      if (piles[mid].back() >= x)
        hi = mid;
      else
        lo = mid + 1;
    }
    if (lo == piles.size())
      piles.push_back({x});
    else
      piles[lo].push_back(x);
    if constexpr (std::is_same_v<T, int>) {
      if (trace) {
        admit(1);
        admit(piles.size());
        PatienceDeal d{x, lo, comparisons, {}};
        for (const auto& p : piles)
          d.tops.push_back(p.back());
        trace->deals.push_back(std::move(d));
      }
    }
  }
  if constexpr (std::is_same_v<T, int>) {
    if (trace) {
      admit(a.size());
      admit(piles.size());
      trace->piles = piles;
    }
  }
  using Item = std::pair<T, std::size_t>;
  std::priority_queue<Item, std::vector<Item>, std::greater<Item>> pq;
  for (std::size_t i = 0; i < piles.size(); ++i)
    pq.emplace(piles[i].back(), i);
  std::vector<T> out;
  out.reserve(a.size());
  while (!pq.empty()) {
    const auto [val, i] = pq.top();
    pq.pop();
    out.push_back(val);
    if (trace) {
      admit(1);
      trace->extracted_piles.push_back(i);
    }
    piles[i].pop_back();
    if (!piles[i].empty())
      pq.emplace(piles[i].back(), i);
  }
  a.swap(out);
  if constexpr (std::is_same_v<T, int>) {
    if (trace) {
      admit(a.size());
      trace->sorted = a;
    }
  }
}
} // namespace patience_detail
/// @brief Sort in place by leftmost top >= value dealing and a min-heap merge.
/// Pile count equals strictly increasing LIS length for a total order, including
/// duplicates. Not stable. O(n log n) comparisons, O(n) working storage. Empty is
/// valid. No state or references survive; concurrent separate inputs are safe.
/// @tparam T Copyable total-order value supporting >= and pair ordering.
/// @param a Borrowed input/output vector, replaced only after complete merge.
/// @par Example
/// @code
/// std::vector<int>a{3,1,2};dm::algorithms::patience_sort(a);std::print("{}\n",a); // [1, 2, 3]
/// @endcode
/// @par References
/// - D. Aldous and P. Diaconis (1999), Longest Increasing Subsequences: From
///   Patience Sorting to the Baik-Deift-Johansson Theorem, doi:10.1090/S0273-0979-99-00796-X.
template<class T>
void patience_sort(std::vector<T>& a) {
  patience_detail::kernel(a, nullptr, 0);
}
/// @brief Retain actual integer deals, complete piles and heap-source ids.
/// Worst-case O(n^2) retained tops; no second deal/replay is used.
/// @param a Owned integer input, copied for lvalues.
/// @param max_entries Retained logical integers/records cap, default one million;
/// counts deals, their tops, piles and their values, extracted ids and sorted
/// values; zero disables policy. Excludes bytes, allocator overhead and peak RAM.
/// @return Owned producer histories and sorted output.
/// @throws std::length_error If history policy or representable storage is exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::patience_sort_trace({2,2,1}).piles.size()); // 1
/// @endcode
inline PatienceTrace patience_sort_trace(std::vector<int> a, std::size_t max_entries = 1000000) {
  PatienceTrace t;
  patience_detail::kernel(a, &t, max_entries);
  return t;
}
} // namespace datamunge::algorithms
