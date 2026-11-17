/// @file exchange_sorts.cpp
/// @brief Canonical producing implementation and actual-site diagnostics; public contracts are in the paired header.
#include <algorithm>
#include <cstddef>
#include <datamunge/algorithms/exchange_sorts.hpp>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

/// @brief File-local validated kernels shared by ordinary and observed entry points.
namespace {
/// @brief Shared original greedy loop; captures values only at actual producing sites.
PancakeResult pancake_kernel(std::vector<int> a, PancakeTrace* trace, std::size_t cap) {
  std::size_t retained = 0;
  /// @brief Admit logical retained history scalars before copying snapshots.
  auto admit = [&](std::size_t n) {
    if (n > std::numeric_limits<std::size_t>::max() - retained ||
        (cap && (retained > cap || n > cap - retained)))
      throw std::length_error("pancake history cap");
    retained += n;
  };
  if (trace) {
    admit(a.size());
    trace->states.push_back(a);
  }
  PancakeResult     out;
  const std::size_t n    = a.size();
  auto              flip = [&](std::size_t k) { // reverse the prefix of length k
    std::reverse(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(k));
    out.flips.push_back(k);
    if (trace) {
      admit(a.size());
      trace->states.push_back(a);
    }
  };
  for (std::size_t size = n; size > 1; --size) {
    // Index of the maximum within the unsorted prefix a[0..size-1].
    std::size_t maxi = 0;
    for (std::size_t i = 1; i < size; ++i)
      if (a[i] > a[maxi])
        maxi = i;
    const auto begin = out.flips.size();
    if (maxi != size - 1) {
      if (maxi != 0)
        flip(maxi + 1);
      flip(size);
    }
    if (trace) {
      admit(a.size());
      admit(1);
      trace->steps.push_back({size, maxi, size - 1, begin, out.flips.size(), a});
    }
  }
  out.sorted = std::move(a);
  return out;
}

} // namespace
PancakeResult pancake_sort(std::vector<int> a) {
  return pancake_kernel(std::move(a), nullptr, 0);
}
PancakeTrace pancake_sort_trace(std::vector<int> a, std::size_t cap) {
  PancakeTrace t;
  t.result = pancake_kernel(std::move(a), &t, cap);
  return t;
}
} // namespace datamunge::algorithms
