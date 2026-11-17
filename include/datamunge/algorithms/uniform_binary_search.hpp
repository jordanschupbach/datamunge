#pragma once
/// @file uniform_binary_search.hpp
/// @brief Checked native uniform binary search and actual producing diagnostics.
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>
namespace datamunge::algorithms {
/// @brief Precompute the fixed descending power-of-two schedule.
/// Uses n/2 rather than doubling in the loop condition, avoiding signed overflow. O(log n) work/storage.
/// @param n Nonnegative array length, up to INT_MAX.
/// @return Largest power of two <=n down to1; empty at n0.
/// @throws std::invalid_argument Negative length.
/// @par Example
/// @code
/// auto steps=dm::algorithms::uniform_search_table(5); //4,2,1
/// @endcode
/// @par References
/// - The accompanying report derives the implemented recurrence.
/// @see Other operations in this header.
inline std::vector<int> uniform_search_table(int n) {
  if (n < 0)
    throw std::invalid_argument("negative search size");
  std::vector<int> steps;
  if (!n)
    return steps;
  int s = 1;
  while (s <= n / 2)
    s *= 2;
  for (; s; s /= 2)
    steps.push_back(s);
  return steps;
}
/// @brief One actual fixed-schedule probe.
/// @par Example
/// @code
/// dm::algorithms::UniformSearchStep r;
/// @endcode
/// @par References
/// - The accompanying report derives the implemented recurrence.
/// @see Other operations in this header.
struct UniformSearchStep {
  int       step{0};         ///< Scheduled positive stride.
  long long before{-1};      ///< Actual running index before probe.
  long long probe{-1};       ///< Candidate index, possibly beyond input.
  bool      accepted{false}; ///< In bounds and key comparison accepted.
  long long after{-1};       ///< Actual running index after probe.
};
/// @brief Actual fixed-schedule search result.
/// @par Example
/// @code
/// dm::algorithms::UniformSearchWork r;
/// @endcode
/// @par References
/// - The accompanying report derives the implemented recurrence.
/// @see Other operations in this header.
struct UniformSearchWork {
  std::size_t index{std::size_t(-1)};    ///< Rightmost equal index, or size_t(-1) if absent.
  std::vector<UniformSearchStep> probes; ///< Actual steps in schedule order.
  std::size_t comparisons{0}; ///< Actual <= probes plus final equality check; validation excluded.
};
/// @brief Search a checked sorted array using the exact precomputed schedule.
/// @tparam T Copyable ordered key type supporting <= and ==.
/// @param a Borrowed nondecreasing keys, size<=INT_MAX; finite built-in floating values required.
/// @param key Borrowed target; built-in floating target must be finite.
/// @param steps Exact uniform_search_table(a.size()) schedule; malformed schedules rejected.
/// @return Actual probes, comparisons and rightmost match or size_t(-1).
/// @throws std::invalid_argument Unsorted input, nonfinite floating data or malformed schedule.
/// @throws std::length_error More than INT_MAX keys.
/// @par Example
/// @code
/// auto r=dm::algorithms::uniform_binary_search_work(std::vector<int>{1,2,2},2,dm::algorithms::uniform_search_table(3)); //index2
/// @endcode
/// Validation scans all keys in O(n); actual probe work is O(log n), stored in O(log n).
/// Inputs unchanged; calls independent. Generic comparison exceptions propagate.
template<class T>
UniformSearchWork
uniform_binary_search_work(const std::vector<T>& a, const T& key, const std::vector<int>& steps) {
  if (a.size() > INT_MAX)
    throw std::length_error("search length");
  if (steps != uniform_search_table(static_cast<int>(a.size())))
    throw std::invalid_argument("search schedule");
  if constexpr (std::is_floating_point_v<T>) {
    if (!std::isfinite(key))
      throw std::invalid_argument("nonfinite key");
    for (auto x : a)
      if (!std::isfinite(x))
        throw std::invalid_argument("nonfinite key");
  }
  if (!std::is_sorted(a.begin(), a.end()))
    throw std::invalid_argument("unsorted search input");
  UniformSearchWork r;
  long long         i = -1;
  for (int step : steps) {
    long long probe    = i + step;
    bool      accepted = false;
    if (probe < static_cast<long long>(a.size())) {
      ++r.comparisons;
      accepted = a[probe] <= key;
    }
    r.probes.push_back({step, i, probe, accepted, accepted ? probe : i});
    if (accepted)
      i = probe;
  }
  if (i >= 0) {
    ++r.comparisons;
    if (a[i] == key)
      r.index = i;
  }
  return r;
}
/// @brief Return the checked fixed-schedule producer's rightmost matching index.
/// @tparam T Ordered key type supporting <= and ==.
/// @param a Borrowed sorted keys.
/// @param key Target value.
/// @param steps Exact canonical schedule for this array size.
/// @return Rightmost equal index, or size_t(-1).
/// @throws std::invalid_argument Invalid input ordering, floating values or schedule.
/// @throws std::length_error More than INT_MAX keys.
/// @par Example
/// @code
/// auto i=dm::algorithms::uniform_binary_search(std::vector<int>{1,2},2,dm::algorithms::uniform_search_table(2)); //1
/// @endcode
template<class T>
std::size_t
uniform_binary_search(const std::vector<T>& a, const T& key, const std::vector<int>& steps) {
  return uniform_binary_search_work(a, key, steps).index;
}

} // namespace datamunge::algorithms
