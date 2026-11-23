#pragma once
/// @file pooled_permutations.hpp
/// @brief Internal fixed-group-size assignment enumeration and sampling.

#include <datamunge/random/random.hpp>
#include <algorithm>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace datamunge::stats::detail {

/// @brief Counts distinct fixed-size subsets using integer cancellation and a caller-supplied limit.
/// @param total Number of distinct pooled rows.
/// @param selected Size of the smaller group.
/// @param limit Largest allowed number of assignments.
/// @return Binomial coefficient C(total, selected).
/// @throws std::length_error The exact reference exceeds the allowed size.
inline std::size_t fixed_assignment_count(std::size_t total, std::size_t selected, std::size_t limit) {
  std::size_t count = 1;
  for (std::size_t i = 1; i <= selected; ++i) {
    const auto factor = total - selected + i;
    const auto divisor = std::gcd(factor, i);
    const auto numerator = factor / divisor;
    count /= i / divisor;
    if (count > limit / numerator)
      throw std::length_error("pooled permutation: exact assignment cap exceeded");
    count *= numerator;
  }
  return count;
}

/// @brief Advances a sorted subset to its lexicographic successor.
/// @param group Current sorted subset, overwritten when a successor exists.
/// @param total Size of the pooled set.
/// @return True if a successor exists.
inline bool next_fixed_assignment(std::vector<std::size_t>& group, std::size_t total) {
  for (std::size_t i = group.size(); i > 0; --i) {
    if (group[i - 1] == total - group.size() + i - 1)
      continue;
    ++group[i - 1];
    for (std::size_t j = i; j < group.size(); ++j)
      group[j] = group[j - 1] + 1;
    return true;
  }
  return false;
}

/// @brief Selects a uniform subset as the prefix of an existing permutation.
/// @param order Current permutation, overwritten by a partial Fisher–Yates pass.
/// @param selected Desired prefix size, no larger than order.size().
/// @param rng Source of unbiased bounded integer draws.
inline void shuffle_assignment(std::vector<std::size_t>& order, std::size_t selected,
                               random::RandomGenerator& rng) {
  for (std::size_t i = 0; i < selected; ++i)
    std::swap(order[i], order[rng.uniform_u64(i, order.size() - 1)]);
}

} // namespace datamunge::stats::detail
