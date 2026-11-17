/// @file steinhaus_johnson_trotter.cpp
/// @brief Single checked largest-mobile generator and actual transition capture.
#include <datamunge/algorithms/steinhaus_johnson_trotter.hpp>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
namespace datamunge::algorithms {
/// @brief Translation-unit-private checked producing kernels; no shared mutable state.
namespace {
/// @brief Generate actual complete permutations after geometry checks.
/// @param n Complete order.
/// @param o Policies.
/// @param capture Retain actual direction states.
/// @return Owned actual stream.
SJTRun run(std::size_t n, const SJTOptions& o, bool capture) {
  const auto limit = std::numeric_limits<std::size_t>::max();
  if (n >= static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error("SJT: int indices");
  std::size_t rows = 1;
  for (std::size_t i = 2; i <= n; ++i) {
    if (rows > limit / i)
      throw std::length_error("SJT: factorial capacity");
    rows *= i;
    if (o.max_permutations && rows > o.max_permutations)
      throw std::length_error("SJT: factorial rows");
  }
  if (o.max_permutations && rows > o.max_permutations)
    throw std::length_error("SJT: rows");
  if (rows > limit / (2 * n + 1) ||
      (o.max_output_values && rows * (2 * n + 1) > o.max_output_values))
    throw std::length_error("SJT: complete output");
  if (capture && (n > limit - 2 || rows > limit / (n + 2) ||
                  (o.max_history_values && rows * (n + 2) > o.max_history_values)))
    throw std::length_error("SJT: complete history");
  SJTRun           r;
  std::vector<int> perm(n), directions(n + 1, -1);
  std::iota(perm.begin(), perm.end(), 1);
  r.permutations.reserve(rows);
  r.permutations.push_back(perm);
  std::vector<std::size_t> position(n + 1);
  r.value_positions.resize(n + 1);
  for (std::size_t value = 1; value <= n; ++value) {
    position[value] = value - 1;
    r.value_positions[value].reserve(rows);
    r.value_positions[value].push_back(value - 1);
  }
  if (capture)
    r.history.push_back({0, 0, -1, -1, directions});
  for (;;) {
    int largest = 0, at = -1;
    for (int i = 0; i < static_cast<int>(n); ++i) {
      if (r.mobile_tests == limit || (o.max_mobile_tests && r.mobile_tests >= o.max_mobile_tests))
        throw std::length_error("SJT: mobile tests");
      ++r.mobile_tests;
      const int value = perm[i], j = i + directions[value];
      if (j >= 0 && j < static_cast<int>(n) && perm[j] < value && value > largest) {
        largest = value;
        at      = i;
      }
    }
    if (!largest)
      break;
    const int to = at + directions[largest];
    std::swap(perm[at], perm[to]);
    position[perm[at]] = static_cast<std::size_t>(at);
    position[perm[to]] = static_cast<std::size_t>(to);
    for (std::size_t value = 1; value <= n; ++value)
      r.value_positions[value].push_back(position[value]);
    ++r.adjacent_swaps;
    for (int value = largest + 1; value <= static_cast<int>(n); ++value) {
      directions[value] = -directions[value];
      if (r.direction_flips == limit)
        throw std::length_error("SJT: flip capacity");
      ++r.direction_flips;
    }
    r.permutations.push_back(perm);
    if (capture)
      r.history.push_back({r.permutations.size() - 1, largest, at, to, directions});
  }
  return r;
}
} // namespace
SJTRun sjt_run(std::size_t n, const SJTOptions& o) {
  return run(n, o, false);
}
SJTRun sjt_trace(std::size_t n, const SJTOptions& o) {
  return run(n, o, true);
}
std::vector<std::vector<int>> sjt_permutations(std::size_t n) {
  return sjt_run(n).permutations;
}
} // namespace datamunge::algorithms
