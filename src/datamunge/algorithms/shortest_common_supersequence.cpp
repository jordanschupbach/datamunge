/// @file shortest_common_supersequence.cpp
/// @brief One checked actual LCS table and deterministic reverse-emission stream.
#include <algorithm>
#include <datamunge/algorithms/shortest_common_supersequence.hpp>
#include <limits>
#include <stdexcept>
#include <utility>
/// @brief Canonical shortest-supersequence implementation.
namespace datamunge::algorithms {
/// @brief Private intrinsic DP and optional capture producer.
/// @brief Private invocation-local checked algorithm helpers.
namespace {
/// @brief Validate complete geometry, fill actual cells and emit one actual shortest sequence.
/// @param a Borrowed first input.
/// @param b Borrowed second input.
/// @param o Policies.
/// @param capture Retain actual inputs/table/backtracking.
/// @return Owned actual output and selected work.
SCSRun
solve(const std::vector<int>& a, const std::vector<int>& b, const SCSOptions& o, bool capture) {
  const auto n = a.size(), m = b.size(), max = std::numeric_limits<std::size_t>::max();
  if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      m > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      (o.max_length && (n > o.max_length || m > o.max_length)) || n == max || m == max ||
      n + 1 > max / (m + 1) || n > max - m)
    throw std::length_error("SCS: input/table capacity");
  const auto cells = (n + 1) * (m + 1);
  if (o.max_cells && cells > o.max_cells)
    throw std::length_error("SCS: intrinsic table policy");
  SCSRun      out;
  std::size_t retained = 0;
  /// @brief Admit actual optional retained scalars/records before their work.
  /// @param count Actual next capture geometry.
  const auto admit = [&](std::size_t count) {
    if (!capture)
      return;
    if (count > max - retained || (o.max_trace_entries && (retained > o.max_trace_entries ||
                                                           count > o.max_trace_entries - retained)))
      throw std::length_error("SCS: history policy");
    retained += count;
  };
  if (capture) {
    if (cells > max - n - m)
      throw std::length_error("SCS: capture capacity");
    admit(cells + n + m);
    out.first  = a;
    out.second = b;
  }
  std::vector<std::vector<int>> table(n + 1, std::vector<int>(m + 1));
  for (std::size_t i = 1; i <= n; ++i)
    for (std::size_t j = 1; j <= m; ++j) {
      ++out.equality_tests;
      if (a[i - 1] == b[j - 1])
        table[i][j] = table[i - 1][j - 1] + 1;
      else {
        ++out.score_comparisons;
        table[i][j] = std::max(table[i - 1][j], table[i][j - 1]);
      }
    }
  out.lcs_length = table[n][m];
  out.sequence.reserve(n + m - out.lcs_length);
  std::size_t i = n, j = m;
  while (i || j) {
    admit(1);
    SCSStep event;
    event.first_size  = i;
    event.second_size = j;
    bool shared       = false;
    if (i && j) {
      ++out.equality_tests;
      shared = a[i - 1] == b[j - 1];
    }
    if (shared) {
      event.choice = SCSChoice::Shared;
      event.value  = a[--i];
      --j;
    } else {
      bool take_first = i && !j;
      if (i && j) {
        ++out.score_comparisons;
        take_first = table[i - 1][j] >= table[i][j - 1];
      }
      if (take_first) {
        event.choice = SCSChoice::First;
        event.value  = a[--i];
      } else {
        event.choice = SCSChoice::Second;
        event.value  = b[--j];
      }
    }
    out.sequence.push_back(event.value);
    if (capture)
      out.steps.push_back(event);
  }
  std::reverse(out.sequence.begin(), out.sequence.end());
  if (capture)
    out.table = std::move(table);
  return out;
}
} // namespace
SCSRun shortest_common_supersequence_run(
    const std::vector<int>& a,
    const std::vector<int>& b,
    const SCSOptions&       o) {
  return solve(a, b, o, false);
}
SCSRun shortest_common_supersequence_trace(
    const std::vector<int>& a,
    const std::vector<int>& b,
    const SCSOptions&       o) {
  return solve(a, b, o, true);
}
std::vector<int> shortest_common_supersequence(
    const std::vector<int>& a,
    const std::vector<int>& b) {
  return solve(a, b, {}, false).sequence;
}
} // namespace datamunge::algorithms
