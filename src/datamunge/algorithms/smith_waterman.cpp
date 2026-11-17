/// @file smith_waterman.cpp
/// @brief Single checked local recurrence and actual maximum-to-zero traceback.
#include <algorithm>
#include <cstdint>
#include <datamunge/algorithms/smith_waterman.hpp>
#include <limits>
#include <stdexcept>
#include <utility>
/// @brief Canonical checked native producers and owned values.
namespace datamunge::algorithms {
/// @brief Private invocation-local checked producing helpers.
namespace {
/// @brief Evaluate the local table and backtrack its actual selected states.
/// @param a First bytes.
/// @param b Second bytes.
/// @param scoring Integer scores.
/// @param o Policies.
/// @param capture Keep actual table/path.
/// @return Actual local result.
SmithWatermanRun run(
    const std::string&          a,
    const std::string&          b,
    AlignmentScoring            scoring,
    const SmithWatermanOptions& o,
    bool                        capture) {
  if (scoring.gap > 0)
    throw std::invalid_argument("smith waterman: nonpositive gap");
  const auto n = a.size(), m = b.size(), limit = std::numeric_limits<std::size_t>::max();
  if (n == limit || m == limit || (n + 1) > limit / (m + 1))
    throw std::length_error("smith waterman: cells capacity");
  const auto cells = (n + 1) * (m + 1);
  if ((o.max_cells && cells > o.max_cells) ||
      (n + 1) > std::vector<std::vector<int>>{}.max_size() ||
      (m + 1) > std::vector<int>{}.max_size())
    throw std::length_error("smith waterman: DP policy");
  std::vector<std::vector<int>> H(n + 1, std::vector<int>(m + 1));
  SmithWatermanRun              r;
  std::size_t                   bi = 0, bj = 0;
  for (std::size_t i = 1; i <= n; ++i)
    for (std::size_t j = 1; j <= m; ++j) {
      const auto         pair  = a[i - 1] == b[j - 1] ? scoring.match : scoring.mismatch;
      const std::int64_t diag  = std::int64_t(H[i - 1][j - 1]) + pair,
                         up    = std::int64_t(H[i - 1][j]) + scoring.gap,
                         left  = std::int64_t(H[i][j - 1]) + scoring.gap,
                         value = std::max({std::int64_t{0}, diag, up, left});
      if (value > std::numeric_limits<int>::max())
        throw std::overflow_error("smith waterman: score capacity");
      H[i][j] = static_cast<int>(value);
      ++r.evaluated_cells;
      if (value > r.alignment.score) {
        r.alignment.score = static_cast<int>(value);
        bi                = i;
        bj                = j;
      }
    }
  r.end_a = bi;
  r.end_b = bj;
  auto i = bi, j = bj;
  while (i && j && H[i][j] > 0) {
    const auto row = i, col = j;
    const int  score = H[i][j];
    char       move;
    const auto pair = a[i - 1] == b[j - 1] ? scoring.match : scoring.mismatch;
    if (std::int64_t(score) == std::int64_t(H[i - 1][j - 1]) + pair) {
      move = 'D';
      r.alignment.a_aligned.push_back(a[--i]);
      r.alignment.b_aligned.push_back(b[--j]);
    } else if (std::int64_t(score) == std::int64_t(H[i - 1][j]) + scoring.gap) {
      move = 'U';
      r.alignment.a_aligned.push_back(a[--i]);
      r.alignment.b_aligned.push_back('-');
    } else {
      move = 'L';
      r.alignment.a_aligned.push_back('-');
      r.alignment.b_aligned.push_back(b[--j]);
    }
    ++r.traceback_steps;
    if (capture) {
      if (o.max_trace_steps && r.traceback_steps > o.max_trace_steps)
        throw std::length_error("smith waterman: trace policy");
      r.traceback.push_back({row, col, score, move});
    }
  }
  r.begin_a = i;
  r.begin_b = j;
  std::reverse(r.alignment.a_aligned.begin(), r.alignment.a_aligned.end());
  std::reverse(r.alignment.b_aligned.begin(), r.alignment.b_aligned.end());
  if (capture)
    r.table = std::move(H);
  return r;
}
} // namespace
SmithWatermanRun smith_waterman_run(
    const std::string&          a,
    const std::string&          b,
    AlignmentScoring            s,
    const SmithWatermanOptions& o) {
  return run(a, b, s, o, false);
}
SmithWatermanRun smith_waterman_trace(
    const std::string&          a,
    const std::string&          b,
    AlignmentScoring            s,
    const SmithWatermanOptions& o) {
  return run(a, b, s, o, true);
}
Alignment smith_waterman(const std::string& a, const std::string& b, AlignmentScoring s) {
  return smith_waterman_run(a, b, s).alignment;
}
} // namespace datamunge::algorithms
