/// @file levenshtein.cpp
/// @brief Shared actual rolling-row byte distance with admitted producing snapshots and diagnostic optimal moves.
#include <algorithm>
#include <datamunge/algorithms/levenshtein.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private safe geometry and one shared row producer.
namespace levenshtein_detail {
/// @brief Validate complete rolling row, selected cell counts and safe distance increments.
/// @param a Original first length.
/// @param b Original second length.
/// @throws std::length_error On full row/work/safe increment geometry.
void geometry(std::size_t a, std::size_t b) {
  const auto max = std::numeric_limits<std::size_t>::max();
  if (a >= max - 1 || b >= max - 1 || a > max - b || (a && b > max / a) ||
      std::min(a, b) + 1 > std::vector<std::size_t>{}.max_size())
    throw std::length_error("Levenshtein complete geometry");
}
/// @brief Evaluate every actual rolling-row cell once and optionally capture the produced rows.
/// @param a Validated original first bytes.
/// @param b Validated original second bytes.
/// @param trace Optional completely admitted capture.
/// @return Actual minimum cost and selected producing counts.
/// @throws std::bad_alloc On runtime/capture allocation failure.
LevenshteinAnalysis run(const std::string& a, const std::string& b, LevenshteinTrace* trace) {
  const auto&              s = a.size() <= b.size() ? a : b;
  const auto&              t = a.size() <= b.size() ? b : a;
  std::vector<std::size_t> prev(s.size() + 1), curr(s.size() + 1);
  for (std::size_t j = 0; j <= s.size(); ++j)
    prev[j] = j;
  LevenshteinAnalysis out;
  if (trace)
    trace->table.push_back(prev);
  for (std::size_t i = 1; i <= t.size(); ++i) {
    curr[0] = i;
    for (std::size_t j = 1; j <= s.size(); ++j) {
      const std::size_t cost = t[i - 1] == s[j - 1] ? 0 : 1;
      out.work.equalities += cost == 0;
      curr[j] = std::min({prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost});
      ++out.work.cells;
    }
    prev.swap(curr);
    ++out.work.rows;
    if (trace)
      trace->table.push_back(prev);
  }
  out.distance = prev.back();
  return out;
}
} // namespace levenshtein_detail
std::size_t levenshtein_distance(const std::string& a, const std::string& b) {
  return levenshtein_analyze(a, b).distance;
}
LevenshteinAnalysis levenshtein_analyze(const std::string& a, const std::string& b) {
  levenshtein_detail::geometry(a.size(), b.size());
  return levenshtein_detail::run(a, b, nullptr);
}
LevenshteinTrace
levenshtein_trace(const std::string& a, const std::string& b, std::size_t max_entries) {
  levenshtein_detail::geometry(a.size(), b.size());
  const auto n = std::max(a.size(), b.size()), m = std::min(a.size(), b.size()),
             max = std::numeric_limits<std::size_t>::max();
  if (n + 1 > max / (m + 1))
    throw std::length_error("Levenshtein full table geometry");
  const auto cells = (n + 1) * (m + 1), sum = n + m;
  if (sum > (max - cells) / 2 || n + 1 > std::vector<std::vector<std::size_t>>{}.max_size() ||
      cells > std::vector<std::size_t>{}.max_size())
    throw std::length_error("Levenshtein capture geometry");
  if (max_entries && cells + 2 * sum > max_entries)
    throw std::length_error("Levenshtein conservative capture quota");
  LevenshteinTrace out;
  out.rows_are_a   = a.size() > b.size();
  out.row_bytes    = out.rows_are_a ? a : b;
  out.column_bytes = out.rows_are_a ? b : a;
  out.table.reserve(n + 1);
  out.path.reserve(sum);
  out.analysis  = levenshtein_detail::run(a, b, &out);
  std::size_t i = n, j = m;
  while (i || j) {
    LevenshteinEdit e;
    e.row                  = i;
    e.column               = j;
    const std::size_t cost = i && j && out.row_bytes[i - 1] == out.column_bytes[j - 1] ? 0 : 1;
    if (i && j && out.table[i][j] == out.table[i - 1][j - 1] + cost) {
      --i;
      --j;
      e.cost = cost;
    } else if (i && out.table[i][j] == out.table[i - 1][j] + 1) {
      --i;
      e.cost = 1;
    } else {
      --j;
      e.cost = 1;
    }
    e.next_row    = i;
    e.next_column = j;
    out.path.push_back(e);
    ++out.analysis.work.traceback_moves;
  }
  out.captured_entries = sum + cells + out.path.size();
  return out;
}
} // namespace datamunge::algorithms
