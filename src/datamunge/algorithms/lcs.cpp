/// @file lcs.cpp
/// @brief One checked full-table LCS producer shared by ordinary, counted and captured APIs.
#include <algorithm>
#include <datamunge/algorithms/lcs.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private checked geometry and shared actual producer.
namespace lcs_detail {
/// @brief Validate complete table and traceback count geometry before allocation.
/// @param n Actual first length.
/// @param m Actual second length.
/// @return Full table cell count including boundaries.
/// @throws std::length_error On int-length, size_t or container count limits.
std::size_t geometry(std::size_t n, std::size_t m) {
  const auto max = std::numeric_limits<std::size_t>::max();
  if (std::min(n, m) > std::numeric_limits<int>::max() || n == max || m == max || n > max - m ||
      (n + 1) > max / (m + 1))
    throw std::length_error("LCS complete geometry");
  const auto cells = (n + 1) * (m + 1);
  if (n + 1 > std::vector<std::vector<int>>{}.max_size() || m + 1 > std::vector<int>{}.max_size() ||
      cells > std::vector<int>{}.max_size())
    throw std::length_error("LCS table geometry");
  return cells;
}
/// @brief Produce the original table and actual traceback once with optional owned capture.
/// @param a Validated first sequence.
/// @param b Validated second sequence.
/// @param capture Optional preadmitted trace.
/// @return Actual forward result and selected work.
/// @throws std::bad_alloc On table/output/capture allocation failure.
LCSAnalysis run(const std::vector<int>& a, const std::vector<int>& b, LCSTrace* capture) {
  std::vector<std::vector<int>> c(a.size() + 1, std::vector<int>(b.size() + 1, 0));
  LCSAnalysis                   out;
  for (std::size_t i = 1; i <= a.size(); ++i)
    for (std::size_t j = 1; j <= b.size(); ++j) {
      c[i][j] = a[i - 1] == b[j - 1] ? c[i - 1][j - 1] + 1 : std::max(c[i - 1][j], c[i][j - 1]);
      ++out.work.cells;
    }
  std::size_t i = a.size(), j = b.size();
  while (i && j) {
    LCSStep step{i, j};
    if (a[i - 1] == b[j - 1]) {
      step.move = LCSMove::Match;
      out.sequence.push_back(a[i - 1]);
      --i;
      --j;
      ++out.work.matches;
    } else if (c[i - 1][j] >= c[i][j - 1]) {
      step.move = LCSMove::Up;
      --i;
    } else {
      step.move = LCSMove::Left;
      --j;
    }
    ++out.work.traceback_steps;
    if (capture)
      capture->steps.push_back(step);
  }
  std::reverse(out.sequence.begin(), out.sequence.end());
  if (capture)
    capture->table = std::move(c);
  return out;
}
} // namespace lcs_detail
std::vector<int> longest_common_subsequence(const std::vector<int>& a, const std::vector<int>& b) {
  return lcs_analyze(a, b).sequence;
}
LCSAnalysis lcs_analyze(const std::vector<int>& a, const std::vector<int>& b) {
  lcs_detail::geometry(a.size(), b.size());
  return lcs_detail::run(a, b, nullptr);
}
LCSTrace lcs_trace(const std::vector<int>& a, const std::vector<int>& b, std::size_t max_entries) {
  const auto cells = lcs_detail::geometry(a.size(), b.size()), sum = a.size() + b.size(),
             max = std::numeric_limits<std::size_t>::max();
  if (sum > (max - cells) / 2)
    throw std::length_error("LCS capture geometry");
  const auto bound = cells + 2 * sum;
  if (max_entries && bound > max_entries)
    throw std::length_error("LCS complete conservative capture quota");
  LCSTrace out;
  out.a = a;
  out.b = b;
  out.steps.reserve(sum);
  out.analysis         = lcs_detail::run(a, b, &out);
  out.captured_entries = sum + cells + out.steps.size();
  return out;
}
} // namespace datamunge::algorithms
