/// @file longest_common_substring.cpp
/// @brief Shared checked rolling-row producer and optional actual DP observations.
#include <algorithm>
#include <datamunge/algorithms/longest_common_substring.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private checked geometry and shared producing loop.
namespace {
/// @brief Add logical counts without wrapping.
/// @param a First nonnegative count.
/// @param b Second nonnegative count.
/// @return Exact representable sum.
/// @throws std::length_error On overflow.
std::size_t add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::length_error("LCSubstring: count overflow");
  return a + b;
}
/// @brief Multiply logical counts without wrapping.
/// @param a First nonnegative count.
/// @param b Second nonnegative count.
/// @return Exact representable product.
/// @throws std::length_error On overflow.
std::size_t mul(std::size_t a, std::size_t b) {
  if (a && b > std::numeric_limits<std::size_t>::max() / a)
    throw std::length_error("LCSubstring: count overflow");
  return a * b;
}
/// @brief Evaluate actual common-suffix cells once, with optional complete admitted capture.
/// @param a First original sequence.
/// @param b Second original sequence.
/// @param trace Optional actual owned input/row/cell output.
/// @param quota Optional complete logical capture limit.
/// @return Actual first row-major maximizing run and selected work.
LCSubstringAnalysis produce(
    const std::vector<int>& a,
    const std::vector<int>& b,
    LCSubstringTrace*       trace,
    std::size_t             quota) {
  const std::size_t n = a.size(), m = b.size(), cols = add(m, 1), cells = mul(n, m);
  if (n > std::size_t(std::numeric_limits<std::ptrdiff_t>::max()) ||
      m > std::size_t(std::numeric_limits<std::ptrdiff_t>::max()) ||
      cols > std::vector<std::size_t>().max_size())
    throw std::length_error("LCSubstring: sequence geometry");
  if (trace) {
    const auto total = add(add(n, m), add(mul(add(n, 1), cols), cells));
    if (quota && total > quota)
      throw std::length_error("LCSubstring: complete trace quota");
    trace->captured_entries = total;
    trace->a                = a;
    trace->b                = b;
    trace->rows.reserve(add(n, 1));
    trace->cells.reserve(cells);
  }
  std::vector<std::size_t> prev(cols, 0), cur(cols, 0);
  if (trace)
    trace->rows.push_back(prev);
  LCSubstringAnalysis result;
  for (std::size_t i = 1; i <= n; ++i) {
    for (std::size_t j = 1; j <= m; ++j) {
      ++result.work.cells;
      const bool equal    = a[i - 1] == b[j - 1];
      const auto diagonal = equal ? prev[j - 1] : 0;
      cur[j]              = equal ? diagonal + 1 : 0;
      equal ? ++result.work.matches : ++result.work.resets;
      const bool record = cur[j] > result.length;
      if (record) {
        result.length  = cur[j];
        result.a_start = i - result.length;
        result.b_start = j - result.length;
        ++result.work.records;
      }
      if (trace)
        trace->cells.push_back({i, j, equal, diagonal, cur[j], record});
    }
    std::swap(prev, cur);
    if (trace)
      trace->rows.push_back(prev);
  }
  result.sequence.assign(
      a.begin() + static_cast<std::ptrdiff_t>(result.a_start),
      a.begin() + static_cast<std::ptrdiff_t>(result.a_start + result.length));
  return result;
}
} // namespace
std::vector<int> longest_common_substring(const std::vector<int>& a, const std::vector<int>& b) {
  return produce(a, b, nullptr, 0).sequence;
}
LCSubstringAnalysis lcsubstring_analyze(const std::vector<int>& a, const std::vector<int>& b) {
  return produce(a, b, nullptr, 0);
}
LCSubstringTrace
lcsubstring_trace(const std::vector<int>& a, const std::vector<int>& b, std::size_t max_entries) {
  LCSubstringTrace t;
  t.analysis = produce(a, b, &t, max_entries);
  return t;
}
std::vector<int> lcsubstring_bytes(const std::string& text) {
  std::vector<int> out;
  out.reserve(text.size());
  for (unsigned char c : text)
    out.push_back(c);
  return out;
}
} // namespace datamunge::algorithms
