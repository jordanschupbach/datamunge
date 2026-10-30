/// @file z_algorithm.cpp
/// @brief Shared instrumented Z-box loop over borrowed bytes or a virtual non-byte-sentinel join.
#include <algorithm>
#include <datamunge/algorithms/z_algorithm.hpp>
#include <limits>
#include <stdexcept>

namespace datamunge::algorithms {
/// @brief Internal random-access sequence preprocessing.
namespace {
/// @brief Compute a Z-array and optional trace over a virtual sequence without copying symbols.
/// @tparam Symbol Random-access callable returning comparable symbol values.
/// @param n Sequence length.
/// @param symbol Symbol at a valid zero-based index, with no mutation.
/// @param trace Optional output frame; null omits trace storage.
/// @return Owned Z-array and actual comparison counters.
template<class Symbol>
ZArrayStats z_preprocess(std::size_t n, const Symbol& symbol, dstruct::DataFrame* trace) {
  ZArrayStats out;
  out.values.assign(n, 0);
  std::vector<double> positions, codes, values, seeds, left_before, right_before, left_after,
      right_after, comparisons;
  std::size_t left = 0, right = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const auto  old_left = left, old_right = right, old_comparisons = out.comparisons;
    std::size_t seed = 0;
    if (i == 0)
      out.values[0] = n;
    else {
      if (i < right) {
        seed = std::min(right - i, out.values[i - left]);
        ++out.box_reuses;
      }
      out.values[i] = seed;
      while (out.values[i] < n - i) {
        ++out.comparisons;
        if (symbol(out.values[i]) != symbol(i + out.values[i]))
          break;
        ++out.values[i];
        ++out.successful_comparisons;
      }
      if (i + out.values[i] > right) {
        left  = i;
        right = i + out.values[i];
      }
    }
    if (trace) {
      positions.push_back(static_cast<double>(i));
      codes.push_back(symbol(i));
      values.push_back(static_cast<double>(out.values[i]));
      seeds.push_back(static_cast<double>(seed));
      left_before.push_back(static_cast<double>(old_left));
      right_before.push_back(static_cast<double>(old_right));
      left_after.push_back(static_cast<double>(left));
      right_after.push_back(static_cast<double>(right));
      comparisons.push_back(static_cast<double>(out.comparisons - old_comparisons));
    }
  }
  if (trace) {
    trace->add_column("position", positions);
    trace->add_column("symbol_code", codes);
    trace->add_column("z", values);
    trace->add_column("seed", seeds);
    trace->add_column("left_before", left_before);
    trace->add_column("right_before", right_before);
    trace->add_column("left_after", left_after);
    trace->add_column("right_after", right_after);
    trace->add_column("comparisons", comparisons);
  }
  return out;
}
} // namespace
ZArrayStats z_array_stats(const std::string& s) {
  /// @brief Borrowed unsigned-byte accessor for the shared preprocessing loop.
  /// @param i Valid index in s.
  /// @return Byte code in [0,255].
  const auto symbol = [&s](std::size_t i) { return static_cast<unsigned char>(s[i]); };
  return z_preprocess(s.size(), symbol, nullptr);
}
std::vector<std::size_t> z_array(const std::string& s) {
  return z_array_stats(s).values;
}
dstruct::DataFrame z_array_trace(const std::string& s) {
  dstruct::DataFrame out;
  /// @brief Borrowed byte accessor for trace generation.
  /// @param i Valid index in s.
  /// @return Unsigned byte code.
  const auto symbol = [&s](std::size_t i) { return static_cast<unsigned char>(s[i]); };
  (void)z_preprocess(s.size(), symbol, &out);
  return out;
}
ZSearchStats z_search_stats(const std::string& text, const std::string& pattern) {
  ZSearchStats out;
  const auto   n = text.size(), m = pattern.size();
  if (m == 0) {
    out.positions.reserve(n + 1);
    for (std::size_t i = 0; i <= n; ++i)
      out.positions.push_back(i);
    return out;
  }
  if (m > n)
    return out;
  if (n == std::numeric_limits<std::size_t>::max() ||
      m > std::numeric_limits<std::size_t>::max() - n - 1)
    throw std::length_error("z_search: virtual sequence length overflow");
  /// @brief Virtual pattern/sentinel/text accessor, keeping sentinel 256 outside the byte alphabet.
  /// @param i Valid index in the virtual sequence of m+1+n symbols.
  /// @return Unsigned pattern/text byte, or 256 at the separator position.
  const auto symbol = [&text, &pattern, m](std::size_t i) -> unsigned int {
    if (i < m)
      return static_cast<unsigned char>(pattern[i]);
    if (i == m)
      return 256;
    return static_cast<unsigned char>(text[i - m - 1]);
  };
  auto work             = z_preprocess(m + 1 + n, symbol, nullptr);
  out.comparisons       = work.comparisons;
  out.box_reuses        = work.box_reuses;
  out.processed_symbols = m + 1 + n;
  for (std::size_t i = m + 1; i < work.values.size(); ++i)
    if (work.values[i] >= m)
      out.positions.push_back(i - m - 1);
  return out;
}
std::vector<std::size_t> z_search(const std::string& text, const std::string& pattern) {
  return z_search_stats(text, pattern).positions;
}
dstruct::DataFrame z_search_summary(
    const dstruct::DataFrame&       frame,
    const std::string&              column,
    const std::vector<std::string>& patterns) {
  const auto&         texts = frame.get_column<std::string>(column);
  std::vector<double> records, bytes, matches, matched_records, comparisons, processed, bounds,
      reuses;
  for (const auto& pattern : patterns) {
    double nbytes = 0, nmatch = 0, nrecords = 0, ncompare = 0, nprocessed = 0, bound = 0,
           nreuse = 0;
    for (std::size_t r = 0; r < frame.nrows(); ++r) {
      const auto& text = texts.at(r);
      const auto  work = z_search_stats(text, pattern);
      nbytes += text.size();
      nmatch += work.positions.size();
      nrecords += !work.positions.empty();
      ncompare += work.comparisons;
      nprocessed += work.processed_symbols;
      nreuse += work.box_reuses;
      if (work.processed_symbols > 0)
        bound += 2.0 * static_cast<double>(work.processed_symbols - 1);
    }
    records.push_back(static_cast<double>(frame.nrows()));
    bytes.push_back(nbytes);
    matches.push_back(nmatch);
    matched_records.push_back(nrecords);
    comparisons.push_back(ncompare);
    processed.push_back(nprocessed);
    bounds.push_back(bound);
    reuses.push_back(nreuse);
  }
  dstruct::DataFrame out;
  out.add_column("pattern", patterns);
  out.add_column("records", records);
  out.add_column("text_bytes", bytes);
  out.add_column("matches", matches);
  out.add_column("matched_records", matched_records);
  out.add_column("comparisons", comparisons);
  out.add_column("processed_symbols", processed);
  out.add_column("comparison_bound", bounds);
  out.add_column("box_reuses", reuses);
  return out;
}
} // namespace datamunge::algorithms
