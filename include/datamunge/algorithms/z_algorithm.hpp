#pragma once

/// @file z_algorithm.hpp
/// @brief Linear Z-box preprocessing, binary-safe exact search and actual comparison diagnostics.
#include <cstddef>
#include <datamunge/dstruct/dataframe.hpp>
#include <string>
#include <vector>

namespace datamunge::algorithms {

/// @brief Owned Z-array and work counters from the same preprocessing loop.
///
/// Comparisons count symbol-equality tests only, excluding bounds checks.
/// successful_comparisons extend a match, and the remaining comparisons fail.
/// box_reuses counts positions inside an existing rightmost box, even when
/// their copied seed is zero. Position zero uses the convention Z[0]=n.
/// @par Example
/// @code
/// const auto result = dm::algorithms::z_array_stats("aaaa");
/// const auto comparisons = result.comparisons; // 3
/// @endcode
/// @see z_array_stats(), z_array_trace()
struct ZArrayStats {
  std::vector<std::size_t>
              values; ///< Z[i] is the longest prefix match at position i; empty for empty input.
  std::size_t comparisons            = 0; ///< Actual symbol equality tests.
  std::size_t successful_comparisons = 0; ///< Equality tests that extend a match.
  std::size_t box_reuses = 0; ///< Positions initialised from the current rightmost Z-box.
};

/// @brief Owned exact matches and preprocessing work for one Z-based search.
///
/// Empty patterns match every boundary without preprocessing. A nonempty
/// pattern longer than the text also returns immediately. Otherwise the
/// processed sequence is a virtual pattern, one non-byte sentinel, and text.
/// @par Example
/// @code
/// const auto result = dm::algorithms::z_search_stats("abababa", "aba");
/// const auto count = result.positions.size(); // 3
/// @endcode
/// @see z_search_stats(), z_search_summary()
struct ZSearchStats {
  std::vector<std::size_t> positions;       ///< Sorted zero-based byte offsets, including overlaps.
  std::size_t              comparisons = 0; ///< Actual symbol equality tests in Z preprocessing.
  std::size_t              processed_symbols =
      0;                      ///< Virtual sequence length, or zero for either early-return case.
  std::size_t box_reuses = 0; ///< Positions reusing a previously established Z-box.
};

/// @brief Fundamental preprocessing: prefix-match lengths at every string position.
///
/// Z[i] is the largest k with s[0..k)=s[i..i+k). Z[0]=s.size(), unlike
/// conventions that set it to zero. Matches are on bytes, including NUL and
/// high-bit bytes, without Unicode normalisation. O(n) time and owned output
/// storage; the input is borrowed and unchanged. Concurrent calls are safe.
/// @param s Any byte string, possibly empty.
/// @return Owned Z-array, or an empty vector when s is empty.
/// @par Example
/// @code
/// const auto z = dm::algorithms::z_array("abacaba"); // {7,0,1,0,3,0,1}
/// @endcode
/// @par References
/// - D. Gusfield (1997), Algorithms on Strings, Trees, and Sequences, chapter 1,
///   https://doi.org/10.1017/CBO9780511574931
/// @see z_array_stats(), z_array_trace(), z_search()
std::vector<std::size_t> z_array(const std::string& s);

/// @brief Compute the Z-array while counting actual symbol tests and box reuses.
///
/// For n>0, at most n-1 successful and n-1 failed comparisons occur, hence
/// comparisons <= 2(n-1). The counters come from the same loop as z_array().
/// @param s Borrowed byte string, possibly empty.
/// @return Owned array and counters, all counters zero for empty or one-byte strings.
/// @par Example
/// @code
/// const auto work = dm::algorithms::z_array_stats("aaaa");
/// const auto reuses = work.box_reuses; // 2
/// @endcode
/// @par References
/// - D. Gusfield (1997), chapter 1, https://doi.org/10.1017/CBO9780511574931
/// @see z_array(), z_array_trace()
ZArrayStats z_array_stats(const std::string& s);

/// @brief Trace the rightmost Z-box and copied seed at each preprocessing position.
///
/// Columns position, symbol_code (0..255), z, seed, left_before, right_before,
/// left_after, right_after and comparisons. Boxes are half-open intervals.
/// Row zero records the Z[0]=n convention with no comparisons and an empty
/// current box; the operational scan starts at position one. O(n) storage.
/// @param s Borrowed bytes; empty input produces an empty table with the same columns.
/// @return Owned diagnostic frame; numeric indices are exact up to 2^53.
/// @par Example
/// @code
/// const auto trace = dm::algorithms::z_array_trace("aaaa");
/// const double seed = trace.double_at("seed", 2); // 2
/// @endcode
/// @par References
/// - D. Gusfield (1997), chapter 1, https://doi.org/10.1017/CBO9780511574931
/// @see z_array_stats()
dstruct::DataFrame z_array_trace(const std::string& s);

/// @brief Find every exact pattern occurrence in linear time using a virtual Z-array join.
///
/// Uses an integer sentinel 256 outside the byte alphabet; all 256 byte values
/// may occur in either input. No joined string is copied. For nonempty patterns
/// that fit, O(n+m) time and temporary storage plus owned matches. Empty patterns
/// match boundaries 0..text.size(), as repeated std::string::find("") does.
/// @param text Borrowed text bytes, unchanged.
/// @param pattern Borrowed pattern bytes, unchanged.
/// @return Sorted zero-based byte offsets, including overlapping occurrences.
/// @throws std::length_error If the virtual sequence length exceeds size_t range.
/// @par Example
/// @code
/// const auto hits = dm::algorithms::z_search("abababa", "aba"); // {0,2,4}
/// @endcode
/// @par References
/// - D. Gusfield (1997), chapter 1, https://doi.org/10.1017/CBO9780511574931
/// @see z_search_stats(), z_array()
std::vector<std::size_t> z_search(const std::string& text, const std::string& pattern);

/// @brief Find every exact occurrence and report the actual Z-preprocessing work.
/// @param text Borrowed text bytes, unchanged.
/// @param pattern Borrowed pattern bytes, unchanged; empty matches every boundary.
/// @return Owned matches, symbol comparisons, processed length and box reuses.
/// @throws std::length_error If the virtual sequence length exceeds size_t range.
/// @par Example
/// @code
/// const auto work = dm::algorithms::z_search_stats("abababa", "aba");
/// const auto length = work.processed_symbols; // 11
/// @endcode
/// @par References
/// - D. Gusfield (1997), chapter 1, https://doi.org/10.1017/CBO9780511574931
/// @see z_search(), z_search_summary()
ZSearchStats z_search_stats(const std::string& text, const std::string& pattern);

/// @brief Summarise Z-search matches and actual work over separate DataFrame records.
///
/// Every pattern is searched within each record independently: occurrences
/// cannot cross record boundaries. Work is O(sum(n_record+m)) per nonempty
/// pattern, plus reported matches; only one record's temporary array is held.
/// Counters exclude bounds checks, column access and output construction.
/// @param frame Borrowed dataset, unchanged.
/// @param column String column with no null values.
/// @param patterns Case-sensitive byte patterns, in desired output order.
/// @return Owned rows per pattern with pattern, records, text_bytes, matches,
///         matched_records, comparisons, processed_symbols, comparison_bound
///         (the sum of 2(L-1) over processed sequences) and box_reuses.
/// @throws std::invalid_argument If the column is not a string column or contains nulls.
/// @throws std::out_of_range If the column is absent.
/// @throws std::length_error If a virtual sequence length exceeds size_t range.
/// @par Example
/// @code
/// const auto table = dm::algorithms::z_search_summary(dm::datasets::titanic(), "Name", {"Miss.",
/// "Mrs."}); const auto rows = table.nrows(); // 2
/// @endcode
/// @par References
/// - D. Gusfield (1997), chapter 1, https://doi.org/10.1017/CBO9780511574931
/// @see z_search_stats(), z_array_stats()
dstruct::DataFrame z_search_summary(
    const dstruct::DataFrame&       frame,
    const std::string&              column,
    const std::vector<std::string>& patterns);
} // namespace datamunge::algorithms
