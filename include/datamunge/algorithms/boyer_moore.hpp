#pragma once
/// @file boyer_moore.hpp
/// @brief Bytewise Boyer-Moore search, strong good-suffix tables and actual alignment traces.
#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace datamunge::algorithms {
/// @brief Owned rightmost-byte and strong good-suffix tables for one pattern.
/// Obtain populated values from boyer_moore_tables(); default field values are
/// placeholders. Tables borrow no pattern storage and can be inspected concurrently.
/// @par Example
/// @code
/// const auto tables=dm::algorithms::boyer_moore_tables("GCGGCG");
/// std::print("{} {}\n",tables.last['C'],tables.good_suffix[0]); // 4 3
/// @endcode
/// @see boyer_moore_tables()
struct BoyerMooreTables {
  std::array<std::ptrdiff_t,256> last{}; ///< Rightmost pattern index per unsigned byte, or -1 if absent in a populated table.
  std::vector<std::ptrdiff_t> good_suffix; ///< Positive shifts indexed by matched-suffix start k; k=0 follows a full match.
};
/// @brief One actual right-to-left scan and the shift that follows it.
/// Full matches use mismatch_index=pattern length, text_byte=0 and bad_character_shift=0;
/// inspect matched to distinguish that sentinel from an actual NUL mismatch byte.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::boyer_moore_trace("XXABC","ABC");
/// std::print("{} {}\n",trace.alignments[0].mismatch_index,trace.alignments[0].shift); // 2 2
/// @endcode
/// @see BoyerMooreTrace
struct BoyerMooreAlignment {
  std::size_t position{0}; ///< Current pattern start offset in the text, in bytes.
  std::size_t mismatch_index{0}; ///< Pattern mismatch index, or pattern length after a full match.
  std::size_t matched_suffix_length{0}; ///< Equal trailing bytes before the mismatch, or the whole pattern length.
  unsigned char text_byte{0}; ///< Actual mismatching text byte, or zero on a full match.
  std::ptrdiff_t bad_character_shift{0}; ///< Raw j-last[c], possibly negative; zero on a full match.
  std::ptrdiff_t good_suffix_shift{0}; ///< Strong shift[j+1], or shift[0] on a full match.
  std::size_t shift{0}; ///< Positive chosen shift: max(1,bad,good), or the full-match border shift.
  std::size_t comparisons{0}; ///< Pattern/text byte comparisons in this alignment, including a failed comparison.
  bool matched{false}; ///< Whether the complete pattern matched at position.
};
/// @brief Owned tables, all matches, actual alignments and search-comparison count.
/// No text or pattern bytes are copied into the trace. Sizes and positions are
/// byte counts; preprocessing comparisons are excluded. Empty/oversized patterns
/// produce no alignments, but the trace still contains their preprocessed tables.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::boyer_moore_trace("aaaa","aa");
/// std::print("{} {}\n",trace.matches.size(),trace.comparisons); // 3 6
/// @endcode
/// @see boyer_moore_trace(), BoyerMooreAlignment
struct BoyerMooreTrace {
  BoyerMooreTables tables; ///< Tables built by the same preprocessing used in ordinary search.
  std::vector<std::size_t> matches; ///< All overlapping occurrence starts, in increasing byte order.
  std::vector<BoyerMooreAlignment> alignments; ///< Actual visited alignments, including the final departure shift.
  std::size_t text_size{0}; ///< Original text byte length.
  std::size_t pattern_size{0}; ///< Original pattern byte length.
  std::size_t comparisons{0}; ///< Sum of alignment byte-comparison counts; excludes preprocessing and loop conditions.
};
/// @brief Preprocess unsigned-byte last occurrences and strong good-suffix shifts.
/// The two-pass border-position construction builds good_suffix[k] for a matched
/// pattern suffix starting at k, with a mismatch just before k when k>0. Index zero
/// holds the full-match period shift. Empty input yields last=-1 and good_suffix={1}.
/// O(m+256) work/storage; input is borrowed and unchanged; no shared state.
/// @param pattern Arbitrary byte string of length at most PTRDIFF_MAX-1.
/// @return Owned 256-entry last table and positive m+1 suffix shifts.
/// @throws std::length_error If the signed border sentinel m+1 is unrepresentable, or vector capacity is exceeded.
/// @par Example
/// @code
/// const auto tables=dm::algorithms::boyer_moore_tables("GCGGCG");
/// std::print("{}\n",tables.good_suffix); // [3, 3, 3, 3, 5, 2, 1]
/// @endcode
/// @par References
/// - R. S. Boyer and J S. Moore (1977), A fast string searching algorithm,
///   Communications of the ACM 20(10), 762-772, doi:10.1145/359842.359859.
///   Uses strong good-suffix preprocessing by border positions.
/// @see boyer_moore_search(), boyer_moore_trace()
BoyerMooreTables boyer_moore_tables(const std::string& pattern);
/// @brief Find every overlapping bytewise occurrence by right-to-left Boyer-Moore scans.
/// Uses bad-character and strong good-suffix shifts from boyer_moore_tables().
/// Empty or oversized patterns return no matches before preprocessing. Supports
/// embedded NUL/high bytes, is case-sensitive and performs no Unicode normalization.
/// Inputs are borrowed and unchanged; owned offsets are sorted; calls share no state.
/// Preprocessing takes O(m+256) work/storage. This all-occurrences implementation
/// does not reuse matched regions across alignments: worst-case search comparisons
/// are O((n-m+1)*m), e.g. repeated identical bytes. Large skips are data-dependent.
/// Output costs O(K) for K matches; the ordinary search does not allocate alignment records.
/// @param text Arbitrary byte string to search.
/// @param pattern Pattern bytes; eligible nonempty patterns must have length at most PTRDIFF_MAX-1.
/// @return Increasing zero-based byte offsets, including overlapping occurrences; empty for an empty pattern.
/// @throws std::length_error If eligible pattern preprocessing cannot represent/allocate its tables.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::boyer_moore_search("aaaa","aa")); // [0, 1, 2]
/// @endcode
/// @par References
/// - R. S. Boyer and J S. Moore (1977), A fast string searching algorithm,
///   doi:10.1145/359842.359859. This version reports all matches without the Galil optimization.
/// @see boyer_moore_trace(), boyer_moore_tables()
std::vector<std::size_t> boyer_moore_search(const std::string& text,const std::string& pattern);
/// @brief Record the same tables, matches, right-to-left decisions and shifts as ordinary search.
/// Always preprocesses the pattern, including when it is empty or longer than text.
/// Search costs match boyer_moore_search(); recording adds O(A+K) storage for A
/// visited alignments and K matches, besides O(m+256) table storage. No bytes are
/// copied, no caller data are mutated and independent calls share no state.
/// @param text Borrowed arbitrary bytes.
/// @param pattern Borrowed bytes with length at most PTRDIFF_MAX-1.
/// @return Owned preprocessing and actual search records, including negative raw bad shifts.
/// @throws std::length_error If pattern preprocessing cannot represent/allocate its tables.
/// @throws std::overflow_error If the total byte-comparison counter exceeds size_t.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::boyer_moore_trace("XXABC","ABC");
/// std::print("{} {}\n",trace.matches,trace.alignments.size()); // [2] 2
/// @endcode
/// @par References
/// - R. S. Boyer and J S. Moore (1977), doi:10.1145/359842.359859;
///   full search comparisons are recorded, with no remembered-match optimization.
/// @see boyer_moore_search(), BoyerMooreTrace
BoyerMooreTrace boyer_moore_trace(const std::string& text,const std::string& pattern);
} // namespace datamunge::algorithms
