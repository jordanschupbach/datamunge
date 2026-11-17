#pragma once
/// @file kmp.hpp
/// @brief Exact byte-string matching with actual producing prefix/search comparison evidence.
#include <cstddef>
#include <string>
#include <vector>
namespace datamunge::algorithms {
/// @brief Producing phase of an actual KMP byte comparison.
/// @par Example
/// @code
/// const auto t=dm::algorithms::kmp_search_trace("aba","a");
/// std::print("{}\n",t.comparisons.front().phase==dm::algorithms::KMPPhase::Text); // true
/// @endcode
/// @see KMPComparison
enum class KMPPhase {
  Prefix, ///< Pattern-prefix table construction, indices are pattern positions.
  Text    ///< Actual text scan, index identifies the unchanged text byte position.
};
/// @brief Actual comparison site in the preserved while-then-if implementation.
/// While is tested only for positive border length, and a successful while test
/// may be repeated by Extend. Both are actual byte comparisons, not distinct reads.
/// @par Example
/// @code
/// const auto t=dm::algorithms::kmp_search_trace("a","a");
/// std::print("{}\n",t.comparisons[0].site==dm::algorithms::KMPSite::Extend); // true
/// @endcode
/// @see KMPComparison
enum class KMPSite {
  While, ///< Actual positive-border mismatch-loop byte comparison.
  Extend ///< Actual final equality comparison after the mismatch loop.
};
/// @brief Owned operands and pre-test border of one producing byte comparison.
/// index is in pattern during Prefix and text during Text. border is the actual
/// pattern index being tested. Bytes are unsigned values0..255; no encoding or
/// Unicode code-point interpretation. Records do not imply one test per text byte.
/// @par Example
/// @code
/// const auto t=dm::algorithms::kmp_search_trace("b","a");
/// std::print("{} {}\n",t.comparisons[0].index,t.comparisons[0].equal); // 0 false
/// @endcode
/// @see KMPSearchTrace
struct KMPComparison {
  KMPPhase      phase{KMPPhase::Prefix}; ///< Actual producing table/text phase.
  KMPSite       site{KMPSite::Extend};   ///< Actual while-predicate or final-extension comparison.
  std::size_t   index{0};                ///< Actual original position of inspected source byte.
  std::size_t   border{0};               ///< Actual pre-comparison pattern position/border length.
  unsigned char source_byte{0};          ///< Actual pattern/text byte at index.
  unsigned char pattern_byte{0};         ///< Actual pattern byte at border.
  bool          equal{false}; ///< Actual equality result; while mismatches trigger fallback.
};
/// @brief Actual logical comparisons, border fallbacks and overlap resets, excluding time/copies.
/// Counts byte tests in the preserved while-then-if branches, including successful
/// tests repeated by Extend. Does not count border-integer tests, table accesses,
/// output writes, validation, allocation or total machine work.
/// @par Example
/// @code
/// const auto a=dm::algorithms::kmp_search_analyze("aaa","a");
/// std::print("{} {}\n",a.work.text_comparisons,a.work.match_resets); // 3 3
/// @endcode
/// @see kmp_search_analyze()
struct KMPWork {
  std::size_t prefix_comparisons{0}; ///< Actual pattern-prefix construction byte tests.
  std::size_t text_comparisons{0};   ///< Actual text-scan byte tests.
  std::size_t prefix_fallbacks{0}; ///< Actual mismatch fallback updates during prefix construction.
  std::size_t text_fallbacks{0};   ///< Actual mismatch fallback updates during text scan.
  std::size_t match_resets{0};     ///< Actual full-match overlap realignments, one per output.
};
/// @brief Owned exact matches and actual selected work from one shared search execution.
/// No input/history copies retained. Empty pattern or longer-than-text returns
/// no matches and zero work, preserving historical early-exit behavior.
/// @par Example
/// @code
/// const auto a=dm::algorithms::kmp_search_analyze("ababa","aba");
/// std::print("{}\n",a.matches.size()); // 2
/// @endcode
/// @see kmp_search_trace()
struct KMPSearchAnalysis {
  std::vector<std::size_t>
          matches; ///< Actual ascending zero-based byte match starts, overlaps retained.
  KMPWork work;    ///< Producing selected comparisons and branch updates.
};
/// @brief Owned original byte strings, actual table, matches and chronological comparisons.
/// No second search or prefix reconstruction. captured_entries counts original
/// input bytes + produced table entries + comparisons + output match identities.
/// Input bytes admitted before copies, table before allocation, each comparison
/// before evaluation and each match before append. Excludes capacities, private
/// runtime table/output copies, struct fields and bytes/RAM. Default million,
/// zero disables optional quota. No partial trace escapes an exception.
/// @par Example
/// @code
/// const auto t=dm::algorithms::kmp_search_trace("aa","a");
/// std::print("{} {}\n",t.matches.size(),t.comparisons.size()); // 2 2
/// @endcode
/// @see kmp_search_trace()
struct KMPSearchTrace {
  std::string              text;    ///< Owned complete original text bytes, including embedded NUL.
  std::string              pattern; ///< Owned complete original pattern bytes.
  std::vector<std::size_t> prefix;  ///< Actual producing table; empty on the historical early exit.
  std::vector<std::size_t> matches; ///< Actual ascending match starts, independently owned.
  KMPWork                  work;    ///< Actual same-execution selected work.
  std::vector<KMPComparison>
              comparisons; ///< Actual prefix comparisons followed by actual text comparisons.
  std::size_t captured_entries{0}; ///< Logical original-byte/table/comparison/match-entry count.
};
/// @brief Build longest-proper-prefix/suffix lengths of every byte prefix.
/// prefix[i]<=i, with prefix[0]=0. On mismatch fall back through previous prefix
/// entries without backing up i. Empty returns empty; arbitrary embedded NUL/high
/// bytes accepted. O(m) work and owned O(m) output; input unchanged and unretained.
/// Independent calls share no mutable state. This is not Unicode normalization.
/// @param pattern Borrowed arbitrary byte string, possibly empty.
/// @return Owned prefix lengths, one per original byte position.
/// @throws std::length_error On vector/count representability limits.
/// @throws std::bad_alloc On table allocation failure.
/// @par Example
/// @code
/// const auto p=dm::algorithms::kmp_prefix_function("ababaca");
/// std::print("{} {}\n",p[4],p[6]); // 3 1
/// @endcode
/// @par References
/// - Knuth, Morris and Pratt(1977), Fast Pattern Matching in Strings, SIAM J.Comput.6(2),323–350, https://doi.org/10.1137/0206024.
/// @see kmp_search(), kmp_search_trace()
std::vector<std::size_t> kmp_prefix_function(const std::string& pattern);
/// @brief Find all exact byte-string occurrences using the shared KMP prefix/scan kernel.
/// Empty pattern or m>n returns empty without building a prefix table. Otherwise
/// overlapping matches retained in ascending original byte offsets, with no text
/// rewinding. Equality is bytewise, including embedded NUL/high bytes. O(n+m)
/// work, O(m+matches) storage, no input mutation/copies or shared mutable state.
/// @param text Borrowed original byte string, possibly empty.
/// @param pattern Borrowed original byte needle, empty means no matches.
/// @return Owned ascending exact occurrence starts.
/// @throws std::length_error On container/count representability limits.
/// @throws std::bad_alloc On table/output allocation failure.
/// @par Example
/// @code
/// const auto m=dm::algorithms::kmp_search("ababa","aba");
/// std::print("{} {}\n",m[0],m[1]); // 0 2
/// @endcode
/// @par References
/// - Knuth, Morris and Pratt(1977), Fast Pattern Matching in Strings, https://doi.org/10.1137/0206024.
/// @see kmp_prefix_function(), kmp_search_analyze()
std::vector<std::size_t> kmp_search(const std::string& text, const std::string& pattern);
/// @brief Return exact matches and actual selected comparison/fallback work without history.
/// Same early-exit, overlap, byte semantics and O(n+m) kernel as kmp_search().
/// No input copies, optional quotas, comparison records or reconstruction.
/// @param text Borrowed original byte string.
/// @param pattern Borrowed original byte needle.
/// @return Owned matches and producing selected work.
/// @throws std::length_error On container/count representability limits.
/// @throws std::bad_alloc On table/output allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::kmp_search_analyze("ababa","aba").matches.size()); // 2
/// @endcode
/// @par References
/// - Knuth, Morris and Pratt(1977), Fast Pattern Matching in Strings, https://doi.org/10.1137/0206024.
/// @see kmp_search(), kmp_search_trace()
KMPSearchAnalysis kmp_search_analyze(const std::string& text, const std::string& pattern);
/// @brief Capture original bytes and every actual producing prefix/text byte comparison.
/// Same exact byte results and preserved while-then-if comparison order as the
/// scalar kernel. Complete input/table and online test/match admission follow
/// KMPSearchTrace. Adds O(n+m+records+matches) owned capture on top of runtime work.
/// @param text Borrowed complete original bytes, unchanged.
/// @param pattern Borrowed complete original needle, unchanged.
/// @param max_entries Logical original-byte/table/test/match ceiling, default1000000, zero off.
/// @return Owned complete actual observations, never a partial result on failure.
/// @throws std::length_error On enabled capture ceiling or container/count overflow.
/// @throws std::bad_alloc On runtime/capture allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::kmp_search_trace("aa","a",8);
/// std::print("{}\n",t.captured_entries); // 8
/// @endcode
/// @par References
/// - Knuth, Morris and Pratt(1977), Fast Pattern Matching in Strings, https://doi.org/10.1137/0206024.
/// @see KMPSearchTrace, kmp_search_analyze()
KMPSearchTrace kmp_search_trace(
    const std::string& text,
    const std::string& pattern,
    std::size_t        max_entries = 1000000);
} // namespace datamunge::algorithms
