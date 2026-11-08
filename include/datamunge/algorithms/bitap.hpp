#pragma once
/// @file bitap.hpp
/// @brief Bytewise Shift-And search with bounded substitutions and canonical state traces.
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace datamunge::algorithms {
/// @brief One fixed-length matching window with its actual Hamming distance.
/// Indices are byte offsets in a half-open interval [begin,end). Empty-pattern
/// matches have begin=end. Records own their numbers and retain no input pointers.
/// @par Example
/// @code
/// const auto matches=dm::algorithms::bitap_matches("ABXC","ABCC",1);
/// std::print("{} {} {}\n",matches[0].begin,matches[0].end,matches[0].substitutions); // 0 4 1
/// @endcode
/// @see bitap_matches(), BitapTrace
struct BitapMatch {
  std::size_t begin{0}; ///< Inclusive first byte offset.
  std::size_t end{0}; ///< Exclusive byte offset after the window.
  int substitutions{0}; ///< Minimum substitution count for this full-length window.
};

/// @brief Canonical post-update Shift-And state after one consumed text byte.
/// In states[d], bit j represents a pattern prefix of length j+1 matching the
/// suffix ending at position with at most d substitutions. One means active;
/// all recorded levels use the same convention. Only pattern-length bits are kept.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::bitap_trace("AABA","ABA");
/// std::print("{}\n",trace.steps.back().states[0]); // 5
/// @endcode
/// @see bitap_trace()
struct BitapStep {
  std::size_t position{0}; ///< Zero-based position of the consumed text byte.
  unsigned char byte{0}; ///< Raw byte, without character-set conversion or case folding.
  std::vector<std::uint64_t> states; ///< Levels zero through the effective substitution budget.
};

/// @brief Owned overlapping matches, chronological prefix states and exact scan-work counts.
/// Empty patterns match every boundary but consume no bytes and record no steps.
/// State storage is O(n*(k+1)); match storage is O(number of reported windows).
/// @par Example
/// @code
/// const auto trace=dm::algorithms::bitap_trace("AABA","ABA",1);
/// std::print("{} {}\n",trace.bytes_scanned,trace.word_updates); // 4 8
/// @endcode
/// @see bitap_trace(), BitapStep
struct BitapTrace {
  std::vector<BitapMatch> matches; ///< Every qualifying overlapping window, in increasing offset order.
  std::vector<BitapStep> steps; ///< One post-update state snapshot per consumed byte.
  std::size_t bytes_scanned{0}; ///< Text bytes consumed; zero for an empty pattern.
  std::size_t word_updates{0}; ///< Number of individual error-level state updates.
  int effective_budget{0}; ///< Requested budget clamped into [0,pattern.size()].
};

/// @brief Find the first exact bytewise occurrence with the historical start-index interface.
/// Empty pattern returns zero. Patterns longer than 63 bytes return -1, the same
/// sentinel as a missing match. Searches are case-sensitive and support embedded
/// NUL and arbitrary bytes; Unicode text is interpreted as encoded bytes.
/// Uses the canonical Shift-And loop without recording. O(n+m+256) word work
/// in the worst case, constant auxiliary storage and no input mutation/shared state.
/// @param text Borrowed byte string.
/// @param pattern Borrowed pattern, supported through 63 bytes.
/// @return First matching start offset as long, zero for empty pattern, or -1.
/// @throws std::overflow_error If a found offset cannot be represented by long.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::bitap_search("abracadabra","cad")); // 4
/// @endcode
/// @par References
/// - R. Baeza-Yates and G. H. Gonnet (1992), A New Approach to Text Searching,
///   Communications of the ACM 35(10), 74–82, doi:10.1145/135239.135243.
/// @see bitap_fuzzy_search(), bitap_matches(), bitap_trace()
long bitap_search(const std::string& text,const std::string& pattern);

/// @brief Find the first Hamming-close window with the historical inclusive-end interface.
/// Allows substitutions only; windows have exactly pattern.size() bytes. Negative
/// budgets mean zero; budgets above the pattern length cannot admit extra windows.
/// Empty pattern returns the historical special value zero. Patterns longer than
/// 62 bytes return -1. Uses the canonical loop; no state vector is recorded.
/// O(n*(k+1)+m+256) word work with k capped at m; constant word-state storage.
/// No mutation or shared state. Byte, case and encoding semantics match bitap_search().
/// @param text Borrowed byte string.
/// @param pattern Borrowed pattern, supported through 62 bytes.
/// @param max_errors Maximum substitutions; clamped into [0,pattern.size()].
/// @return Inclusive ending offset of the first full-length match, zero for empty pattern, or -1.
/// @throws std::overflow_error If a found ending offset cannot be represented by long.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::bitap_fuzzy_search("abracadabra","cid",1)); // 6
/// @endcode
/// @par References
/// - R. Baeza-Yates and G. H. Gonnet (1992), A New Approach to Text Searching,
///   doi:10.1145/135239.135243 (this implementation uses substitutions only).
/// @see bitap_search(), bitap_matches()
long bitap_fuzzy_search(const std::string& text,const std::string& pattern,int max_errors);

/// @brief Return all overlapping byte windows within a substitution budget.
/// Supports patterns through 63 bytes. Empty patterns match all n+1 boundaries,
/// with empty intervals and zero substitutions. For nonempty patterns, reports
/// actual minimum substitutions even when the supplied budget is larger. Uses
/// O(n*(k+1)+m+256) word work, fixed word-state storage and owned output storage.
/// No input mutation, retained pointers or shared mutable state.
/// @param text Borrowed byte string; embedded NUL and non-ASCII bytes are supported.
/// @param pattern Borrowed pattern of at most 63 bytes.
/// @param max_substitutions Budget, default zero, clamped into [0,pattern.size()].
/// @return Owned match records ordered by their start offsets; overlaps are retained.
/// @throws std::invalid_argument If the pattern exceeds 63 bytes.
/// @par Example
/// @code
/// const auto matches=dm::algorithms::bitap_matches("AAAA","AA");
/// std::print("{} {}\n",matches.size(),matches.back().begin); // 3 2
/// @endcode
/// @par References
/// - R. Baeza-Yates and G. H. Gonnet (1992), A New Approach to Text Searching,
///   doi:10.1145/135239.135243.
/// @see bitap_trace(), BitapMatch
std::vector<BitapMatch> bitap_matches(const std::string& text,const std::string& pattern,
                                    int max_substitutions=0);

/// @brief Record the same coupled Shift-And loop used by the first-match and all-match APIs.
/// Each consumed byte updates k+1 states. Records every state with one=active,
/// independent of the older implementation's complemented-bit convention.
/// Empty-pattern and byte semantics match bitap_matches(). O(n*(k+1)) trace
/// storage in addition to matches; no input mutation or shared state.
/// @param text Borrowed byte string.
/// @param pattern Pattern of at most 63 bytes, possibly empty.
/// @param max_substitutions Budget, default zero, clamped into [0,pattern.size()].
/// @return Owned matches, state snapshots, effective budget and exact scan counts.
/// @throws std::invalid_argument If the pattern exceeds 63 bytes.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::bitap_trace("AABA","ABA");
/// std::print("{} {}\n",trace.matches.front().begin,trace.steps.back().states[0]); // 1 5
/// @endcode
/// @par References
/// - R. Baeza-Yates and G. H. Gonnet (1992), A New Approach to Text Searching,
///   doi:10.1145/135239.135243.
/// @see bitap_matches(), BitapStep
BitapTrace bitap_trace(const std::string& text,const std::string& pattern,int max_substitutions=0);
} // namespace datamunge::algorithms
