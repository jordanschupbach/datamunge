#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace datamunge::algorithms {

/// @brief The prefix function (a.k.a. failure function) of @p pattern, the core of the
///        Knuth-Morris-Pratt algorithm. For each position i, pi[i] is the length of the longest
///        *proper* prefix of pattern[0..i] that is also a suffix of pattern[0..i] -- "proper"
///        meaning strictly shorter than i+1, so pi[i] <= i. It is computed in O(m) time (m the
///        pattern length) by a single left-to-right scan that itself reuses the values already
///        found: on a mismatch the candidate length falls back to pi of the previous match,
///        which amortizes the total work to linear. On a mismatch during matching, pi tells the
///        automaton how far the pattern can be slid forward without missing any occurrence.
///
/// @param pattern the string to analyze.
/// @return a vector pi of the same length as @p pattern (empty for an empty pattern).
std::vector<std::size_t> kmp_prefix_function(const std::string& pattern);

/// @brief The Knuth-Morris-Pratt exact string-matching algorithm (Knuth, Morris & Pratt 1977).
///        Finds every starting index in @p text at which @p pattern occurs, in O(n + m) time
///        (n = text length, m = pattern length). It first builds the pattern's prefix function
///        (@ref kmp_prefix_function), then scans @p text once, keeping the length q of the
///        longest pattern prefix that matches a suffix of the text read so far. On a character
///        mismatch it does *not* back up in the text: it slides the pattern by consulting
///        pi[q-1], reusing the fact that the failure function already records how a matched
///        proper prefix realigns with itself. Overlapping matches are reported (searching for
///        "aba" in "ababa" yields both 0 and 2). By convention an *empty pattern* returns no
///        matches (the caller can treat an empty needle as it prefers).
///
/// @param text   the string to search within.
/// @param pattern the string to search for.
/// @return the sorted, 0-indexed start positions of all occurrences of @p pattern in @p text.
std::vector<std::size_t> kmp_search(const std::string& text, const std::string& pattern);

} // namespace datamunge::algorithms
