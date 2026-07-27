#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace datamunge::algorithms {

/// @brief The Boyer-Moore exact string-matching algorithm (Boyer & Moore 1977). Finds every
///        starting index in @p text at which @p pattern occurs. Unlike a left-to-right scan,
///        Boyer-Moore aligns the pattern against the text and compares them *right to left*; on a
///        mismatch it slides the pattern forward by the *maximum* of two precomputed heuristics,
///        so most text characters are never even examined -- giving sublinear average-case time.
///
///        The two heuristics are:
///        - the *bad-character rule*: on a mismatch, align the mismatching text byte with its
///          rightmost occurrence in the pattern (or shift the whole pattern past that byte if it
///          never occurs). Precomputed as a 256-entry last-occurrence table over unsigned char.
///        - the *(strong) good-suffix rule*: the suffix already matched before the mismatch
///          reappears elsewhere in the pattern (preceded by a different character), or a prefix of
///          the pattern equals a suffix of that matched part -- shift to realign with it.
///          Precomputed as a shift table of length m+1 by the classic border-position construction.
///
///        Both shifts are always safe (they never skip an occurrence); taking the larger of the
///        two is what makes the algorithm fast. Overlapping matches are reported (searching for
///        "aa" in "aaaa" yields 0, 1, and 2). By convention an *empty pattern* returns no matches,
///        as does a pattern longer than the text.
///
/// @param text    the string to search within.
/// @param pattern the string to search for.
/// @return the sorted, 0-indexed start positions of all occurrences of @p pattern in @p text.
std::vector<std::size_t> boyer_moore_search(const std::string& text, const std::string& pattern);

} // namespace datamunge::algorithms
