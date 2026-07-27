#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace datamunge::algorithms {

/// @brief Fundamental preprocessing: the Z-array of a string.
///
/// z_array(s)[i] is the length of the longest substring starting at position i
/// that is *also a prefix* of s. By convention this implementation sets
/// z[0] = s.size() (the whole string is trivially a prefix of itself); some
/// texts instead leave z[0] undefined or 0, so callers that scan z should start
/// at i = 1. The array is built in a single left-to-right pass in O(n) time and
/// O(n) space using the rightmost Z-box [l, r) window: values that fall inside a
/// previously computed box are reused from their mirror position, and only
/// comparisons that push the right edge r forward ever do explicit character
/// work, which bounds the total comparisons by 2n.
///
/// @param s the string to preprocess.
/// @return z, where z[i] is the prefix-match length at position i and z[0] = s.size().
std::vector<std::size_t> z_array(const std::string& s);

/// @brief Linear-time exact string matching via the Z-array.
///
/// Returns every 0-based start index i in @p text at which @p pattern occurs, in
/// increasing order (overlapping occurrences included). It forms the string
/// pattern + '\x01' + text, computes its Z-array once, and reports every
/// concatenation position in the text region whose z-value reaches
/// pattern.size() -- each such position marks a full copy of the pattern. The
/// separator byte '\x01' is assumed to occur in neither pattern nor text; it
/// keeps a prefix match from spanning the pattern/text boundary. Total work is
/// O(n + m) for a text of length n and a pattern of length m.
///
/// An empty pattern is treated as matching at every position 0..text.size()
/// inclusive (the same set of positions std::string::find("") reports).
///
/// @param text the text to search.
/// @param pattern the pattern to locate.
/// @return the sorted start indices of every occurrence.
std::vector<std::size_t> z_search(const std::string& text, const std::string& pattern);

} // namespace datamunge::algorithms
