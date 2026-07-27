#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace datamunge::algorithms {

/// @brief The Rabin-Karp string-matching algorithm (Karp & Rabin 1987). Finds every position in
///        @p text at which @p pattern occurs, using a *polynomial rolling hash*. Each length-m
///        window of the text is treated as an m-digit number in some base b, taken modulo a large
///        prime p; sliding the window one character to the right updates its hash in O(1) --
///        subtract the leading digit's contribution, multiply by the base, add the incoming digit
///        -- rather than re-reading the whole window. When a window's hash equals the pattern's,
///        that only *suggests* a match (two distinct strings can collide modulo p), so the actual
///        characters are compared to confirm it; the returned positions are therefore always
///        exact. Expected running time is O(n + m) when collisions are rare, degrading to O(n*m)
///        in the worst case (adversarial or unlucky inputs that force a verification at every
///        window). Overlapping occurrences are all reported.
///
/// @param text the string to search within.
/// @param pattern the string to search for.
/// @return the sorted list of start indices at which @p pattern occurs in @p text. By convention
///         an empty @p pattern matches at every position, yielding {0, 1, ..., text.size()}
///         (mirroring std::string::find("")). If @p pattern is longer than @p text, the result is
///         empty.
std::vector<std::size_t> rabin_karp_search(const std::string& text, const std::string& pattern);

} // namespace datamunge::algorithms
