#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace datamunge::algorithms {

/// @brief The longest palindromic substring of a string: its start index and length.
struct PalindromeResult {
    /// @brief 0-based start index in the original string (0 when the input is empty).
    std::size_t start{0};
    /// @brief Length of the substring (0 when the input is empty).
    std::size_t length{0};
};

/// @brief Manacher's algorithm: the longest palindromic substring in linear time.
///
/// Finds the *leftmost* longest palindromic substring of @p s -- among all
/// substrings that read the same forwards and backwards, one of maximum length,
/// and of those the one with the smallest start index (so "babad" yields "bab",
/// not "aba"). The empty string yields {start = 0, length = 0}, and a
/// single-character string yields that character.
///
/// The work is done by palindrome_radii(): its Manacher radius array over the
/// '#'-interleaved transform gives, at every center, the length of the
/// original-string palindrome centered there, from which the maximum is a single
/// linear scan. Total time and space are O(n) for a string of length n.
///
/// @param s the string to search.
/// @return the leftmost longest palindromic substring, as a {start, length} pair.
PalindromeResult longest_palindrome(const std::string& s);

/// @brief Manacher's radius array over the separator-interleaved transform of @p s.
///
/// To treat odd- and even-length palindromes uniformly, the input is transformed
/// into t = "^#a#b#c#...#$": a '#' is inserted before every character and after
/// the last, and distinct sentinels '^' and '$' are placed at the two ends. For a
/// string of length n this makes |t| = 2n + 3; every palindrome of s -- odd or
/// even -- becomes an odd-length palindrome of t centered on some single position,
/// and the sentinels (which match nothing) let center expansion run without any
/// bounds checks.
///
/// The returned array p has one entry per position of t (p.size() == 2n + 3).
/// p[i] is the radius of the longest palindrome of t centered at i, i.e. the
/// largest r with t[i-r .. i+r] a palindrome. With this particular transform p[i]
/// equals *exactly the length of the original-string palindrome* centered at t[i],
/// and that palindrome begins in s at index (i - p[i]) / 2. The two sentinel ends
/// have radius 0.
///
/// Linearity comes from the standard mirror trick: the algorithm tracks the
/// rightmost palindrome discovered so far as [center, right]. For a position i
/// inside it (i < right), p[i] is initialized to min(p[2*center - i], right - i) --
/// the radius of i's mirror about center, capped so it cannot poke past the known
/// boundary -- before any further character comparison. Every comparison that does
/// happen strictly advances right, and right only ever moves forward across the
/// whole run, so the total comparison work is O(n).
///
/// @param s the string to preprocess.
/// @return the Manacher radius array over the transform of @p s (length 2*s.size() + 3).
std::vector<std::size_t> palindrome_radii(const std::string& s);

} // namespace datamunge::algorithms
