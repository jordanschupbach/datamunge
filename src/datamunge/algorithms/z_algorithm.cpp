#include <datamunge/algorithms/z_algorithm.hpp>

#include <algorithm>

namespace datamunge::algorithms {

std::vector<std::size_t> z_array(const std::string& s) {
    const std::size_t n = s.size();
    std::vector<std::size_t> z(n, 0);
    if (n == 0) return z;
    z[0] = n; // convention: the whole string is a prefix of itself

    // [l, r) is the rightmost Z-box discovered so far: a half-open interval known
    // to match a prefix of s, with r - 1 the farthest matched position.
    std::size_t l = 0;
    std::size_t r = 0;
    for (std::size_t i = 1; i < n; ++i) {
        if (i < r) {
            // i lies inside the current Z-box, so its mirror position i - l already
            // carries a match against the prefix; reuse it, but never claim past r.
            z[i] = std::min(r - i, z[i - l]);
        }
        // Extend the match beyond what we could reuse by explicit comparison.
        while (i + z[i] < n && s[z[i]] == s[i + z[i]]) {
            ++z[i];
        }
        // If this match reaches farther right than any before, it is the new box.
        if (i + z[i] > r) {
            l = i;
            r = i + z[i];
        }
    }
    return z;
}

std::vector<std::size_t> z_search(const std::string& text, const std::string& pattern) {
    const std::size_t m = pattern.size();
    const std::size_t n = text.size();
    std::vector<std::size_t> matches;

    if (m == 0) {
        // The empty string occurs at every position, including one past the end.
        matches.reserve(n + 1);
        for (std::size_t i = 0; i <= n; ++i) matches.push_back(i);
        return matches;
    }
    if (m > n) return matches; // pattern cannot fit

    // pattern + separator + text; the separator ('\x01') is assumed to appear in
    // neither string, so no prefix match spans the boundary.
    std::string joined;
    joined.reserve(m + 1 + n);
    joined += pattern;
    joined += '\x01';
    joined += text;

    const std::vector<std::size_t> z = z_array(joined);
    // Positions in the text region start at m + 1 (after pattern and separator).
    for (std::size_t i = m + 1; i < joined.size(); ++i) {
        if (z[i] >= m) matches.push_back(i - (m + 1));
    }
    return matches;
}

} // namespace datamunge::algorithms
