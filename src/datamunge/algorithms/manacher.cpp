#include <datamunge/algorithms/manacher.hpp>

namespace datamunge::algorithms {

std::vector<std::size_t> palindrome_radii(const std::string& s) {
    // Transform s -> t = ^ # s0 # s1 # ... # s_{n-1} # $. The interleaved '#'
    // turn every even-length palindrome into an odd-length one centered on a '#',
    // and the distinct sentinels '^'/'$' (which match nothing) let center
    // expansion run without any bounds checks. |t| = 2n + 3.
    std::string t;
    t.reserve(2 * s.size() + 3);
    t.push_back('^');
    for (const char c : s) {
        t.push_back('#');
        t.push_back(c);
    }
    t.push_back('#');
    t.push_back('$');

    const std::size_t m = t.size();
    std::vector<std::size_t> p(m, 0);

    // [center, right] is the palindrome with the furthest-reaching right edge seen
    // so far. right only ever advances, which is what makes the whole pass O(n).
    std::size_t center = 0;
    std::size_t right = 0;

    for (std::size_t i = 1; i + 1 < m; ++i) {
        if (i < right) {
            // Reuse the mirror's radius, capped so it cannot exceed the known
            // boundary. mirror = 2*center - i is a valid non-negative index
            // because i <= right implies i - center <= right - center, hence
            // mirror = center - (i - center) >= 2*center - right = left >= 0.
            const std::size_t mirror = 2 * center - i;
            const std::size_t bound = right - i;
            p[i] = p[mirror] < bound ? p[mirror] : bound;
        }
        // Attempt to grow past whatever was reused. The sentinels guarantee this
        // stops before either index runs off the string.
        while (t[i + p[i] + 1] == t[i - p[i] - 1]) {
            ++p[i];
        }
        // If this palindrome reaches further right than any before, it becomes the
        // new reference [center, right].
        if (i + p[i] > right) {
            center = i;
            right = i + p[i];
        }
    }
    return p;
}

PalindromeResult longest_palindrome(const std::string& s) {
    if (s.empty()) {
        return {0, 0};
    }
    const std::vector<std::size_t> p = palindrome_radii(s);

    // p[i] is exactly the length of the original-string palindrome centered at
    // t[i]. Scanning left to right and keeping only strictly longer palindromes
    // returns the first (smallest-index) occurrence of the maximum -- which, since
    // start = (i - p[i]) / 2 increases with i for a fixed length, is the leftmost
    // longest palindromic substring.
    std::size_t best_center = 0;
    std::size_t best_len = 0;
    for (std::size_t i = 1; i + 1 < p.size(); ++i) {
        if (p[i] > best_len) {
            best_len = p[i];
            best_center = i;
        }
    }

    PalindromeResult result;
    result.length = best_len;
    result.start = (best_center - best_len) / 2;
    return result;
}

} // namespace datamunge::algorithms
