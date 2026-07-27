#include <datamunge/algorithms/kmp.hpp>

namespace datamunge::algorithms {

std::vector<std::size_t> kmp_prefix_function(const std::string& pattern) {
    const std::size_t m = pattern.size();
    std::vector<std::size_t> pi(m, 0);
    // k tracks the length of the current longest proper prefix-suffix of pattern[0..i-1].
    std::size_t k = 0;
    for (std::size_t i = 1; i < m; ++i) {
        // On a mismatch, fall back through shorter candidate borders until one extends or we
        // reach length 0. Each fall-back strictly decreases k, so the total work is O(m).
        while (k > 0 && pattern[i] != pattern[k])
            k = pi[k - 1];
        if (pattern[i] == pattern[k]) // the candidate prefix extends by one character
            ++k;
        pi[i] = k;
    }
    return pi;
}

std::vector<std::size_t> kmp_search(const std::string& text, const std::string& pattern) {
    std::vector<std::size_t> matches;
    const std::size_t n = text.size();
    const std::size_t m = pattern.size();
    if (m == 0 || m > n) // empty pattern: no matches by convention; longer than text: impossible
        return matches;

    const auto pi = kmp_prefix_function(pattern);
    // q = number of characters of pattern currently matched against the text suffix ending at i-1.
    std::size_t q = 0;
    for (std::size_t i = 0; i < n; ++i) {
        // Mismatch: slide the pattern by consulting the failure function, never rewinding text.
        while (q > 0 && text[i] != pattern[q])
            q = pi[q - 1];
        if (text[i] == pattern[q]) // next pattern character matches
            ++q;
        if (q == m) { // a full occurrence ends at i, so it starts at i - m + 1
            matches.push_back(i - m + 1);
            q = pi[q - 1]; // realign to allow overlapping matches
        }
    }
    return matches;
}

} // namespace datamunge::algorithms
