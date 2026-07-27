#include <datamunge/algorithms/rabin_karp.hpp>

#include <cstdint>

namespace datamunge::algorithms {

std::vector<std::size_t> rabin_karp_search(const std::string& text, const std::string& pattern) {
    std::vector<std::size_t> matches;
    const std::size_t n = text.size();
    const std::size_t m = pattern.size();

    // Convention: the empty pattern occurs at every position, {0, 1, ..., n} (n + 1 empty
    // occurrences), mirroring std::string::find("").
    if (m == 0) {
        matches.resize(n + 1);
        for (std::size_t i = 0; i <= n; ++i) matches[i] = i;
        return matches;
    }
    if (m > n) return matches; // pattern longer than text: no occurrence is possible

    constexpr std::uint64_t kBase = 256;           // one "digit" per byte of the alphabet
    constexpr std::uint64_t kMod  = 1000000007ULL; // a large prime keeps spurious collisions rare

    // high = kBase^(m-1) mod kMod: the positional weight of the window's leading byte, i.e. the
    // contribution that must be removed when that byte slides out of the window.
    std::uint64_t high = 1;
    for (std::size_t i = 1; i < m; ++i) high = (high * kBase) % kMod;

    // Horner-evaluate the pattern's hash and the hash of the first window of the text.
    std::uint64_t pattern_hash = 0;
    std::uint64_t window_hash  = 0;
    for (std::size_t i = 0; i < m; ++i) {
        pattern_hash = (pattern_hash * kBase + static_cast<unsigned char>(pattern[i])) % kMod;
        window_hash  = (window_hash * kBase + static_cast<unsigned char>(text[i])) % kMod;
    }

    for (std::size_t start = 0;; ++start) {
        // A hash match only *suggests* equality; comparing the characters rules out a spurious
        // collision, so the recorded positions are always exact.
        if (window_hash == pattern_hash && text.compare(start, m, pattern) == 0)
            matches.push_back(start);

        if (start + m == n) break; // the last window has been processed

        // Slide the window one byte right in O(1): drop the leading byte's weighted contribution,
        // shift everything up by one base position, then fold in the incoming byte.
        const std::uint64_t leading  = static_cast<unsigned char>(text[start]);
        const std::uint64_t incoming = static_cast<unsigned char>(text[start + m]);
        window_hash = (window_hash + kMod - (leading * high) % kMod) % kMod;
        window_hash = (window_hash * kBase + incoming) % kMod;
    }
    return matches;
}

} // namespace datamunge::algorithms
