#include <datamunge/algorithms/bitap.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace datamunge::algorithms {

namespace {
constexpr std::size_t ALPHABET = 256;
} // namespace

long bitap_search(const std::string& text, const std::string& pattern) {
    const std::size_t m = pattern.size();
    if (m == 0) return 0;
    if (m > 63) return -1; // does not fit a 64-bit state word

    // mask[c] has bit j set where pattern[j] == c (shift-and convention: 1 = match).
    std::uint64_t mask[ALPHABET] = {0};
    for (std::size_t j = 0; j < m; ++j)
        mask[static_cast<unsigned char>(pattern[j])] |= (std::uint64_t{1} << j);

    const std::uint64_t accept = std::uint64_t{1} << (m - 1);
    std::uint64_t       r      = 0; // bit j set => pattern[0..j] matches text ending here
    for (std::size_t i = 0; i < text.size(); ++i) {
        r = ((r << 1) | 1ULL) & mask[static_cast<unsigned char>(text[i])];
        if (r & accept) return static_cast<long>(i - m + 1);
    }
    return -1;
}

long bitap_fuzzy_search(const std::string& text, const std::string& pattern, int max_errors) {
    const std::size_t m = pattern.size();
    if (m == 0) return 0;
    if (m > 62) return -1;
    if (max_errors < 0) max_errors = 0;
    const std::size_t k = static_cast<std::size_t>(max_errors);

    // shift-OR convention here (0 = match). pattern_mask[c] has bit j = 0 where pattern[j] == c.
    std::uint64_t pattern_mask[ALPHABET];
    for (std::size_t c = 0; c < ALPHABET; ++c) pattern_mask[c] = ~std::uint64_t{0};
    for (std::size_t j = 0; j < m; ++j)
        pattern_mask[static_cast<unsigned char>(pattern[j])] &= ~(std::uint64_t{1} << j);

    // R[d] tracks matches with exactly up to d errors; bit m clear in R[k] => a match ended here.
    std::vector<std::uint64_t> R(k + 1);
    for (std::size_t d = 0; d <= k; ++d) R[d] = ~std::uint64_t{1};

    const std::uint64_t accept = std::uint64_t{1} << m;
    for (std::size_t i = 0; i < text.size(); ++i) {
        const std::uint64_t cmask   = pattern_mask[static_cast<unsigned char>(text[i])];
        std::uint64_t       old_rd1 = R[0]; // R[d-1] at the previous text position
        R[0] |= cmask;
        R[0] <<= 1;
        for (std::size_t d = 1; d <= k; ++d) {
            const std::uint64_t tmp = R[d];
            R[d]    = (old_rd1 & (R[d] | cmask)) << 1;
            old_rd1 = tmp;
        }
        if ((R[k] & accept) == 0) return static_cast<long>(i);
    }
    return -1;
}

} // namespace datamunge::algorithms
