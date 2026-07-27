#include <datamunge/algorithms/boyer_moore.hpp>

#include <algorithm>
#include <array>
#include <cstddef>

namespace datamunge::algorithms {

namespace {

// Bad-character table: last[c] is the rightmost index at which byte c occurs in the pattern, or
// -1 if c never occurs. Indexed by unsigned char over the full 256-entry byte range.
std::array<std::ptrdiff_t, 256> bad_character_table(const std::string& pattern) {
    std::array<std::ptrdiff_t, 256> last;
    last.fill(-1);
    for (std::size_t i = 0; i < pattern.size(); ++i)
        last[static_cast<unsigned char>(pattern[i])] = static_cast<std::ptrdiff_t>(i);
    return last;
}

// Strong good-suffix shift table. shift[k] is how far to advance the pattern when the suffix
// pattern[k..m-1] has already matched the text and either pattern[k-1] mismatched (k > 0) or the
// whole pattern matched (k == 0). Built by the classic two-pass border-position (bpos)
// construction (Gusfield): case 1 handles a re-occurrence of the matched suffix elsewhere in the
// pattern, case 2 handles the shorter suffix that is only a prefix of the pattern.
std::vector<std::ptrdiff_t> good_suffix_table(const std::string& pattern) {
    const std::ptrdiff_t m = static_cast<std::ptrdiff_t>(pattern.size());
    std::vector<std::ptrdiff_t> shift(static_cast<std::size_t>(m) + 1, 0);
    std::vector<std::ptrdiff_t> bpos(static_cast<std::size_t>(m) + 1, 0);

    // Case 1: the matched suffix occurs again, preceded by a different character.
    std::ptrdiff_t i = m, j = m + 1;
    bpos[static_cast<std::size_t>(i)] = j;
    while (i > 0) {
        while (j <= m && pattern[static_cast<std::size_t>(i - 1)] != pattern[static_cast<std::size_t>(j - 1)]) {
            if (shift[static_cast<std::size_t>(j)] == 0)
                shift[static_cast<std::size_t>(j)] = j - i;
            j = bpos[static_cast<std::size_t>(j)];
        }
        --i;
        --j;
        bpos[static_cast<std::size_t>(i)] = j;
    }

    // Case 2: only a prefix of the pattern matches the (shorter) matched suffix.
    j = bpos[0];
    for (i = 0; i <= m; ++i) {
        if (shift[static_cast<std::size_t>(i)] == 0)
            shift[static_cast<std::size_t>(i)] = j;
        if (i == j)
            j = bpos[static_cast<std::size_t>(j)];
    }
    return shift;
}

} // namespace

std::vector<std::size_t> boyer_moore_search(const std::string& text, const std::string& pattern) {
    std::vector<std::size_t> matches;
    const std::size_t n = text.size();
    const std::size_t m = pattern.size();
    if (m == 0 || m > n) // empty pattern: no matches by convention; longer than text: impossible
        return matches;

    const auto last = bad_character_table(pattern);
    const auto shift = good_suffix_table(pattern);

    std::size_t s = 0; // left end of the pattern's current alignment against the text
    while (s <= n - m) {
        // Scan the pattern RIGHT-TO-LEFT against the current alignment.
        std::size_t j = m;
        while (j > 0 && pattern[j - 1] == text[s + j - 1])
            --j;

        if (j == 0) { // every character matched: record the occurrence
            matches.push_back(s);
            s += static_cast<std::size_t>(shift[0]); // good-suffix shift for a full match (>= 1)
        } else {
            const std::size_t mi = j - 1; // pattern index of the mismatch
            // Bad-character rule: align the mismatching text byte with its rightmost pattern
            // occurrence (or shift the whole pattern past it). May be <= 0, hence the max below.
            const std::ptrdiff_t bad =
                static_cast<std::ptrdiff_t>(mi) - last[static_cast<unsigned char>(text[s + mi])];
            // Good-suffix rule: reuse the already-matched suffix pattern[mi+1..m-1] (>= 1).
            const std::ptrdiff_t good = shift[mi + 1];
            // Advance by the larger safe shift; the floor of 1 guarantees forward progress.
            s += static_cast<std::size_t>(std::max<std::ptrdiff_t>({std::ptrdiff_t{1}, bad, good}));
        }
    }
    return matches;
}

} // namespace datamunge::algorithms
