#include <datamunge/algorithms/string_search_extra.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace datamunge::algorithms {

namespace {

constexpr int kAlphabet = 256;

inline unsigned char uc(char c) { return static_cast<unsigned char>(c); }

// Boyer-Moore good-suffix (bmGs) preprocessing via the suffix/border table.
void compute_good_suffix(const std::string& x, std::vector<int>& bmGs) {
    const int        m = static_cast<int>(x.size());
    std::vector<int> suff(m);
    suff[m - 1] = m;
    int f = 0, g = m - 1;
    for (int i = m - 2; i >= 0; --i) {
        if (i > g && suff[i + m - 1 - f] < i - g) {
            suff[i] = suff[i + m - 1 - f];
        } else {
            if (i < g) g = i;
            f = i;
            while (g >= 0 && x[g] == x[g + m - 1 - f]) --g;
            suff[i] = f - g;
        }
    }
    bmGs.assign(m, m);
    int j = 0;
    for (int i = m - 1; i >= 0; --i)
        if (suff[i] == i + 1)
            for (; j < m - 1 - i; ++j)
                if (bmGs[j] == m) bmGs[j] = m - 1 - i;
    for (int i = 0; i <= m - 2; ++i) bmGs[m - 1 - suff[i]] = m - 1 - i;
}

} // namespace

std::vector<std::size_t> boyer_moore_horspool_search(const std::string& text, const std::string& pattern) {
    std::vector<std::size_t> result;
    const std::size_t        n = text.size(), m = pattern.size();
    if (m == 0 || m > n) return result;

    // Bad-character shift: distance to the last-but-one occurrence of each char.
    std::array<std::size_t, kAlphabet> shift;
    shift.fill(m);
    for (std::size_t i = 0; i + 1 < m; ++i) shift[uc(pattern[i])] = m - 1 - i;

    std::size_t s = 0;
    while (s <= n - m) {
        std::size_t j = m - 1;
        while (j != static_cast<std::size_t>(-1) && text[s + j] == pattern[j]) --j;
        if (j == static_cast<std::size_t>(-1)) result.push_back(s);
        s += shift[uc(text[s + m - 1])];
    }
    return result;
}

std::vector<std::size_t> zhu_takaoka_search(const std::string& text, const std::string& pattern) {
    std::vector<std::size_t> result;
    const int                n = static_cast<int>(text.size()), m = static_cast<int>(pattern.size());
    if (m == 0 || m > n) return result;
    if (m == 1) { // digram table is undefined for length 1: scan directly
        for (int i = 0; i < n; ++i)
            if (text[i] == pattern[0]) result.push_back(static_cast<std::size_t>(i));
        return result;
    }

    // Two-character bad-character shift ztBc[a][b], indexed by the digram at the
    // end of the current window.
    std::vector<int> ztBc(kAlphabet * kAlphabet, m);
    auto             at = [](int a, int b) { return a * kAlphabet + b; };
    for (int a = 0; a < kAlphabet; ++a) ztBc[at(a, uc(pattern[0]))] = m - 1;
    for (int i = 1; i < m - 1; ++i) ztBc[at(uc(pattern[i - 1]), uc(pattern[i]))] = m - 1 - i;

    std::vector<int> bmGs;
    compute_good_suffix(pattern, bmGs);

    int j = 0;
    while (j <= n - m) {
        int i = m - 1;
        while (i >= 0 && pattern[i] == text[i + j]) --i;
        if (i < 0) {
            result.push_back(static_cast<std::size_t>(j));
            j += bmGs[0];
        } else {
            j += std::max(bmGs[i], ztBc[at(uc(text[j + m - 2]), uc(text[j + m - 1]))]);
        }
    }
    return result;
}

} // namespace datamunge::algorithms
