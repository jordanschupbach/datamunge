#include <datamunge/algorithms/string_metrics.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace datamunge::algorithms {

std::size_t levenshtein_distance(const std::string& a, const std::string& b) {
    // Work with the shorter string along the row so the two rolling rows use O(min(|a|,|b|)) space.
    const std::string& s = a.size() <= b.size() ? a : b; // indexes columns
    const std::string& t = a.size() <= b.size() ? b : a; // indexes rows
    const std::size_t  n = s.size();

    std::vector<std::size_t> prev(n + 1), curr(n + 1);
    for (std::size_t j = 0; j <= n; ++j) prev[j] = j; // D(0,j) = j
    for (std::size_t i = 1; i <= t.size(); ++i) {
        curr[0] = i; // D(i,0) = i
        for (std::size_t j = 1; j <= n; ++j) {
            const std::size_t cost = (t[i - 1] == s[j - 1]) ? 0 : 1;
            curr[j] = std::min({prev[j] + 1,        // deletion
                                curr[j - 1] + 1,    // insertion
                                prev[j - 1] + cost}); // match / substitution
        }
        std::swap(prev, curr);
    }
    return prev[n];
}

std::size_t damerau_levenshtein_distance(const std::string& a, const std::string& b) {
    const std::size_t n = a.size();
    const std::size_t m = b.size();
    // Lowrance-Wagner table with a one-cell border on the top and left holding a "infinity"
    // sentinel (n + m is an upper bound on any real distance, so it never wins a min).
    const std::size_t          inf = n + m;
    std::vector<std::vector<std::size_t>> d(n + 2, std::vector<std::size_t>(m + 2, 0));
    d[0][0] = inf;
    for (std::size_t i = 0; i <= n; ++i) {
        d[i + 1][0] = inf;
        d[i + 1][1] = i;
    }
    for (std::size_t j = 0; j <= m; ++j) {
        d[0][j + 1] = inf;
        d[1][j + 1] = j;
    }

    // da[c] = the largest row i (1-based) at which byte c has appeared in a so far.
    std::array<std::size_t, 256> da{};
    for (std::size_t i = 1; i <= n; ++i) {
        std::size_t db = 0; // largest column j at which a[0..i-1] last matched b
        for (std::size_t j = 1; j <= m; ++j) {
            const std::size_t i1   = da[static_cast<unsigned char>(b[j - 1])];
            const std::size_t j1   = db;
            std::size_t       cost = 1;
            if (a[i - 1] == b[j - 1]) {
                cost = 0;
                db   = j;
            }
            d[i + 1][j + 1] = std::min({
                d[i][j] + cost,                                // substitution / match
                d[i + 1][j] + 1,                               // insertion
                d[i][j + 1] + 1,                               // deletion
                d[i1][j1] + (i - i1 - 1) + 1 + (j - j1 - 1)});  // transposition of a block
        }
        da[static_cast<unsigned char>(a[i - 1])] = i;
    }
    return d[n + 1][m + 1];
}

std::size_t hamming_distance(const std::string& a, const std::string& b) {
    if (a.size() != b.size())
        throw std::invalid_argument("hamming_distance: strings must have equal length");
    std::size_t d = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) ++d;
    return d;
}

double jaro_similarity(const std::string& a, const std::string& b) {
    if (a.empty() && b.empty()) return 1.0;
    if (a.empty() || b.empty()) return 0.0;

    const std::size_t la = a.size();
    const std::size_t lb = b.size();
    // Matching window: characters count as matching only if within floor(max/2)-1 positions.
    std::size_t window = std::max(la, lb) / 2;
    if (window > 0) --window;

    std::vector<bool> a_match(la, false);
    std::vector<bool> b_match(lb, false);
    std::size_t       matches = 0;
    for (std::size_t i = 0; i < la; ++i) {
        const std::size_t lo = i > window ? i - window : 0;
        const std::size_t hi = std::min(i + window + 1, lb); // exclusive
        for (std::size_t j = lo; j < hi; ++j) {
            if (!b_match[j] && a[i] == b[j]) {
                a_match[i] = true;
                b_match[j] = true;
                ++matches;
                break;
            }
        }
    }
    if (matches == 0) return 0.0;

    // Count transpositions: walk the matched characters of both strings in order and tally the
    // positions where they disagree; the number of transpositions is half that tally.
    std::size_t t = 0;
    std::size_t k = 0;
    for (std::size_t i = 0; i < la; ++i) {
        if (!a_match[i]) continue;
        while (!b_match[k]) ++k;
        if (a[i] != b[k]) ++t;
        ++k;
    }
    const double m = static_cast<double>(matches);
    return (m / static_cast<double>(la) + m / static_cast<double>(lb) + (m - static_cast<double>(t) / 2.0) / m) / 3.0;
}

double jaro_winkler_similarity(const std::string& a, const std::string& b, double prefix_scale) {
    const double j = jaro_similarity(a, b);
    // Common prefix length, capped at 4.
    std::size_t       l    = 0;
    const std::size_t maxl = std::min<std::size_t>({a.size(), b.size(), 4});
    while (l < maxl && a[l] == b[l]) ++l;
    return j + static_cast<double>(l) * prefix_scale * (1.0 - j);
}

double dice_coefficient(const std::string& a, const std::string& b) {
    if (a == b) return 1.0;                     // identical (covers the both-empty and short cases)
    if (a.size() < 2 || b.size() < 2) return 0.0; // no bigrams and not equal -> nothing in common

    std::unordered_map<std::string, int> counts;
    counts.reserve(a.size());
    for (std::size_t i = 0; i + 1 < a.size(); ++i) ++counts[a.substr(i, 2)];

    std::size_t intersection = 0;
    for (std::size_t i = 0; i + 1 < b.size(); ++i) {
        auto it = counts.find(b.substr(i, 2));
        if (it != counts.end() && it->second > 0) {
            --it->second;
            ++intersection;
        }
    }
    const std::size_t total = (a.size() - 1) + (b.size() - 1);
    return 2.0 * static_cast<double>(intersection) / static_cast<double>(total);
}

} // namespace datamunge::algorithms
