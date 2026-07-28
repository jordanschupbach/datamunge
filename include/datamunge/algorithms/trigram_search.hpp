#pragma once

// Trigram search: fuzzy string matching for when the exact spelling is unknown.
// A string is represented by the set of its 3-character substrings (trigrams);
// two strings are similar when their trigram sets overlap. The Jaccard overlap
// gives a similarity score robust to typos, insertions, and small edits.

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

// The set of trigrams of `s`: lowercased and padded with two leading and one
// trailing space so that the start and end of the string are captured.
inline std::set<std::string> trigrams(const std::string& s) {
    std::string t = "  ";
    for (char c : s) t.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    t.push_back(' ');
    std::set<std::string> out;
    for (std::size_t i = 0; i + 3 <= t.size(); ++i) out.insert(t.substr(i, 3));
    return out;
}

// Jaccard similarity of the two strings' trigram sets, in [0, 1].
inline double trigram_similarity(const std::string& a, const std::string& b) {
    const std::set<std::string> A = trigrams(a), B = trigrams(b);
    if (A.empty() && B.empty()) return 1.0;
    std::size_t inter = 0;
    for (const auto& g : A)
        if (B.count(g)) ++inter;
    const std::size_t uni = A.size() + B.size() - inter;
    return uni == 0 ? 0.0 : static_cast<double>(inter) / static_cast<double>(uni);
}

// Rank `candidates` by trigram similarity to `query`, best first. Ties keep the
// original order (stable).
inline std::vector<std::pair<double, std::size_t>> trigram_search(const std::vector<std::string>& candidates,
                                                                  const std::string&              query) {
    std::vector<std::pair<double, std::size_t>> scored;
    scored.reserve(candidates.size());
    for (std::size_t i = 0; i < candidates.size(); ++i)
        scored.emplace_back(trigram_similarity(candidates[i], query), i);
    std::stable_sort(scored.begin(), scored.end(),
                     [](const auto& x, const auto& y) { return x.first > y.first; });
    return scored;
}

} // namespace datamunge::algorithms
