#include <gtest/gtest.h>

#include <datamunge/algorithms/z_algorithm.hpp>

#include <algorithm>
#include <random>
#include <string>
#include <vector>

using datamunge::algorithms::z_array;
using datamunge::algorithms::z_search;

namespace {

// Naive O(n^2) reference: at each i, compare s[0..] with s[i..] character by
// character. Uses the same z[0] = n convention as z_array.
std::vector<std::size_t> naive_z(const std::string& s) {
    const std::size_t n = s.size();
    std::vector<std::size_t> z(n, 0);
    if (n == 0) return z;
    z[0] = n;
    for (std::size_t i = 1; i < n; ++i) {
        std::size_t k = 0;
        while (i + k < n && s[k] == s[i + k]) ++k;
        z[i] = k;
    }
    return z;
}

// Naive reference search using a std::string::find loop; advancing by one keeps
// overlapping matches. Empty pattern matches at every position 0..size inclusive.
std::vector<std::size_t> naive_search(const std::string& text, const std::string& pattern) {
    std::vector<std::size_t> out;
    if (pattern.empty()) {
        for (std::size_t i = 0; i <= text.size(); ++i) out.push_back(i);
        return out;
    }
    std::size_t pos = text.find(pattern, 0);
    while (pos != std::string::npos) {
        out.push_back(pos);
        pos = text.find(pattern, pos + 1);
    }
    return out;
}

std::string random_string(std::size_t len, std::size_t alphabet, std::mt19937_64& rng) {
    std::string s;
    s.reserve(len);
    for (std::size_t i = 0; i < len; ++i)
        s.push_back(static_cast<char>('a' + static_cast<char>(rng() % alphabet)));
    return s;
}

} // namespace

TEST(ZAlgorithm, ZArrayOfAllEqualCharacters) {
    // Convention here: z[0] = n. "aaaaa" -> {5, 4, 3, 2, 1}.
    EXPECT_EQ(z_array("aaaaa"), (std::vector<std::size_t>{5, 4, 3, 2, 1}));
}

TEST(ZAlgorithm, ZArrayClassicStrings) {
    // Self-similar string with prefix matches of length 6 and 3 nested inside.
    EXPECT_EQ(z_array("aabaabaab"), (std::vector<std::size_t>{9, 1, 0, 6, 1, 0, 3, 1, 0}));
    // The textbook "abacaba".
    EXPECT_EQ(z_array("abacaba"), (std::vector<std::size_t>{7, 0, 1, 0, 3, 0, 1}));
}

TEST(ZAlgorithm, ZArrayEdgeCases) {
    EXPECT_EQ(z_array(""), (std::vector<std::size_t>{}));
    EXPECT_EQ(z_array("a"), (std::vector<std::size_t>{1}));
    // No character repeats the prefix, so every interior value is 0.
    EXPECT_EQ(z_array("abcde"), (std::vector<std::size_t>{5, 0, 0, 0, 0}));
}

TEST(ZAlgorithm, SearchFindsOverlappingMatches) {
    EXPECT_EQ(z_search("abababa", "aba"), (std::vector<std::size_t>{0, 2, 4}));
    EXPECT_EQ(z_search("aaaaa", "aa"), (std::vector<std::size_t>{0, 1, 2, 3}));
    EXPECT_EQ(z_search("aaaaa", "aaa"), (std::vector<std::size_t>{0, 1, 2}));
}

TEST(ZAlgorithm, SearchBoundaryMatches) {
    EXPECT_EQ(z_search("abcabc", "abc"), (std::vector<std::size_t>{0, 3})); // start and end
    EXPECT_EQ(z_search("abc", "abc"), (std::vector<std::size_t>{0}));       // whole text
    EXPECT_EQ(z_search("xxxxxab", "ab"), (std::vector<std::size_t>{5}));    // at the very end
}

TEST(ZAlgorithm, SearchNotPresentIsEmpty) {
    EXPECT_TRUE(z_search("ababab", "abc").empty());
    EXPECT_TRUE(z_search("abc", "abcd").empty()); // pattern longer than text
    EXPECT_TRUE(z_search("", "a").empty());
}

TEST(ZAlgorithm, EmptyPatternMatchesEverywhere) {
    EXPECT_EQ(z_search("ab", ""), (std::vector<std::size_t>{0, 1, 2}));
    EXPECT_EQ(z_search("", ""), (std::vector<std::size_t>{0}));
}

TEST(ZAlgorithm, SearchMatchesNaiveOverManyRandomTrials) {
    std::mt19937_64 rng(20240927);
    for (int trial = 0; trial < 3000; ++trial) {
        const std::size_t alphabet = 2 + rng() % 2; // 2 or 3 symbols -> frequent overlaps
        const std::string text = random_string(rng() % 30, alphabet, rng);
        // Half the time draw the pattern from a real substring so matches are common.
        std::string pattern;
        if (!text.empty() && (rng() & 1)) {
            const std::size_t start = rng() % text.size();
            const std::size_t len = 1 + rng() % (text.size() - start);
            pattern = text.substr(start, len);
        } else {
            pattern = random_string(rng() % 5, alphabet, rng);
        }
        EXPECT_EQ(z_search(text, pattern), naive_search(text, pattern))
            << "trial " << trial << " text=\"" << text << "\" pattern=\"" << pattern << "\"";
    }
}

TEST(ZAlgorithm, ZArrayMatchesNaiveOverManyRandomTrials) {
    std::mt19937_64 rng(11);
    for (int trial = 0; trial < 3000; ++trial) {
        const std::size_t alphabet = 1 + rng() % 3; // 1..3 symbols
        const std::string s = random_string(rng() % 40, alphabet, rng);
        EXPECT_EQ(z_array(s), naive_z(s)) << "trial " << trial << " s=\"" << s << "\"";
    }
}
