#include <gtest/gtest.h>

#include <datamunge/algorithms/kmp.hpp>

#include <random>
#include <string>
#include <vector>

using datamunge::algorithms::kmp_prefix_function;
using datamunge::algorithms::kmp_search;

namespace {

// Naive O(n*m) reference: every start index where pattern occurs, via std::string::find.
std::vector<std::size_t> naive_search(const std::string& text, const std::string& pattern) {
    std::vector<std::size_t> out;
    if (pattern.empty()) return out;
    std::size_t pos = text.find(pattern, 0);
    while (pos != std::string::npos) {
        out.push_back(pos);
        pos = text.find(pattern, pos + 1); // +1 so overlapping matches are found
    }
    return out;
}

} // namespace

TEST(Kmp, PrefixFunctionClassicPattern) {
    // CLRS's canonical example: pattern "ababaca".
    const std::vector<std::size_t> expected = {0, 0, 1, 2, 3, 0, 1};
    EXPECT_EQ(kmp_prefix_function("ababaca"), expected);
}

TEST(Kmp, PrefixFunctionEdgeCases) {
    EXPECT_TRUE(kmp_prefix_function("").empty());
    EXPECT_EQ(kmp_prefix_function("a"), (std::vector<std::size_t>{0}));
    // All-equal string: pi[i] = i.
    EXPECT_EQ(kmp_prefix_function("aaaa"), (std::vector<std::size_t>{0, 1, 2, 3}));
    // No repeated structure: all zeros.
    EXPECT_EQ(kmp_prefix_function("abcde"), (std::vector<std::size_t>{0, 0, 0, 0, 0}));
}

TEST(Kmp, FindsOverlappingMatches) {
    EXPECT_EQ(kmp_search("aaaaa", "aaa"), (std::vector<std::size_t>{0, 1, 2}));
    EXPECT_EQ(kmp_search("ababababa", "aba"), (std::vector<std::size_t>{0, 2, 4, 6}));
}

TEST(Kmp, FindsNonOverlappingMatches) {
    EXPECT_EQ(kmp_search("ababacaababaca", "ababaca"), (std::vector<std::size_t>{0, 7}));
    EXPECT_EQ(kmp_search("hello world", "o"), (std::vector<std::size_t>{4, 7}));
}

TEST(Kmp, PatternNotPresentYieldsEmpty) {
    EXPECT_TRUE(kmp_search("abcabcabc", "abcd").empty());
    EXPECT_TRUE(kmp_search("the quick brown fox", "zzz").empty());
}

TEST(Kmp, PatternLongerThanTextYieldsEmpty) {
    EXPECT_TRUE(kmp_search("abc", "abcdef").empty());
    EXPECT_TRUE(kmp_search("", "a").empty());
}

TEST(Kmp, EmptyPatternYieldsEmptyByConvention) {
    EXPECT_TRUE(kmp_search("anything", "").empty());
    EXPECT_TRUE(kmp_search("", "").empty());
}

TEST(Kmp, WholeTextMatch) {
    EXPECT_EQ(kmp_search("abcabc", "abcabc"), (std::vector<std::size_t>{0}));
}

TEST(Kmp, RandomCrossCheckAgainstNaiveReference) {
    std::mt19937_64 rng(20240927);
    const std::string alphabet = "ab"; // tiny alphabet -> many matches and overlaps to stress KMP
    auto rand_string = [&](std::size_t len) {
        std::string s;
        s.reserve(len);
        for (std::size_t k = 0; k < len; ++k)
            s.push_back(alphabet[rng() % alphabet.size()]);
        return s;
    };
    for (int trial = 0; trial < 3000; ++trial) {
        const std::size_t n = rng() % 40;      // text length 0..39
        const std::size_t m = 1 + rng() % 6;   // pattern length 1..6
        const std::string text = rand_string(n);
        const std::string pattern = rand_string(m);
        EXPECT_EQ(kmp_search(text, pattern), naive_search(text, pattern))
            << "trial " << trial << " text='" << text << "' pattern='" << pattern << "'";
    }
}
