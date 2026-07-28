#include <gtest/gtest.h>

#include <datamunge/algorithms/string_search_extra.hpp>

#include <cstddef>
#include <random>
#include <string>
#include <vector>

using namespace datamunge::algorithms;

namespace {

// Naive all-occurrences reference.
std::vector<std::size_t> naive(const std::string& text, const std::string& pat) {
    std::vector<std::size_t> r;
    if (pat.empty() || pat.size() > text.size()) return r;
    for (std::size_t i = 0; i + pat.size() <= text.size(); ++i)
        if (text.compare(i, pat.size(), pat) == 0) r.push_back(i);
    return r;
}

std::string random_string(std::mt19937_64& rng, int len, int alphabet) {
    std::string s;
    for (int i = 0; i < len; ++i) s.push_back(static_cast<char>('a' + rng() % alphabet));
    return s;
}

} // namespace

TEST(StringSearchExtra, RandomAgainstNaive) {
    std::mt19937_64 rng(1);
    for (int alphabet : {2, 3, 4, 26}) {
        for (int t = 0; t < 4000; ++t) {
            const std::string text = random_string(rng, 1 + static_cast<int>(rng() % 200), alphabet);
            const std::string pat  = random_string(rng, 1 + static_cast<int>(rng() % 8), alphabet);
            const auto        want = naive(text, pat);
            EXPECT_EQ(boyer_moore_horspool_search(text, pat), want) << "horspool a=" << alphabet;
            EXPECT_EQ(zhu_takaoka_search(text, pat), want) << "zhu-takaoka a=" << alphabet;
        }
    }
}

TEST(StringSearchExtra, OverlappingMatches) {
    // "aaaa" in "aaaaaa" occurs at 0,1,2.
    const std::string text = "aaaaaa", pat = "aaaa";
    const std::vector<std::size_t> want = {0, 1, 2};
    EXPECT_EQ(boyer_moore_horspool_search(text, pat), want);
    EXPECT_EQ(zhu_takaoka_search(text, pat), want);
}

TEST(StringSearchExtra, KnownAndEdgeCases) {
    EXPECT_EQ(boyer_moore_horspool_search("abracadabra", "abra"), (std::vector<std::size_t>{0, 7}));
    EXPECT_EQ(zhu_takaoka_search("abracadabra", "abra"), (std::vector<std::size_t>{0, 7}));

    // Single-character pattern.
    EXPECT_EQ(boyer_moore_horspool_search("mississippi", "s"), (std::vector<std::size_t>{2, 3, 5, 6}));
    EXPECT_EQ(zhu_takaoka_search("mississippi", "s"), (std::vector<std::size_t>{2, 3, 5, 6}));

    // No match, empty pattern, and pattern longer than text.
    EXPECT_TRUE(boyer_moore_horspool_search("hello", "xyz").empty());
    EXPECT_TRUE(zhu_takaoka_search("hello", "").empty());
    EXPECT_TRUE(zhu_takaoka_search("hi", "hello").empty());

    // Whole-text match.
    EXPECT_EQ(zhu_takaoka_search("pattern", "pattern"), (std::vector<std::size_t>{0}));
    EXPECT_EQ(boyer_moore_horspool_search("pattern", "pattern"), (std::vector<std::size_t>{0}));
}
