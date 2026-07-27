#include <gtest/gtest.h>

#include <datamunge/algorithms/boyer_moore.hpp>

#include <cstddef>
#include <random>
#include <string>
#include <vector>

using datamunge::algorithms::boyer_moore_search;

namespace {

// Independent reference: every occurrence via a std::string::find loop. This is the correctness
// oracle the randomized tests below cross-check the good-suffix table against.
std::vector<std::size_t> naive_search(const std::string& text, const std::string& pattern) {
    std::vector<std::size_t> res;
    if (pattern.empty()) // mirror boyer_moore_search's convention: empty pattern -> no matches
        return res;
    std::size_t pos = text.find(pattern, 0);
    while (pos != std::string::npos) {
        res.push_back(pos);
        pos = text.find(pattern, pos + 1); // pos + 1 to catch overlapping occurrences
    }
    return res;
}

std::string random_string(std::size_t len, unsigned alphabet, std::mt19937_64& rng) {
    std::string s;
    s.reserve(len);
    for (std::size_t i = 0; i < len; ++i)
        s.push_back(static_cast<char>('a' + static_cast<int>(rng() % alphabet)));
    return s;
}

} // namespace

TEST(BoyerMoore, FindsAllOverlappingMatches) {
    EXPECT_EQ(boyer_moore_search("aaaa", "aa"), (std::vector<std::size_t>{0, 1, 2}));
    EXPECT_EQ(boyer_moore_search("aaaaa", "aaa"), (std::vector<std::size_t>{0, 1, 2}));
    EXPECT_EQ(boyer_moore_search("ababa", "aba"), (std::vector<std::size_t>{0, 2}));
}

TEST(BoyerMoore, KnownSearches) {
    // Classic bad-character example: three occurrences, two of them adjacent.
    EXPECT_EQ(boyer_moore_search("AABAACAADAABAABA", "AABA"), (std::vector<std::size_t>{0, 9, 12}));
    EXPECT_EQ(boyer_moore_search("abracadabra", "abra"), (std::vector<std::size_t>{0, 7}));
    // Gusfield's good-suffix example: exactly one occurrence at index 5.
    EXPECT_EQ(boyer_moore_search("GCATCGCAGAGAGTATACAGTACG", "GCAGAGAG"), (std::vector<std::size_t>{5}));
}

TEST(BoyerMoore, SingleCharacterPattern) {
    EXPECT_EQ(boyer_moore_search("banana", "a"), (std::vector<std::size_t>{1, 3, 5}));
    EXPECT_EQ(boyer_moore_search("banana", "n"), (std::vector<std::size_t>{2, 4}));
    EXPECT_TRUE(boyer_moore_search("banana", "z").empty());
}

TEST(BoyerMoore, NotPresentReturnsEmpty) {
    EXPECT_TRUE(boyer_moore_search("hello world", "xyz").empty());
    EXPECT_TRUE(boyer_moore_search("abcabcabc", "abcd").empty());
    EXPECT_TRUE(boyer_moore_search("aaaaaaaaaa", "aaaab").empty());
}

TEST(BoyerMoore, NeedleInAHaystack) {
    std::string haystack = std::string(1000, 'a') + "needle" + std::string(1000, 'a');
    const auto r = boyer_moore_search(haystack, "needle");
    ASSERT_EQ(r.size(), 1u);
    EXPECT_EQ(r[0], 1000u);
}

TEST(BoyerMoore, WorstCaseLikePatterns) {
    // The textbook worst case for right-to-left scanning: pattern a^(m-1)b over a run of a's.
    const std::string text(200, 'a');
    EXPECT_TRUE(boyer_moore_search(text, "aaaaab").empty());
    const std::string t2 = std::string(30, 'a') + "aaaaab" + std::string(30, 'a') + "aaaaab";
    EXPECT_EQ(boyer_moore_search(t2, "aaaaab"), naive_search(t2, "aaaaab"));
}

TEST(BoyerMoore, RepeatedCharactersExerciseGoodSuffix) {
    // Periodic pattern and text: the good-suffix rule (not bad-character) drives the shifts here.
    const std::string text = "abababababcababababab";
    const std::string pat = "ababab";
    EXPECT_EQ(boyer_moore_search(text, pat), naive_search(text, pat));
    EXPECT_EQ(boyer_moore_search("aabaabaabaab", "aabaab"), naive_search("aabaabaabaab", "aabaab"));
}

TEST(BoyerMoore, EmptyPatternAndOversizedPattern) {
    EXPECT_TRUE(boyer_moore_search("abc", "").empty()); // empty pattern -> no matches by convention
    EXPECT_TRUE(boyer_moore_search("", "").empty());
    EXPECT_TRUE(boyer_moore_search("ab", "abc").empty()); // pattern longer than text
    EXPECT_TRUE(boyer_moore_search("", "a").empty());
}

TEST(BoyerMoore, RandomCrossCheckSmallAlphabet) {
    std::mt19937_64 rng(12345);
    for (int trial = 0; trial < 2000; ++trial) {
        const unsigned alphabet = 2u + static_cast<unsigned>(rng() % 3u); // 2..4 symbols so matches occur
        const std::size_t tlen = 1 + rng() % 40;
        const std::size_t plen = 1 + rng() % 6;
        const std::string text = random_string(tlen, alphabet, rng);
        const std::string pat = random_string(plen, alphabet, rng);
        ASSERT_EQ(boyer_moore_search(text, pat), naive_search(text, pat))
            << "trial " << trial << " text=" << text << " pat=" << pat;
    }
}

TEST(BoyerMoore, RandomCrossCheckBinaryAlphabet) {
    // A two-symbol alphabet maximizes periodicity, the hardest case for the good-suffix table.
    std::mt19937_64 rng(2024987);
    for (int trial = 0; trial < 2000; ++trial) {
        const std::size_t tlen = 1 + rng() % 60;
        const std::size_t plen = 1 + rng() % 8;
        const std::string text = random_string(tlen, 2u, rng);
        const std::string pat = random_string(plen, 2u, rng);
        ASSERT_EQ(boyer_moore_search(text, pat), naive_search(text, pat))
            << "trial " << trial << " text=" << text << " pat=" << pat;
    }
}
