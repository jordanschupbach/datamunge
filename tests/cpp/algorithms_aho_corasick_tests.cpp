#include <gtest/gtest.h>

#include <datamunge/algorithms/aho_corasick.hpp>

#include <algorithm>
#include <random>
#include <string>
#include <utility>
#include <vector>

using datamunge::algorithms::aho_corasick_search;
using datamunge::algorithms::AhoCorasickMatch;

namespace {

// A comparable, order-independent view of a match set: the sorted multiset of (position, pattern).
std::vector<std::pair<std::size_t, std::size_t>> as_pairs(const std::vector<AhoCorasickMatch>& ms) {
    std::vector<std::pair<std::size_t, std::size_t>> out;
    out.reserve(ms.size());
    for (const auto& m : ms) out.emplace_back(m.position, m.pattern_index);
    std::sort(out.begin(), out.end());
    return out;
}

// Naive O(|text| * #patterns) reference: run each pattern's own std::string::find independently and
// collect the (start position, pattern index) of every occurrence, overlaps included.
std::vector<std::pair<std::size_t, std::size_t>> naive_reference(const std::string&              text,
                                                                 const std::vector<std::string>& patterns) {
    std::vector<std::pair<std::size_t, std::size_t>> out;
    for (std::size_t pi = 0; pi < patterns.size(); ++pi) {
        const std::string& p = patterns[pi];
        if (p.empty()) continue; // empty patterns match nowhere, by convention
        std::size_t pos = text.find(p, 0);
        while (pos != std::string::npos) {
            out.emplace_back(pos, pi);
            pos = text.find(p, pos + 1); // +1 so overlapping occurrences are all found
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

} // namespace

TEST(AhoCorasick, ClassicUshersExample) {
    // The canonical Aho-Corasick example. Expected occurrences (start position @ text):
    //   she @ 1 (pattern 1), he @ 2 (pattern 0), hers @ 2 (pattern 3).
    const std::vector<std::string> patterns = {"he", "she", "his", "hers"};
    const auto matches = aho_corasick_search("ushers", patterns);

    const std::vector<std::pair<std::size_t, std::size_t>> expected = {
        {1, 1}, // she
        {2, 0}, // he
        {2, 3}, // hers
    };
    EXPECT_EQ(as_pairs(matches), expected);

    // "his" (pattern 2) never occurs in "ushers".
    for (const auto& m : matches) EXPECT_NE(m.pattern_index, 2u);
}

TEST(AhoCorasick, EmissionOrderIsByEndPosition) {
    // The documented ordering: matches come out in order of increasing END position, and within a
    // shared end position from the longest match down its suffix chain. Both "she" and "he" end at
    // text index 3; "she" (longer) is emitted before "he".
    const std::vector<std::string> patterns = {"he", "she", "his", "hers"};
    const auto matches = aho_corasick_search("ushers", patterns);
    ASSERT_EQ(matches.size(), 3u);
    EXPECT_EQ(matches[0].position, 1u);
    EXPECT_EQ(matches[0].pattern_index, 1u); // she, ends at index 3
    EXPECT_EQ(matches[1].position, 2u);
    EXPECT_EQ(matches[1].pattern_index, 0u); // he, ends at index 3
    EXPECT_EQ(matches[2].position, 2u);
    EXPECT_EQ(matches[2].pattern_index, 3u); // hers, ends at index 5
}

TEST(AhoCorasick, PatternThatIsSuffixOfAnotherIsAlsoReported) {
    // "a" is a suffix of "ba"; both must be reported. In "aba": a@0, ba@1, a@2.
    const std::vector<std::string> patterns = {"a", "ba"};
    const std::vector<std::pair<std::size_t, std::size_t>> expected = {
        {0, 0}, // a
        {1, 1}, // ba
        {2, 0}, // a
    };
    EXPECT_EQ(as_pairs(aho_corasick_search("aba", patterns)), expected);
}

TEST(AhoCorasick, OverlappingMatches) {
    // Overlapping occurrences of a single pattern are all reported.
    const std::vector<std::string> patterns = {"aa", "aaa"};
    const std::vector<std::pair<std::size_t, std::size_t>> expected = {
        {0, 0}, // aa @ 0
        {0, 1}, // aaa @ 0
        {1, 0}, // aa @ 1
        {1, 1}, // aaa @ 1
        {2, 0}, // aa @ 2
    };
    EXPECT_EQ(as_pairs(aho_corasick_search("aaaa", patterns)), expected);
}

TEST(AhoCorasick, PatternNotPresentYieldsNothingForThatPattern) {
    const std::vector<std::string> patterns = {"cat", "dog", "bird"};
    const auto matches = aho_corasick_search("the cat sat on the dog", patterns);
    // "bird" (index 2) is absent; only cat@4 and dog@19 occur.
    const std::vector<std::pair<std::size_t, std::size_t>> expected = {{4, 0}, {19, 1}};
    EXPECT_EQ(as_pairs(matches), expected);
}

TEST(AhoCorasick, DuplicatePatternsEachReportedUnderTheirOwnIndex) {
    // Two identical patterns at different indices: each occurrence is reported once per index.
    const std::vector<std::string> patterns = {"ab", "ab"};
    const auto matches = aho_corasick_search("abab", patterns);
    const std::vector<std::pair<std::size_t, std::size_t>> expected = {
        {0, 0}, {0, 1}, {2, 0}, {2, 1},
    };
    EXPECT_EQ(as_pairs(matches), expected);
}

TEST(AhoCorasick, EmptyPatternsAndEmptyInputsAreHandled) {
    EXPECT_TRUE(aho_corasick_search("abc", {}).empty());              // no patterns
    EXPECT_TRUE(aho_corasick_search("", {"a", "b"}).empty());         // empty text
    EXPECT_TRUE(aho_corasick_search("abc", {"", ""}).empty());        // empty patterns skipped
    // An empty pattern mixed with a real one: only the real one matches.
    const auto matches = aho_corasick_search("abc", {"", "b"});
    ASSERT_EQ(matches.size(), 1u);
    EXPECT_EQ(matches[0].position, 1u);
    EXPECT_EQ(matches[0].pattern_index, 1u);
}

TEST(AhoCorasick, PatternLongerThanTextYieldsNothing) {
    EXPECT_TRUE(aho_corasick_search("ab", {"abcdef"}).empty());
}

TEST(AhoCorasick, RandomCrossCheckAgainstNaiveReference) {
    std::mt19937_64 rng(20240927);
    const std::string alphabet = "abc"; // small alphabet -> many matches, overlaps, shared suffixes
    auto rand_string = [&](std::size_t len) {
        std::string s;
        s.reserve(len);
        for (std::size_t k = 0; k < len; ++k) s.push_back(alphabet[rng() % alphabet.size()]);
        return s;
    };
    for (int trial = 0; trial < 3000; ++trial) {
        const std::size_t n = rng() % 50; // text length 0..49
        const std::string text = rand_string(n);

        const std::size_t              num_patterns = 1 + rng() % 6; // 1..6 patterns
        std::vector<std::string>       patterns;
        patterns.reserve(num_patterns);
        for (std::size_t k = 0; k < num_patterns; ++k) patterns.push_back(rand_string(1 + rng() % 4));

        EXPECT_EQ(as_pairs(aho_corasick_search(text, patterns)), naive_reference(text, patterns))
            << "trial " << trial << " text='" << text << "'";
    }
}
