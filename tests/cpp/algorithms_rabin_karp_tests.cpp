#include <gtest/gtest.h>

#include <datamunge/algorithms/rabin_karp.hpp>

#include <cstddef>
#include <random>
#include <string>
#include <vector>

using datamunge::algorithms::rabin_karp_search;

namespace {

// Reference matcher: a straightforward std::string::find loop. It reports every start index
// (including overlapping ones, since we advance by a single character) and, by our convention,
// treats the empty pattern as occurring at every position.
std::vector<std::size_t> naive_search(const std::string& text, const std::string& pattern) {
    std::vector<std::size_t> out;
    if (pattern.empty()) {
        for (std::size_t i = 0; i <= text.size(); ++i) out.push_back(i);
        return out;
    }
    for (std::size_t i = text.find(pattern); i != std::string::npos; i = text.find(pattern, i + 1))
        out.push_back(i);
    return out;
}

} // namespace

TEST(RabinKarp, FindsOverlappingOccurrences) {
    // "aba" occurs at 0 and 2 in "abababbb"; the trailing "bbb" is a decoy that collides under a
    // small modulus but never matches the actual characters.
    EXPECT_EQ(rabin_karp_search("abababbb", "aba"), (std::vector<std::size_t>{0, 2}));
    // "aa" overlaps itself all across a run of a's.
    EXPECT_EQ(rabin_karp_search("aaaa", "aa"), (std::vector<std::size_t>{0, 1, 2}));
    EXPECT_EQ(rabin_karp_search("aaaaa", "aaa"), (std::vector<std::size_t>{0, 1, 2}));
}

TEST(RabinKarp, ReturnsEmptyWhenAbsent) {
    EXPECT_TRUE(rabin_karp_search("hello world", "xyz").empty());
    EXPECT_TRUE(rabin_karp_search("abcabcabc", "abcd").empty());
    // Pattern longer than the text.
    EXPECT_TRUE(rabin_karp_search("ab", "abc").empty());
}

TEST(RabinKarp, FindsMultipleSeparatedOccurrences) {
    EXPECT_EQ(rabin_karp_search("the cat sat on the mat", "at"),
              (std::vector<std::size_t>{5, 9, 20}));
    EXPECT_EQ(rabin_karp_search("abcabcabc", "abc"), (std::vector<std::size_t>{0, 3, 6}));
}

TEST(RabinKarp, WholeStringAndSingleCharacter) {
    EXPECT_EQ(rabin_karp_search("datamunge", "datamunge"), (std::vector<std::size_t>{0}));
    EXPECT_EQ(rabin_karp_search("mississippi", "s"), (std::vector<std::size_t>{2, 3, 5, 6}));
    EXPECT_TRUE(rabin_karp_search("", "a").empty());
}

TEST(RabinKarp, EmptyPatternMatchesEveryPosition) {
    // Convention documented on rabin_karp_search: the empty pattern occurs at {0..n}.
    EXPECT_EQ(rabin_karp_search("abc", ""), (std::vector<std::size_t>{0, 1, 2, 3}));
    EXPECT_EQ(rabin_karp_search("", ""), (std::vector<std::size_t>{0}));
}

TEST(RabinKarp, VerificationDefeatsHashCollisions) {
    // Every reported position must genuinely spell out the pattern -- if character verification
    // were skipped, a spurious hash collision could sneak a wrong index in here.
    std::mt19937_64 rng(1234567);
    for (int trial = 0; trial < 400; ++trial) {
        const std::size_t n = 1 + rng() % 60;
        const std::size_t m = 1 + rng() % 6;
        std::string text(n, 'a'), pattern(m, 'a');
        for (auto& c : text) c = static_cast<char>('a' + rng() % 3); // tiny alphabet -> collisions likely
        for (auto& c : pattern) c = static_cast<char>('a' + rng() % 3);
        for (const std::size_t pos : rabin_karp_search(text, pattern)) {
            ASSERT_LE(pos + m, n + 1u) << "trial " << trial; // within bounds
            EXPECT_EQ(text.compare(pos, m, pattern), 0) << "trial " << trial << " pos " << pos;
        }
    }
}

TEST(RabinKarp, MatchesNaiveReferenceOverManyRandomTrials) {
    // Small alphabet + many trials makes hash collisions common; the results must still agree
    // exactly with the naive std::string::find reference.
    std::mt19937_64 rng(987654321);
    for (int trial = 0; trial < 5000; ++trial) {
        const std::size_t n = rng() % 40;
        const std::size_t m = 1 + rng() % 5;
        std::string text(n, 'a'), pattern(m, 'a');
        for (auto& c : text) c = static_cast<char>('a' + rng() % 2);    // alphabet {a, b}
        for (auto& c : pattern) c = static_cast<char>('a' + rng() % 2);
        ASSERT_EQ(rabin_karp_search(text, pattern), naive_search(text, pattern))
            << "trial " << trial << " text=\"" << text << "\" pattern=\"" << pattern << "\"";
    }
}
