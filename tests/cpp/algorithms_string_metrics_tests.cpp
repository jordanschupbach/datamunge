#include <gtest/gtest.h>

#include <datamunge/algorithms/string_metrics.hpp>

#include <algorithm>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using datamunge::algorithms::damerau_levenshtein_distance;
using datamunge::algorithms::dice_coefficient;
using datamunge::algorithms::hamming_distance;
using datamunge::algorithms::jaro_similarity;
using datamunge::algorithms::jaro_winkler_similarity;
using datamunge::algorithms::levenshtein_distance;

namespace {

// Straightforward full-matrix Wagner-Fischer reference (no space optimisation) to cross-check the
// production two-row implementation.
std::size_t naive_levenshtein(const std::string& a, const std::string& b) {
    const std::size_t                     n = a.size();
    const std::size_t                     m = b.size();
    std::vector<std::vector<std::size_t>> d(n + 1, std::vector<std::size_t>(m + 1, 0));
    for (std::size_t i = 0; i <= n; ++i) d[i][0] = i;
    for (std::size_t j = 0; j <= m; ++j) d[0][j] = j;
    for (std::size_t i = 1; i <= n; ++i)
        for (std::size_t j = 1; j <= m; ++j) {
            const std::size_t cost = (a[i - 1] == b[j - 1]) ? 0 : 1;
            d[i][j] = std::min({d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost});
        }
    return d[n][m];
}

std::string random_string(std::mt19937& rng, std::size_t max_len, char alphabet_size) {
    std::uniform_int_distribution<std::size_t> len_dist(0, max_len);
    std::uniform_int_distribution<int>         chr_dist(0, alphabet_size - 1);
    std::string                                s(len_dist(rng), 'a');
    for (char& c : s) c = static_cast<char>('a' + chr_dist(rng));
    return s;
}

} // namespace

// ------------------------------------------------------------------------------------------------
// Levenshtein
// ------------------------------------------------------------------------------------------------

TEST(Levenshtein, KnownValues) {
    EXPECT_EQ(levenshtein_distance("kitten", "sitting"), 3u);  // canonical textbook example
    EXPECT_EQ(levenshtein_distance("flaw", "lawn"), 2u);
    EXPECT_EQ(levenshtein_distance("saturday", "sunday"), 3u);
    EXPECT_EQ(levenshtein_distance("", ""), 0u);
    EXPECT_EQ(levenshtein_distance("abc", ""), 3u);
    EXPECT_EQ(levenshtein_distance("", "abc"), 3u);
    EXPECT_EQ(levenshtein_distance("abc", "abc"), 0u);
}

TEST(Levenshtein, MetricProperties) {
    std::mt19937 rng(12345);
    for (int trial = 0; trial < 4000; ++trial) {
        const std::string a = random_string(rng, 8, 4);
        const std::string b = random_string(rng, 8, 4);
        const std::string c = random_string(rng, 8, 4);

        const std::size_t dab = levenshtein_distance(a, b);
        EXPECT_EQ(dab, naive_levenshtein(a, b)) << a << " | " << b;      // matches reference
        EXPECT_EQ(dab, levenshtein_distance(b, a));                       // symmetry
        EXPECT_EQ(dab == 0, a == b);                                      // identity of indiscernibles
        // Triangle inequality d(a,c) <= d(a,b) + d(b,c).
        EXPECT_LE(levenshtein_distance(a, c), dab + levenshtein_distance(b, c));
        EXPECT_LE(dab, std::max(a.size(), b.size()));                     // bounded by the longer string
    }
}

// ------------------------------------------------------------------------------------------------
// Damerau-Levenshtein
// ------------------------------------------------------------------------------------------------

TEST(DamerauLevenshtein, TranspositionCostsOne) {
    EXPECT_EQ(damerau_levenshtein_distance("ab", "ba"), 1u);   // one adjacent transposition
    EXPECT_EQ(damerau_levenshtein_distance("abcd", "abdc"), 1u);
    EXPECT_EQ(levenshtein_distance("ab", "ba"), 2u);           // plain Levenshtein pays two
}

TEST(DamerauLevenshtein, UnrestrictedBeatsOSA) {
    // The classic witness where the *unrestricted* distance (2: transpose CA->AC, insert B) is less
    // than the optimal-string-alignment restriction would give (3).
    EXPECT_EQ(damerau_levenshtein_distance("CA", "ABC"), 2u);
}

TEST(DamerauLevenshtein, KnownAndBounds) {
    EXPECT_EQ(damerau_levenshtein_distance("", ""), 0u);
    EXPECT_EQ(damerau_levenshtein_distance("abc", ""), 3u);
    EXPECT_EQ(damerau_levenshtein_distance("", "xyz"), 3u);
    EXPECT_EQ(damerau_levenshtein_distance("data", "data"), 0u);

    std::mt19937 rng(99);
    for (int trial = 0; trial < 4000; ++trial) {
        const std::string a = random_string(rng, 8, 4);
        const std::string b = random_string(rng, 8, 4);
        const std::size_t dl = damerau_levenshtein_distance(a, b);
        EXPECT_EQ(dl, damerau_levenshtein_distance(b, a));            // symmetry
        EXPECT_EQ(dl == 0, a == b);                                   // zero iff equal
        EXPECT_LE(dl, levenshtein_distance(a, b));                    // never worse than Levenshtein
    }
}

// ------------------------------------------------------------------------------------------------
// Hamming
// ------------------------------------------------------------------------------------------------

TEST(Hamming, KnownValues) {
    EXPECT_EQ(hamming_distance("karolin", "kathrin"), 3u);
    EXPECT_EQ(hamming_distance("1011101", "1001001"), 2u);
    EXPECT_EQ(hamming_distance("abc", "abc"), 0u);
    EXPECT_EQ(hamming_distance("", ""), 0u);
}

TEST(Hamming, LengthMismatchThrows) {
    EXPECT_THROW(hamming_distance("abc", "ab"), std::invalid_argument);
}

TEST(Hamming, EqualsLevenshteinWhenNoIndels) {
    // For equal-length strings the Hamming distance is an upper bound on the Levenshtein distance.
    std::mt19937                       rng(7);
    std::uniform_int_distribution<int> chr_dist(0, 3);
    for (int trial = 0; trial < 2000; ++trial) {
        const std::string a = random_string(rng, 8, 4);
        std::string       b(a.size(), 'a'); // exactly the same length as a
        for (char& c : b) c = static_cast<char>('a' + chr_dist(rng));
        EXPECT_GE(hamming_distance(a, b), levenshtein_distance(a, b));
    }
}

// ------------------------------------------------------------------------------------------------
// Jaro / Jaro-Winkler
// ------------------------------------------------------------------------------------------------

TEST(Jaro, KnownValues) {
    EXPECT_NEAR(jaro_similarity("MARTHA", "MARHTA"), 0.944444, 1e-5);
    EXPECT_NEAR(jaro_similarity("DIXON", "DICKSONX"), 0.766666, 1e-5);
    EXPECT_NEAR(jaro_similarity("DWAYNE", "DUANE"), 0.822222, 1e-5);
    EXPECT_DOUBLE_EQ(jaro_similarity("abc", "abc"), 1.0);
    EXPECT_DOUBLE_EQ(jaro_similarity("", ""), 1.0);
    EXPECT_DOUBLE_EQ(jaro_similarity("abc", ""), 0.0);
    EXPECT_DOUBLE_EQ(jaro_similarity("abc", "xyz"), 0.0); // no shared characters
}

TEST(JaroWinkler, KnownValues) {
    EXPECT_NEAR(jaro_winkler_similarity("MARTHA", "MARHTA"), 0.961111, 1e-5);
    EXPECT_NEAR(jaro_winkler_similarity("DIXON", "DICKSONX"), 0.813333, 1e-5);
    EXPECT_DOUBLE_EQ(jaro_winkler_similarity("abc", "abc"), 1.0);
}

TEST(JaroWinkler, BoundedAndBoosts) {
    std::mt19937 rng(2024);
    for (int trial = 0; trial < 4000; ++trial) {
        const std::string a = random_string(rng, 10, 5);
        const std::string b = random_string(rng, 10, 5);
        const double      j  = jaro_similarity(a, b);
        const double      jw = jaro_winkler_similarity(a, b);
        EXPECT_GE(j, -1e-12);
        EXPECT_LE(j, 1.0 + 1e-12);
        EXPECT_GE(jw, j - 1e-12);        // the prefix bonus never lowers the score
        EXPECT_LE(jw, 1.0 + 1e-12);      // and stays within [0,1]
        EXPECT_NEAR(j, jaro_similarity(b, a), 1e-12); // symmetric
    }
}

// ------------------------------------------------------------------------------------------------
// Sorensen-Dice
// ------------------------------------------------------------------------------------------------

TEST(Dice, KnownValues) {
    EXPECT_DOUBLE_EQ(dice_coefficient("night", "nacht"), 0.25);      // share only the "ht" bigram
    EXPECT_DOUBLE_EQ(dice_coefficient("abc", "abc"), 1.0);
    EXPECT_DOUBLE_EQ(dice_coefficient("", ""), 1.0);                 // identical (both empty)
    EXPECT_DOUBLE_EQ(dice_coefficient("a", "a"), 1.0);              // identical single chars
    EXPECT_DOUBLE_EQ(dice_coefficient("a", "b"), 0.0);              // no bigrams, not equal
    EXPECT_DOUBLE_EQ(dice_coefficient("abc", "xyz"), 0.0);          // disjoint bigrams
}

TEST(Dice, MultisetIntersection) {
    // "aa" has one bigram {aa}; "aaaa" has three {aa,aa,aa}; the intersection with multiplicity is 1.
    EXPECT_DOUBLE_EQ(dice_coefficient("aa", "aaaa"), 2.0 * 1.0 / (1.0 + 3.0));
}

TEST(Dice, SymmetricAndBounded) {
    std::mt19937 rng(555);
    for (int trial = 0; trial < 3000; ++trial) {
        const std::string a = random_string(rng, 10, 4);
        const std::string b = random_string(rng, 10, 4);
        const double      d  = dice_coefficient(a, b);
        EXPECT_NEAR(d, dice_coefficient(b, a), 1e-12);
        EXPECT_GE(d, -1e-12);
        EXPECT_LE(d, 1.0 + 1e-12);
    }
}
