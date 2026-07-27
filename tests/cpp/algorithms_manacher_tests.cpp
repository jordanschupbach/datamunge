#include <gtest/gtest.h>

#include <datamunge/algorithms/manacher.hpp>

#include <random>
#include <string>
#include <utility>
#include <vector>

using datamunge::algorithms::longest_palindrome;
using datamunge::algorithms::palindrome_radii;
using datamunge::algorithms::PalindromeResult;

namespace {

bool is_palindrome(const std::string& s) {
    for (std::size_t i = 0, j = s.size(); i + 1 < j; ++i) {
        --j;
        if (s[i] != s[j]) return false;
    }
    return true;
}

// Independent O(n^2) reference: the leftmost longest palindromic substring by
// center expansion, returning {start, length}.
std::pair<std::size_t, std::size_t> brute_longest(const std::string& s) {
    const long n = static_cast<long>(s.size());
    long best_start = 0;
    long best_len = 0;
    auto expand = [&](long l, long r) {
        while (l >= 0 && r < n && s[static_cast<std::size_t>(l)] == s[static_cast<std::size_t>(r)]) {
            --l;
            ++r;
        }
        // palindrome is s[l+1 .. r-1]: start l+1, length (r-1)-(l+1)+1 = r-l-1
        return std::pair<long, long>{l + 1, r - l - 1};
    };
    for (long c = 0; c < n; ++c) {
        for (long type = 0; type < 2; ++type) { // odd (c,c) then even (c,c+1)
            const auto [st, len] = expand(c, c + type);
            if (len > best_len || (len == best_len && st < best_start)) {
                best_len = len;
                best_start = st;
            }
        }
    }
    return {static_cast<std::size_t>(best_start), static_cast<std::size_t>(best_len)};
}

// Independent O(|t|^2) reference for the Manacher radius array: expand about every
// center of the '#'-interleaved transform directly.
std::vector<std::size_t> brute_radii(const std::string& s) {
    std::string t = "^";
    for (const char c : s) {
        t.push_back('#');
        t.push_back(c);
    }
    t += "#$";
    const long m = static_cast<long>(t.size());
    std::vector<std::size_t> p(t.size(), 0);
    for (long i = 1; i + 1 < m; ++i) {
        long r = 0;
        while (i - r - 1 >= 0 && i + r + 1 < m &&
               t[static_cast<std::size_t>(i - r - 1)] == t[static_cast<std::size_t>(i + r + 1)]) {
            ++r;
        }
        p[static_cast<std::size_t>(i)] = static_cast<std::size_t>(r);
    }
    return p;
}

} // namespace

TEST(Manacher, KnownLongestPalindromes) {
    const auto babad = longest_palindrome("babad");
    EXPECT_EQ(babad.start, 0u); // "bab" is the leftmost of the two length-3 palindromes
    EXPECT_EQ(babad.length, 3u);
    EXPECT_EQ(std::string("babad").substr(babad.start, babad.length), "bab");

    const auto cbbd = longest_palindrome("cbbd");
    EXPECT_EQ(cbbd.start, 1u);
    EXPECT_EQ(cbbd.length, 2u);
    EXPECT_EQ(std::string("cbbd").substr(cbbd.start, cbbd.length), "bb");

    const auto single = longest_palindrome("a");
    EXPECT_EQ(single.start, 0u);
    EXPECT_EQ(single.length, 1u);

    const auto empty = longest_palindrome("");
    EXPECT_EQ(empty.start, 0u);
    EXPECT_EQ(empty.length, 0u);

    const auto aaaa = longest_palindrome("aaaa");
    EXPECT_EQ(aaaa.start, 0u);
    EXPECT_EQ(aaaa.length, 4u);
}

TEST(Manacher, FullyPalindromicString) {
    const std::string s = "abacaba"; // itself a palindrome of length 7
    const auto r = longest_palindrome(s);
    EXPECT_EQ(r.start, 0u);
    EXPECT_EQ(r.length, s.size());
    EXPECT_EQ(s.substr(r.start, r.length), s);

    // The peak radius sits at the exact center of the transform (index n + 1) and
    // equals the whole length.
    const auto p = palindrome_radii(s);
    EXPECT_EQ(p[s.size() + 1], s.size());
}

TEST(Manacher, RadiiTransformInvariants) {
    const std::string s = "abacaba";
    const auto p = palindrome_radii(s);
    ASSERT_EQ(p.size(), 2 * s.size() + 3); // |t| = 2n + 3
    EXPECT_EQ(p.front(), 0u);              // '^' sentinel
    EXPECT_EQ(p.back(), 0u);               // '$' sentinel
    // Hand-computed radii for ^#a#b#a#c#a#b#a#$
    const std::vector<std::size_t> expected = {0, 0, 1, 0, 3, 0, 1, 0, 7,
                                               0, 1, 0, 3, 0, 1, 0, 0};
    EXPECT_EQ(p, expected);
}

TEST(Manacher, RadiiMatchBruteForceExpansion) {
    for (const std::string& s : {std::string(""), std::string("a"), std::string("aa"),
                                 std::string("abba"), std::string("abacabad"),
                                 std::string("mississippi"), std::string("forgeeksskeegfor")}) {
        EXPECT_EQ(palindrome_radii(s), brute_radii(s)) << "s = " << s;
    }
}

TEST(Manacher, MatchesBruteForceOnRandomStrings) {
    std::mt19937_64 rng(20240927);
    for (int trial = 0; trial < 4000; ++trial) {
        const std::size_t len = rng() % 18;             // 0..17 chars
        const int alphabet = 2 + static_cast<int>(rng() % 3); // {2,3,4}-letter alphabet
        std::string s;
        s.reserve(len);
        for (std::size_t k = 0; k < len; ++k)
            s.push_back(static_cast<char>('a' + static_cast<int>(rng() % static_cast<unsigned>(alphabet))));

        const PalindromeResult got = longest_palindrome(s);
        const auto [ref_start, ref_len] = brute_longest(s);

        ASSERT_EQ(got.length, ref_len) << "trial " << trial << " s = " << s;
        ASSERT_EQ(got.start, ref_start) << "trial " << trial << " s = " << s;
        // The returned substring must actually be a palindrome.
        ASSERT_TRUE(is_palindrome(s.substr(got.start, got.length))) << "trial " << trial << " s = " << s;
    }
}
