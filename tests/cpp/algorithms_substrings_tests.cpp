#include <gtest/gtest.h>

#include <datamunge/algorithms/substrings.hpp>

#include <algorithm>
#include <random>
#include <set>
#include <string>
#include <vector>

using datamunge::algorithms::build_suffix_tree;
using datamunge::algorithms::distinct_substring_count;
using datamunge::algorithms::kadane;
using datamunge::algorithms::longest_common_substring;
using datamunge::algorithms::MaxSubarray;
using datamunge::algorithms::suffix_tree_contains;
using datamunge::algorithms::wildcard_match;

namespace {

double brute_max_subarray(const std::vector<int>& a) {
    double best = a.empty() ? 0.0 : a[0];
    for (std::size_t i = 0; i < a.size(); ++i) {
        double s = 0;
        for (std::size_t j = i; j < a.size(); ++j) {
            s += a[j];
            best = std::max(best, s);
        }
    }
    return best;
}

std::size_t lcsubstr_len_ref(const std::vector<int>& a, const std::vector<int>& b) {
    std::vector<std::vector<int>> d(a.size() + 1, std::vector<int>(b.size() + 1, 0));
    std::size_t                   best = 0;
    for (std::size_t i = 1; i <= a.size(); ++i)
        for (std::size_t j = 1; j <= b.size(); ++j)
            if (a[i - 1] == b[j - 1]) {
                d[i][j] = d[i - 1][j - 1] + 1;
                best    = std::max(best, static_cast<std::size_t>(d[i][j]));
            }
    return best;
}

bool contiguous_in(const std::vector<int>& sub, const std::vector<int>& seq) {
    if (sub.empty()) return true;
    if (sub.size() > seq.size()) return false;
    for (std::size_t i = 0; i + sub.size() <= seq.size(); ++i)
        if (std::equal(sub.begin(), sub.end(), seq.begin() + static_cast<std::ptrdiff_t>(i))) return true;
    return false;
}

// Recursive wildcard reference with memoization.
bool wildcard_ref(const std::string& t, const std::string& p) {
    const std::size_t n = t.size(), m = p.size();
    std::vector<std::vector<int>> memo(n + 1, std::vector<int>(m + 1, -1)); // -1 unknown, 0/1 result
    auto solve = [&](auto&& self, std::size_t i, std::size_t j) -> bool {
        if (memo[i][j] != -1) return memo[i][j] == 1;
        bool res;
        if (j == m)
            res = (i == n);
        else if (p[j] == '*')
            res = self(self, i, j + 1) || (i < n && self(self, i + 1, j));
        else
            res = (i < n && (p[j] == '?' || p[j] == t[i])) && self(self, i + 1, j + 1);
        memo[i][j] = res ? 1 : 0;
        return res;
    };
    return solve(solve, 0, 0);
}

std::set<std::string> brute_distinct_substrings(const std::string& s) {
    std::set<std::string> out;
    for (std::size_t i = 0; i < s.size(); ++i)
        for (std::size_t len = 1; i + len <= s.size(); ++len) out.insert(s.substr(i, len));
    return out;
}

std::string random_string(std::mt19937& rng, std::size_t max_len, const std::string& alphabet) {
    std::uniform_int_distribution<std::size_t> len(0, max_len);
    std::uniform_int_distribution<std::size_t> ch(0, alphabet.size() - 1);
    std::string                                s(len(rng), 'a');
    for (char& c : s) c = alphabet[ch(rng)];
    return s;
}

} // namespace

// ------------------------------------------------------------------------------------------------
// Kadane
// ------------------------------------------------------------------------------------------------

TEST(Kadane, KnownAndRange) {
    const MaxSubarray r = kadane({-2, 1, -3, 4, -1, 2, 1, -5, 4}); // classic -> 6 over [3,7)
    EXPECT_DOUBLE_EQ(r.sum, 6.0);
    EXPECT_EQ(r.begin, 3u);
    EXPECT_EQ(r.end, 7u);
    EXPECT_TRUE(kadane({}).begin == 0u && kadane({}).end == 0u);
    EXPECT_DOUBLE_EQ(kadane({-3, -1, -2}).sum, -1.0); // all negative -> least-negative element
}

TEST(Kadane, MatchesBruteForce) {
    std::mt19937                       rng(1);
    std::uniform_int_distribution<int> len(1, 20), val(-9, 9);
    for (int t = 0; t < 5000; ++t) {
        std::vector<int> a(static_cast<std::size_t>(len(rng)));
        for (int& x : a) x = val(rng);
        const MaxSubarray r = kadane(a);
        EXPECT_DOUBLE_EQ(r.sum, brute_max_subarray(a));
        ASSERT_LT(r.begin, r.end);
        double s = 0;
        for (std::size_t i = r.begin; i < r.end; ++i) s += a[i];
        EXPECT_DOUBLE_EQ(s, r.sum); // the reported range really sums to the reported value
    }
}

// ------------------------------------------------------------------------------------------------
// Longest common substring
// ------------------------------------------------------------------------------------------------

TEST(LongestCommonSubstring, KnownAndProperties) {
    std::mt19937 rng(2);
    for (int t = 0; t < 5000; ++t) {
        std::uniform_int_distribution<int>         len(0, 12), val(0, 3);
        std::vector<int>                           a(static_cast<std::size_t>(len(rng)));
        std::vector<int>                           b(static_cast<std::size_t>(len(rng)));
        for (int& x : a) x = val(rng);
        for (int& x : b) x = val(rng);
        const std::vector<int> r = longest_common_substring(a, b);
        EXPECT_EQ(r.size(), lcsubstr_len_ref(a, b));
        EXPECT_TRUE(contiguous_in(r, a));
        EXPECT_TRUE(contiguous_in(r, b));
    }
}

// ------------------------------------------------------------------------------------------------
// Wildcard matching
// ------------------------------------------------------------------------------------------------

TEST(WildcardMatch, KnownExamples) {
    EXPECT_TRUE(wildcard_match("abcde", "a*e"));
    EXPECT_TRUE(wildcard_match("abcde", "a?cd?"));
    EXPECT_TRUE(wildcard_match("", "*"));
    EXPECT_TRUE(wildcard_match("anything", "*"));
    EXPECT_FALSE(wildcard_match("abcde", "a*f"));
    EXPECT_FALSE(wildcard_match("abc", "abcd"));
    EXPECT_FALSE(wildcard_match("abc", "?"));
    EXPECT_TRUE(wildcard_match("aaa", "a*a"));
    EXPECT_TRUE(wildcard_match("mississippi", "m*i*s*p*i"));
}

TEST(WildcardMatch, MatchesRecursiveReference) {
    std::mt19937 rng(3);
    for (int t = 0; t < 20000; ++t) {
        const std::string text    = random_string(rng, 9, "ab");
        const std::string pattern = random_string(rng, 8, "ab?*");
        EXPECT_EQ(wildcard_match(text, pattern), wildcard_ref(text, pattern))
            << "text=" << text << " pattern=" << pattern;
    }
}

// ------------------------------------------------------------------------------------------------
// Ukkonen suffix tree
// ------------------------------------------------------------------------------------------------

TEST(SuffixTree, ContainsAndDistinctCountKnown) {
    const auto tree = build_suffix_tree("banana");
    EXPECT_TRUE(suffix_tree_contains(tree, "ana"));
    EXPECT_TRUE(suffix_tree_contains(tree, "banana"));
    EXPECT_TRUE(suffix_tree_contains(tree, "nan"));
    EXPECT_TRUE(suffix_tree_contains(tree, ""));
    EXPECT_FALSE(suffix_tree_contains(tree, "anas"));
    EXPECT_FALSE(suffix_tree_contains(tree, "x"));
    // distinct substrings of "banana": a,an,ana,anan,anana,b,ba,ban,bana,banan,banana,n,na,nan,nana = 15
    EXPECT_EQ(distinct_substring_count(tree), 15u);
}

TEST(SuffixTree, MatchesBruteForceOverRandomStrings) {
    std::mt19937 rng(4);
    for (int t = 0; t < 4000; ++t) {
        const std::string s    = random_string(rng, 12, "abc");
        const auto        tree = build_suffix_tree(s);
        const auto        subs = brute_distinct_substrings(s);

        EXPECT_EQ(distinct_substring_count(tree), subs.size()) << "s=" << s;

        // Every genuine substring is found; a batch of random queries agrees with std::string::find.
        for (const std::string& sub : subs) EXPECT_TRUE(suffix_tree_contains(tree, sub)) << s << " / " << sub;
        for (int q = 0; q < 20; ++q) {
            const std::string p = random_string(rng, 5, "abcd");
            EXPECT_EQ(suffix_tree_contains(tree, p), s.find(p) != std::string::npos) << s << " ? " << p;
        }
    }
}
