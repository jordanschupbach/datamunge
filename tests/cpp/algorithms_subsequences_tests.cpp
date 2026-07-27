#include <gtest/gtest.h>

#include <datamunge/algorithms/subsequences.hpp>

#include <algorithm>
#include <random>
#include <set>
#include <tuple>
#include <vector>

using datamunge::algorithms::longest_common_subsequence;
using datamunge::algorithms::longest_increasing_subsequence;
using datamunge::algorithms::ruzzo_tompa;
using datamunge::algorithms::ScoringSegment;
using datamunge::algorithms::shortest_common_supersequence;

namespace {

bool is_subsequence(const std::vector<int>& sub, const std::vector<int>& seq) {
    std::size_t i = 0;
    for (int v : seq)
        if (i < sub.size() && sub[i] == v) ++i;
    return i == sub.size();
}

std::size_t lcs_len_ref(const std::vector<int>& a, const std::vector<int>& b) {
    std::vector<std::vector<int>> c(a.size() + 1, std::vector<int>(b.size() + 1, 0));
    for (std::size_t i = 1; i <= a.size(); ++i)
        for (std::size_t j = 1; j <= b.size(); ++j)
            c[i][j] = (a[i - 1] == b[j - 1]) ? c[i - 1][j - 1] + 1
                                             : std::max(c[i - 1][j], c[i][j - 1]);
    return static_cast<std::size_t>(c[a.size()][b.size()]);
}

std::size_t lis_len_ref(const std::vector<int>& a) {
    std::vector<int> tails;
    for (int x : a) {
        auto it = std::lower_bound(tails.begin(), tails.end(), x);
        if (it == tails.end()) tails.push_back(x); else *it = x;
    }
    return tails.size();
}

std::vector<int> random_seq(std::mt19937& rng, std::size_t max_len, int alphabet) {
    std::uniform_int_distribution<std::size_t> len(0, max_len);
    std::uniform_int_distribution<int>         val(0, alphabet - 1);
    std::vector<int>                           s(len(rng));
    for (int& x : s) x = val(rng);
    return s;
}

// Brute-force Ruzzo-Tompa by the definition, O(n^4): all maximal-scoring segments as (begin,end).
std::set<std::pair<std::size_t, std::size_t>> brute_ruzzo_tompa(const std::vector<double>& s) {
    const std::size_t   n = s.size();
    std::vector<double> pre(n + 1, 0.0);
    for (std::size_t i = 0; i < n; ++i) pre[i + 1] = pre[i] + s[i];
    auto score = [&](std::size_t b, std::size_t e) { return pre[e] - pre[b]; };

    // "self-maximal": positive score and strictly greater than every proper sub-interval.
    std::set<std::pair<std::size_t, std::size_t>> selfmax;
    for (std::size_t b = 0; b < n; ++b)
        for (std::size_t e = b + 1; e <= n; ++e) {
            if (score(b, e) <= 0.0) continue;
            bool ok = true;
            for (std::size_t i = b; i < e && ok; ++i)
                for (std::size_t j = i + 1; j <= e; ++j) {
                    if (i == b && j == e) continue;
                    if (score(i, j) >= score(b, e)) { ok = false; break; }
                }
            if (ok) selfmax.insert({b, e});
        }
    // Keep only those not contained in a larger self-maximal of at least equal score.
    std::set<std::pair<std::size_t, std::size_t>> result;
    for (const auto& [b, e] : selfmax) {
        bool maximal = true;
        for (const auto& [b2, e2] : selfmax) {
            if (std::tie(b2, e2) == std::tie(b, e)) continue;
            if (b2 <= b && e2 >= e && score(b2, e2) >= score(b, e)) { maximal = false; break; }
        }
        if (maximal) result.insert({b, e});
    }
    return result;
}

} // namespace

// ------------------------------------------------------------------------------------------------
// Longest common subsequence
// ------------------------------------------------------------------------------------------------

TEST(LCS, KnownAndSubsequenceOfBoth) {
    const std::vector<int> a = {1, 3, 4, 1, 2, 1, 3}; // "AGCAT"-like
    const std::vector<int> b = {3, 4, 1, 2, 1};
    const std::vector<int> lcs = longest_common_subsequence(a, b);
    EXPECT_EQ(lcs.size(), lcs_len_ref(a, b));
    EXPECT_TRUE(is_subsequence(lcs, a));
    EXPECT_TRUE(is_subsequence(lcs, b));
}

TEST(LCS, RandomProperties) {
    std::mt19937 rng(11);
    for (int t = 0; t < 4000; ++t) {
        const auto a = random_seq(rng, 10, 4);
        const auto b = random_seq(rng, 10, 4);
        const auto lcs = longest_common_subsequence(a, b);
        EXPECT_EQ(lcs.size(), lcs_len_ref(a, b));
        EXPECT_TRUE(is_subsequence(lcs, a));
        EXPECT_TRUE(is_subsequence(lcs, b));
    }
}

// ------------------------------------------------------------------------------------------------
// Longest increasing subsequence
// ------------------------------------------------------------------------------------------------

TEST(LIS, KnownExample) {
    // Classic: LIS length 6 (e.g. 0 2 6 9 11 15).
    const std::vector<int> a = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};
    const std::vector<int> lis = longest_increasing_subsequence(a);
    EXPECT_EQ(lis.size(), 6u);
    EXPECT_EQ(lis.size(), lis_len_ref(a));
    EXPECT_TRUE(is_subsequence(lis, a));
    EXPECT_TRUE(std::is_sorted(lis.begin(), lis.end())); // increasing
    for (std::size_t i = 1; i < lis.size(); ++i) EXPECT_LT(lis[i - 1], lis[i]); // strictly
}

TEST(LIS, RandomProperties) {
    std::mt19937 rng(22);
    for (int t = 0; t < 4000; ++t) {
        const auto a = random_seq(rng, 14, 8);
        const auto lis = longest_increasing_subsequence(a);
        EXPECT_EQ(lis.size(), lis_len_ref(a));
        EXPECT_TRUE(is_subsequence(lis, a));
        for (std::size_t i = 1; i < lis.size(); ++i) EXPECT_LT(lis[i - 1], lis[i]);
    }
}

// ------------------------------------------------------------------------------------------------
// Shortest common supersequence
// ------------------------------------------------------------------------------------------------

TEST(SCS, ContainsBothAndOptimalLength) {
    std::mt19937 rng(33);
    for (int t = 0; t < 4000; ++t) {
        const auto a = random_seq(rng, 9, 4);
        const auto b = random_seq(rng, 9, 4);
        const auto scs = shortest_common_supersequence(a, b);
        EXPECT_TRUE(is_subsequence(a, scs)); // a is a subsequence of the supersequence
        EXPECT_TRUE(is_subsequence(b, scs));
        EXPECT_EQ(scs.size(), a.size() + b.size() - lcs_len_ref(a, b)); // optimal length
    }
}

// ------------------------------------------------------------------------------------------------
// Ruzzo-Tompa
// ------------------------------------------------------------------------------------------------

TEST(RuzzoTompa, KnownExamples) {
    // Two positive islands separated by a deep negative valley stay separate.
    auto segs = ruzzo_tompa({4, -5, 3});
    ASSERT_EQ(segs.size(), 2u);
    EXPECT_EQ(segs[0].begin, 0u); EXPECT_EQ(segs[0].end, 1u); EXPECT_DOUBLE_EQ(segs[0].score, 4.0);
    EXPECT_EQ(segs[1].begin, 2u); EXPECT_EQ(segs[1].end, 3u); EXPECT_DOUBLE_EQ(segs[1].score, 3.0);

    // A shallow valley is bridged into one island.
    segs = ruzzo_tompa({4, -1, 3});
    ASSERT_EQ(segs.size(), 1u);
    EXPECT_EQ(segs[0].begin, 0u); EXPECT_EQ(segs[0].end, 3u); EXPECT_DOUBLE_EQ(segs[0].score, 6.0);

    EXPECT_TRUE(ruzzo_tompa({-1, -2, -3}).empty()); // nothing positive
}

TEST(RuzzoTompa, MatchesBruteForceAndIsDisjointPositive) {
    std::mt19937                       rng(44);
    std::uniform_int_distribution<int> len(0, 10);
    std::uniform_int_distribution<int> val(-4, 4);
    for (int t = 0; t < 6000; ++t) {
        std::vector<double> s(static_cast<std::size_t>(len(rng)));
        for (double& x : s) x = val(rng);

        const auto segs = ruzzo_tompa(s);

        std::set<std::pair<std::size_t, std::size_t>> got;
        std::size_t prev_end = 0;
        for (const auto& seg : segs) {
            EXPECT_LT(seg.begin, seg.end);
            EXPECT_GT(seg.score, 0.0);                 // positive score
            EXPECT_GE(seg.begin, prev_end);            // disjoint and left-to-right
            prev_end = seg.end;
            double sum = 0.0;
            for (std::size_t i = seg.begin; i < seg.end; ++i) sum += s[i];
            EXPECT_DOUBLE_EQ(sum, seg.score);          // reported score matches the values
            got.insert({seg.begin, seg.end});
        }
        EXPECT_EQ(got, brute_ruzzo_tompa(s));          // exactly the maximal segments
    }
}
