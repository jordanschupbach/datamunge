#include <gtest/gtest.h>

#include <datamunge/algorithms/bitap.hpp>
#include <datamunge/algorithms/merge.hpp>
#include <datamunge/algorithms/ternary_search.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace datamunge::algorithms;

TEST(Merge, SimpleMergeMatchesStdMerge) {
    std::mt19937 rng(101);
    for (int trial = 0; trial < 200; ++trial) {
        std::vector<int> a(rng() % 25), b(rng() % 25);
        for (int& x : a) x = static_cast<int>(rng() % 40);
        for (int& x : b) x = static_cast<int>(rng() % 40);
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        std::vector<int> want;
        std::merge(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(want));
        EXPECT_EQ(simple_merge(a, b), want);
    }
}

TEST(Merge, KWayMergeMatchesSortedConcatenation) {
    std::mt19937 rng(2024);
    for (int trial = 0; trial < 150; ++trial) {
        const std::size_t             k = 1 + rng() % 8;
        std::vector<std::vector<int>> lists(k);
        std::vector<int>              all;
        for (auto& lst : lists) {
            lst.resize(rng() % 12);
            for (int& x : lst) {
                x = static_cast<int>(rng() % 100);
                all.push_back(x);
            }
            std::sort(lst.begin(), lst.end());
        }
        std::sort(all.begin(), all.end());
        EXPECT_EQ(k_way_merge(lists), all) << "k=" << k;
    }
    EXPECT_EQ(k_way_merge({{1, 4, 7}, {2, 5, 8}, {3, 6, 9}}),
              (std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9}));
}

TEST(Merge, UnionMergeMatchesSortedSetUnion) {
    std::mt19937 rng(55);
    for (int trial = 0; trial < 150; ++trial) {
        const std::size_t             k = 1 + rng() % 6;
        std::vector<std::vector<int>> lists(k);
        std::set<int>                 uniq;
        for (auto& lst : lists) {
            lst.resize(rng() % 12);
            for (int& x : lst) {
                x = static_cast<int>(rng() % 30);
                uniq.insert(x);
            }
            std::sort(lst.begin(), lst.end());
        }
        const std::vector<int> want(uniq.begin(), uniq.end());
        EXPECT_EQ(union_merge(lists), want) << "k=" << k;
    }
    EXPECT_EQ(union_merge({{1, 2, 2, 3}, {2, 3, 4}, {4, 4, 5}}),
              (std::vector<int>{1, 2, 3, 4, 5}));
}

TEST(TernarySearch, FindsAnalyticOptima) {
    const auto mx = ternary_search_max([](double x) { return -(x - 2.0) * (x - 2.0) + 5.0; }, -10, 10);
    EXPECT_NEAR(mx.x, 2.0, 1e-6);
    EXPECT_NEAR(mx.value, 5.0, 1e-6);

    const auto mn = ternary_search_min([](double x) { return (x - 3.0) * (x - 3.0) + 1.0; }, -10, 10);
    EXPECT_NEAR(mn.x, 3.0, 1e-6);
    EXPECT_NEAR(mn.value, 1.0, 1e-6);

    const auto s = ternary_search_max([](double x) { return std::sin(x); }, 0.0, M_PI);
    EXPECT_NEAR(s.x, M_PI / 2.0, 1e-6);
    EXPECT_NEAR(s.value, 1.0, 1e-9);
}

TEST(TernarySearch, ConvergesLogarithmically) {
    const auto r = ternary_search_max([](double x) { return -(x - 1.0) * (x - 1.0); }, -10, 10, 1e-9);
    EXPECT_LT(r.iterations, 70);
    EXPECT_NEAR(r.x, 1.0, 1e-6);
}

TEST(Bitap, ExactSearchMatchesStdFind) {
    std::mt19937 rng(9);
    std::uniform_int_distribution<int> ch('a', 'c');
    for (int trial = 0; trial < 3000; ++trial) {
        std::string t(1 + rng() % 20, 'x'), p(1 + rng() % 5, 'x');
        for (char& c : t) c = static_cast<char>(ch(rng));
        for (char& c : p) c = static_cast<char>(ch(rng));
        const auto pos  = t.find(p);
        const long want = (pos == std::string::npos) ? -1 : static_cast<long>(pos);
        EXPECT_EQ(bitap_search(t, p), want) << "t=" << t << " p=" << p;
    }
    EXPECT_EQ(bitap_search("abracadabra", "cad"), 4);
    EXPECT_EQ(bitap_search("abracadabra", "xyz"), -1);
    EXPECT_EQ(bitap_search("hello", ""), 0);
}

namespace {
// Oracle: first ending index of a length-m window within k substitutions (Hamming) of pattern.
long hamming_oracle(const std::string& t, const std::string& p, int k) {
    const long m = static_cast<long>(p.size());
    for (long i = 0; i + m <= static_cast<long>(t.size()); ++i) {
        int mm = 0;
        for (long j = 0; j < m; ++j) mm += (t[i + j] != p[j]);
        if (mm <= k) return i + m - 1;
    }
    return -1;
}
} // namespace

TEST(Bitap, FuzzySearchMatchesHammingOracle) {
    std::mt19937 rng(31);
    std::uniform_int_distribution<int> ch('a', 'd'), plen(1, 6), tlen(4, 16), kd(0, 3);
    for (int trial = 0; trial < 8000; ++trial) {
        const int   k = kd(rng);
        std::string t(tlen(rng), 'x'), p(plen(rng), 'x');
        for (char& c : t) c = static_cast<char>(ch(rng));
        for (char& c : p) c = static_cast<char>(ch(rng));
        EXPECT_EQ(bitap_fuzzy_search(t, p, k), hamming_oracle(t, p, k)) << "t=" << t << " p=" << p << " k=" << k;
    }
    // k=0 reduces to exact matching.
    EXPECT_EQ(bitap_fuzzy_search("abracadabra", "cad", 0), 6); // ending index of "cad"
    // one substitution: "cid" matches "cad" (a->i) ending at index 6.
    EXPECT_EQ(bitap_fuzzy_search("abracadabra", "cid", 1), 6);
    // too many needed: "xyz" never within 1 substitution.
    EXPECT_EQ(bitap_fuzzy_search("abracadabra", "xyz", 1), -1);
}
