#include <gtest/gtest.h>

#include <datamunge/algorithms/gabow_scc.hpp>
#include <datamunge/algorithms/graph_connectivity.hpp>
#include <datamunge/algorithms/library_sort.hpp>
#include <datamunge/geometry/segment_intersection.hpp>
#include <datamunge/geometry/shamos_hoey.hpp>

#include <algorithm>
#include <map>
#include <random>
#include <vector>

using namespace datamunge::algorithms;
using namespace datamunge::geometry;

// ---------------- Gabow SCC ----------------

namespace {
bool same_partition(const std::vector<int>& a, const std::vector<int>& b) {
    if (a.size() != b.size()) return false;
    std::map<int, int> ab, ba;
    for (std::size_t i = 0; i < a.size(); ++i) {
        auto ia = ab.find(a[i]); if (ia == ab.end()) ab[a[i]] = b[i]; else if (ia->second != b[i]) return false;
        auto ib = ba.find(b[i]); if (ib == ba.end()) ba[b[i]] = a[i]; else if (ib->second != a[i]) return false;
    }
    return true;
}
} // namespace

TEST(GabowScc, AgreesWithKosaraju) {
    std::mt19937 rng(1);
    for (int t = 0; t < 3000; ++t) {
        const int n = 2 + static_cast<int>(rng() % 11);
        std::vector<std::vector<int>> adj(n);
        for (int u = 0; u < n; ++u)
            for (int v = 0; v < n; ++v)
                if (u != v && rng() % 4 == 0) adj[u].push_back(v);
        EXPECT_TRUE(same_partition(gabow_scc(n, adj), kosaraju_scc(n, adj))) << "trial " << t;
    }
}

TEST(GabowScc, HandcraftedComponents) {
    // 0<->1 form one component; 2->3->4->2 form another; 5 alone.
    std::vector<std::vector<int>> adj(6);
    adj[0] = {1}; adj[1] = {0};
    adj[2] = {3}; adj[3] = {4}; adj[4] = {2};
    // vertex 5: no edges
    const auto c = gabow_scc(6, adj);
    EXPECT_EQ(c[0], c[1]);
    EXPECT_EQ(c[2], c[3]);
    EXPECT_EQ(c[3], c[4]);
    EXPECT_NE(c[0], c[2]);
    EXPECT_NE(c[0], c[5]);
    EXPECT_NE(c[2], c[5]);
}

// ---------------- Shamos-Hoey ----------------

TEST(ShamosHoey, AgreesWithBruteForce) {
    std::mt19937                           rng(1);
    std::uniform_real_distribution<double> u(0, 100);
    for (int t = 0; t < 3000; ++t) {
        const int n = 2 + static_cast<int>(rng() % 12);
        std::vector<std::pair<Point2D, Point2D>> segs;
        for (int i = 0; i < n; ++i) segs.push_back({{u(rng), u(rng)}, {u(rng), u(rng)}});
        bool brute = false;
        for (int i = 0; i < n && !brute; ++i)
            for (int j = i + 1; j < n; ++j)
                if (segments_intersect(segs[i].first, segs[i].second, segs[j].first, segs[j].second)) { brute = true; break; }
        EXPECT_EQ(any_segments_intersect(segs), brute) << "trial " << t;
    }
}

TEST(ShamosHoey, DisjointVsCrossing) {
    // Two parallel horizontal segments: no intersection.
    EXPECT_FALSE(any_segments_intersect({{{0, 0}, {5, 0}}, {{0, 2}, {5, 2}}}));
    // An X: they cross.
    EXPECT_TRUE(any_segments_intersect({{{0, 0}, {5, 5}}, {{0, 5}, {5, 0}}}));
}

// ---------------- Library sort ----------------

TEST(LibrarySort, MatchesStdSort) {
    std::mt19937 rng(1);
    for (int t = 0; t < 4000; ++t) {
        const int          n = static_cast<int>(rng() % 120);
        std::vector<int>   a(n);
        std::uniform_int_distribution<int> d(-50, 50);
        for (auto& v : a) v = d(rng);
        auto ref = a;
        std::sort(ref.begin(), ref.end());
        EXPECT_EQ(library_sort(a), ref) << "n=" << n;
    }
}

TEST(LibrarySort, EdgeCases) {
    EXPECT_TRUE(library_sort({}).empty());
    EXPECT_EQ(library_sort({7}), (std::vector<int>{7}));
    EXPECT_EQ(library_sort({3, 3, 3, 3}), (std::vector<int>{3, 3, 3, 3}));
    EXPECT_EQ(library_sort({5, 4, 3, 2, 1}), (std::vector<int>{1, 2, 3, 4, 5}));
}
