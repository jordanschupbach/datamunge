#include <gtest/gtest.h>

#include <datamunge/algorithms/arc_consistency.hpp>
#include <datamunge/algorithms/ellipsoid.hpp>
#include <datamunge/algorithms/exact_cover.hpp>
#include <datamunge/algorithms/min_conflicts.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <set>
#include <vector>

using namespace datamunge::algorithms;

TEST(Ac3, LeavesDomainsArcConsistent) {
    // Variables 0,1,2 with domains {1..5}; constraint i<j on arcs (0,1),(1,2) and reverse.
    std::vector<std::vector<int>> domains{{1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}};
    std::vector<std::pair<int, int>> arcs{{0, 1}, {1, 0}, {1, 2}, {2, 1}};
    auto less = [](int, int a, int, int b) { return a < b; }; // value(i) < value(j) required for arc (i,j)
    // Actually enforce x_i < x_{i+1}: for arc (i,j) with i<j require a<b, for (j,i) require a>b.
    auto rel = [](int i, int a, int j, int b) { return i < j ? a < b : a > b; };

    const Ac3Result r = ac3(domains, arcs, rel);
    ASSERT_TRUE(r.consistent);
    // Every arc must now be consistent: each value has a support.
    for (const auto& [i, j] : arcs)
        for (int a : r.domains[i]) {
            bool sup = false;
            for (int b : r.domains[j]) if (rel(i, a, j, b)) sup = true;
            EXPECT_TRUE(sup) << "value " << a << " of var " << i << " unsupported on arc(" << i << "," << j << ")";
        }
    // x0<x1<x2 with domain 1..5 forces x0<=3, x2>=3.
    EXPECT_LE(*std::max_element(r.domains[0].begin(), r.domains[0].end()), 3);
    EXPECT_GE(*std::min_element(r.domains[2].begin(), r.domains[2].end()), 3);
}

TEST(Ac3, DetectsInconsistency) {
    // x0 in {1}, x1 in {1}, constraint x0 != x1 -> no support -> inconsistent.
    std::vector<std::vector<int>>    domains{{1}, {1}};
    std::vector<std::pair<int, int>> arcs{{0, 1}, {1, 0}};
    auto neq = [](int, int a, int, int b) { return a != b; };
    EXPECT_FALSE(ac3(domains, arcs, neq).consistent);
}

namespace {
bool is_exact_cover(int cols, const std::vector<std::vector<int>>& rows, const std::vector<int>& chosen) {
    std::vector<int> count(cols, 0);
    for (int r : chosen)
        for (int c : rows[r]) ++count[c];
    for (int c = 0; c < cols; ++c) if (count[c] != 1) return false;
    return true;
}
// Brute-force: does any subset of rows exactly cover all columns?
bool brute_cover_exists(int cols, const std::vector<std::vector<int>>& rows) {
    const int n = static_cast<int>(rows.size());
    for (int mask = 0; mask < (1 << n); ++mask) {
        std::vector<int> chosen;
        for (int i = 0; i < n; ++i) if (mask & (1 << i)) chosen.push_back(i);
        if (is_exact_cover(cols, rows, chosen)) return true;
    }
    return false;
}
} // namespace

TEST(ExactCover, DlxMatchesBruteForceAndReturnsValidCover) {
    // Knuth's classic example: 7 columns, rows as column-index subsets; has a unique cover {rows 0,3,4}.
    const std::vector<std::vector<int>> knuth{{0, 3, 6}, {0, 3}, {3, 4, 6}, {2, 4, 5}, {1, 6}, {2, 5}};
    const ExactCoverResult              kr = exact_cover(7, knuth);
    EXPECT_TRUE(kr.solved);
    EXPECT_TRUE(is_exact_cover(7, knuth, kr.rows));

    std::mt19937 rng(7);
    for (int trial = 0; trial < 300; ++trial) {
        const int                     cols = 2 + rng() % 6;
        const int                     nr   = 1 + rng() % 8;
        std::vector<std::vector<int>> rr(nr);
        for (auto& row : rr) {
            std::set<int> s;
            const int     k = 1 + rng() % cols;
            for (int j = 0; j < k; ++j) s.insert(rng() % cols);
            row.assign(s.begin(), s.end());
        }
        const ExactCoverResult res = exact_cover(cols, rr);
        EXPECT_EQ(res.solved, brute_cover_exists(cols, rr));
        if (res.solved) EXPECT_TRUE(is_exact_cover(cols, rr, res.rows));
    }
}

TEST(MinConflicts, SolvesNQueens) {
    for (int n : {8, 16, 25, 50, 100}) {
        const NQueensResult r = min_conflicts_nqueens(n, 2000, 12345u + n);
        ASSERT_TRUE(r.solved) << "n=" << n;
        ASSERT_EQ(static_cast<int>(r.queens.size()), n);
        // No two queens share a column or diagonal (rows are distinct by construction).
        for (int a = 0; a < n; ++a)
            for (int b = a + 1; b < n; ++b) {
                EXPECT_NE(r.queens[a], r.queens[b]) << "same column";
                EXPECT_NE(std::abs(r.queens[a] - r.queens[b]), b - a) << "same diagonal";
            }
    }
}

TEST(Ellipsoid, ConvergesToConvexMinimum) {
    // Minimize a shifted, scaled convex quadratic f(x) = sum a_i (x_i - x*_i)^2.
    const std::vector<double> star{2.0, -3.0, 0.5};
    const std::vector<double> a{1.0, 4.0, 2.0};
    auto f = [&](const std::vector<double>& x) {
        double s = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i) s += a[i] * (x[i] - star[i]) * (x[i] - star[i]);
        return s;
    };
    auto grad = [&](const std::vector<double>& x) {
        std::vector<double> g(x.size());
        for (std::size_t i = 0; i < x.size(); ++i) g[i] = 2.0 * a[i] * (x[i] - star[i]);
        return g;
    };
    const EllipsoidResult r = ellipsoid_minimize(f, grad, {0.0, 0.0, 0.0}, 20.0, 2000);
    for (std::size_t i = 0; i < star.size(); ++i) EXPECT_NEAR(r.x[i], star[i], 1e-3);
    EXPECT_NEAR(r.value, 0.0, 1e-5);

    // A nonsmooth convex objective: f(x) = |x-1| + |x+2| (1-D), minimized on [-2,1].
    auto g1 = [](const std::vector<double>& x) { return std::fabs(x[0] - 1.0) + std::fabs(x[0] + 2.0); };
    auto sg1 = [](const std::vector<double>& x) {
        double s = (x[0] > 1.0 ? 1.0 : -1.0) + (x[0] > -2.0 ? 1.0 : -1.0);
        return std::vector<double>{s};
    };
    const EllipsoidResult r1 = ellipsoid_minimize(g1, sg1, {5.0}, 10.0, 200);
    EXPECT_NEAR(r1.value, 3.0, 1e-2); // min value is the distance 3 between the two kinks
}
