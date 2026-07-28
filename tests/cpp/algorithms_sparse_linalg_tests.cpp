#include <gtest/gtest.h>

#include <datamunge/algorithms/sparse_linalg.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <set>
#include <vector>

using namespace datamunge::algorithms;

namespace {

// A random connected symmetric graph on n nodes.
std::vector<std::vector<int>> random_graph(std::mt19937& rng, int n, double density) {
    std::vector<std::set<int>> s(n);
    for (int i = 1; i < n; ++i) { const int j = rng() % i; s[i].insert(j); s[j].insert(i); } // spanning tree (connected)
    std::uniform_real_distribution<double> u(0, 1);
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            if (u(rng) < density) { s[i].insert(j); s[j].insert(i); }
    std::vector<std::vector<int>> a(n);
    for (int i = 0; i < n; ++i) a[i].assign(s[i].begin(), s[i].end());
    return a;
}

bool is_permutation(const std::vector<int>& p, int n) {
    if (static_cast<int>(p.size()) != n) return false;
    std::vector<char> seen(n, 0);
    for (int x : p) { if (x < 0 || x >= n || seen[x]) return false; seen[x] = 1; }
    return true;
}

// Bandwidth of the pattern under a permutation (inv[orig] = new position).
int bandwidth(const std::vector<std::vector<int>>& adj, const std::vector<int>& perm) {
    const int        n = static_cast<int>(adj.size());
    std::vector<int> pos(n);
    for (int k = 0; k < n; ++k) pos[perm[k]] = k;
    int bw = 0;
    for (int i = 0; i < n; ++i)
        for (int j : adj[i]) bw = std::max(bw, std::abs(pos[i] - pos[j]));
    return bw;
}

} // namespace

TEST(SparseLinalg, CuthillMcKeeReducesBandwidth) {
    std::mt19937 rng(1);
    std::vector<int> identity;
    int              improved = 0, total = 0;
    for (int t = 0; t < 200; ++t) {
        const int n = 8 + rng() % 30;
        const auto g = random_graph(rng, n, 0.12);
        identity.resize(n);
        for (int i = 0; i < n; ++i) identity[i] = i;

        const auto perm = cuthill_mckee(g, false);
        ASSERT_TRUE(is_permutation(perm, n));
        const int bw0 = bandwidth(g, identity), bw1 = bandwidth(g, perm);
        ++total;
        if (bw1 <= bw0) ++improved;

        EXPECT_TRUE(is_permutation(cuthill_mckee(g, true), n)); // RCM also a valid permutation
    }
    EXPECT_GT(improved, total * 9 / 10) << "Cuthill-McKee should not worsen bandwidth in the vast majority";
}

namespace {
// Count fill (nonzeros of L below diagonal) for a natural-order symbolic Cholesky of a permuted graph.
int fill_count(const std::vector<std::vector<int>>& adj, const std::vector<int>& perm) {
    const int        n = static_cast<int>(adj.size());
    std::vector<int> pos(n);
    for (int k = 0; k < n; ++k) pos[perm[k]] = k;
    // Relabel the graph by perm, then symbolic Cholesky.
    std::vector<std::vector<int>> g(n);
    for (int i = 0; i < n; ++i)
        for (int j : adj[i]) g[pos[i]].push_back(pos[j]);
    const auto L = symbolic_cholesky(g);
    int total = 0;
    for (const auto& row : L) total += static_cast<int>(row.size());
    return total;
}
} // namespace

TEST(SparseLinalg, MinimumDegreeReducesFillVsNatural) {
    std::mt19937 rng(2);
    std::vector<int> identity;
    long long        better = 0, ties = 0, worse = 0;
    for (int t = 0; t < 200; ++t) {
        const int n = 10 + rng() % 25;
        const auto g = random_graph(rng, n, 0.10);
        identity.resize(n);
        for (int i = 0; i < n; ++i) identity[i] = i;

        const auto perm = minimum_degree_ordering(g);
        ASSERT_TRUE(is_permutation(perm, n));

        const int f_nat = fill_count(g, identity);
        const int f_md  = fill_count(g, perm);
        if (f_md < f_nat) ++better; else if (f_md == f_nat) ++ties; else ++worse;
    }
    EXPECT_GT(better + ties, worse * 3) << "minimum degree should usually not increase fill";
}

TEST(SparseLinalg, SymbolicCholeskyCoversNumericFactor) {
    std::mt19937 rng(3);
    for (int t = 0; t < 100; ++t) {
        const int  n = 6 + rng() % 12;
        const auto g = random_graph(rng, n, 0.25);
        const auto Lpat = symbolic_cholesky(g);

        // Build an SPD matrix with this pattern: M = pattern*small + diagonal dominance.
        std::vector<std::vector<double>> M(n, std::vector<double>(n, 0.0));
        std::uniform_real_distribution<double> val(-1.0, 1.0);
        for (int i = 0; i < n; ++i)
            for (int j : g[i]) if (j > i) { const double v = val(rng); M[i][j] = v; M[j][i] = v; }
        for (int i = 0; i < n; ++i) M[i][i] = n + 1.0; // strong diagonal -> SPD

        // Numeric Cholesky; L(i,j) for j<i.
        std::vector<std::vector<double>> L(n, std::vector<double>(n, 0.0));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j <= i; ++j) {
                double s = M[i][j];
                for (int k = 0; k < j; ++k) s -= L[i][k] * L[j][k];
                L[i][j] = (i == j) ? std::sqrt(s) : s / L[j][j];
            }
        // Every numeric nonzero below the diagonal must be in the symbolic pattern.
        for (int i = 0; i < n; ++i) {
            std::set<int> pat(Lpat[i].begin(), Lpat[i].end());
            for (int j = 0; j < i; ++j)
                if (std::fabs(L[i][j]) > 1e-9) EXPECT_TRUE(pat.count(j)) << "numeric L(" << i << "," << j << ") not predicted";
        }
    }
}

TEST(SparseLinalg, CannonEqualsNaiveMatmul) {
    std::mt19937                          rng(4);
    std::uniform_real_distribution<double> val(-5.0, 5.0);
    for (int t = 0; t < 300; ++t) {
        const int          n = 1 + rng() % 9;
        std::vector<double> A(n * n), B(n * n);
        for (double& x : A) x = val(rng);
        for (double& x : B) x = val(rng);
        std::vector<double> want(n * n, 0.0);
        for (int i = 0; i < n; ++i)
            for (int k = 0; k < n; ++k)
                for (int j = 0; j < n; ++j) want[i * n + j] += A[i * n + k] * B[k * n + j];
        const auto got = cannon_matmul(A, B, n);
        for (int e = 0; e < n * n; ++e) EXPECT_NEAR(got[e], want[e], 1e-9) << "n=" << n << " e=" << e;
    }
}
