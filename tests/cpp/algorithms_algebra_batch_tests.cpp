#include <gtest/gtest.h>

#include <datamunge/algorithms/berlekamp.hpp>
#include <datamunge/algorithms/lll.hpp>
#include <datamunge/algorithms/pollard_kangaroo.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

using namespace datamunge::algorithms;

// ---------------- LLL ----------------

namespace {

double dot(const std::vector<long long>& a, const std::vector<long long>& b) {
    double s = 0;
    for (std::size_t i = 0; i < a.size(); ++i) s += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    return s;
}

// Gram determinant det(B B^T): invariant of the lattice under unimodular change.
double gram_det(const std::vector<std::vector<long long>>& b) {
    const std::size_t n = b.size();
    std::vector<std::vector<double>> g(n, std::vector<double>(n));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) g[i][j] = dot(b[i], b[j]);
    // Gaussian elimination determinant.
    double det = 1;
    for (std::size_t i = 0; i < n; ++i) {
        std::size_t piv = i;
        for (std::size_t r = i + 1; r < n; ++r)
            if (std::fabs(g[r][i]) > std::fabs(g[piv][i])) piv = r;
        if (std::fabs(g[piv][i]) < 1e-9) return 0;
        std::swap(g[i], g[piv]);
        if (piv != i) det = -det;
        det *= g[i][i];
        for (std::size_t r = i + 1; r < n; ++r) {
            const double f = g[r][i] / g[i][i];
            for (std::size_t c = i; c < n; ++c) g[r][c] -= f * g[i][c];
        }
    }
    return det;
}

// Verify the output is LLL-reduced (size-reduced + Lovasz), delta = 0.75.
bool is_lll_reduced(const std::vector<std::vector<long long>>& b, double delta) {
    const std::size_t n = b.size();
    std::vector<std::vector<double>> bstar(n), mu(n, std::vector<double>(n, 0));
    auto d = [&](const std::vector<double>& x, const std::vector<double>& y) {
        double s = 0; for (std::size_t i = 0; i < x.size(); ++i) s += x[i] * y[i]; return s;
    };
    std::vector<std::vector<double>> bd(n);
    for (std::size_t i = 0; i < n; ++i) { bd[i].assign(b[i].begin(), b[i].end()); }
    for (std::size_t i = 0; i < n; ++i) {
        bstar[i] = bd[i];
        for (std::size_t j = 0; j < i; ++j) {
            mu[i][j] = d(bd[i], bstar[j]) / d(bstar[j], bstar[j]);
            for (std::size_t k = 0; k < bd[i].size(); ++k) bstar[i][k] -= mu[i][j] * bstar[j][k];
        }
    }
    for (std::size_t i = 1; i < n; ++i)
        for (std::size_t j = 0; j < i; ++j)
            if (std::fabs(mu[i][j]) > 0.5 + 1e-6) return false;
    for (std::size_t i = 1; i < n; ++i) {
        const double lhs = d(bstar[i], bstar[i]);
        const double rhs = (delta - mu[i][i - 1] * mu[i][i - 1]) * d(bstar[i - 1], bstar[i - 1]);
        if (lhs < rhs - 1e-6) return false;
    }
    return true;
}

} // namespace

TEST(AlgebraBatch, LllReducedAndLatticePreserved) {
    std::mt19937_64 rng(1);
    for (int t = 0; t < 200; ++t) {
        const int n = 2 + static_cast<int>(rng() % 3); // dimension 2..4
        std::vector<std::vector<long long>> b(n, std::vector<long long>(n));
        for (auto& row : b)
            for (auto& c : row) c = static_cast<long long>(rng() % 200) - 100;
        const double det0 = std::fabs(gram_det(b));
        if (det0 < 1e-6) continue; // skip degenerate

        const auto r = lll_reduce(b, 0.75);
        EXPECT_TRUE(is_lll_reduced(r, 0.75)) << "not reduced, t=" << t;
        // Same lattice => same Gram determinant.
        EXPECT_NEAR(std::fabs(gram_det(r)), det0, det0 * 1e-6 + 1e-6) << "det changed, t=" << t;
    }
}

TEST(AlgebraBatch, LllKnownShortVector) {
    // A skewed basis of Z^2 should reduce to something close to the standard basis.
    std::vector<std::vector<long long>> b = {{1, 1}, {1, 0}};
    auto r = lll_reduce(b, 0.75);
    EXPECT_TRUE(is_lll_reduced(r, 0.75));
    // Shortest reduced vector has norm^2 == 1.
    double best = 1e18;
    for (auto& v : r) best = std::min(best, dot(v, v));
    EXPECT_NEAR(best, 1.0, 1e-9);
}

// ---------------- Berlekamp root finding ----------------

namespace {

std::int64_t modp(std::int64_t a, std::int64_t p) { return ((a % p) + p) % p; }
std::int64_t evalp(const BPoly& f, std::int64_t x, std::int64_t p) {
    std::int64_t acc = 0;
    for (std::size_t i = f.size(); i-- > 0;) acc = modp(acc * x + f[i], p);
    return acc;
}

} // namespace

TEST(AlgebraBatch, BerlekampRootsMatchBruteForce) {
    std::mt19937_64                 rng(2);
    const std::vector<std::int64_t> primes = {5, 7, 11, 13, 31, 101, 257};
    for (std::int64_t p : primes) {
        for (int t = 0; t < 300; ++t) {
            const int deg = 1 + static_cast<int>(rng() % 6);
            BPoly     f(deg + 1);
            for (auto& c : f) c = static_cast<std::int64_t>(rng() % static_cast<std::uint64_t>(p));
            f.back() = 1 + static_cast<std::int64_t>(rng() % static_cast<std::uint64_t>(p - 1));

            std::vector<std::int64_t> want;
            for (std::int64_t r = 0; r < p; ++r)
                if (evalp(f, r, p) == 0) want.push_back(r);
            const auto got = berlekamp_roots(f, p);
            EXPECT_EQ(got, want) << "p=" << p;
        }
    }
}

TEST(AlgebraBatch, BerlekampModularSquareRoot) {
    // Roots of x^2 - a give the modular square roots of a.
    const std::int64_t p = 1009;
    for (std::int64_t r = 1; r < 50; ++r) {
        const std::int64_t a    = modp(r * r, p);
        const auto         root = berlekamp_roots({modp(-a, p), 0, 1}, p);
        ASSERT_EQ(root.size(), 2u) << "a=" << a;
        for (std::int64_t s : root) EXPECT_EQ(modp(s * s, p), a);
    }
    // A non-residue has no square root: x^2 - 3 mod 7 (3 not a QR mod 7).
    EXPECT_TRUE(berlekamp_roots({modp(-3, 7), 0, 1}, 7).empty());
}

// ---------------- Pollard kangaroo ----------------

namespace {
std::uint64_t pmod(std::uint64_t a, std::uint64_t e, std::uint64_t m) {
    unsigned __int128 r = 1; a %= m;
    while (e) { if (e & 1) r = r * a % m; a = (unsigned __int128)a * a % m; e >>= 1; }
    return static_cast<std::uint64_t>(r);
}
} // namespace

TEST(AlgebraBatch, KangarooRecoversDiscreteLog) {
    // Prime P, generator g; pick secret x in an interval, solve g^x = h.
    struct Case { std::uint64_t P, g; };
    const std::vector<Case> cases = {{1000003, 2}, {1000033, 5}, {2000003, 2}, {104729, 6}};
    std::mt19937_64         rng(7);
    for (const auto& c : cases) {
        for (int t = 0; t < 40; ++t) {
            const std::uint64_t a = 1 + rng() % 100000;
            const std::uint64_t b = a + 1 + rng() % 50000; // interval [a, b]
            const std::uint64_t x = a + rng() % (b - a + 1);
            const std::uint64_t h = pmod(c.g, x, c.P);
            const auto          r = pollard_kangaroo(c.g, h, c.P, a, b);
            ASSERT_TRUE(r.found) << "P=" << c.P << " x=" << x;
            // Solution must satisfy the congruence and lie in the interval
            // (the log is unique mod ord(g), so we check g^x, not x itself).
            EXPECT_EQ(pmod(c.g, r.x, c.P), h) << "P=" << c.P << " x=" << x;
            EXPECT_GE(r.x, a);
            EXPECT_LE(r.x, b);
        }
    }
}
