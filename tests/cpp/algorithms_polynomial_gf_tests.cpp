#include <gtest/gtest.h>

#include <datamunge/algorithms/polynomial_gf.hpp>

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

using namespace datamunge::algorithms;

namespace {

std::int64_t mod(std::int64_t a, std::int64_t p) { return ((a % p) + p) % p; }

GFPoly reduce(GFPoly f, std::int64_t p) {
    for (auto& c : f) c = mod(c, p);
    while (!f.empty() && f.back() == 0) f.pop_back();
    return f;
}

GFPoly mul(const GFPoly& a, const GFPoly& b, std::int64_t p) {
    if (a.empty() || b.empty()) return {};
    GFPoly r(a.size() + b.size() - 1, 0);
    for (std::size_t i = 0; i < a.size(); ++i)
        for (std::size_t j = 0; j < b.size(); ++j)
            r[i + j] = mod(r[i + j] + a[i] * b[j], p);
    return reduce(r, p);
}

GFPoly add(const GFPoly& a, const GFPoly& b, std::int64_t p) {
    GFPoly r(std::max(a.size(), b.size()), 0);
    for (std::size_t i = 0; i < a.size(); ++i) r[i] = a[i];
    for (std::size_t i = 0; i < b.size(); ++i) r[i] = mod(r[i] + b[i], p);
    return reduce(r, p);
}

std::int64_t eval(const GFPoly& f, std::int64_t x, std::int64_t p) {
    std::int64_t acc = 0;
    for (std::size_t i = f.size(); i-- > 0;) acc = mod(acc * x + f[i], p);
    return acc;
}

GFPoly random_poly(std::mt19937_64& rng, int deg, std::int64_t p) {
    GFPoly f(deg + 1);
    for (auto& c : f) c = static_cast<std::int64_t>(rng() % static_cast<std::uint64_t>(p));
    f.back() = 1 + static_cast<std::int64_t>(rng() % static_cast<std::uint64_t>(p - 1)); // nonzero lead
    return f;
}

} // namespace

TEST(PolynomialGF, DivModIdentity) {
    std::mt19937_64                 rng(1);
    const std::vector<std::int64_t> primes = {2, 3, 5, 7, 11, 13, 101};
    for (std::int64_t p : primes) {
        for (int t = 0; t < 5000; ++t) {
            const int    da = static_cast<int>(rng() % 12);
            const int    db = 1 + static_cast<int>(rng() % 6);
            const GFPoly a  = random_poly(rng, da, p);
            const GFPoly b  = random_poly(rng, db, p);
            const auto   qr = poly_divmod_gf(a, b, p);
            // a == q*b + r  and  deg(r) < deg(b)
            const GFPoly recon = add(mul(qr.quotient, b, p), qr.remainder, p);
            EXPECT_EQ(recon, reduce(a, p)) << "p=" << p;
            EXPECT_LT(static_cast<int>(qr.remainder.size()) - 1, static_cast<int>(b.size()) - 1) << "p=" << p;
        }
    }
}

TEST(PolynomialGF, ChienFindsAllRoots) {
    std::mt19937_64                 rng(2);
    const std::vector<std::int64_t> primes = {5, 7, 11, 13, 31, 101};
    for (std::int64_t p : primes) {
        for (int t = 0; t < 300; ++t) {
            const GFPoly f = random_poly(rng, 1 + static_cast<int>(rng() % 6), p);
            // Brute-force reference set of roots.
            std::vector<std::int64_t> want;
            for (std::int64_t r = 0; r < p; ++r)
                if (eval(f, r, p) == 0) want.push_back(r);
            std::vector<std::int64_t> got = chien_search(f, p);
            std::sort(got.begin(), got.end());
            EXPECT_EQ(got, want) << "p=" << p;
        }
    }
}

TEST(PolynomialGF, ChienKnownRoots) {
    // x^2 - 1 = (x-1)(x+1) over GF(7): roots {1, 6}.
    auto r = chien_search({mod(-1, 7), 0, 1}, 7);
    std::sort(r.begin(), r.end());
    EXPECT_EQ(r, (std::vector<std::int64_t>{1, 6}));
    // x^2 + 1 has no root mod 7 (7 = 3 mod 4).
    EXPECT_TRUE(chien_search({1, 0, 1}, 7).empty());
}

TEST(PolynomialGF, CantorZassenhausSplitsDistinctRoots) {
    std::mt19937_64                 rng(3);
    const std::vector<std::int64_t> primes = {3, 5, 7, 11, 13, 101};
    for (std::int64_t p : primes) {
        for (int t = 0; t < 300; ++t) {
            // Product over a set of DISTINCT roots => monic squarefree, all linear.
            std::vector<std::int64_t> roots;
            const int                 k = 2 + static_cast<int>(rng() % 4);
            for (int i = 0; i < k && static_cast<std::int64_t>(roots.size()) < p; ++i) {
                std::int64_t r = static_cast<std::int64_t>(rng() % static_cast<std::uint64_t>(p));
                if (std::find(roots.begin(), roots.end(), r) == roots.end()) roots.push_back(r);
            }
            GFPoly f{1};
            for (std::int64_t r : roots) f = mul(f, GFPoly{mod(-r, p), 1}, p); // (x - r)

            const auto factors = cantor_zassenhaus(f, p);
            // Every factor must be linear (x - r) for one of our roots.
            EXPECT_EQ(factors.size(), roots.size()) << "p=" << p;
            GFPoly prod{1};
            for (const auto& fac : factors) {
                EXPECT_EQ(fac.size(), 2u) << "p=" << p;      // linear
                EXPECT_EQ(fac.back(), 1);                    // monic
                prod = mul(prod, fac, p);
            }
            EXPECT_EQ(prod, reduce(f, p)) << "p=" << p;
        }
    }
}

TEST(PolynomialGF, CantorZassenhausKnown) {
    // x^2 - 3 over GF(7): 3 is not a QR mod 7 (squares are {1,2,4}), so it is
    // irreducible -> one factor.
    auto f1 = cantor_zassenhaus({mod(-3, 7), 0, 1}, 7);
    EXPECT_EQ(f1.size(), 1u);
    // x^2 - 1 over GF(7) = (x-1)(x+1): two linear factors.
    auto f2 = cantor_zassenhaus({mod(-1, 7), 0, 1}, 7);
    EXPECT_EQ(f2.size(), 2u);
    // Product of two distinct irreducible quadratics over GF(3):
    //   (x^2+1)(x^2+x+2), both irreducible mod 3 -> two degree-2 factors.
    GFPoly a{1, 0, 1};    // x^2 + 1
    GFPoly b{2, 1, 1};    // x^2 + x + 2
    GFPoly prod = mul(a, b, 3);
    auto   f3   = cantor_zassenhaus(prod, 3);
    EXPECT_EQ(f3.size(), 2u);
    for (const auto& fac : f3) EXPECT_EQ(fac.size(), 3u); // both quadratic
}
