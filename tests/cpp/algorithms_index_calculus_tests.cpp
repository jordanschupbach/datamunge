#include <gtest/gtest.h>

#include <datamunge/algorithms/index_calculus.hpp>

#include <cstdint>
#include <random>
#include <vector>

using namespace datamunge::algorithms;

namespace {

std::uint64_t pmod(std::uint64_t a, std::uint64_t e, std::uint64_t m) {
    unsigned __int128 r = 1;
    a %= m;
    while (e) { if (e & 1) r = r * a % m; a = static_cast<unsigned __int128>(a) * a % m; e >>= 1; }
    return static_cast<std::uint64_t>(r);
}

// Smallest primitive root of the prime p.
std::uint64_t primitive_root(std::uint64_t p) {
    std::uint64_t              n = p - 1;
    std::vector<std::uint64_t> fac;
    for (std::uint64_t f = 2; f * f <= n; ++f)
        if (n % f == 0) { fac.push_back(f); while (n % f == 0) n /= f; }
    if (n > 1) fac.push_back(n);
    for (std::uint64_t g = 2; g < p; ++g) {
        bool ok = true;
        for (std::uint64_t q : fac)
            if (pmod(g, (p - 1) / q, p) == 1) { ok = false; break; }
        if (ok) return g;
    }
    return 0;
}

} // namespace

TEST(IndexCalculus, RecoversDiscreteLog) {
    const std::vector<std::uint64_t> primes = {10007, 100003, 1000003, 1299709, 15485863};
    std::mt19937_64                  rng(1);
    for (std::uint64_t p : primes) {
        const std::uint64_t g = primitive_root(p);
        ASSERT_NE(g, 0u);
        for (int t = 0; t < 5; ++t) {
            const std::uint64_t x = 1 + rng() % (p - 2);
            const std::uint64_t h = pmod(g, x, p);
            const auto          r = index_calculus(g, h, p);
            ASSERT_TRUE(r.found) << "p=" << p << " x=" << x;
            EXPECT_EQ(pmod(g, r.x, p), h) << "p=" << p << " x=" << x;
        }
    }
}

TEST(IndexCalculus, KnownAndTrivial) {
    // g^0 = 1 and g^1 = g.
    const std::uint64_t p = 1000003, g = primitive_root(p);
    EXPECT_TRUE(index_calculus(g, 1, p).found);
    EXPECT_EQ(index_calculus(g, 1, p).x, 0u);
    auto r = index_calculus(g, g, p);
    EXPECT_TRUE(r.found);
    EXPECT_EQ(pmod(g, r.x, p), g % p);

    // A concrete case: recover x from h = g^x.
    const std::uint64_t x = 424242;
    const std::uint64_t h = pmod(g, x, p);
    const auto          got = index_calculus(g, h, p);
    ASSERT_TRUE(got.found);
    EXPECT_EQ(pmod(g, got.x, p), h);
}
