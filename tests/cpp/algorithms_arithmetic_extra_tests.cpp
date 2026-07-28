#include <gtest/gtest.h>

#include <datamunge/algorithms/arithmetic_extra.hpp>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

using namespace datamunge::algorithms;

namespace {
bool is_prime(std::uint64_t n) {
    if (n < 2) return false;
    for (std::uint64_t d = 2; d * d <= n; ++d)
        if (n % d == 0) return false;
    return true;
}
} // namespace

TEST(Arithmetic, GoldschmidtMatchesTrueQuotient) {
    std::mt19937                          rng(1);
    std::uniform_real_distribution<double> d(-1e6, 1e6);
    for (int t = 0; t < 100000; ++t) {
        const double n = d(rng);
        double       den = d(rng);
        if (den == 0.0) den = 1.0;
        EXPECT_NEAR(goldschmidt_division(n, den), n / den, std::fabs(n / den) * 1e-12 + 1e-12);
    }
    EXPECT_NEAR(goldschmidt_division(1.0, 3.0), 1.0 / 3.0, 1e-15);
    EXPECT_NEAR(goldschmidt_division(-22.0, 7.0), -22.0 / 7.0, 1e-14);
}

TEST(Arithmetic, MontgomeryMultiplyEqualsModMul) {
    std::mt19937_64 rng(2);
    for (int t = 0; t < 200000; ++t) {
        std::uint64_t n = (rng() % 100000000ULL) | 1ULL; // odd
        if (n < 3) n = 3;
        const std::uint64_t a = rng() % n;
        const std::uint64_t b = rng() % n;
        const std::uint64_t want = static_cast<std::uint64_t>((static_cast<__uint128_t>(a) * b) % n);
        EXPECT_EQ(montgomery_multiply(a, b, n), want) << "a=" << a << " b=" << b << " n=" << n;
    }
    // large odd modulus near 2^63
    const std::uint64_t n = 9223372036854775783ULL; // prime
    EXPECT_EQ(montgomery_multiply(n - 1, n - 1, n),
              static_cast<std::uint64_t>((static_cast<__uint128_t>(n - 1) * (n - 1)) % n));
}

TEST(Arithmetic, CipollaFindsModularSquareRoot) {
    const std::uint64_t primes[] = {13, 101, 1009, 65537, 1000003, 2147483647ULL};
    std::mt19937_64     rng(3);
    for (std::uint64_t p : primes) {
        int residues = 0, nonresidues = 0;
        for (int t = 0; t < 200; ++t) {
            const std::uint64_t x = rng() % p;
            const std::uint64_t n = static_cast<std::uint64_t>((static_cast<__uint128_t>(x) * x) % p);
            const long long     r = cipolla_sqrt(n, p);
            ASSERT_GE(r, 0) << "n=" << n << " is a residue (x^2) but Cipolla returned -1, p=" << p;
            EXPECT_EQ(static_cast<std::uint64_t>((static_cast<__uint128_t>(r) * r) % p), n) << "p=" << p;
            ++residues;
        }
        EXPECT_GT(residues, 0);
        // A guaranteed non-residue check for p = 4k+3 (a small non-residue exists).
        for (std::uint64_t n = 2; n < p; ++n) {
            const long long r = cipolla_sqrt(n, p);
            if (r < 0) { ++nonresidues; break; } // found a non-residue -> correctly rejected
        }
        if (p % 4 == 3) EXPECT_GT(nonresidues, 0);
    }
}

TEST(Arithmetic, AdditionChainIsValidAndShort) {
    // Verify: every element is the sum of two earlier ones, chain starts at 1, ends at e.
    auto valid = [](const std::vector<std::uint64_t>& c, std::uint64_t e) {
        if (c.empty() || c.front() != 1 || c.back() != e) return false;
        for (std::size_t k = 1; k < c.size(); ++k) {
            bool ok = false;
            for (std::size_t i = 0; i < k && !ok; ++i)
                for (std::size_t j = 0; j <= i && !ok; ++j)
                    if (c[i] + c[j] == c[k]) ok = true;
            if (!ok) return false;
        }
        return true;
    };
    for (std::uint64_t e = 1; e <= 300; ++e) {
        const auto c = addition_chain(e);
        EXPECT_TRUE(valid(c, e)) << "invalid chain for e=" << e;
    }
    // Known optimal lengths (multiplications = chain.size()-1):
    EXPECT_EQ(addition_chain(15).size() - 1, 5u); // beats binary's 6
    EXPECT_EQ(addition_chain(1).size(), 1u);
    EXPECT_EQ(addition_chain(2).size() - 1, 1u);
    EXPECT_LE(addition_chain(255).size() - 1, 10u); // optimal is 10 (binary is 13)
}
