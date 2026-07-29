#include <gtest/gtest.h>

#include <datamunge/algorithms/modular_exponentiation.hpp>

#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

using datamunge::algorithms::mod_pow;

namespace {

// Naive O(exponent) reference: repeated multiply, reducing mod m at every step. The 128-bit
// accumulator keeps each product exact for any 64-bit modulus. Kept small-exponent for speed.
std::uint64_t mod_pow_naive(std::uint64_t base, std::uint64_t exponent, std::uint64_t modulus) {
    if (modulus == 1) return 0;
    std::uint64_t acc = 1 % modulus;
    const std::uint64_t b = base % modulus;
    for (std::uint64_t i = 0; i < exponent; ++i)
        acc = static_cast<std::uint64_t>((static_cast<unsigned __int128>(acc) * b) % modulus);
    return acc;
}

} // namespace

TEST(ModPow, KnownValues) {
    EXPECT_EQ(mod_pow(2, 10, 1000), 24u); // 1024 mod 1000
    EXPECT_EQ(mod_pow(3, 0, 7), 1u);      // any base^0 == 1 (mod > 1)
    EXPECT_EQ(mod_pow(7, 256, 13), 9u);   // 7 has order 12 mod 13; 256 mod 12 == 4; 7^4 == 9
    EXPECT_EQ(mod_pow(7, 13, 1000), 407u);
    EXPECT_EQ(mod_pow(0, 5, 97), 0u);
    EXPECT_EQ(mod_pow(1, 1000000, 97), 1u);
}

TEST(ModPow, MatchesNaiveReferenceOnRandomSmallExponents) {
    std::mt19937_64 rng(20240927);
    std::uniform_int_distribution<std::uint64_t> mod_dist(2, (1u << 30) - 1); // ~30-bit moduli
    std::uniform_int_distribution<std::uint64_t> base_dist(0, ~0ull);
    std::uniform_int_distribution<std::uint64_t> exp_dist(0, 4000);
    for (int trial = 0; trial < 5000; ++trial) {
        const std::uint64_t m = mod_dist(rng);
        const std::uint64_t base = base_dist(rng);
        const std::uint64_t exp = exp_dist(rng);
        EXPECT_EQ(mod_pow(base, exp, m), mod_pow_naive(base, exp, m))
            << "base=" << base << " exp=" << exp << " mod=" << m;
    }
}

TEST(ModPow, FermatLittleTheorem) {
    // For prime p and a not divisible by p, a^(p-1) == 1 (mod p).
    const std::vector<std::uint64_t> primes = {2, 3, 5, 7, 11, 13, 101, 7919, 1000000007};
    std::mt19937_64 rng(12345);
    for (const std::uint64_t p : primes) {
        std::uniform_int_distribution<std::uint64_t> a_dist(1, ~0ull);
        for (int trial = 0; trial < 200; ++trial) {
            std::uint64_t a = a_dist(rng);
            if (a % p == 0) continue; // p must not divide a
            EXPECT_EQ(mod_pow(a, p - 1, p), 1u) << "a=" << a << " p=" << p;
        }
    }
}

TEST(ModPow, LargeModulusOverflowSafety) {
    // Moduli near 2^62: a single 64-bit multiply of two operands this large overflows, so this
    // exercises the 128-bit intermediate. Cross-check against an independent left-to-right
    // square-and-multiply that also uses __int128 mulmod.
    auto independent = [](std::uint64_t base, std::uint64_t exp, std::uint64_t m) {
        auto mulmod = [](std::uint64_t x, std::uint64_t y, std::uint64_t mm) {
            return static_cast<std::uint64_t>((static_cast<unsigned __int128>(x) * y) % mm);
        };
        std::uint64_t acc = 1 % m;
        for (int bit = 63; bit >= 0; --bit) { // scan exponent bits high-to-low
            acc = mulmod(acc, acc, m);
            if ((exp >> bit) & 1u) acc = mulmod(acc, base % m, m);
        }
        return acc;
    };

    std::mt19937_64 rng(999);
    const std::uint64_t m0 = (1ull << 62) + 135; // ~2^62
    std::uniform_int_distribution<std::uint64_t> mod_dist((1ull << 61), (1ull << 62));
    std::uniform_int_distribution<std::uint64_t> big(0, ~0ull);
    for (int trial = 0; trial < 2000; ++trial) {
        const std::uint64_t m = (trial == 0) ? m0 : (mod_dist(rng) | 1ull); // keep odd-ish, all valid
        const std::uint64_t base = big(rng);
        const std::uint64_t exp = big(rng);
        EXPECT_EQ(mod_pow(base, exp, m), independent(base, exp, m))
            << "base=" << base << " exp=" << exp << " mod=" << m;
    }

    // A concrete large-value anchor computed independently above.
    EXPECT_EQ(mod_pow(1234567890123456789ull, 987654321ull, (1ull << 62) + 135),
              independent(1234567890123456789ull, 987654321ull, (1ull << 62) + 135));
}

TEST(ModPow, EdgeCases) {
    // modulus == 1: every residue is 0.
    EXPECT_EQ(mod_pow(0, 0, 1), 0u);
    EXPECT_EQ(mod_pow(123, 456, 1), 0u);
    EXPECT_EQ(mod_pow(~0ull, ~0ull, 1), 0u);

    // exponent == 0: result is 1 for modulus > 1.
    EXPECT_EQ(mod_pow(0, 0, 7), 1u);
    EXPECT_EQ(mod_pow(999999, 0, 2), 1u);

    // base larger than modulus is reduced first.
    EXPECT_EQ(mod_pow(1000, 1, 7), 1000u % 7u);
}

TEST(ModPow, RejectsZeroModulus) {
    EXPECT_THROW((void)mod_pow(2, 3, 0), std::invalid_argument);
    EXPECT_THROW((void)mod_pow(0, 0, 0), std::invalid_argument);
}
