#include <gtest/gtest.h>

#include <datamunge/algorithms/number_theory.hpp>

#include <cstdint>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using datamunge::algorithms::baby_step_giant_step;
using datamunge::algorithms::binary_gcd;
using datamunge::algorithms::DiscreteLog;
using datamunge::algorithms::karatsuba_multiply;
using datamunge::algorithms::ModSqrt;
using datamunge::algorithms::tonelli_shanks;

namespace {

std::uint64_t powmod(std::uint64_t b, std::uint64_t e, std::uint64_t m) {
    std::uint64_t r = 1 % m;
    b %= m;
    while (e) {
        if (e & 1ULL) r = static_cast<std::uint64_t>((static_cast<__uint128_t>(r) * b) % m);
        b = static_cast<std::uint64_t>((static_cast<__uint128_t>(b) * b) % m);
        e >>= 1;
    }
    return r;
}

bool is_prime(std::uint64_t n) {
    if (n < 2) return false;
    for (std::uint64_t d = 2; d * d <= n; ++d)
        if (n % d == 0) return false;
    return true;
}

// Independent schoolbook multiplier on decimal strings.
std::string ref_multiply(const std::string& A, const std::string& B) {
    std::vector<int> a, b;
    for (auto it = A.rbegin(); it != A.rend(); ++it) a.push_back(*it - '0');
    for (auto it = B.rbegin(); it != B.rend(); ++it) b.push_back(*it - '0');
    std::vector<long long> acc(a.size() + b.size(), 0);
    for (std::size_t i = 0; i < a.size(); ++i)
        for (std::size_t j = 0; j < b.size(); ++j) acc[i + j] += static_cast<long long>(a[i]) * b[j];
    std::string out;
    long long   carry = 0;
    std::vector<int> res;
    for (std::size_t i = 0; i < acc.size(); ++i) {
        long long cur = acc[i] + carry;
        res.push_back(static_cast<int>(cur % 10));
        carry = cur / 10;
    }
    while (carry) { res.push_back(static_cast<int>(carry % 10)); carry /= 10; }
    while (res.size() > 1 && res.back() == 0) res.pop_back();
    for (auto it = res.rbegin(); it != res.rend(); ++it) out.push_back(static_cast<char>('0' + *it));
    return out;
}

std::string random_number(std::mt19937& rng, std::size_t len) {
    std::uniform_int_distribution<int> d(0, 9), first(1, 9);
    std::string                        s(1, static_cast<char>('0' + first(rng)));
    for (std::size_t i = 1; i < len; ++i) s.push_back(static_cast<char>('0' + d(rng)));
    return s;
}

} // namespace

// ------------------------------------------------------------------------------------------------
// Binary GCD
// ------------------------------------------------------------------------------------------------

TEST(BinaryGCD, MatchesStdGcd) {
    EXPECT_EQ(binary_gcd(0, 0), 0u);
    EXPECT_EQ(binary_gcd(12, 0), 12u);
    EXPECT_EQ(binary_gcd(0, 7), 7u);
    EXPECT_EQ(binary_gcd(54, 24), 6u);
    EXPECT_EQ(binary_gcd(1071, 462), 21u);

    std::mt19937_64 rng(1);
    for (int t = 0; t < 200000; ++t) {
        const std::uint64_t a = rng() % 1000000000ULL;
        const std::uint64_t b = rng() % 1000000000ULL;
        EXPECT_EQ(binary_gcd(a, b), std::gcd(a, b));
    }
}

// ------------------------------------------------------------------------------------------------
// Tonelli-Shanks
// ------------------------------------------------------------------------------------------------

TEST(TonelliShanks, KnownAndRootProperty) {
    EXPECT_TRUE(tonelli_shanks(10, 13).exists);
    EXPECT_EQ(tonelli_shanks(10, 13).root, 6u); // 6^2 = 36 = 10 (mod 13); smaller of {6,7}
    EXPECT_FALSE(tonelli_shanks(5, 7).exists);  // 5 is a non-residue mod 7

    const std::vector<std::uint64_t> primes = {3, 5, 7, 11, 13, 17, 101, 1009, 7919, 1000003,
                                               2147483647ULL}; // includes p % 8 == 1 cases
    std::mt19937_64 rng(2);
    for (std::uint64_t p : primes) {
        for (int t = 0; t < 500; ++t) {
            const std::uint64_t n  = rng() % p;
            const ModSqrt       ms = tonelli_shanks(n, p);
            const bool          qr = n == 0 || powmod(n, (p - 1) / 2, p) == 1;
            EXPECT_EQ(ms.exists, qr) << "n=" << n << " p=" << p;
            if (ms.exists) {
                EXPECT_LE(ms.root, p - ms.root); // the smaller root
                EXPECT_EQ(powmod(ms.root, 2, p), n % p) << "n=" << n << " p=" << p;
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Karatsuba multiplication
// ------------------------------------------------------------------------------------------------

TEST(Karatsuba, KnownAndEdgeCases) {
    EXPECT_EQ(karatsuba_multiply("0", "12345"), "0");
    EXPECT_EQ(karatsuba_multiply("12345", "0"), "0");
    EXPECT_EQ(karatsuba_multiply("1", "999"), "999");
    EXPECT_EQ(karatsuba_multiply("007", "008"), "56");        // leading zeros tolerated
    EXPECT_EQ(karatsuba_multiply("99999", "99999"), "9999800001");        // (10^5-1)^2
    EXPECT_EQ(karatsuba_multiply("1111111", "1111111"), "1234567654321"); // repunit square
}

TEST(Karatsuba, MatchesSchoolbookReference) {
    std::mt19937 rng(3);
    for (int t = 0; t < 3000; ++t) {
        std::uniform_int_distribution<std::size_t> len(1, 120);
        const std::string a = random_number(rng, len(rng));
        const std::string b = random_number(rng, len(rng));
        EXPECT_EQ(karatsuba_multiply(a, b), ref_multiply(a, b)) << a << " * " << b;
    }
}

// ------------------------------------------------------------------------------------------------
// Baby-step giant-step
// ------------------------------------------------------------------------------------------------

TEST(BabyStepGiantStep, RecoversExponent) {
    std::mt19937_64                  rng(4);
    const std::vector<std::uint64_t> primes = {7, 11, 13, 101, 1009, 7919, 104729};
    for (std::uint64_t p : primes) {
        ASSERT_TRUE(is_prime(p));
        for (int t = 0; t < 200; ++t) {
            const std::uint64_t g = 2 + rng() % (p - 2); // in [2, p-1]
            const std::uint64_t x = rng() % (p - 1);
            const std::uint64_t h = powmod(g, x, p);
            const DiscreteLog   r = baby_step_giant_step(g, h, p);
            ASSERT_TRUE(r.exists) << "g=" << g << " h=" << h << " p=" << p;
            EXPECT_EQ(powmod(g, r.exponent, p), h) << "g=" << g << " p=" << p;
        }
    }
}

TEST(BabyStepGiantStep, RejectsNonMembers) {
    // g = 2 has order 3 modulo 7 (powers {1,2,4}); 3 is not a power of 2, so no logarithm exists.
    const DiscreteLog r = baby_step_giant_step(2, 3, 7);
    EXPECT_FALSE(r.exists);
    EXPECT_TRUE(baby_step_giant_step(2, 4, 7).exists); // 2^2 = 4
    EXPECT_EQ(baby_step_giant_step(2, 1, 7).exponent, 0u); // 2^0 = 1
}
