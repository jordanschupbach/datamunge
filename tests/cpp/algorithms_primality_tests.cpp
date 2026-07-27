#include <gtest/gtest.h>

#include <datamunge/algorithms/primality.hpp>

#include <cstdint>
#include <numeric>
#include <vector>

using datamunge::algorithms::atkin_primes;
using datamunge::algorithms::fermat_probable_prime;
using datamunge::algorithms::lucas_primality_test;
using datamunge::algorithms::sundaram_primes;

namespace {

bool is_prime(std::uint64_t n) {
    if (n < 2) return false;
    for (std::uint64_t d = 2; d * d <= n; ++d)
        if (n % d == 0) return false;
    return true;
}

std::vector<std::uint64_t> primes_upto(std::uint64_t n) {
    std::vector<std::uint64_t> out;
    for (std::uint64_t k = 2; k <= n; ++k)
        if (is_prime(k)) out.push_back(k);
    return out;
}

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

} // namespace

// ------------------------------------------------------------------------------------------------
// Fermat
// ------------------------------------------------------------------------------------------------

TEST(Fermat, PrimesAlwaysPassCompositesCaught) {
    for (std::uint64_t p : primes_upto(3000)) EXPECT_TRUE(fermat_probable_prime(p)) << p; // one-sided: never wrong on primes
    for (std::uint64_t c : {4u, 6u, 9u, 15u, 25u, 49u, 91u, 133u, 200u}) EXPECT_FALSE(fermat_probable_prime(c)) << c;
    EXPECT_FALSE(fermat_probable_prime(0));
    EXPECT_FALSE(fermat_probable_prime(1));
    EXPECT_TRUE(fermat_probable_prime(2));
}

TEST(Fermat, CarmichaelNumberFoolsEveryCoprimeBase) {
    // 561 = 3*11*17 is a Carmichael number: a^560 == 1 (mod 561) for every base coprime to 561,
    // so a Fermat test restricted to coprime bases cannot distinguish it from a prime.
    const std::uint64_t n = 561;
    EXPECT_FALSE(is_prime(n));
    for (std::uint64_t a = 2; a < n; ++a)
        if (std::gcd(a, n) == 1) EXPECT_EQ(powmod(a, n - 1, n), 1u) << "a=" << a;
}

// ------------------------------------------------------------------------------------------------
// Lucas (deterministic)
// ------------------------------------------------------------------------------------------------

TEST(Lucas, MatchesTrialDivision) {
    for (std::uint64_t n = 2; n <= 3000; ++n) EXPECT_EQ(lucas_primality_test(n), is_prime(n)) << n;
    EXPECT_FALSE(lucas_primality_test(0));
    EXPECT_FALSE(lucas_primality_test(1));
    // A few larger primes and composites.
    for (std::uint64_t p : {104729u, 1299709u}) EXPECT_TRUE(lucas_primality_test(p));
    for (std::uint64_t c : {104730u, 1299707u}) EXPECT_EQ(lucas_primality_test(c), is_prime(c));
}

// ------------------------------------------------------------------------------------------------
// Sieves of Sundaram and Atkin
// ------------------------------------------------------------------------------------------------

TEST(Sundaram, GeneratesExactlyThePrimes) {
    for (std::uint64_t N : {0u, 1u, 2u, 3u, 4u, 10u, 97u, 100u, 1000u, 10000u})
        EXPECT_EQ(sundaram_primes(N), primes_upto(N)) << "N=" << N;
}

TEST(Atkin, GeneratesExactlyThePrimes) {
    for (std::uint64_t N : {0u, 1u, 2u, 3u, 4u, 5u, 10u, 97u, 100u, 1000u, 10000u})
        EXPECT_EQ(atkin_primes(N), primes_upto(N)) << "N=" << N;
}

TEST(Sieves, AgreeWithEachOther) {
    for (std::uint64_t N : {50u, 500u, 5000u}) EXPECT_EQ(atkin_primes(N), sundaram_primes(N)) << "N=" << N;
}
