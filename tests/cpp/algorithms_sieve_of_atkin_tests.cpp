#include <gtest/gtest.h>

#include <datamunge/algorithms/primality.hpp>
#include <datamunge/algorithms/sieve_of_atkin.hpp>
#include <datamunge/algorithms/sieve_of_eratosthenes.hpp>

#include <cmath>
#include <stdexcept>

using namespace datamunge::algorithms;

TEST(SieveOfAtkin, MatchesEratosthenesAcrossLimits) {
    for (std::uint64_t n : {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 11u, 12u, 13u, 59u, 60u, 61u, 1000u, 99991u, 250000u})
        EXPECT_EQ(atkin_primes(n), primes_up_to(n)) << n;
}

TEST(SieveOfAtkin, ParityTheoremOnSquarefreeNumbers) {
    // For squarefree n > 3 with n mod 12 in {1, 5, 7, 11}: n is prime iff its form count is odd.
    const auto sieve = prime_sieve(20000);
    for (std::uint64_t n = 5; n <= 20000; ++n) {
        AtkinForm form;
        if (!atkin_form_for(n, form)) continue;
        bool squarefree = true;
        for (std::uint64_t d = 2; d * d <= n; ++d)
            if (n % (d * d) == 0) squarefree = false;
        if (!squarefree) continue;
        EXPECT_EQ(atkin_form_representations(n, form) % 2 == 1, static_cast<bool>(sieve[n])) << n;
    }
}

TEST(SieveOfAtkin, RepresentationCountsMatchDivisorFormula) {
    // n = 1 mod 4: #{x, y >= 1 : 4x^2 + y^2 = n} = (d1(n) - d3(n)) / 2 for non-square n.
    for (std::uint64_t n = 5; n < 5000; n += 4) {
        const auto r = static_cast<std::uint64_t>(std::sqrt(static_cast<double>(n)));
        if (r * r == n) continue;
        long d1 = 0, d3 = 0;
        for (std::uint64_t d = 1; d <= n; ++d)
            if (n % d == 0) (d % 4 == 1 ? d1 : (d % 4 == 3 ? d3 : d1)) += (d % 2 == 1);
        EXPECT_EQ(static_cast<long>(atkin_form_representations(n, AtkinForm::FourXSquaredPlusYSquared)), (d1 - d3) / 2)
            << n;
    }
    EXPECT_EQ(atkin_form_representations(11, AtkinForm::ThreeXSquaredMinusYSquared), 1u);  // 3*4 - 1
    EXPECT_EQ(atkin_form_representations(7, AtkinForm::ThreeXSquaredPlusYSquared), 1u);    // 3 + 4
}

TEST(SieveOfAtkin, WorkCounts) {
    const auto a = atkin_sieve_work(1000000);
    const auto e = eratosthenes_sieve_work(1000000);
    EXPECT_EQ(a.primes, 78498u);
    EXPECT_EQ(e.primes, 78498u);
    EXPECT_NEAR(static_cast<double>(a.lattice_points) / 1e6, 1.036, 0.01);
    EXPECT_LT(a.toggles, a.lattice_points);
    EXPECT_GT(e.crossings, 2000000u);
    EXPECT_LT(e.crossings, 2600000u);
}

TEST(PrimeCounting, LiAndRiemannR) {
    EXPECT_NEAR(logarithmic_integral(1e6), 78627.5491594622, 1e-6);
    EXPECT_NEAR(logarithmic_integral(2.0), 1.04516378011749, 1e-12);
    EXPECT_NEAR(riemann_prime_count(1e6), 78527.3994291277, 1e-6);
    EXPECT_NEAR(riemann_prime_count(1e9), 50847455.4277214, 1e-3);
    EXPECT_THROW(logarithmic_integral(1.0), std::invalid_argument);
}
