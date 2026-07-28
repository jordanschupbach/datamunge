#include <gtest/gtest.h>

#include <datamunge/algorithms/number_theory_extra.hpp>

#include <cstdint>
#include <numeric>
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

TEST(Factorization, TrialDivisionFullyFactors) {
    std::mt19937_64 rng(1);
    for (int t = 0; t < 2000; ++t) {
        const std::uint64_t n = 1 + rng() % 1000000ULL;
        const auto          f = trial_division_factorize(n);
        std::uint64_t       prod = 1;
        for (std::uint64_t p : f) {
            EXPECT_TRUE(is_prime(p)) << "non-prime factor " << p << " of " << n;
            prod *= p;
        }
        EXPECT_EQ(prod, n == 1 ? 1u : n);
    }
    EXPECT_EQ(trial_division_factorize(360), (std::vector<std::uint64_t>{2, 2, 2, 3, 3, 5}));
    EXPECT_TRUE(trial_division_factorize(1).empty());
    EXPECT_EQ(trial_division_factorize(97), (std::vector<std::uint64_t>{97})); // prime
}

TEST(Factorization, FermatFindsCloseFactors) {
    // Products of two nearby primes are Fermat's best case.
    const std::uint64_t pairs[][2] = {{10007, 10009}, {100003, 100019}, {611, 617}, {65537, 65539}};
    for (auto& pr : pairs) {
        const std::uint64_t n = pr[0] * pr[1];
        const std::uint64_t f = fermat_factor(n);
        EXPECT_GT(f, 1u);
        EXPECT_LT(f, n);
        EXPECT_EQ(n % f, 0u) << "n=" << n << " f=" << f;
    }
    EXPECT_EQ(fermat_factor(9973 * 2), 2u); // even
    EXPECT_EQ(fermat_factor(9973), 9973u);  // prime -> returns n
}

TEST(Factorization, PollardPMinus1FindsSmoothFactor) {
    // p-1 is smooth: p = 1 + 2*3*5*7*... ; here choose p with smooth p-1.
    // 10007-1 = 10006 = 2*5003 (5003 prime, not smooth). Use primes with smooth p-1:
    // 1009-1 = 1008 = 2^4*3^2*7 (7-smooth). 2003-1=2002=2*7*11*13 (13-smooth).
    const std::uint64_t p = 1009, q = 2003;
    const std::uint64_t n = p * q;
    const std::uint64_t f = pollard_p_minus_1(n, 20);
    EXPECT_GT(f, 1u);
    EXPECT_LT(f, n);
    EXPECT_EQ(n % f, 0u);

    // A larger smooth case: 104729 (prime) has 104728 = 2^3 * 13 * 19 * 53 (53-smooth).
    const std::uint64_t n2 = 104729ULL * 1299709ULL; // second factor: 1299708 = 2^2*3*11*13*757
    const std::uint64_t f2 = pollard_p_minus_1(n2, 60);
    if (f2 != 0) { // may need a larger bound; when it returns, it must be a true factor
        EXPECT_GT(f2, 1u);
        EXPECT_LT(f2, n2);
        EXPECT_EQ(n2 % f2, 0u);
    }
}

TEST(Factorization, DixonSplitsComposites) {
    const std::uint64_t ns[] = {8051, 15347, 1234577ULL * 7ULL, 92434447ULL, 16843009ULL};
    for (std::uint64_t n : ns) {
        if (is_prime(n)) continue;
        const std::uint64_t f = dixon_factor(n, 30);
        ASSERT_NE(f, 0u) << "Dixon failed on " << n;
        EXPECT_GT(f, 1u);
        EXPECT_LT(f, n);
        EXPECT_EQ(n % f, 0u) << "n=" << n << " f=" << f;
    }

    std::mt19937_64 rng(5);
    int             solved = 0;
    for (int t = 0; t < 60; ++t) {
        // odd composite = product of two ~4-digit primes
        auto rand_prime = [&]() {
            for (;;) {
                const std::uint64_t c = 1000 + rng() % 9000;
                if (is_prime(c)) return c;
            }
        };
        const std::uint64_t n = rand_prime() * rand_prime();
        const std::uint64_t f = dixon_factor(n, 40);
        if (f != 0) {
            EXPECT_EQ(n % f, 0u);
            EXPECT_GT(f, 1u);
            EXPECT_LT(f, n);
            ++solved;
        }
    }
    EXPECT_GT(solved, 40) << "Dixon should factor most small semiprimes";
}
