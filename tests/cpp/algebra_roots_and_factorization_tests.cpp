#include <gtest/gtest.h>

#include <datamunge/algebra/algebra.hpp>

#include <algorithm>
#include <cmath>

using namespace datamunge::algebra;

TEST(Sturm, RootCountMatchesKnownRootsOfACubic) {
    // p(x) = (x+2)(x)(x-2) = x^3 - 4x, roots at -2, 0, 2.
    const Polynomial p(std::vector<double>{0.0, -4.0, 0.0, 1.0});
    EXPECT_EQ(sturm_root_count(p, -10.0, 10.0), 3);
    EXPECT_EQ(sturm_root_count(p, -1.0, 1.0), 1);   // only root 0
    EXPECT_EQ(sturm_root_count(p, 0.5, 10.0), 1);   // only root 2
    EXPECT_EQ(sturm_root_count(p, -10.0, -0.5), 1); // only root -2
}

TEST(Sturm, IsolateAndRefineRecoverAllRealRootsOfACubic) {
    const Polynomial p(std::vector<double>{0.0, -4.0, 0.0, 1.0}); // x^3 - 4x
    const auto roots = real_roots(p, 1e-10);
    ASSERT_EQ(roots.size(), 3u);
    std::vector<double> sorted = roots;
    std::sort(sorted.begin(), sorted.end());
    EXPECT_NEAR(sorted[0], -2.0, 1e-8);
    EXPECT_NEAR(sorted[1], 0.0, 1e-8);
    EXPECT_NEAR(sorted[2], 2.0, 1e-8);
}

TEST(Sturm, NoRealRootsIsolatesNothing) {
    // x^2 + 1 has no real roots.
    const Polynomial p(std::vector<double>{1.0, 0.0, 1.0});
    EXPECT_TRUE(real_roots(p).empty());
}

TEST(Descartes, SignChangesBoundsPositiveRoots) {
    // (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6, coefficients (asc): -6, 11, -6, 1 -> signs -,+,-,+: 3 changes.
    const Polynomial p(std::vector<double>{-6.0, 11.0, -6.0, 1.0});
    EXPECT_EQ(descartes_sign_changes(p), 3);
}

TEST(Descartes, NegativeRootBoundViaFlippedCoefficients) {
    // (x+1)(x+2)(x+3) = x^3 + 6x^2 + 11x + 6, all coefficients positive at x -> 0 sign changes
    // for positive roots, and p(-x) = -x^3+6x^2-11x+6 flips to give exactly 3 changes.
    const Polynomial p(std::vector<double>{6.0, 11.0, 6.0, 1.0});
    EXPECT_EQ(descartes_sign_changes(p), 0);
    EXPECT_EQ(descartes_negative_root_bound(p), 3);
}

TEST(SquareFreeFactorization, RecoversMultiplicitiesOfARepeatedRootPolynomial) {
    // p(x) = (x-1)^3 * (x-2)
    const Polynomial x_minus_1(std::vector<double>{-1.0, 1.0});
    const Polynomial x_minus_2(std::vector<double>{-2.0, 1.0});
    Polynomial p = x_minus_1.multiply(x_minus_1).multiply(x_minus_1).multiply(x_minus_2);

    const auto factors = square_free_factorization(p);
    ASSERT_EQ(factors.size(), 2u);

    bool found_mult_1 = false, found_mult_3 = false;
    for (const auto& f : factors) {
        if (f.multiplicity == 1) {
            found_mult_1 = true;
            EXPECT_EQ(f.factor.degree(), 1);
        } else if (f.multiplicity == 3) {
            found_mult_3 = true;
            EXPECT_EQ(f.factor.degree(), 1);
        }
    }
    EXPECT_TRUE(found_mult_1);
    EXPECT_TRUE(found_mult_3);

    // Reconstruct and check it matches p up to a leading-coefficient scale.
    Polynomial reconstructed(1.0);
    for (const auto& f : factors)
        for (int i = 0; i < f.multiplicity; ++i) reconstructed = reconstructed.multiply(f.factor);
    const double scale = p.coefficient(p.degree()) / reconstructed.coefficient(reconstructed.degree());
    reconstructed = reconstructed.scale(scale);
    ASSERT_EQ(reconstructed.degree(), p.degree());
    for (int i = 0; i <= p.degree(); ++i) EXPECT_NEAR(reconstructed.coefficient(i), p.coefficient(i), 1e-8);
}

TEST(SquareFreeFactorization, AlreadySquareFreePolynomialHasAllMultiplicityOne) {
    const Polynomial p(std::vector<double>{-6.0, 11.0, -6.0, 1.0}); // (x-1)(x-2)(x-3)
    const auto factors = square_free_factorization(p);
    for (const auto& f : factors) EXPECT_EQ(f.multiplicity, 1);
}

TEST(Modular, ModPowMatchesBruteForceForSmallCases) {
    EXPECT_DOUBLE_EQ(mod_pow(3.0, 4.0, 7.0), 4.0); // 3^4 = 81 = 11*7+4
    EXPECT_DOUBLE_EQ(mod_pow(2.0, 10.0, 1000.0), 24.0); // 1024 mod 1000
    EXPECT_DOUBLE_EQ(mod_pow(5.0, 0.0, 13.0), 1.0);
}

TEST(Modular, ModInverseSatisfiesProductCongruentToOne) {
    const double inv = mod_inverse(3.0, 7.0); // 3*5=15=1 mod 7
    EXPECT_DOUBLE_EQ(inv, 5.0);
    EXPECT_THROW((void)mod_inverse(2.0, 4.0), std::invalid_argument); // gcd(2,4)=2
}

TEST(Modular, CrtReconstructsAKnownValue) {
    // x = 2 mod 3, x = 3 mod 5, x = 2 mod 7 -> x = 23 (mod 105)
    const double x = crt({2.0, 3.0, 2.0}, {3.0, 5.0, 7.0});
    EXPECT_DOUBLE_EQ(x, 23.0);
}

TEST(Modular, CrtRoundTripsRandomizedCases) {
    for (long long value : {0LL, 1LL, 17LL, 41LL, 59LL}) {
        const std::vector<double> moduli{7.0, 11.0, 13.0};
        std::vector<double> remainders;
        for (double m : moduli) remainders.push_back(static_cast<double>(value % static_cast<long long>(m)));
        EXPECT_DOUBLE_EQ(crt(remainders, moduli), static_cast<double>(value));
    }
}

namespace {
// Multiplies two GF(p) polynomials (ascending-degree long long coefficient vectors) using
// plain integer arithmetic then reducing mod p -- an independent reimplementation from the
// library's own detail::gf_mul, used only to verify berlekamp_factor()'s reconstruction.
std::vector<long long> gf_multiply_reference(const std::vector<long long>& a, const std::vector<long long>& b, long long p) {
    std::vector<long long> r(a.size() + b.size() - 1, 0);
    for (std::size_t i = 0; i < a.size(); ++i)
        for (std::size_t j = 0; j < b.size(); ++j) r[i + j] += a[i] * b[j];
    for (auto& c : r) {
        c %= p;
        if (c < 0) c += p;
    }
    while (r.size() > 1 && r.back() == 0) r.pop_back();
    return r;
}

bool has_root_in_gf_p(const std::vector<long long>& poly, long long p) {
    for (long long x = 0; x < p; ++x) {
        long long value = 0, power = 1;
        for (long long c : poly) {
            value = (value + c * power) % p;
            power = (power * x) % p;
        }
        if (value % p == 0) return true;
    }
    return false;
}
} // namespace

TEST(BerlekampFactor, FactorsAKnownProductOfThreeIrreduciblesModFive) {
    // f = (x-1)(x-2)(x^2+2) mod 5 -- x^2+2 has no root mod 5 (squares mod 5 are {0,1,4}), so
    // it's irreducible; the whole product is square-free (three distinct irreducible factors).
    const std::vector<long long> f{4, 4, 4, 2, 1}; // x^4 + 2x^3 + 4x^2 + 4x + 4, built by hand
                                                    // multiplication of (x-1)(x-2)(x^2+2) mod 5
    const long long p = 5;

    const auto factors = berlekamp_factor(f, p);
    ASSERT_EQ(factors.size(), 3u);

    std::vector<int> degrees;
    for (const auto& factor : factors) degrees.push_back(static_cast<int>(factor.size()) - 1);
    std::sort(degrees.begin(), degrees.end());
    EXPECT_EQ(degrees, (std::vector<int>{1, 1, 2}));

    // Every factor should be monic.
    for (const auto& factor : factors) EXPECT_EQ(factor.back(), 1);

    // No degree-1 factor should coincide with the degree-2 factor having a root (sanity check
    // that the degree-2 piece really is irreducible, i.e. has no root mod p).
    for (const auto& factor : factors)
        if (factor.size() == 3) EXPECT_FALSE(has_root_in_gf_p(factor, p));

    // Reconstruction: product of all factors mod p must equal f (up to leading coefficient,
    // which is already 1 on both sides since f and every factor are monic).
    std::vector<long long> product{1};
    for (const auto& factor : factors) product = gf_multiply_reference(product, factor, p);
    ASSERT_EQ(product.size(), f.size());
    for (std::size_t i = 0; i < f.size(); ++i) EXPECT_EQ(product[i], ((f[i] % p) + p) % p);
}

TEST(BerlekampFactor, IrreduciblePolynomialFactorsAsItself) {
    // x^2 + 1 mod 3: squares mod 3 are {0,1}, so -1=2 is not a square -> irreducible.
    const std::vector<long long> f{1, 0, 1};
    const auto factors = berlekamp_factor(f, 3);
    ASSERT_EQ(factors.size(), 1u);
    EXPECT_EQ(factors[0], f);
}

TEST(ComplexRoots, RecoversAllRootsOfARealCubic) {
    // (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6
    const Polynomial p(std::vector<double>{-6.0, 11.0, -6.0, 1.0});
    const auto roots = complex_roots(p);
    ASSERT_EQ(roots.size(), 3u);

    std::vector<double> real_parts;
    for (const auto& r : roots) {
        EXPECT_NEAR(r.im, 0.0, 1e-6);
        real_parts.push_back(r.re);
    }
    std::sort(real_parts.begin(), real_parts.end());
    EXPECT_NEAR(real_parts[0], 1.0, 1e-8);
    EXPECT_NEAR(real_parts[1], 2.0, 1e-8);
    EXPECT_NEAR(real_parts[2], 3.0, 1e-8);
}

TEST(ComplexRoots, RecoversAConjugatePairForXSquaredPlusOne) {
    const Polynomial p(std::vector<double>{1.0, 0.0, 1.0}); // x^2 + 1, roots +-i
    const auto roots = complex_roots(p);
    ASSERT_EQ(roots.size(), 2u);
    EXPECT_NEAR(roots[0].re, 0.0, 1e-8);
    EXPECT_NEAR(roots[1].re, 0.0, 1e-8);
    EXPECT_NEAR(std::fabs(roots[0].im), 1.0, 1e-8);
    EXPECT_NEAR(roots[0].im, -roots[1].im, 1e-8);
}

TEST(ComplexRoots, RejectsDegreeZeroPolynomial) {
    EXPECT_THROW(complex_roots(Polynomial(5.0)), std::invalid_argument);
}

TEST(RationalRoots, FindsAllRootsOfAKnownCubic) {
    // 2x^3 - x^2 - 2x + 1 = (x-1)(x+1)(2x-1), roots 1, -1, 1/2.
    const Polynomial p(std::vector<double>{1.0, -2.0, -1.0, 2.0});
    const auto roots = rational_roots(p);
    ASSERT_EQ(roots.size(), 3u);
    EXPECT_NEAR(roots[0], -1.0, 1e-9);
    EXPECT_NEAR(roots[1], 0.5, 1e-9);
    EXPECT_NEAR(roots[2], 1.0, 1e-9);
}

TEST(RationalRoots, HandlesARootAtExactlyZero) {
    // x^2 * (x - 2) = x^3 - 2x^2, roots 0 (double) and 2.
    const Polynomial p(std::vector<double>{0.0, 0.0, -2.0, 1.0});
    const auto roots = rational_roots(p);
    ASSERT_EQ(roots.size(), 2u);
    EXPECT_NEAR(roots[0], 0.0, 1e-9);
    EXPECT_NEAR(roots[1], 2.0, 1e-9);
}

TEST(RationalRoots, NoRationalRootsForAnIrreducibleQuadratic) {
    const Polynomial p(std::vector<double>{1.0, 0.0, 1.0}); // x^2 + 1
    EXPECT_TRUE(rational_roots(p).empty());
}

TEST(RationalRoots, RejectsNonIntegerCoefficients) {
    const Polynomial p(std::vector<double>{0.5, 1.0});
    EXPECT_THROW(rational_roots(p), std::invalid_argument);
}

namespace {
Polynomial rebuild(const RealFactorization& rf) {
    Polynomial result(rf.leading_coefficient);
    for (const auto& f : rf.factors)
        for (int i = 0; i < f.multiplicity; ++i) result = result.multiply(f.factor);
    return result;
}
} // namespace

TEST(FactorOverReals, SplitsAMixedRealAndComplexRootPolynomial) {
    // (x^2 + 1)(x - 2): one real linear factor, one irreducible real quadratic factor.
    const Polynomial p = Polynomial(std::vector<double>{1.0, 0.0, 1.0}).multiply(Polynomial(std::vector<double>{-2.0, 1.0}));
    const auto rf = factor_over_reals(p);
    ASSERT_EQ(rf.num_factors(), 2u);

    bool found_linear = false, found_quadratic = false;
    for (const auto& f : rf.factors) {
        EXPECT_EQ(f.multiplicity, 1);
        if (f.factor.degree() == 1) found_linear = true;
        if (f.factor.degree() == 2) found_quadratic = true;
    }
    EXPECT_TRUE(found_linear);
    EXPECT_TRUE(found_quadratic);

    const Polynomial rebuilt = rebuild(rf);
    ASSERT_EQ(rebuilt.degree(), p.degree());
    for (int i = 0; i <= p.degree(); ++i) EXPECT_NEAR(rebuilt.coefficient(i), p.coefficient(i), 1e-6);
}

TEST(FactorOverReals, RecoversARepeatedRealRootWithItsMultiplicity) {
    // (x-1)^2 (x-3) -- this exact shape once tripped a floating-point robustness bug in
    // poly_gcd() (a near-zero-but-not-exactly-zero remainder failed to terminate the Euclidean
    // algorithm, collapsing the GCD to a spurious constant); this is as much a regression test
    // for that fix as for factor_over_reals() itself.
    const Polynomial x_minus_1(std::vector<double>{-1.0, 1.0});
    const Polynomial x_minus_3(std::vector<double>{-3.0, 1.0});
    const Polynomial p = x_minus_1.multiply(x_minus_1).multiply(x_minus_3);

    const auto rf = factor_over_reals(p);
    ASSERT_EQ(rf.num_factors(), 2u);

    bool found_mult_2 = false, found_mult_1 = false;
    for (const auto& f : rf.factors) {
        EXPECT_EQ(f.factor.degree(), 1);
        if (f.multiplicity == 2) found_mult_2 = true;
        if (f.multiplicity == 1) found_mult_1 = true;
    }
    EXPECT_TRUE(found_mult_2);
    EXPECT_TRUE(found_mult_1);

    const Polynomial rebuilt = rebuild(rf);
    ASSERT_EQ(rebuilt.degree(), p.degree());
    for (int i = 0; i <= p.degree(); ++i) EXPECT_NEAR(rebuilt.coefficient(i), p.coefficient(i), 1e-6);
}

TEST(PolyGcd, HandlesNearZeroRemainderNoise) {
    // Regression test for the same poly_gcd() bug: gcd(p, p') must recover the true repeated
    // factor (x-1), not collapse to a spurious near-zero constant.
    const Polynomial x_minus_1(std::vector<double>{-1.0, 1.0});
    const Polynomial x_minus_3(std::vector<double>{-3.0, 1.0});
    const Polynomial p = x_minus_1.multiply(x_minus_1).multiply(x_minus_3);
    const Polynomial g = poly_gcd(p, p.derivative());
    ASSERT_EQ(g.degree(), 1);
    EXPECT_NEAR(g.coefficient(0), -1.0, 1e-6);
    EXPECT_NEAR(g.coefficient(1), 1.0, 1e-6);
}
