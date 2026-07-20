#include <gtest/gtest.h>

#include <datamunge/algebra/algebra.hpp>

using namespace datamunge::algebra;

TEST(Polynomial, EvaluateMatchesHandComputedValue) {
    // p(x) = 2x^2 + 3x + 1
    const Polynomial p(std::vector<double>{1.0, 3.0, 2.0});
    EXPECT_DOUBLE_EQ(p.evaluate(2.0), 2 * 4 + 3 * 2 + 1);
    EXPECT_DOUBLE_EQ(p.evaluate(0.0), 1.0);
    EXPECT_EQ(p.degree(), 2);
}

TEST(Polynomial, ZeroPolynomialIsTrimmedAndReportsDegreeZero) {
    const Polynomial p(std::vector<double>{0.0, 0.0, 0.0});
    EXPECT_TRUE(p.is_zero());
    EXPECT_EQ(p.degree(), 0);
}

TEST(Polynomial, TrailingZerosAreTrimmed) {
    const Polynomial p(std::vector<double>{1.0, 2.0, 0.0, 0.0});
    EXPECT_EQ(p.degree(), 1);
    EXPECT_DOUBLE_EQ(p.coefficient(1), 2.0);
    EXPECT_DOUBLE_EQ(p.coefficient(5), 0.0);
}

TEST(Polynomial, AddSubtractMultiplyMatchHandComputation) {
    const Polynomial a(std::vector<double>{1.0, 1.0});  // x + 1
    const Polynomial b(std::vector<double>{-1.0, 1.0}); // x - 1
    const Polynomial sum = a.add(b);                    // 2x
    const Polynomial diff = a.subtract(b);               // 2
    const Polynomial prod = a.multiply(b);                // x^2 - 1

    EXPECT_EQ(sum, Polynomial(std::vector<double>{0.0, 2.0}));
    EXPECT_EQ(diff, Polynomial(2.0));
    EXPECT_EQ(prod, Polynomial(std::vector<double>{-1.0, 0.0, 1.0}));
}

TEST(Polynomial, DerivativeAndAntiderivativeAreInverseUpToConstant) {
    const Polynomial p(std::vector<double>{5.0, 3.0, 2.0}); // 2x^2 + 3x + 5
    const Polynomial d = p.derivative();                    // 4x + 3
    EXPECT_EQ(d, Polynomial(std::vector<double>{3.0, 4.0}));

    const Polynomial a = d.antiderivative(); // 2x^2 + 3x (zero constant term)
    EXPECT_EQ(a, Polynomial(std::vector<double>{0.0, 3.0, 2.0}));
}

TEST(Polynomial, DivmodSatisfiesDividendEqualsQuotientTimesDivisorPlusRemainder) {
    const Polynomial dividend(std::vector<double>{-1.0, 0.0, 1.0}); // x^2 - 1
    const Polynomial divisor(std::vector<double>{-1.0, 1.0});       // x - 1
    auto [q, r] = dividend.divmod(divisor);

    EXPECT_EQ(q, Polynomial(std::vector<double>{1.0, 1.0})); // x + 1
    EXPECT_TRUE(r.is_zero());

    const Polynomial reconstructed = q.multiply(divisor).add(r);
    EXPECT_EQ(reconstructed, dividend);
}

TEST(Polynomial, DivmodWithNonzeroRemainderSatisfiesTheDivisionIdentity) {
    const Polynomial dividend(std::vector<double>{1.0, 0.0, 1.0}); // x^2 + 1
    const Polynomial divisor(std::vector<double>{-1.0, 1.0});      // x - 1
    auto [q, r] = dividend.divmod(divisor);

    EXPECT_TRUE(r.is_zero() || r.degree() < divisor.degree());
    const Polynomial reconstructed = q.multiply(divisor).add(r);
    EXPECT_EQ(reconstructed, dividend);
}

TEST(Polynomial, QuotientAndRemainderMatchDivmod) {
    const Polynomial dividend(std::vector<double>{1.0, 0.0, 1.0}); // x^2 + 1
    const Polynomial divisor(std::vector<double>{-1.0, 1.0});      // x - 1
    auto [q, r] = dividend.divmod(divisor);
    EXPECT_EQ(poly_quotient(dividend, divisor), q);
    EXPECT_EQ(poly_remainder(dividend, divisor), r);
}

TEST(Polynomial, DivisionByZeroPolynomialThrows) {
    const Polynomial p(1.0);
    EXPECT_THROW((void)p.divmod(Polynomial(0.0)), std::invalid_argument);
}

TEST(PolyGcd, GcdOfCoprimePolynomialsIsOne) {
    const Polynomial a(std::vector<double>{1.0, 1.0});  // x + 1
    const Polynomial b(std::vector<double>{-1.0, 1.0}); // x - 1
    const Polynomial g = poly_gcd(a, b);
    EXPECT_EQ(g, Polynomial(1.0));
}

TEST(PolyGcd, GcdOfPolynomialsSharingAFactor) {
    // a = (x-1)(x-2), b = (x-1)(x-3) -> gcd should be monic (x-1)
    const Polynomial a(std::vector<double>{2.0, -3.0, 1.0});
    const Polynomial b(std::vector<double>{3.0, -4.0, 1.0});
    const Polynomial g = poly_gcd(a, b);
    EXPECT_EQ(g, Polynomial(std::vector<double>{-1.0, 1.0}));
}

TEST(PolyGcd, PseudoRemainderSequenceAgreesWithEuclideanGcd) {
    const Polynomial a(std::vector<double>{2.0, -3.0, 1.0});
    const Polynomial b(std::vector<double>{3.0, -4.0, 1.0});
    const Polynomial g1 = poly_gcd(a, b);
    const Polynomial g2 = poly_gcd_pseudo_remainder_sequence(a, b);
    ASSERT_EQ(g1.degree(), g2.degree());
    for (int i = 0; i <= g1.degree(); ++i) EXPECT_NEAR(g1.coefficient(i), g2.coefficient(i), 1e-9);
}

TEST(PolyGcd, ExtendedGcdSatisfiesBezoutIdentity) {
    const Polynomial a(std::vector<double>{2.0, -3.0, 1.0});  // (x-1)(x-2)
    const Polynomial b(std::vector<double>{-6.0, 11.0, -6.0, 1.0}); // (x-1)(x-2)(x-3)
    const auto result = poly_extended_gcd(a, b);

    const Polynomial reconstructed = result.s.multiply(a).add(result.t.multiply(b));
    ASSERT_EQ(reconstructed.degree(), result.gcd.degree());
    for (int i = 0; i <= result.gcd.degree(); ++i) EXPECT_NEAR(reconstructed.coefficient(i), result.gcd.coefficient(i), 1e-9);
}

TEST(Interpolation, LagrangeReconstructsAKnownQuadratic) {
    // p(x) = x^2 - x + 1 at x = 0, 1, 2 -> y = 1, 1, 3
    const std::vector<double> xs{0.0, 1.0, 2.0};
    const std::vector<double> ys{1.0, 1.0, 3.0};
    const Polynomial p = lagrange_interpolate(xs, ys);

    EXPECT_NEAR(p.evaluate(0.0), 1.0, 1e-9);
    EXPECT_NEAR(p.evaluate(1.0), 1.0, 1e-9);
    EXPECT_NEAR(p.evaluate(2.0), 3.0, 1e-9);
    EXPECT_NEAR(p.evaluate(3.0), 7.0, 1e-9); // 9 - 3 + 1
}

TEST(Interpolation, NewtonAgreesWithLagrangeOnTheSamePoints) {
    const std::vector<double> xs{-1.0, 0.5, 2.0, 4.0};
    const std::vector<double> ys{3.0, -1.0, 2.0, 10.0};
    const Polynomial lagrange = lagrange_interpolate(xs, ys);
    const Polynomial newton = newton_interpolate(xs, ys);

    for (double x : {-2.0, 0.0, 1.5, 3.3, 5.0}) EXPECT_NEAR(lagrange.evaluate(x), newton.evaluate(x), 1e-6);
}

TEST(Interpolation, DuplicateXValuesThrow) {
    EXPECT_THROW((void)lagrange_interpolate({1.0, 1.0}, {2.0, 3.0}), std::invalid_argument);
    EXPECT_THROW((void)newton_interpolate({1.0, 1.0}, {2.0, 3.0}), std::invalid_argument);
}

TEST(Resultant, ZeroResultantMeansACommonRoot) {
    // (x-1)(x-2) and (x-2)(x-3) share the root x=2.
    const Polynomial a(std::vector<double>{2.0, -3.0, 1.0});
    const Polynomial b(std::vector<double>{6.0, -5.0, 1.0});
    EXPECT_NEAR(resultant(a, b), 0.0, 1e-6);
}

TEST(Resultant, NonzeroResultantForCoprimePolynomials) {
    const Polynomial a(std::vector<double>{1.0, 1.0});  // x + 1
    const Polynomial b(std::vector<double>{-1.0, 1.0}); // x - 1
    EXPECT_NE(resultant(a, b), 0.0);
}

TEST(Discriminant, ZeroDiscriminantMeansARepeatedRoot) {
    // (x-1)^2 = x^2 - 2x + 1
    const Polynomial p(std::vector<double>{1.0, -2.0, 1.0});
    EXPECT_NEAR(discriminant(p), 0.0, 1e-6);
}

TEST(Discriminant, QuadraticDiscriminantMatchesTheClassicalFormula) {
    // ax^2+bx+c has discriminant b^2-4ac.
    const double a = 2.0, b = 5.0, c = -3.0;
    const Polynomial p(std::vector<double>{c, b, a});
    EXPECT_NEAR(discriminant(p), b * b - 4 * a * c, 1e-6);
}

TEST(RationalFunction, ReducesCommonFactors) {
    // (x^2-1)/(x-1) should simplify to x+1
    const Polynomial num(std::vector<double>{-1.0, 0.0, 1.0});
    const Polynomial den(std::vector<double>{-1.0, 1.0});
    const RationalFunction f(num, den);
    EXPECT_EQ(f.denominator(), Polynomial(1.0));
    EXPECT_EQ(f.numerator(), Polynomial(std::vector<double>{1.0, 1.0}));
}

TEST(RationalFunction, ArithmeticMatchesEvaluationAtSamplePoints) {
    const RationalFunction f(Polynomial(std::vector<double>{0.0, 1.0}), Polynomial(1.0)); // x/1
    const RationalFunction g(Polynomial(1.0), Polynomial(std::vector<double>{1.0, 1.0})); // 1/(x+1)

    const RationalFunction sum = f.add(g);
    const RationalFunction prod = f.multiply(g);
    for (double x : {0.5, 2.0, -0.25}) {
        EXPECT_NEAR(sum.evaluate(x), x + 1.0 / (x + 1.0), 1e-9);
        EXPECT_NEAR(prod.evaluate(x), x * (1.0 / (x + 1.0)), 1e-9);
    }
}

TEST(RationalFunction, EvaluatingAtAPoleThrows) {
    const RationalFunction f(Polynomial(1.0), Polynomial(std::vector<double>{-1.0, 1.0})); // 1/(x-1)
    EXPECT_THROW((void)f.evaluate(1.0), std::domain_error);
}
