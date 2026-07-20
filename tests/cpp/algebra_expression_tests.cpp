#include <gtest/gtest.h>

#include <datamunge/algebra/algebra.hpp>

#include <cmath>

using namespace datamunge::algebra;

TEST(Expr, EvaluateArithmeticExpression) {
    // (x + 2) * (x - 1)
    const Expr x = Expr::variable("x");
    const Expr e = x.add(Expr::constant(2.0)).multiply(x.subtract(Expr::constant(1.0)));
    EXPECT_DOUBLE_EQ(e.evaluate({"x"}, {3.0}), (3.0 + 2.0) * (3.0 - 1.0));
    EXPECT_DOUBLE_EQ(e.evaluate({"x"}, {-2.0}), (-2.0 + 2.0) * (-2.0 - 1.0));
}

TEST(Expr, MissingBindingThrows) {
    const Expr x = Expr::variable("x");
    EXPECT_THROW((void)x.evaluate({"y"}, {1.0}), std::invalid_argument);
}

TEST(Expr, DifferentiatePolynomialMatchesPowerRule) {
    // d/dx (x^3) = 3x^2
    const Expr x = Expr::variable("x");
    const Expr e = x.power(Expr::constant(3.0));
    const Expr d = e.differentiate("x").simplify();
    for (double v : {-2.0, 0.5, 4.0}) EXPECT_NEAR(d.evaluate({"x"}, {v}), 3.0 * v * v, 1e-9);
}

TEST(Expr, DifferentiateProductAndQuotientRules) {
    const Expr x = Expr::variable("x");
    // d/dx (x * sin(x)) = sin(x) + x*cos(x)
    const Expr product = x.multiply(x.sin());
    const Expr dproduct = product.differentiate("x");
    for (double v : {0.3, 1.7, -1.1}) EXPECT_NEAR(dproduct.evaluate({"x"}, {v}), std::sin(v) + v * std::cos(v), 1e-9);

    // d/dx (1 / x) = -1/x^2
    const Expr quotient = Expr::constant(1.0).divide(x);
    const Expr dquotient = quotient.differentiate("x");
    for (double v : {0.5, 2.0, -3.0}) EXPECT_NEAR(dquotient.evaluate({"x"}, {v}), -1.0 / (v * v), 1e-9);
}

TEST(Expr, DifferentiateExpAndLogAreConsistentAtASamplePoint) {
    const Expr x = Expr::variable("x");
    const Expr e = x.exp();
    for (double v : {0.0, 1.0, 2.5}) EXPECT_NEAR(e.differentiate("x").evaluate({"x"}, {v}), std::exp(v), 1e-9);

    const Expr l = x.log();
    for (double v : {0.5, 1.0, 10.0}) EXPECT_NEAR(l.differentiate("x").evaluate({"x"}, {v}), 1.0 / v, 1e-9);
}

TEST(Expr, DifferentiatingNonConstantExponentThrows) {
    const Expr x = Expr::variable("x");
    const Expr e = x.power(x); // x^x, not a constant exponent
    EXPECT_THROW((void)e.differentiate("x"), std::invalid_argument);
}

TEST(Expr, SimplifyFoldsConstantsAndAppliesIdentities) {
    const Expr x = Expr::variable("x");
    const Expr zero_add = x.add(Expr::constant(0.0)).simplify();
    EXPECT_EQ(zero_add.to_string(), x.to_string());

    const Expr one_mul = x.multiply(Expr::constant(1.0)).simplify();
    EXPECT_EQ(one_mul.to_string(), x.to_string());

    const Expr zero_mul = x.multiply(Expr::constant(0.0)).simplify();
    EXPECT_DOUBLE_EQ(zero_mul.evaluate({}, {}), 0.0);

    const Expr folded = Expr::constant(2.0).add(Expr::constant(3.0)).simplify();
    EXPECT_DOUBLE_EQ(folded.evaluate({}, {}), 5.0);
}

TEST(Expr, SimplifyPreservesSemanticsAgainstUnsimplifiedEvaluation) {
    const Expr x = Expr::variable("x");
    const Expr e = x.multiply(Expr::constant(1.0)).add(Expr::constant(0.0)).power(Expr::constant(1.0));
    const Expr simplified = e.simplify();
    for (double v : {-3.0, 0.0, 4.5}) EXPECT_NEAR(e.evaluate({"x"}, {v}), simplified.evaluate({"x"}, {v}), 1e-9);
}

TEST(Expr, ToStringProducesAReadableExpression) {
    const Expr x = Expr::variable("x");
    const Expr e = x.add(Expr::constant(1.0));
    EXPECT_EQ(e.to_string(), "(x + 1)");
}
