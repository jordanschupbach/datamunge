#include <gtest/gtest.h>

#include <datamunge/algebra/algebra.hpp>

#include <cmath>
#include <type_traits>

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

TEST(ExprOperators, NumericConstantsConvertWithoutNumericConversionOfExpressions) {
    static_assert(std::is_convertible_v<int, Expr>);
    static_assert(std::is_convertible_v<float, Expr>);
    static_assert(std::is_convertible_v<long double, Expr>);
    static_assert(!std::is_convertible_v<Expr, double>);
    static_assert(!std::is_convertible_v<std::string, Expr>);
    static_assert(!std::is_convertible_v<const char*, Expr>);
    Expr scalar = 2.5;
    EXPECT_DOUBLE_EQ(scalar.evaluate({}, {}), 2.5);
    scalar = -7;
    EXPECT_DOUBLE_EQ(scalar.evaluate({}, {}), -7);
    const Expr precise = 0.125L;
    EXPECT_DOUBLE_EQ(precise.evaluate({}, {}), 0.125);
}

TEST(ExprOperators, AllBinaryOperatorsAcceptLiteralsOnEitherSideAndExpressions) {
    const auto x = Expr::variable("x");
    const auto y = Expr::variable("y");
    EXPECT_DOUBLE_EQ((x + 2).evaluate({"x"}, {3}), 5);
    EXPECT_DOUBLE_EQ((2 + x).evaluate({"x"}, {3}), 5);
    EXPECT_DOUBLE_EQ((x - 2).evaluate({"x"}, {3}), 1);
    EXPECT_DOUBLE_EQ((2 - x).evaluate({"x"}, {3}), -1);
    EXPECT_DOUBLE_EQ((x * 2).evaluate({"x"}, {3}), 6);
    EXPECT_DOUBLE_EQ((2 * x).evaluate({"x"}, {3}), 6);
    EXPECT_DOUBLE_EQ((x / 2).evaluate({"x"}, {3}), 1.5);
    EXPECT_DOUBLE_EQ((2 / x).evaluate({"x"}, {4}), 0.5);
    EXPECT_DOUBLE_EQ((x ^ 2).evaluate({"x"}, {3}), 9);
    EXPECT_DOUBLE_EQ((2 ^ x).evaluate({"x"}, {3}), 8);
    EXPECT_DOUBLE_EQ((x + y).evaluate({"x", "y"}, {3, 2}), 5);
    EXPECT_DOUBLE_EQ((x - y).evaluate({"x", "y"}, {3, 2}), 1);
    EXPECT_DOUBLE_EQ((x * y).evaluate({"x", "y"}, {3, 2}), 6);
    EXPECT_DOUBLE_EQ((x / y).evaluate({"x", "y"}, {3, 2}), 1.5);
    EXPECT_DOUBLE_EQ((x ^ y).evaluate({"x", "y"}, {3, 2}), 9);
    EXPECT_DOUBLE_EQ((x ^ -2).evaluate({"x"}, {2}), 0.25);
    EXPECT_DOUBLE_EQ((x ^ 0.5).evaluate({"x"}, {4}), 2);
    EXPECT_DOUBLE_EQ((2.5f * x + 3u - 0.5L).evaluate({"x"}, {2}), 7.5);
}

TEST(ExprOperators, OperatorPolynomialMatchesCalculusAndExistingMethodTrees) {
    const auto x = Expr::variable("x");
    const auto f = (x ^ 3) - 2 * x + 1;
    const auto named = x.power(Expr::constant(3)).subtract(Expr::constant(2).multiply(x)).add(Expr::constant(1));
    EXPECT_EQ(f.to_string(), named.to_string());
    const auto derivative = f.differentiate("x").simplify();
    const auto primitive = f.integrate("x").simplify();
    for (double value : {-2., -1., 0., 1., 2., 3.5}) {
        EXPECT_DOUBLE_EQ(f.evaluate({"x"}, {value}), value * value * value - 2 * value + 1);
        EXPECT_DOUBLE_EQ(derivative.evaluate({"x"}, {value}), 3 * value * value - 2);
        EXPECT_NEAR(primitive.differentiate("x").evaluate({"x"}, {value}),
                    f.evaluate({"x"}, {value}), 1e-12);
    }
    EXPECT_NEAR(definite_integral(f, "x", 0, 2), 2, 1e-12);
    EXPECT_EQ((f * 1 + 0).simplify().to_string(), f.to_string());
    EXPECT_EQ(f.to_tex(), "x^{3} - 2 \\cdot x + 1");
    const auto wave = (2 * x).sin();
    EXPECT_NEAR(wave.differentiate("x").simplify().evaluate({"x"}, {0.3}), 2 * std::cos(0.6), 1e-12);
}

TEST(ExprOperators, OwnOperandsAfterTemporaryDestructionReassignmentAndSelfAssignment) {
    const auto temporary = (Expr::variable("x") + 1) / (Expr::variable("x") - 2);
    EXPECT_DOUBLE_EQ(temporary.evaluate({"x"}, {3}), 4);
    Expr term = Expr::variable("x") + 1;
    const auto product = term * term;
    term = 100;
    EXPECT_DOUBLE_EQ(product.evaluate({"x"}, {2}), 9);
    Expr evolving = Expr::variable("x");
    evolving = evolving * evolving;
    evolving = evolving + 1;
    const auto snapshot = evolving;
    evolving = 0;
    EXPECT_DOUBLE_EQ(snapshot.evaluate({"x"}, {2}), 5);
}

TEST(ExprOperators, RetainsCppPowerPrecedenceAndLeftAssociativity) {
    const auto x = Expr::variable("x");
    const auto polynomial = (x ^ 3) - 2 * x + 1;
    // C++ parses the entire additive expression as the exponent without parentheses.
    const auto ungrouped = x ^ 3 - 2 * x + 1;
    EXPECT_DOUBLE_EQ(polynomial.evaluate({"x"}, {2}), 5);
    EXPECT_DOUBLE_EQ(ungrouped.evaluate({"x"}, {2}), 1);
    EXPECT_THROW((void)ungrouped.differentiate("x"), std::invalid_argument);
    const auto left = x ^ 2 ^ 3;
    const auto right = x ^ (Expr(2) ^ 3);
    EXPECT_DOUBLE_EQ(left.evaluate({"x"}, {2}), 64);
    EXPECT_DOUBLE_EQ(right.evaluate({"x"}, {2}), 256);
    EXPECT_EQ(2 ^ 3, 1); // Ordinary integral operands keep C++ XOR semantics.
}

TEST(ExprOperators, UnarySignsAndDeferredDivisionRetainExistingSemantics) {
    const auto x = Expr::variable("x");
    EXPECT_EQ((+x).to_string(), x.to_string());
    EXPECT_DOUBLE_EQ((-x).evaluate({"x"}, {3}), -3);
    EXPECT_DOUBLE_EQ(((-x) ^ 2).evaluate({"x"}, {3}), 9);
    EXPECT_DOUBLE_EQ((-(x ^ 2)).evaluate({"x"}, {3}), -9);
    EXPECT_DOUBLE_EQ((x - (x + 1)).evaluate({"x"}, {3}), -1);
    EXPECT_EQ((x + 0).to_string(), "(x + 0)");
    const auto zero_division = 1 / (x - x);
    EXPECT_TRUE(std::isinf(zero_division.evaluate({"x"}, {3})));
    const auto variable_power = x ^ x;
    EXPECT_DOUBLE_EQ(variable_power.evaluate({"x"}, {3}), 27);
    EXPECT_THROW((void)variable_power.differentiate("x"), std::invalid_argument);
}
