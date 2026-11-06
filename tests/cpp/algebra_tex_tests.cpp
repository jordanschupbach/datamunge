/// @file algebra_tex_tests.cpp
/// @brief Algebra TeX notation, grouping, precision, ownership and literal-name regressions.
#include <gtest/gtest.h>

#include <datamunge/algebra/algebra.hpp>
#include <limits>
#include <locale>
#include <type_traits>

namespace {
using datamunge::Tex;
using datamunge::algebra::Expr;
using datamunge::algebra::Polynomial;
using datamunge::algebra::RationalFunction;
const auto x = Expr::variable("x");
const auto y = Expr::variable("y");
const auto z = Expr::variable("z");
} // namespace

TEST(AlgebraTex, FormatsArithmeticAndRespectsSubtractionAndDivisionGrouping) {
  EXPECT_EQ(x.add(y).to_tex(), "x + y");
  EXPECT_EQ(x.subtract(y.add(z)).to_tex(), R"(x - \left(y + z\right))");
  EXPECT_EQ(x.subtract(y.subtract(z)).to_tex(), R"(x - \left(y - z\right))");
  EXPECT_EQ(x.add(y).multiply(z).to_tex(), R"(\left(x + y\right) \cdot z)");
  EXPECT_EQ(x.multiply(y.add(z)).to_tex(), R"(x \cdot \left(y + z\right))");
  EXPECT_EQ(x.add(y).divide(z.subtract(x)).to_tex(), R"(\frac{x + y}{z - x})");
  EXPECT_EQ(x.divide(y.divide(z)).to_tex(), R"(\frac{x}{\frac{y}{z}})");
  EXPECT_EQ(x.divide(y).divide(z).to_tex(), R"(\frac{\frac{x}{y}}{z})");
}

TEST(AlgebraTex, DistinguishesNegativeBasesNestedPowersAndGroupedExponents) {
  EXPECT_EQ(x.negate().power(Expr::constant(2)).to_tex(), R"(\left(-x\right)^{2})");
  EXPECT_EQ(x.power(Expr::constant(2)).negate().to_tex(), R"(-x^{2})");
  EXPECT_EQ(Expr::constant(-2).power(x).to_tex(), R"(\left(-2\right)^{x})");
  EXPECT_EQ(x.power(Expr::constant(-12)).to_tex(), R"(x^{-12})");
  EXPECT_EQ(x.power(y.add(z)).to_tex(), R"(x^{y + z})");
  EXPECT_EQ(x.power(y).power(z).to_tex(), R"(\left(x^{y}\right)^{z})");
  EXPECT_EQ(x.power(y.power(z)).to_tex(), R"(x^{y^{z}})");
  EXPECT_EQ(x.divide(y).power(z).to_tex(), R"(\left(\frac{x}{y}\right)^{z})");
  EXPECT_EQ(x.add(y).negate().to_tex(), R"(-\left(x + y\right))");
  EXPECT_EQ(x.negate().negate().to_tex(), R"(-\left(-x\right))");
  EXPECT_EQ(x.multiply(y.negate()).to_tex(), R"(x \cdot \left(-y\right))");
}

TEST(AlgebraTex, UsesNamedFunctionsAndEscapesLogicalNames) {
  EXPECT_EQ(x.add(y).sin().to_tex(), R"(\sin\left(x + y\right))");
  EXPECT_EQ(x.cos().to_tex(), R"(\cos\left(x\right))");
  EXPECT_EQ(x.exp().to_tex(), R"(\exp\left(x\right))");
  EXPECT_EQ(x.log().to_tex(), R"(\log\left(x\right))");
  EXPECT_EQ(Expr::variable("rate_1").to_tex(), R"(\text{rate\_1})");
  EXPECT_EQ(Expr::variable(R"(\alpha%{x})").to_tex(), R"(\text{\textbackslash{}alpha\%\{x\}})");
  EXPECT_EQ(Expr::variable("café").to_tex(), "\\text{café}");
  EXPECT_EQ(Expr::variable("").to_tex(), R"(\text{})");
}

TEST(AlgebraTex, ProducesOwnedTexValuesFromCalculusResults) {
  const auto cube       = x.power(Expr::constant(3));
  auto       derivative = cube.differentiate("x").simplify();
  const auto result     = derivative.to_tex();
  static_assert(std::is_same_v<std::decay_t<decltype(result)>, Tex>);
  derivative = Expr::constant(0);
  EXPECT_EQ(result, R"(3 \cdot x^{2})");
  EXPECT_EQ(result.underbrace("f'(x)").display(), "$$\n\\underbrace{3 \\cdot x^{2}}_{f'(x)}\n$$");
  EXPECT_EQ(cube.integrate("x").simplify().to_tex(), R"(\frac{x^{4}}{4})");
  EXPECT_EQ(cube.to_string(), "(x ^ 3)");
  EXPECT_DOUBLE_EQ(cube.evaluate({"x"}, {2}), 8);
}

TEST(AlgebraTex, UsesLocaleIndependentPrecisionAndScientificNotation) {
  struct CommaDecimal : std::numpunct<char> {
    char do_decimal_point() const override { return ','; }
  };
  struct RestoreLocale {
    std::locale original = std::locale();
    ~RestoreLocale() { std::locale::global(original); }
  } restore;
  std::locale::global(std::locale(std::locale::classic(), new CommaDecimal));
  EXPECT_EQ(Expr::constant(1.23456789).to_tex(), "1.234568");
  EXPECT_EQ(Expr::constant(1.23456789).to_tex(10), "1.23456789");
  EXPECT_EQ(Expr::constant(-0.).to_tex(), "0");
  EXPECT_EQ(Expr::constant(2e20).to_tex(), R"(2 \times 10^{20})");
  EXPECT_EQ(Expr::constant(-2e-20).to_tex(), R"(-2 \times 10^{-20})");
  EXPECT_EQ(Expr::constant(1e20).power(x).to_tex(), R"(\left(1 \times 10^{20}\right)^{x})");
  EXPECT_EQ(Polynomial({1.25, 2e-20}).to_tex(), R"(2 \times 10^{-20}\,x + 1.25)");
  EXPECT_EQ(Expr::constant(std::numeric_limits<double>::infinity()).to_tex(), R"(\infty)");
  EXPECT_EQ(Expr::constant(-std::numeric_limits<double>::infinity()).to_tex(), R"(-\infty)");
  EXPECT_EQ(Expr::constant(std::numeric_limits<double>::quiet_NaN()).to_tex(), R"(\mathrm{NaN})");
}

TEST(AlgebraTex, FormatsPolynomialsWithConventionalSignsAndCustomSymbols) {
  const Polynomial p({-1, 2, 0, -1});
  EXPECT_EQ(p.to_tex(), R"(-{x}^{3} + 2\,x - 1)");
  EXPECT_EQ(p.to_tex("h"), R"(-{h}^{3} + 2\,h - 1)");
  EXPECT_EQ(Polynomial({0, 0, 1}).to_tex(R"(\theta)"), R"({\theta}^{2})");
  EXPECT_EQ(Polynomial().to_tex(), "0");
  EXPECT_EQ(Polynomial(1).to_tex(), "1");
  EXPECT_EQ(Polynomial(-1).to_tex(), "-1");
  EXPECT_EQ(Polynomial({0, -1}).to_tex(), "-x");
  EXPECT_EQ(Polynomial({0, 0, 0, 1.23456789}).to_tex("h", 10), R"(1.23456789\,{h}^{3})");
  EXPECT_EQ(p.to_string(), "-x^3 + 2x - 1");
  EXPECT_DOUBLE_EQ(p.evaluate(2), -5);
}

TEST(AlgebraTex, GroupsRationalFunctionsAndOmitsReducedUnitDenominators) {
  const RationalFunction fraction(Polynomial({1, 1}), Polynomial({-1, 1}));
  EXPECT_EQ(fraction.to_tex(), R"(\frac{x + 1}{x - 1})");
  EXPECT_EQ(fraction.to_tex("h"), R"(\frac{h + 1}{h - 1})");
  const RationalFunction reduced(Polynomial({-1, 0, 1}), Polynomial({-1, 1}));
  EXPECT_EQ(reduced.to_tex(), "x + 1");
  EXPECT_EQ(RationalFunction().to_tex(), "0");
  EXPECT_EQ(fraction.to_string(), "(x + 1) / (x - 1)");
  EXPECT_DOUBLE_EQ(fraction.evaluate(2), 3);
  EXPECT_DOUBLE_EQ(reduced.evaluate(1), 2);
}

TEST(AlgebraTex, RejectsInvalidPrecisionAndEmptyPolynomialSymbols) {
  for (int precision : {-1, 0, 18}) {
    EXPECT_THROW(x.to_tex(precision), std::invalid_argument);
    EXPECT_THROW(Polynomial().to_tex("x", precision), std::invalid_argument);
    EXPECT_THROW(RationalFunction().to_tex("x", precision), std::invalid_argument);
  }
  EXPECT_THROW(Polynomial().to_tex(""), std::invalid_argument);
  EXPECT_THROW(RationalFunction().to_tex(""), std::invalid_argument);
  EXPECT_NO_THROW(x.to_tex(1));
  EXPECT_NO_THROW(x.to_tex(std::numeric_limits<double>::max_digits10));
}
