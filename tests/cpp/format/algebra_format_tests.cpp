// std::format / std::print support for datamunge::algebra.
#include <dm/dm.hpp>

#include <gtest/gtest.h>

#include <format>
#include <string>

namespace alg = dm::algebra;

static_assert(std::formattable<alg::Polynomial, char>);
static_assert(std::formattable<alg::MultivariatePolynomial, char>);
static_assert(std::formattable<alg::RationalFunction, char>);
static_assert(std::formattable<alg::Expr, char>);
static_assert(std::formattable<alg::Complex, char>);
static_assert(std::formattable<alg::RealFactor, char>);
static_assert(std::formattable<alg::SquareFreeFactor, char>);
static_assert(std::formattable<alg::RealFactorization, char>);
static_assert(std::formattable<alg::PolyExtendedGcdResult, char>);
static_assert(std::formattable<alg::RealRootIntervals, char>);
static_assert(std::formattable<alg::MonomialOrder, char>);

TEST(AlgebraFormat, UsesToString) {
    const alg::Polynomial p({-1.0, 0.0, 1.0});
    EXPECT_EQ(std::format("{}", p), "x^2 - 1");
    EXPECT_EQ(std::format("{:>9}", p), "  x^2 - 1");
    const alg::RationalFunction f(alg::Polynomial({1.0, 1.0}), alg::Polynomial({-1.0, 1.0}));
    EXPECT_EQ(std::format("{}", f), f.to_string());
    const alg::MultivariatePolynomial m(2, {{1, 0}, {0, 2}}, {3.0, -1.0});
    EXPECT_EQ(std::format("{}", m), m.to_string());
    const auto e = alg::Expr::variable("x").multiply(alg::Expr::constant(2)).sin();
    EXPECT_EQ(std::format("{}", e), e.to_string());
    EXPECT_EQ(std::format("{}", e), "sin((x * 2))");
}

TEST(AlgebraFormat, ResultsAndEnums) {
    EXPECT_EQ(std::format("{}", alg::Complex{1, -2.5}), "Complex(1, -2.5)");
    EXPECT_EQ(std::format("{}", alg::RealFactor{alg::Polynomial({-1.0, 1.0}), 2}),
              "RealFactor(factor=(x - 1), multiplicity=2)");
    EXPECT_EQ(std::format("{}", alg::SquareFreeFactor{alg::Polynomial({1.0, 1.0}), 1}),
              "SquareFreeFactor(factor=(x + 1), multiplicity=1)");
    alg::RealFactorization rf;
    rf.leading_coefficient = 2;
    rf.factors = {{alg::Polynomial({-1.0, 1.0}), 2}, {alg::Polynomial({1.0, 0.0, 1.0}), 1}};
    EXPECT_EQ(std::format("{}", rf), "2 * (x - 1)^2 * (x^2 + 1)");
    const alg::PolyExtendedGcdResult g{alg::Polynomial(1.0), alg::Polynomial(2.0), alg::Polynomial({0.0, 1.0})};
    EXPECT_EQ(std::format("{}", g), "PolyExtendedGcdResult(gcd=(1), s=(2), t=(x))");
    EXPECT_EQ(std::format("{}", alg::RealRootIntervals{{-2, 0.5}, {-1, 1}}), "RealRootIntervals([-2, -1], [0.5, 1])");
    EXPECT_EQ(std::format("{}", alg::RealRootIntervals{}), "RealRootIntervals()");
    EXPECT_EQ(std::format("{}", alg::MonomialOrder::Grevlex), "Grevlex");
}
