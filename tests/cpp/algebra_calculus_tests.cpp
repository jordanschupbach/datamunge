#include <gtest/gtest.h>

#include <datamunge/algebra/algebra.hpp>

#include <cmath>
#include <vector>

using namespace datamunge::algebra;

namespace {

// The primary oracle for Expr::integrate(): the fundamental theorem of calculus says
// differentiating the returned antiderivative must reproduce the original expression at every
// point, independent of the antiderivative's particular symbolic form.
void expect_integral_round_trips(const Expr& f, const std::string& var, const std::vector<double>& sample_points) {
    const Expr antiderivative = f.integrate(var);
    const Expr recovered = antiderivative.differentiate(var).simplify();
    for (double x : sample_points) {
        EXPECT_NEAR(recovered.evaluate({var}, {x}), f.evaluate({var}, {x}), 1e-8)
            << "at " << var << " = " << x;
    }
}

const std::vector<double> kSamplePoints = {0.3, 0.7, 1.1, 1.9, 2.4};
const std::vector<double> kPositiveSamplePoints = {0.3, 0.7, 1.1, 1.9, 2.4}; // safe for log()

} // namespace

TEST(ExprIntegrate, ConstantAndPowerRule) {
    const Expr x = Expr::variable("x");
    expect_integral_round_trips(Expr::constant(3.0), "x", kSamplePoints);
    expect_integral_round_trips(x, "x", kSamplePoints);
    expect_integral_round_trips(x.power(Expr::constant(2.0)), "x", kSamplePoints);
    expect_integral_round_trips(x.power(Expr::constant(-1.0)), "x", kPositiveSamplePoints);
}

TEST(ExprIntegrate, LinearSubstitutionForPowerSinCosExp) {
    const Expr x = Expr::variable("x");
    const Expr linear = x.multiply(Expr::constant(2.0)).add(Expr::constant(1.0)); // 2x + 1
    expect_integral_round_trips(linear.power(Expr::constant(3.0)), "x", kSamplePoints);
    expect_integral_round_trips(linear.sin(), "x", kSamplePoints);
    expect_integral_round_trips(linear.cos(), "x", kSamplePoints);
    expect_integral_round_trips(x.multiply(Expr::constant(2.0)).exp(), "x", kSamplePoints);
}

TEST(ExprIntegrate, LogRuleAndConstantOverLinear) {
    const Expr x = Expr::variable("x");
    expect_integral_round_trips(x.log(), "x", kPositiveSamplePoints);
    const Expr linear = x.multiply(Expr::constant(2.0)).add(Expr::constant(3.0));
    expect_integral_round_trips(linear.log(), "x", kSamplePoints);
    expect_integral_round_trips(Expr::constant(5.0).divide(linear), "x", kSamplePoints);
}

TEST(ExprIntegrate, TabularIntegrationByParts) {
    const Expr x = Expr::variable("x");
    expect_integral_round_trips(x.multiply(x.sin()), "x", kSamplePoints);
    expect_integral_round_trips(x.power(Expr::constant(2.0)).multiply(x.exp()), "x", kSamplePoints);
    const Expr linear = x.multiply(Expr::constant(2.0)).add(Expr::constant(1.0));
    expect_integral_round_trips(x.multiply(linear.cos()), "x", kSamplePoints);
}

TEST(ExprIntegrate, ThrowsForUnsupportedForms) {
    const Expr x = Expr::variable("x");
    EXPECT_THROW(x.sin().multiply(x.cos()).integrate("x"), std::invalid_argument);
    EXPECT_THROW(x.divide(x.power(Expr::constant(2.0)).add(Expr::constant(1.0))).integrate("x"), std::invalid_argument);
    EXPECT_THROW(Expr::constant(2.0).power(x).integrate("x"), std::invalid_argument);
}

TEST(DefiniteIntegral, MatchesKnownClosedForms) {
    const Expr x = Expr::variable("x");
    EXPECT_NEAR(definite_integral(x.sin(), "x", 0.0, M_PI), 2.0, 1e-8);
    EXPECT_NEAR(definite_integral(x.power(Expr::constant(2.0)), "x", 0.0, 1.0), 1.0 / 3.0, 1e-8);
}

TEST(DefiniteIntegral, HandlesFormsIntegrateCannotSolveSymbolically) {
    const Expr x = Expr::variable("x");
    const Expr f = x.sin().multiply(x.cos()); // = 0.5 sin(2x); antiderivative -0.25 cos(2x)
    const double numeric = definite_integral(f, "x", 0.0, 1.0);
    const double exact = 0.25 * (1.0 - std::cos(2.0));
    EXPECT_NEAR(numeric, exact, 1e-8);
}

TEST(DefiniteIntegral, NegatesWhenBoundsAreReversed) {
    const Expr x = Expr::variable("x");
    const double forward = definite_integral(x, "x", 0.0, 2.0);
    const double backward = definite_integral(x, "x", 2.0, 0.0);
    EXPECT_NEAR(forward, -backward, 1e-10);
}

TEST(TaylorSeries, MatchesKnownMaclaurinCoefficientsForExp) {
    const Expr x = Expr::variable("x");
    const Polynomial t = taylor_series(x.exp(), "x", 0.0, 5);
    // exp(x) = sum x^k / k!
    double factorial = 1.0;
    for (int k = 0; k <= 5; ++k) {
        EXPECT_NEAR(t.coefficient(k), 1.0 / factorial, 1e-10);
        factorial *= static_cast<double>(k + 1);
    }
}

TEST(TaylorSeries, ApproximatesAroundANonzeroCenter) {
    const Expr x = Expr::variable("x");
    const Polynomial t = taylor_series(x.sin(), "x", M_PI / 2.0, 4);
    // Evaluate at (x - center): a small offset should closely match sin(center + offset).
    for (double offset : {-0.2, -0.05, 0.05, 0.2}) {
        EXPECT_NEAR(t.evaluate(offset), std::sin(M_PI / 2.0 + offset), 1e-4);
    }
}

TEST(TaylorSeries, RejectsNegativeOrder) {
    const Expr x = Expr::variable("x");
    EXPECT_THROW(taylor_series(x, "x", 0.0, -1), std::invalid_argument);
}
