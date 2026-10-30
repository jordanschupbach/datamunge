#include <gtest/gtest.h>

#include <cmath>
#include <dm/dm.hpp>
#include <limits>
#include <stdexcept>
#include <vector>
namespace {
const auto square   = [](const std::vector<double>& p, double) { return p[0] * p[0]; };
const auto square_j = [](const std::vector<double>& p, double) {
  return std::vector<double>{2 * p[0]};
};
} // namespace
TEST(GaussNewtonDiagnostics, ExactLinearStepAndCurrentFinalSSE) {
  const auto f = [](const std::vector<double>& p, double x) { return p[0] + p[1] * x; };
  const auto j = [](const std::vector<double>&, double x) { return std::vector<double>{1, x}; };
  const std::vector<double> x = {-1, 0, 1}, y = {0, 2, 4}, p = {0, 0};
  const auto                a = dm::algorithms::gauss_newton_linearize(f, j, x, y, p);
  EXPECT_NEAR(a.step[0], 2, 1e-14);
  EXPECT_NEAR(a.step[1], 2, 1e-14);
  EXPECT_EQ(a.sse, 20);
  EXPECT_LT(a.linearized_sse, 1e-26);
  EXPECT_EQ(a.gradient_inf, 6);
  const auto fit = dm::algorithms::gauss_newton(f, j, x, y, p, 1, 1e-12);
  EXPECT_EQ(fit.iterations, 1);
  EXPECT_TRUE(fit.step_tolerance_met);
  EXPECT_LT(fit.sse, 1e-26);
  ASSERT_EQ(fit.history.size(), 2);
  EXPECT_EQ(fit.history.back().sse, fit.sse);
  EXPECT_EQ(fit.trace().nrows(), 2);
  EXPECT_NEAR(fit.trace().double_at("p1", 1), 2, 1e-14);
  EXPECT_EQ(p, (std::vector<double>{0, 0}));
}
TEST(GaussNewtonDiagnostics, BudgetExhaustionAndZeroBudgetReturnConsistentStates) {
  const auto fit = dm::algorithms::gauss_newton(square, square_j, {0}, {2}, {1}, 1, 1e-14);
  EXPECT_EQ(fit.iterations, 1);
  EXPECT_FALSE(fit.step_tolerance_met);
  EXPECT_NEAR(fit.params[0], 1.5, 1e-14);
  EXPECT_NEAR(fit.sse, .0625, 1e-14);
  ASSERT_EQ(fit.residuals.size(), 1);
  EXPECT_NEAR(fit.residuals[0], .25, 1e-14);
  EXPECT_NEAR(fit.history.back().next_step_norm, 1. / 12, 1e-14);
  const auto initial = dm::algorithms::gauss_newton(square, square_j, {0}, {2}, {1}, 0);
  EXPECT_EQ(initial.iterations, 0);
  EXPECT_EQ(initial.params[0], 1);
  EXPECT_EQ(initial.sse, 1);
  EXPECT_FALSE(initial.step_tolerance_met);
  EXPECT_EQ(initial.history.size(), 1);
  EXPECT_EQ(dm::algorithms::GaussNewtonResult{}.trace().nrows(), 0);
}
TEST(GaussNewtonDiagnostics, CentralDifferencesWorkAtZeroAndAgreeWithAnalyticSteps) {
  const auto f = [](const std::vector<double>& p, double x) {
    return p[0] + p[1] * std::exp(p[2] * x);
  };
  const auto j = [](const std::vector<double>& p, double x) {
    const double e = std::exp(p[2] * x);
    return std::vector<double>{1, e, p[1] * x * e};
  };
  const std::vector<double> x = {-1, 0, .3, 1, 2}, y = {1, 2, 3, 4, 5}, p = {0, 2, .4};
  const auto                a = dm::algorithms::gauss_newton_linearize(f, x, y, p),
             b                = dm::algorithms::gauss_newton_linearize(f, j, x, y, p);
  for (std::size_t r = 0; r < x.size(); ++r)
    for (std::size_t c = 0; c < p.size(); ++c)
      EXPECT_NEAR(a.jacobian(r, c), b.jacobian(r, c), 2e-9);
  for (std::size_t c = 0; c < p.size(); ++c)
    EXPECT_NEAR(a.step[c], b.step[c], 2e-8);
}
TEST(GaussNewtonDiagnostics, FullStepCanIncreaseLossWhileLinearModelPredictsZero) {
  const auto fit = dm::algorithms::gauss_newton(square, square_j, {0}, {1}, {.1}, 1);
  EXPECT_NEAR(fit.params[0], 5.05, 1e-12);
  EXPECT_NEAR(fit.history.front().sse, .9801, 1e-14);
  EXPECT_LT(fit.history.front().linearized_sse, 1e-26);
  EXPECT_NEAR(fit.sse, 600.37250625, 1e-9);
  EXPECT_GT(fit.sse, fit.history.front().sse);
}
TEST(GaussNewtonDiagnostics, ZeroResidualQuadraticAndNonzeroResidualLinearConvergence) {
  const auto zero = dm::algorithms::gauss_newton(square, square_j, {0}, {2}, {1}, 5, 1e-15);
  for (std::size_t i = 0; i + 1 < zero.history.size(); ++i) {
    const double p = zero.history[i].params[0], e = p - std::sqrt(2.);
    EXPECT_NEAR(zero.history[i + 1].params[0] - std::sqrt(2.), e * e / (2 * p), 4e-16);
  }
  const auto f = [](const std::vector<double>& p, double x) { return x == 0 ? p[0] : p[0] * p[0]; };
  const auto j = [](const std::vector<double>& p, double x) {
    return std::vector<double>{x == 0 ? 1 : 2 * p[0]};
  };
  const auto fit = dm::algorithms::gauss_newton(f, j, {0, 1}, {0, 1}, {1}, 40, 1e-13);
  EXPECT_TRUE(fit.step_tolerance_met);
  EXPECT_NEAR(fit.params[0], std::sqrt(.5), 2e-13);
  EXPECT_NEAR(fit.sse, .75, 1e-14);
  const double e8 = fit.history[8].params[0] - std::sqrt(.5),
               e9 = fit.history[9].params[0] - std::sqrt(.5);
  EXPECT_NEAR(e9 / e8, 1. / 3, 2e-5);
}
TEST(GaussNewtonDiagnostics, QRColumnScalingPreservesSeverelyDifferentParameterUnits) {
  const auto f = [](const std::vector<double>& p, double x) {
    return 1e12 * p[0] + 1e-9 * p[1] * x;
  };
  const auto j = [](const std::vector<double>&, double x) {
    return std::vector<double>{1e12, 1e-9 * x};
  };
  const auto a = dm::algorithms::gauss_newton_linearize(f, j, {-1, 0, 1}, {0, 2, 4}, {0, 0});
  EXPECT_NEAR(a.step[0], 2e-12, 1e-26);
  EXPECT_NEAR(a.step[1], 2e9, 1e-5);
  EXPECT_LT(a.linearized_sse, 1e-25);
}
TEST(GaussNewtonDiagnostics, RankFailureIsExplicit) {
  EXPECT_THROW(
      (void)dm::algorithms::gauss_newton(square, square_j, {0}, {1}, {0}), std::runtime_error);
  const auto f = [](const std::vector<double>& p, double) { return p[0] + p[1]; };
  const auto j = [](const std::vector<double>&, double) { return std::vector<double>{1, 1}; };
  EXPECT_THROW(
      (void)dm::algorithms::gauss_newton(f, j, {0, 1}, {1, 2}, {0, 0}), std::runtime_error);
}
TEST(GaussNewtonDiagnostics, InvalidInputsAndCallbackFailuresAreChecked) {
  using dm::algorithms::gauss_newton;
  EXPECT_THROW((void)gauss_newton(square, {}, {}, {1}), std::invalid_argument);
  EXPECT_THROW((void)gauss_newton(square, {0, 1}, {1}, {1}), std::invalid_argument);
  EXPECT_THROW((void)gauss_newton(square, {0}, {1}, {}), std::invalid_argument);
  EXPECT_THROW((void)gauss_newton(square, {0}, {1}, {1, 2}), std::invalid_argument);
  EXPECT_THROW((void)gauss_newton(square, {0}, {1}, {1}, -1), std::invalid_argument);
  for (double v :
       {0., -1., std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
    EXPECT_THROW((void)gauss_newton(square, {0}, {1}, {1}, 2, v), std::invalid_argument);
  EXPECT_THROW(
      (void)gauss_newton(square, {std::numeric_limits<double>::quiet_NaN()}, {1}, {1}),
      std::invalid_argument);
  const auto bad = [](const std::vector<double>&, double) {
    return std::numeric_limits<double>::infinity();
  };
  EXPECT_THROW((void)gauss_newton(bad, {0}, {1}, {1}), std::invalid_argument);
  const auto j = [](const std::vector<double>&, double) { return std::vector<double>{}; };
  EXPECT_THROW((void)gauss_newton(square, j, {0}, {1}, {1}), std::invalid_argument);
  const auto throwing = [](const std::vector<double>&, double) -> double {
    throw std::domain_error("model domain");
  };
  EXPECT_THROW((void)gauss_newton(throwing, {0}, {1}, {1}), std::domain_error);
  const auto log = [](const std::vector<double>& p, double) { return std::log(p[0]); };
  EXPECT_THROW((void)gauss_newton(log, {0}, {-20}, {1e-9}), std::invalid_argument);
}
TEST(GaussNewtonDiagnostics, Misra1aDataAndNistCertifiedFit) {
  const auto d = dm::datasets::misra1a();
  ASSERT_EQ(d.nrows(), 14);
  EXPECT_EQ(d.ncols(), 2);
  EXPECT_DOUBLE_EQ(d.double_at("pressure", 0), 77.6);
  EXPECT_DOUBLE_EQ(d.double_at("volume", 13), 81.78);
  EXPECT_EQ(dm::datasets::load("misra1a").nrows(), 14);
  EXPECT_EQ(d.null_count("volume"), 0);
  const auto f = [](const std::vector<double>& p, double x) {
    return -p[0] * std::expm1(-p[1] * x / 1000);
  };
  const auto j = [](const std::vector<double>& p, double x) {
    const double t = x / 1000;
    return std::vector<double>{-std::expm1(-p[1] * t), p[0] * t * std::exp(-p[1] * t)};
  };
  const auto x = d.numeric_column("pressure").values(), y = d.numeric_column("volume").values();
  for (const auto& start : std::vector<std::vector<double>>{{500, .1}, {250, .5}}) {
    const auto a = dm::algorithms::gauss_newton(f, j, x, y, start, 100, 1e-10),
               b = dm::algorithms::gauss_newton(f, x, y, start, 100, 1e-9);
    EXPECT_NEAR(a.params[0], 238.94212918, 2e-7);
    EXPECT_NEAR(a.params[1], .55015643181, 1e-9);
    EXPECT_NEAR(a.sse, .12455138894, 1e-11);
    EXPECT_NEAR(a.sse, b.sse, 1e-11);
    EXPECT_NEAR(a.params[0], b.params[0], 2e-6);
    double ss = 0;
    for (std::size_t i = 0; i < x.size(); ++i)
      ss += std::pow(f(a.params, x[i]) - y[i], 2);
    EXPECT_NEAR(a.sse, ss, 1e-13);
  }
}

TEST(GaussNewtonDiagnostics, AbsoluteStepStoppingIsNotAStationarityCertificate) {
  const auto f   = [](const std::vector<double>& p, double) { return 1e12 * p[0]; };
  const auto j   = [](const std::vector<double>&, double) { return std::vector<double>{1e12}; };
  const auto fit = dm::algorithms::gauss_newton(f, j, {0}, {1}, {0}, 10, 1e-12);
  EXPECT_TRUE(fit.step_tolerance_met);
  EXPECT_EQ(fit.iterations, 0);
  EXPECT_EQ(fit.sse, 1);
  EXPECT_EQ(fit.history[0].gradient_inf, 1e12);
}
