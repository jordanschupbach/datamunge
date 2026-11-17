#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/inverse_iteration.hpp>
#include <datamunge/algorithms/inverse_iteration_analysis.hpp>
#include <limits>
#include <random>
#include <stdexcept>
using namespace datamunge::algorithms;
namespace {
using Mat = std::vector<std::vector<double>>;
void check(const Mat& a, const InverseIterationTrace& t) {
  ASSERT_FALSE(t.steps.empty());
  EXPECT_EQ(t.run.solves, t.steps.size() - 1);
  EXPECT_EQ(t.run.factorizations, t.run.solves ? 1 : 0);
  EXPECT_EQ(t.run.pair.vector, t.steps.back().pair.vector);
  EXPECT_EQ(t.run.pair.value, t.steps.back().pair.value);
  EXPECT_EQ(t.run.residual, t.steps.back().residual);
  long double normA = 0;
  for (auto& row : a)
    for (auto v : row)
      normA = std::hypot(normA, (long double)v);
  for (std::size_t k = 0; k < t.steps.size(); ++k) {
    const auto& s = t.steps[k];
    EXPECT_EQ(s.iteration, k);
    long double xx = 0, dot = 0, r2 = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      long double ax = 0;
      for (std::size_t j = 0; j < a.size(); ++j)
        ax += (long double)a[i][j] * s.pair.vector[j];
      dot += (long double)s.pair.vector[i] * ax;
      xx += (long double)s.pair.vector[i] * s.pair.vector[i];
      const auto e = ax - (long double)s.pair.value * s.pair.vector[i];
      r2 += e * e;
    }
    EXPECT_NEAR((double)xx, 1, 5e-16);
    EXPECT_NEAR(s.pair.value, (double)(dot / xx), 2e-14 * (1 + std::abs(s.pair.value)));
    EXPECT_NEAR(s.residual, (double)std::sqrt(r2), 2e-14 * (1 + s.residual));
    const auto den = (normA + std::abs((long double)s.pair.value)) * std::sqrt(xx);
    EXPECT_NEAR(s.relative_residual, den == 0 ? 0 : (double)(std::sqrt(r2) / den), 1e-15);
  }
}
} // namespace
TEST(InverseIterationTrace, DiagonalEveryActualIterateAndSign) {
  const Mat a{{2, 0}, {0, 5}};
  for (double mu : {1.9, 2.1, 3., 4., 4.9, 5.1}) {
    const auto t = inverse_iteration_trace(a, mu, {12});
    check(a, t);
    long double u = 1 / std::sqrt(2.L), v = u;
    for (const auto& s : t.steps) {
      EXPECT_NEAR(s.pair.vector[0], (double)u, 4e-15);
      EXPECT_NEAR(s.pair.vector[1], (double)v, 4e-15);
      u /= (2 - mu);
      v /= (5 - mu);
      const auto n = std::hypot(u, v);
      u /= n;
      v /= n;
    }
    const auto r = inverse_iteration_run(a, mu, {12});
    EXPECT_EQ(r.pair.vector, t.run.pair.vector);
    EXPECT_EQ(r.residual, t.run.residual);
    EXPECT_EQ(inverse_iteration(a, mu, 12).vector, r.pair.vector);
  }
}
TEST(InverseIterationTrace, KnownRotatedSpectrumAndSpectralRecurrence) {
  std::mt19937                           rng(175);
  std::uniform_real_distribution<double> angle(-3., 3.);
  for (int trial = 0; trial < 400; ++trial) {
    const double theta = angle(rng), c = std::cos(theta), s = std::sin(theta);
    const double d0 = -2., d1 = 5., mu = (trial % 2 ? -1.9 : 4.9);
    const Mat    a{
           {d0 * c * c + d1 * s * s, (d0 - d1) * c * s}, {(d0 - d1) * c * s, d0 * s * s + d1 * c * c}};
    const auto t = inverse_iteration_trace(a, mu, {8});
    check(a, t);
    long double u = (c + s) / std::sqrt(2.L), v = (c - s) / std::sqrt(2.L);
    for (const auto& step : t.steps) {
      EXPECT_NEAR(step.pair.vector[0], (double)(c * u - s * v), 1e-12);
      EXPECT_NEAR(step.pair.vector[1], (double)(s * u + c * v), 1e-12);
      u /= (d0 - mu);
      v /= (d1 - mu);
      const auto n = std::hypot(u, v);
      u /= n;
      v /= n;
    }
    EXPECT_NEAR(t.run.pair.value, trial % 2 ? d0 : d1, 1e-10);
    EXPECT_LT(t.run.residual, 1e-10);
  }
}
TEST(InverseIterationTrace, PivotingAndOriginalRoundedBasis) {
  const auto pivot = inverse_iteration_trace({{0, 1}, {1, 0}}, 0.1, {8}, {1, 2});
  check({{0, 1}, {1, 0}}, pivot);
  const double q[3][3] = {{.6667, -.6667, .3333}, {.6667, .3333, -.6667}, {.3333, .6667, .6667}};
  const double d[]     = {2, 5, 9};
  Mat          a(3, std::vector<double>(3));
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
        a[i][j] += q[i][k] * d[k] * q[j][k];
  const auto t = inverse_iteration_trace(a, 5.05, {60});
  check(a, t);
  EXPECT_NEAR(t.run.pair.value, 5.0003333148092, 1e-12);
  EXPECT_LT(t.run.residual, 1e-14);
  EXPECT_GT(std::abs(t.run.pair.value - 5), .0003);
}
TEST(InverseIterationTrace, MissingComponentAndEqualDistanceAreNotPromises) {
  const auto missing = inverse_iteration_trace({{2, 0}, {0, 5}}, 2.1, {12}, {0, 1});
  EXPECT_EQ(missing.run.pair.value, 5);
  EXPECT_EQ(missing.run.residual, 0);
  const auto tie = inverse_iteration_trace({{2, 0}, {0, 5}}, 3.5, {8});
  for (auto& s : tie.steps) {
    EXPECT_NEAR(s.pair.value, 3.5, 1e-15);
    EXPECT_NEAR(s.residual, 1.5, 1e-15);
  }
}
TEST(InverseIterationTrace, CompleteCapsAdmissionAndOwnership) {
  Mat                 a{{2, 0}, {0, 5}};
  std::vector<double> start{1, 2};
  const auto          t = inverse_iteration_trace(a, 2.1, {4, 2, 10}, start);
  a[0][0]               = 100;
  start[0]              = 9;
  EXPECT_LT(t.run.pair.value, 3);
  EXPECT_THROW(inverse_iteration_trace({{2, 0}, {0, 5}}, 2.1, {4, 2, 9}), std::length_error);
  EXPECT_THROW(inverse_iteration_trace({{2, 0}, {0, 5}}, 2.1, {0, 1, 0}), std::length_error);
  EXPECT_NO_THROW(inverse_iteration_trace({{2, 0}, {0, 5}}, 2.1, {4, 0, 0}));
  EXPECT_NO_THROW(inverse_iteration_run({{2, 0}, {0, 5}}, 2.1, {4, 2, 1}));
  // Complete-history admission takes precedence over factorization at a singular shift.
  EXPECT_THROW(inverse_iteration_trace({{2, 0}, {0, 5}}, 2, {4, 2, 9}), std::length_error);
  EXPECT_THROW(
      inverse_iteration_trace({{2}}, 1, {std::numeric_limits<std::size_t>::max(), 0, 0}),
      std::length_error);
}
TEST(InverseIterationTrace, InvalidInputsSingularityAndZeroSolves) {
  const double nan = std::numeric_limits<double>::quiet_NaN(),
               inf = std::numeric_limits<double>::infinity();
  EXPECT_THROW(inverse_iteration({}, 1), std::invalid_argument);
  EXPECT_THROW(inverse_iteration({{1, 2}, {3}}, 1), std::invalid_argument);
  EXPECT_THROW(inverse_iteration({{nan}}, 1), std::invalid_argument);
  EXPECT_THROW(inverse_iteration({{1}}, inf), std::invalid_argument);
  EXPECT_THROW(inverse_iteration({{1}}, 0, -1), std::invalid_argument);
  EXPECT_THROW(inverse_iteration_run({{1}}, 0, {2}, {1, 2}), std::invalid_argument);
  EXPECT_THROW(inverse_iteration_run({{1}}, 0, {2}, {nan}), std::invalid_argument);
  EXPECT_THROW(inverse_iteration_run({{1}}, 0, {2}, {0}), std::domain_error);
  EXPECT_THROW(inverse_iteration({{2, 0}, {0, 5}}, 2), std::domain_error);
  const auto t = inverse_iteration_trace({{2, 0}, {0, 5}}, 2, {0});
  check({{2, 0}, {0, 5}}, t);
  EXPECT_EQ(t.run.factorizations, 0);
  EXPECT_NEAR(t.run.pair.value, 3.5, 1e-15);
  EXPECT_EQ(inverse_iteration({{7}}, 6).value, 7);
}
TEST(InverseIterationTrace, RobustNormalizationAndNumericalRange) {
  for (double scale : {1e-300, 1e300}) {
    const Mat  a{{2 * scale, 0}, {0, 5 * scale}};
    const auto t = inverse_iteration_trace(a, 2.1 * scale, {8}, {1e300, 1e300});
    check(a, t);
    EXPECT_NEAR(t.run.pair.value / scale, 2, 1e-12);
    EXPECT_LT(t.run.relative_residual, 1e-11);
  }
  const auto m = std::numeric_limits<double>::max();
  EXPECT_THROW(inverse_iteration_run({{m, m}, {m, m}}, 0, {0}), std::overflow_error);
  const auto zero = inverse_iteration_trace({{0, 0}, {0, 0}}, 0, {0});
  EXPECT_EQ(zero.run.relative_residual, 0);
}

TEST(InverseIterationTrace, FixturesGraphicsAndCompleteDisplayCaps) {
  const auto original = inverse_iteration_fixture();
  const auto exact    = inverse_iteration_fixture(false);
  EXPECT_NE(original, exact);
  const auto t = inverse_iteration_trace(exact, 5.05, {8});
  EXPECT_NEAR(t.run.pair.value, 5, 1e-14);
  EXPECT_FALSE(inverse_iteration_diagram(t).to_svg().empty());
  EXPECT_FALSE(inverse_iteration_residual_plot({t}, {"actual"}, 1e-18, 9).to_svg().empty());
  EXPECT_THROW(inverse_iteration_diagram(t, 0), std::out_of_range);
  EXPECT_THROW(inverse_iteration_diagram(t, 9), std::out_of_range);
  EXPECT_THROW(inverse_iteration_residual_plot({t}, {"actual"}, 1e-18, 8), std::length_error);
  EXPECT_THROW(inverse_iteration_residual_plot({t}, {"actual"}, 0), std::invalid_argument);
  EXPECT_THROW(inverse_iteration_residual_plot({t}, {}), std::invalid_argument);
  auto bad              = t;
  bad.steps[1].residual = -1;
  EXPECT_THROW(inverse_iteration_residual_plot({bad}, {"edited"}), std::invalid_argument);
  EXPECT_THROW(inverse_iteration_diagram(bad), std::invalid_argument);
  bad                    = t;
  bad.steps[1].iteration = 4;
  EXPECT_THROW(inverse_iteration_residual_plot({bad}, {"edited"}), std::invalid_argument);
  EXPECT_THROW(inverse_iteration_diagram(bad), std::invalid_argument);
}
