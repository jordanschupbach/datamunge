#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/nn/optimizer_analysis.hpp>
#include <limits>
using namespace datamunge::nn;
TEST(OptimizerAnalysis, FourPrescribedGradientsExposeVelocityAndNesterovConvention) {
  const datamunge::linalg::DenseMatrix<double> g(4, 1, {1, 1, -1, 0});
  for (auto kind : {OptimizerKind::Sgd, OptimizerKind::Momentum, OptimizerKind::Nesterov}) {
    OptimizerConfig c;
    c.kind                = kind;
    const auto   t        = optimizer_gradient_trace({0}, g, {.1, .1, .1, .05}, c);
    const double expected = kind == OptimizerKind::Sgd        ? -.1
                            : kind == OptimizerKind::Momentum ? -.39295
                                                              : -.453655;
    EXPECT_NEAR(t.double_at("after", 3), expected, 1e-14);
    if (kind != OptimizerKind::Sgd) {
      EXPECT_NEAR(t.double_at("velocity", 2), .71, 1e-14);
      EXPECT_NEAR(t.double_at("velocity", 3), .639, 1e-14);
    }
  }
}
TEST(OptimizerAnalysis, GradientDescentQuadraticMatchesEveryClosedFormIterate) {
  const auto t = optimizer_quadratic_trace({1, 10}, {2, -3}, 20, .05);
  for (std::size_t i = 0; i <= 20; ++i) {
    EXPECT_NEAR(t.double_at("weight", 2 * i), 2 * std::pow(.95, i), 1e-13);
    EXPECT_NEAR(t.double_at("weight", 2 * i + 1), -3 * std::pow(.5, i), 1e-13);
    EXPECT_NEAR(
        t.double_at("loss", 2 * i),
        .5 * (4 * std::pow(.95, 2 * i) + 90 * std::pow(.5, 2 * i)),
        1e-12);
  }
}
TEST(OptimizerAnalysis, MomentumAndNesterovMatchIndependentScalarRecurrences) {
  for (auto kind : {OptimizerKind::Momentum, OptimizerKind::Nesterov}) {
    OptimizerConfig c;
    c.kind       = kind;
    c.momentum   = .6;
    const auto t = optimizer_quadratic_trace({3}, {2}, 25, .1, c);
    double     x = 2, v = 0;
    for (std::size_t i = 0; i <= 25; ++i) {
      EXPECT_NEAR(t.double_at("weight", i), x, 1e-14);
      EXPECT_NEAR(t.double_at("velocity", i), v, 1e-14);
      double gradient = 3 * x;
      v               = .6 * v + gradient;
      x -= .1 * (kind == OptimizerKind::Momentum ? v : gradient + .6 * v);
    }
  }
}
TEST(OptimizerAnalysis, ModeRadiiExposeDistinctStabilityBoundaries) {
  EXPECT_DOUBLE_EQ(optimizer_mode_radius(OptimizerKind::Sgd, 2), 1);
  EXPECT_NEAR(optimizer_mode_radius(OptimizerKind::Momentum, 1, .9), std::sqrt(.9), 1e-15);
  EXPECT_NEAR(optimizer_mode_radius(OptimizerKind::Momentum, 3.8, .9), 1, 1e-14);
  EXPECT_NEAR(optimizer_mode_radius(OptimizerKind::Nesterov, 3.8 / 2.8, .9), 1, 1e-15);
  EXPECT_DOUBLE_EQ(optimizer_mode_radius(OptimizerKind::Nesterov, 1, .9), 0);
  EXPECT_GT(optimizer_mode_radius(OptimizerKind::Nesterov, 2, .9), 1);
}
TEST(OptimizerAnalysis, ShapeFiniteAndRuleGuardsAreExplicit) {
  using M = datamunge::linalg::DenseMatrix<double>;
  EXPECT_EQ(optimizer_gradient_trace({1}, M(0, 1), {.1}).nrows(), 0);
  EXPECT_EQ(optimizer_quadratic_trace({1}, {1}, 0, .1).nrows(), 1);
  EXPECT_THROW((void)optimizer_gradient_trace({}, M(1, 0), {.1}), std::invalid_argument);
  EXPECT_THROW((void)optimizer_gradient_trace({1}, M(1, 2), {.1}), std::invalid_argument);
  EXPECT_THROW((void)optimizer_gradient_trace({1}, M(1, 1), {}), std::invalid_argument);
  EXPECT_THROW((void)optimizer_quadratic_trace({0}, {1}, 2, .1), std::invalid_argument);
  EXPECT_THROW((void)optimizer_quadratic_trace({1}, {1}, 2, -.1), std::invalid_argument);
  OptimizerConfig c;
  c.kind = OptimizerKind::Adam;
  EXPECT_THROW((void)optimizer_quadratic_trace({1}, {1}, 2, .1, c), std::invalid_argument);
  EXPECT_THROW((void)optimizer_mode_radius(OptimizerKind::Momentum, -1), std::invalid_argument);
  EXPECT_THROW((void)optimizer_mode_radius(OptimizerKind::Momentum, 1, 1), std::invalid_argument);
  EXPECT_THROW(
      (void)optimizer_gradient_trace({1}, M(1, 1, {std::numeric_limits<double>::infinity()}), {.1}),
      std::invalid_argument);
  EXPECT_THROW((void)optimizer_quadratic_trace({1}, {1e200}, 1, .1), std::overflow_error);
}
