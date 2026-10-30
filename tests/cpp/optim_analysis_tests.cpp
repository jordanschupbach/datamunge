#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/optim/optim.hpp>
namespace o = datamunge::optim;
TEST(OptimizationAnalysis, QuadraticRotationCouplingAndAnalyticDerivatives) {
  o::QuadraticObjective f({2, 8}, {1, -2}, 3, .4);
  const std::vector<double> x{2, 1};
  auto g = f.gradient(x);
  auto h = f.hessian(x);
  for (std::size_t j = 0; j < 2; ++j) {
    auto a = x, b = x;
    a[j] += 1e-5;
    b[j] -= 1e-5;
    EXPECT_NEAR(g[j], (f.evaluate(a) - f.evaluate(b)) / 2e-5, 1e-7);
    for (std::size_t k = 0; k < 2; ++k)
      EXPECT_NEAR(h[k][j], (f.gradient(a)[k] - f.gradient(b)[k]) / 2e-5, 1e-7);
  }
  EXPECT_DOUBLE_EQ(f.evaluate({1, -2}), 0);
  EXPECT_THROW((o::QuadraticObjective({0}, {0})), std::invalid_argument);
}
TEST(OptimizationAnalysis, RosenbrockAndRastriginDerivatives) {
  for (const auto* family : {"rosenbrock", "rastrigin", "bumpy"}) {
    o::SmoothBenchmark f(family);
    const std::vector<double> x{.4, -.2};
    auto g = f.gradient(x);
    auto h = f.hessian(x);
    for (std::size_t j = 0; j < 2; ++j) {
      auto a = x, b = x;
      a[j] += 1e-5;
      b[j] -= 1e-5;
      EXPECT_NEAR(g[j], (f.evaluate(a) - f.evaluate(b)) / 2e-5, 1e-6);
      for (std::size_t k = 0; k < 2; ++k)
        EXPECT_NEAR(h[k][j], (f.gradient(a)[k] - f.gradient(b)[k]) / 2e-5,
                    1e-5);
    }
  }
}
TEST(OptimizationAnalysis, TracePreservesCallOrderAndSeparatesPostRunSampling) {
  o::QuadraticObjective f({2}, {0});
  o::GradientDescentOptions opts;
  opts.step_size = .25;
  opts.max_iterations = 1;
  opts.tolerance = 0;
  const auto r = o::trace_optimize<o::DifferentiableFunction>(
      f, o::GradientDescent(opts), {2});
  EXPECT_EQ(r.trace.calls(), 3);
  EXPECT_DOUBLE_EQ(r.coordinates[0], 1);
  EXPECT_DOUBLE_EQ(r.value, 1);
  EXPECT_DOUBLE_EQ(r.checked_value, 1);
  EXPECT_DOUBLE_EQ(r.gradient_norm, 2);
  const auto v = r.trace.values();
  EXPECT_EQ(v.nrows(), 2);
  EXPECT_DOUBLE_EQ(v.double_at("value", 0), 4);
  EXPECT_DOUBLE_EQ(v.double_at("value", 1), 1);
  const auto q = r.trace.sample(f, "gradient");
  EXPECT_EQ(q.nrows(), 1);
  EXPECT_DOUBLE_EQ(q.double_at("x0", 0), 2);
  EXPECT_DOUBLE_EQ(q.double_at("value", 0), 4);
  EXPECT_EQ(r.trace.calls(), 3);
}
TEST(OptimizationAnalysis, AdaptiveFirstUpdatesUseActualEpsilonAndMoments) {
  o::QuadraticObjective f({2}, {0});
  o::AdaGradOptions a;
  a.step_size = .1;
  a.epsilon = .01;
  a.max_iterations = 1;
  const auto r =
      o::trace_optimize<o::DifferentiableFunction>(f, o::AdaGrad(a), {2});
  EXPECT_NEAR(r.coordinates[0], 2 - .4 / std::sqrt(16.01), 1e-14);
  o::RMSPropOptions m;
  m.step_size = .1;
  m.decay_rate = .9;
  m.epsilon = .01;
  m.max_iterations = 1;
  const auto s =
      o::trace_optimize<o::DifferentiableFunction>(f, o::RMSProp(m), {2});
  EXPECT_NEAR(s.coordinates[0], 2 - .4 / (std::sqrt(1.6) + .01), 1e-14);
}
TEST(OptimizationAnalysis, TraceDoesNotInventConvergenceAndCapturesFailures) {
  o::QuadraticObjective f({1}, {0});
  o::AdaGradOptions opts;
  opts.max_iterations = 0;
  const auto r =
      o::trace_optimize<o::DifferentiableFunction>(f, o::AdaGrad(opts), {10});
  EXPECT_TRUE(r.exception.empty());
  EXPECT_DOUBLE_EQ(r.gradient_norm, 10);
  opts.epsilon = 0;
  const auto bad =
      o::trace_optimize<o::DifferentiableFunction>(f, o::AdaGrad(opts), {10});
  EXPECT_FALSE(bad.exception.empty());
  EXPECT_TRUE(std::isnan(bad.value));
  EXPECT_EQ(bad.coordinates, std::vector<double>{10});
  EXPECT_THROW(
      (o::trace_optimize<o::DifferentiableFunction>(f, o::AdaGrad(), {})),
      std::invalid_argument);
}
TEST(OptimizationAnalysis, EqualityAndInequalityFixturesHaveExactKKTControls) {
  o::EqualityQuadratic f({2, 0}, 1);
  EXPECT_DOUBLE_EQ(f.evaluate({1.5, -.5}), .25);
  EXPECT_DOUBLE_EQ(f.constraints({1.5, -.5})[0], 0);
  auto g = f.gradient({1.5, -.5});
  EXPECT_DOUBLE_EQ(g[0] + .5, 0);
  EXPECT_DOUBLE_EQ(g[1] + .5, 0);
  o::SQPOptions opts;
  opts.step_size = .5;
  const auto r = o::trace_optimize<o::EqualityConstrainedFunction>(
      f, o::SQP(opts), {0, 0});
  EXPECT_NEAR(r.coordinates[0], 1.5, 1e-6);
  EXPECT_NEAR(r.coordinates[1], -.5, 1e-6);
  o::NonnegativeQuadratic q({-1, 2});
  o::InteriorPointOptions ip;
  ip.step_size = .001;
  ip.max_outer_iterations = 3;
  ip.inner_iterations = 100;
  const auto s = o::trace_optimize<o::InequalityConstrainedFunction>(
      q, o::InteriorPoint(ip), {.5, .5});
  EXPECT_TRUE(s.exception.empty());
  for (double x : s.coordinates) EXPECT_GT(x, 0);
}
TEST(OptimizationAnalysis, SummedLeastSquaresComponentsAndMeanSVRGSnapshot) {
  o::LeastSquaresObjective f({{1}, {1}}, {2, 2});
  EXPECT_DOUBLE_EQ(f.evaluate({0}), 4);
  EXPECT_DOUBLE_EQ(f.gradient({0})[0], -4);
  o::SVRGOptions s;
  s.step_size = .1;
  s.max_epochs = 1;
  s.inner_iterations = 2;
  s.tolerance = 0;
  const auto r =
      o::trace_optimize<o::DifferentiableSeparableFunction>(f, o::SVRG(s), {0});
  EXPECT_NEAR(r.coordinates[0], .38, 1e-14);
  EXPECT_TRUE(r.exception.empty());
  o::SGDOptions g;
  g.batch_size = 0;
  std::vector<double> x{0};
  EXPECT_THROW(o::SGD(g).optimize(f, x), std::invalid_argument);
}
TEST(OptimizationAnalysis,
     ProximalSoftThresholdAndZeroSmoothGradientIsNotTheCriterion) {
  o::L1LeastSquaresObjective f({{1, 0}, {0, 1}}, {3, -1}, .5);
  const std::vector<double> optimum{2, 0};
  EXPECT_NEAR(f.evaluate(optimum), 1.5, 1e-14);
  auto g = f.gradient(optimum);
  EXPECT_NEAR(g[0], -.5, 1e-14);
  auto z = optimum;
  for (std::size_t j = 0; j < 2; ++j) z[j] -= g[j];
  EXPECT_EQ(f.proximal(z, 1), optimum);
  const auto r = o::trace_optimize<o::ProximalFunction>(
      f, o::FISTA({1, 1000, 1e-10}), {0, 0});
  EXPECT_NEAR(r.coordinates[0], 2, 1e-7);
  EXPECT_NEAR(r.coordinates[1], 0, 1e-7);
  EXPECT_FALSE(r.trace.sample(f, "gradient").nrows() == 0);
}
TEST(OptimizationAnalysis, AnalyticCurveJacobiansAndResidualOnlyTracing) {
  for (const auto* family : {"exponential", "logistic"}) {
    const auto d = o::optimization_curve_data(family);
    o::CurveResidual f(d, family);
    std::vector<double> x = std::string(family) == "exponential"
                                ? std::vector<double>{2, .6}
                                : std::vector<double>{4, 1, 2.5};
    auto j = f.jacobian(x);
    for (std::size_t k = 0; k < x.size(); ++k) {
      auto a = x, b = x;
      a[k] += 1e-6;
      b[k] -= 1e-6;
      auto ra = f.residuals(a), rb = f.residuals(b);
      for (std::size_t i = 0; i < ra.size(); ++i)
        EXPECT_NEAR(j[i][k], (ra[i] - rb[i]) / 2e-6, 1e-8);
    }
    const auto r =
        o::trace_optimize<o::ResidualFunction>(f, o::LevenbergMarquardt(), x);
    EXPECT_TRUE(r.exception.empty());
    EXPECT_NEAR(r.value, r.checked_value, 1e-12);
    EXPECT_GT(r.trace.values().nrows(), 1);
  }
}
TEST(OptimizationAnalysis, ValueOnlyAndBoundedOptimizerInterfacesRemainNative) {
  o::QuadraticObjective f({2, 8}, {1, -2});
  auto a = o::trace_optimize<o::ArbitraryFunction>(f, o::NelderMead(), {0, 0});
  EXPECT_TRUE(a.exception.empty());
  EXPECT_LT(a.value, 1e-12);
  o::PSOOptions opts;
  opts.max_iterations = 100;
  auto b = o::trace_optimize<o::ArbitraryFunction>(f, o::PSO(opts), {0, 0},
                                                   std::vector<double>{-3, -3},
                                                   std::vector<double>{3, 3});
  EXPECT_TRUE(b.exception.empty());
  EXPECT_LT(b.value, 1e-3);
}

TEST(OptimizationAnalysis, ComponentIndicesAndMeanCurvatureAreInspectable) {
  o::LeastSquaresObjective f({{1}, {2}}, {1, 3});
  EXPECT_NEAR(f.mean_loss_lipschitz(), 2.5, 1e-14);
  o::SVRGOptions c;
  c.step_size = .1;
  c.max_epochs = 1;
  c.inner_iterations = 2;
  c.tolerance = 0;
  const auto run =
      o::trace_optimize<o::DifferentiableSeparableFunction>(f, o::SVRG(c), {0});
  const auto req = run.trace.requests();
  EXPECT_EQ(req.nrows(), run.trace.calls());
  o::OptimizationTrace manual;
  manual.record("value", {1}, 1);
  EXPECT_THROW(manual.record("value", {1, 2}, 1), std::invalid_argument);
  EXPECT_EQ(manual.calls(), 1);
  const auto terms = run.trace.sample(f, "term_gradient");
  ASSERT_EQ(terms.nrows(), 4);
  EXPECT_DOUBLE_EQ(terms.double_at("component", 0),
                   terms.double_at("component", 1));
  EXPECT_DOUBLE_EQ(terms.double_at("component", 2),
                   terms.double_at("component", 3));
  EXPECT_NEAR(terms.double_at("x0", 2), .35, 1e-14);
  const double expected = terms.double_at("component", 2) == 0 ? .665 : .56;
  EXPECT_NEAR(run.coordinates[0], expected, 1e-14);
  EXPECT_THROW(f.gradient_term({0}, 2), std::invalid_argument);
  o::LeastSquaresObjective zero({{0}}, {1});
  EXPECT_DOUBLE_EQ(zero.mean_loss_lipschitz(), 0);
}
TEST(OptimizationAnalysis,
     OriginalCurveRecipesHaveAnalyticOffsetAndLogisticJacobians) {
  for (const auto* model : {"exponential_offset", "logistic"}) {
    const auto data = o::optimization_curve_data(model, 40, 42);
    o::CurveResidual f(data, model);
    const std::vector<double> p = model == std::string("logistic")
                                      ? std::vector<double>{10, 1.5, 5}
                                      : std::vector<double>{5, .7, 1};
    const auto errors = o::check_residual_jacobian(f, p);
    for (std::size_t i = 0; i < errors.nrows(); ++i)
      EXPECT_LT(errors.double_at("max_absolute_error", i), 1e-7);
    EXPECT_EQ(f.predict(p).size(), 40);
  }
}
