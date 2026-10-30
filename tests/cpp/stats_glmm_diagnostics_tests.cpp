#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/stats/glm.hpp>
#include <datamunge/stats/glmm.hpp>
#include <limits>
#include <numeric>
using datamunge::dstruct::DataFrame;
using datamunge::linalg::DenseMatrix;
using namespace datamunge::stats;
namespace {
DataFrame counts() {
  DataFrame d;
  d.add_column("y", std::vector<double>{1, 3, 5, 7});
  d.add_column("g", std::vector<std::string>{"A", "A", "B", "B"});
  return d;
}
DataFrame binary() {
  DataFrame d;
  d.add_column("y", std::vector<double>{0, 0, 0, 1, 0, 1, 1, 1});
  d.add_column("g", std::vector<std::string>{"A", "A", "A", "A", "B", "B", "B", "B"});
  return d;
}
GLMMOptions poisson(double tau, std::size_t iterations = 50) {
  GLMMOptions opt;
  opt.family                    = GLMMFamily::Poisson;
  opt.fixed_random_intercept_sd = tau;
  opt.max_iterations            = iterations;
  return opt;
}
} // namespace
TEST(GLMMDiagnostics, PoissonFirstUpdateHasHandComputedShrinkage) {
  const GLMM m(counts(), "y ~ 1 + (1|g)", poisson(1, 1));
  const auto w = m.working_data();
  EXPECT_EQ(m.iterations(), 1u);
  EXPECT_FALSE(m.converged());
  EXPECT_NEAR(m.coefficients()[0], std::log(4.0), 1e-12);
  EXPECT_NEAR(m.random_effects()[0][0], -4.0 / 9, 1e-12);
  EXPECT_NEAR(m.random_effects()[1][0], 4.0 / 9, 1e-12);
  EXPECT_NEAR(w.double_at("weight", 0), 4, 1e-12);
  EXPECT_NEAR(w.double_at("z", 0), std::log(4.0) - .75, 1e-12);
  EXPECT_DOUBLE_EQ(m.model_summary().double_at("groups", 0), 2);
  EXPECT_EQ(m.random_effects_frame().nrows(), 2u);
  EXPECT_EQ(m.iteration_history().nrows(), 1u);
}
TEST(GLMMDiagnostics, BinomialFirstUpdateCannotConvergeOnFixedEffectsAlone) {
  GLMMOptions opt;
  opt.fixed_random_intercept_sd = 1;
  opt.max_iterations            = 1;
  const GLMM m(binary(), "y ~ 1 + (1|g)", opt);
  EXPECT_NEAR(m.coefficients()[0], 0, 1e-12);
  EXPECT_NEAR(m.random_effects()[0][0], -.5, 1e-12);
  EXPECT_NEAR(m.random_effects()[1][0], .5, 1e-12);
  EXPECT_FALSE(m.converged());
  const auto w = m.working_data();
  EXPECT_NEAR(w.double_at("weight", 0), .25, 1e-12);
  EXPECT_NEAR(w.double_at("z", 0), -2, 1e-12);
}
TEST(GLMMDiagnostics, EstimatedFirstWorkingVarianceMatchesAnalyticMaximum) {
  GLMMOptions opt;
  opt.family         = GLMMFamily::Poisson;
  opt.max_iterations = 1;
  const GLMM m(counts(), "y ~ 1 + (1|g)", opt);
  EXPECT_NEAR(m.random_effect_covariance()(0, 0), .125, 1e-7);
  EXPECT_NEAR(m.random_effects()[0][0], -.25, 1e-7);
  for (double tau : {0.0, .2, .5, 1.0}) {
    // Two independent group-mean variances are tau^2+1/8; contrast variance is fixed.
    const double variance = tau * tau + .125;
    const double expected = 4 * std::log(2 * std::acos(-1.0)) +
                            2 * (-std::log(4.0) + std::log(.25 + 2 * tau * tau)) + 1.0 +
                            .5 / variance;
    EXPECT_NEAR(m.working_criterion_at({tau}), expected, 1e-11);
  }
  EXPECT_THROW((void)m.working_criterion_at({1, 2}), std::invalid_argument);
  EXPECT_THROW((void)m.working_data(m.iterations()), std::out_of_range);
}
TEST(GLMMDiagnostics, KnownVariancePqlFixedPointSatisfiesPenalizedPoissonScores) {
  auto opt = poisson(.8);
  opt.tol  = 1e-10;
  const GLMM m(counts(), "y ~ 1 + (1|g)", opt);
  ASSERT_TRUE(m.converged());
  double total = 0;
  for (std::size_t g = 0; g < 2; ++g) {
    double sum = 0;
    for (std::size_t i = 2 * g; i < 2 * g + 2; ++i)
      sum += counts().double_at("y", i) - m.fitted_values()[i];
    EXPECT_NEAR(sum, m.random_effects()[g][0] / (.8 * .8), 1e-8);
    total += sum;
  }
  EXPECT_NEAR(total, 0, 1e-8);
  EXPECT_GT(m.iterations(), 1u);
}
TEST(GLMMDiagnostics, ZeroKnownVarianceReducesToGlm) {
  const GLMM mixed(counts(), "y ~ 1 + (1|g)", poisson(0));
  const GLM  ordinary(counts(), "y ~ 1", {.family = GLMFamily::Poisson});
  EXPECT_TRUE(mixed.converged());
  EXPECT_TRUE(mixed.covariance_at_boundary());
  EXPECT_NEAR(mixed.coefficients()[0], ordinary.coefficients()[0], 1e-9);
  EXPECT_NEAR(mixed.covariance()(0, 0), ordinary.covariance()(0, 0), 1e-9);
  EXPECT_DOUBLE_EQ(mixed.random_effects()[0][0], 0);
  EXPECT_NEAR(mixed.aic() - mixed.working_criterion(), 2, 1e-12);
}
TEST(GLMMDiagnostics, DenseWorkingModelChecksCorrelatedRandomSlopeProducts) {
  auto        d = simulate_glmm_random_intercepts(GLMMFamily::Poisson, 3, 4, 1, .4, .5, 11);
  GLMMOptions opt;
  opt.family             = GLMMFamily::Poisson;
  opt.max_iterations     = 1;
  opt.de_population_size = 10;
  opt.de_max_generations = 20;
  const GLMM          m(d, "y ~ x + (1+x|group)", opt);
  const auto          work = m.working_data();
  const auto          n    = d.nrows();
  DenseMatrix<double> X(n, 2), Z(n, 2);
  for (std::size_t i = 0; i < n; ++i) {
    X(i, 0) = Z(i, 0) = 1;
    X(i, 1) = Z(i, 1) = d.double_at("x", i);
  }
  // An explicitly non-diagonal factor exercises Z*Lambda rather than Z*Lambda-transpose.
  const DenseMatrix<double> L(2, 2, {.7, 0, .4, .5});
  const auto                D = L * L.transpose();
  DenseMatrix<double>       V(n, n, 0.0);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) {
      if (i == j)
        V(i, j) = 1 / work.double_at("weight", i);
      if (d.string_at("group", i) == d.string_at("group", j))
        for (std::size_t a = 0; a < 2; ++a)
          for (std::size_t b = 0; b < 2; ++b)
            V(i, j) += Z(i, a) * D(a, b) * Z(j, b);
    }
  const auto chol = datamunge::linalg::cholesky(V);
  ASSERT_TRUE(chol.ok);
  const auto wz     = chol.solve(work.numeric_column("z").values());
  const auto wx     = chol.solve(X);
  const auto info   = X.transpose() * wx;
  const auto beta   = datamunge::linalg::cholesky(info).solve(X.transpose() * wz);
  auto       r      = work.numeric_column("z").values();
  const auto fitted = X * beta;
  for (std::size_t i = 0; i < n; ++i)
    r[i] -= fitted[i];
  const auto wr = chol.solve(r);
  double     criterion =
      n * std::log(2 * std::acos(-1.0)) + std::inner_product(r.begin(), r.end(), wr.begin(), 0.0);
  for (std::size_t i = 0; i < n; ++i)
    criterion += 2 * std::log(chol.L(i, i));
  EXPECT_NEAR(m.working_criterion_at({.7, .4, .5}), criterion, 1e-10);
  // Check the actual first-step beta and BLUPs by a second dense solve at fitted D.
  const auto& actualD = m.random_effect_covariance();
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) {
      V(i, j) = i == j ? 1 / work.double_at("weight", i) : 0;
      if (d.string_at("group", i) == d.string_at("group", j))
        for (std::size_t a = 0; a < 2; ++a)
          for (std::size_t b = 0; b < 2; ++b)
            V(i, j) += Z(i, a) * actualD(a, b) * Z(j, b);
    }
  const auto c    = datamunge::linalg::cholesky(V);
  const auto invX = c.solve(X);
  const auto invz = c.solve(work.numeric_column("z").values());
  const auto B    = datamunge::linalg::cholesky(X.transpose() * invX).solve(X.transpose() * invz);
  for (std::size_t j = 0; j < 2; ++j)
    EXPECT_NEAR(m.coefficients()[j], B[j], 1e-9);
  r             = work.numeric_column("z").values();
  const auto XB = X * B;
  for (std::size_t i = 0; i < n; ++i)
    r[i] -= XB[i];
  const auto u = c.solve(r);
  for (std::size_t g = 0; g < 3; ++g)
    for (std::size_t a = 0; a < 2; ++a) {
      double b = 0;
      for (std::size_t i = 4 * g; i < 4 * g + 4; ++i)
        for (std::size_t k = 0; k < 2; ++k)
          b += actualD(a, k) * Z(i, k) * u[i];
      EXPECT_NEAR(m.random_effects()[g][a], b, 1e-9);
    }
}
TEST(GLMMDiagnostics, NumericGroupKeysKeepCloseDistinctValuesSeparate) {
  auto d = counts();
  d.remove_column("g");
  d.add_column("g", std::vector<double>{1.0000001, 1.0000001, 1.0000002, 1.0000002});
  const GLMM m(d, "y ~ 1 + (1|g)", poisson(1));
  ASSERT_EQ(m.num_groups(), 2u);
  const auto p = m.predict(d);
  EXPECT_NE(p[0], p[2]);
  EXPECT_NEAR(p[0], m.fitted_values()[0], 1e-9);
}
TEST(GLMMDiagnostics, PredictionsDistinguishKnownUnseenAndMissingGroups) {
  const GLMM m(counts(), "y ~ 1 + (1|g)", poisson(1));
  DataFrame  q;
  q.add_column("g", std::vector<std::optional<std::string>>{"A", "new", std::nullopt});
  const auto p = m.prediction_frame(q);
  EXPECT_EQ(p.string_at("group_status", 0), "known");
  EXPECT_EQ(p.string_at("group_status", 1), "unseen");
  EXPECT_EQ(p.string_at("group_status", 2), "missing");
  EXPECT_NEAR(
      p.double_at("fit", 0), std::exp(m.coefficients()[0] + m.random_effects()[0][0]), 1e-12);
  EXPECT_NEAR(p.double_at("fit", 1), std::exp(m.coefficients()[0]), 1e-12);
  EXPECT_DOUBLE_EQ(p.double_at("fit", 1), p.double_at("fit", 2));
  EXPECT_NE(p.double_at("fit", 0), p.double_at("fixed_fit", 0));
}
TEST(GLMMDiagnostics, NullRandomSlopesAreOmittedInsteadOfSilentlySetToZero) {
  auto        d = simulate_glmm_random_intercepts(GLMMFamily::Poisson, 3, 4, 1, .4, .5, 11);
  GLMMOptions opt;
  opt.family             = GLMMFamily::Poisson;
  opt.max_iterations     = 1;
  opt.de_population_size = 8;
  opt.de_max_generations = 5;
  const GLMM m(d, "y ~ 1 + (1+x|group)", opt);
  DataFrame  q;
  q.add_column("group", std::vector<std::string>{"g1", "unknown"});
  q.add_column("x", std::vector<std::optional<double>>{std::nullopt, std::nullopt});
  const auto p = m.prediction_frame(q);
  EXPECT_TRUE(std::isnan(p.double_at("fit", 0)));
  EXPECT_EQ(p.string_at("group_status", 0), "incomplete");
  EXPECT_TRUE(std::isfinite(p.double_at("fit", 1)));
  q.remove_column("x");
  EXPECT_THROW((void)m.predict(q), std::invalid_argument);
  EXPECT_TRUE(std::isfinite(m.predict_fixed(q)[0]));
}
TEST(GLMMDiagnostics, JointOmissionRetainsOriginalIndices) {
  DataFrame d;
  d.add_column("y", std::vector<std::optional<double>>{1, 2, std::nullopt, 3, 4, 5});
  d.add_column("g", std::vector<std::optional<std::string>>{"A", std::nullopt, "A", "A", "B", "B"});
  const GLMM m(d, "y ~ 1 + (1|g)", poisson(.5));
  EXPECT_EQ(m.kept_rows(), (std::vector<std::size_t>{0, 3, 4, 5}));
  EXPECT_DOUBLE_EQ(m.working_data().double_at("row", 1), 3);
  EXPECT_DOUBLE_EQ(m.training_details().double_at("row", 2), 4);
}
TEST(GLMMDiagnostics, BinaryScoresHaveClosedFormLosses) {
  GLMMOptions opt;
  opt.fixed_random_intercept_sd = 0;
  const auto d                  = binary();
  const GLMM m(d, "y ~ g + (1|g)", opt);
  const auto s = m.binomial_score(d);
  EXPECT_DOUBLE_EQ(s.double_at("n", 0), 8);
  EXPECT_DOUBLE_EQ(s.double_at("accuracy", 0), .75);
  EXPECT_NEAR(s.double_at("brier", 0), .1875, 1e-9);
  EXPECT_NEAR(s.double_at("log_loss", 0), -(.25 * std::log(.25) + .75 * std::log(.75)), 1e-9);
  EXPECT_THROW((void)m.binomial_score(d, 1), std::invalid_argument);
}
TEST(GLMMDiagnostics, GeneratorKeepsKnownGroupEffectsAcrossClusterSizes) {
  const auto a = simulate_glmm_random_intercepts(GLMMFamily::Poisson, 4, 4, 1, .3, .5, 42);
  const auto b = simulate_glmm_random_intercepts(GLMMFamily::Poisson, 4, 10, 1, .3, .5, 42);
  for (std::size_t g = 0; g < 4; ++g) {
    EXPECT_DOUBLE_EQ(a.double_at("true_effect", 4 * g), b.double_at("true_effect", 10 * g));
    for (std::size_t i = 4 * g; i < 4 * g + 4; ++i) {
      EXPECT_DOUBLE_EQ(a.double_at("true_effect", i), a.double_at("true_effect", 4 * g));
      EXPECT_DOUBLE_EQ(a.double_at("y", i), std::floor(a.double_at("y", i)));
    }
  }
}
TEST(GLMMDiagnostics, ControlsAndNonfiniteDomainsAreRejected) {
  for (double tol : {0.0, -1.0, std::numeric_limits<double>::infinity()}) {
    auto opt = poisson(1);
    opt.tol  = tol;
    EXPECT_THROW(GLMM(counts(), "y ~ 1 + (1|g)", opt), std::invalid_argument);
  }
  auto opt           = poisson(1);
  opt.max_iterations = 0;
  EXPECT_THROW(GLMM(counts(), "y ~ 1 + (1|g)", opt), std::invalid_argument);
  opt = poisson(-1);
  EXPECT_THROW(GLMM(counts(), "y ~ 1 + (1|g)", opt), std::invalid_argument);
  auto d = counts();
  d.remove_column("y");
  d.add_column("y", std::vector<double>{1, 2, 3, std::numeric_limits<double>::quiet_NaN()});
  EXPECT_THROW(GLMM(d, "y ~ 1 + (1|g)", poisson(1)), std::invalid_argument);
  d.remove_column("y");
  d.add_column("y", std::vector<double>{.5, 1, 2, 3});
  EXPECT_THROW(GLMM(d, "y ~ 1 + (1|g)", poisson(1)), std::invalid_argument);
  EXPECT_THROW(GLMM(d, "y ~ 1 + (1|g)"), std::invalid_argument);
}
TEST(GLMMDiagnostics, WorkingCriteriaAreExplicitlyLabelled) {
  const GLMM m(counts(), "y ~ 1 + (1|g)", poisson(1));
  EXPECT_DOUBLE_EQ(m.deviance(), m.working_criterion());
  EXPECT_NE(m.summary().find("not marginal GLMM"), std::string::npos);
  EXPECT_THROW((void)m.random_effect_correlation(1, 0), std::out_of_range);
}

TEST(GLMMDiagnostics, ScoringDropsNullResponsesBeforeValidatingQueryPredictors) {
  auto d = binary();
  d.add_column("x", std::vector<double>{0, 0, 0, 0, 1, 1, 1, 1});
  GLMMOptions opt;
  opt.fixed_random_intercept_sd = 0;
  const GLMM m(d, "y ~ x + (1|g)", opt);
  d.remove_column("y");
  d.add_column("y", std::vector<std::optional<double>>{std::nullopt, 0, 0, 1, 0, 1, 1, 1});
  d.remove_column("x");
  d.add_column(
      "x", std::vector<double>{std::numeric_limits<double>::infinity(), 0, 0, 0, 1, 1, 1, 1});
  const auto result = m.binomial_score(d);
  EXPECT_DOUBLE_EQ(result.double_at("n", 0), 7);
  EXPECT_NEAR(result.double_at("brier", 0), 1.4375 / 7, 1e-9);
}
