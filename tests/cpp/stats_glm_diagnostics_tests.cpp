#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/datasets/datasets.hpp>
#include <datamunge/stats/glm.hpp>
#include <limits>
#include <numeric>
using datamunge::dstruct::DataFrame;
using namespace datamunge::stats;
namespace {
DataFrame binary_groups() {
  DataFrame d;
  d.add_column("x", std::vector<double>{0, 0, 0, 0, 1, 1, 1, 1});
  d.add_column("y", std::vector<double>{0, 0, 0, 1, 0, 1, 1, 1});
  return d;
}
} // namespace
TEST(GLMDiagnostics, TwoBinaryGroupsHaveClosedFormCoefficientsAndInformation) {
  const GLM m(binary_groups(), "y ~ x", {.family = GLMFamily::Binomial});
  ASSERT_TRUE(m.converged());
  EXPECT_FALSE(m.boundary());
  EXPECT_NEAR(m.coefficients()[0], -std::log(3.0), 1e-10);
  EXPECT_NEAR(m.coefficients()[1], 2 * std::log(3.0), 1e-10);
  EXPECT_NEAR(m.covariance()(0, 0), 4.0 / 3, 1e-9);
  EXPECT_NEAR(m.covariance()(0, 1), -4.0 / 3, 1e-9);
  EXPECT_NEAR(m.covariance()(1, 1), 8.0 / 3, 1e-9);
  EXPECT_NEAR(std::accumulate(m.leverage().begin(), m.leverage().end(), 0.0), 2, 1e-12);
  const auto ci = m.predict(binary_groups(), GLMPredictionInterval::Confidence);
  EXPECT_NEAR(ci.se_fit[0], std::sqrt(3.0) / 8, 1e-10);
  EXPECT_EQ(m.coefficient_table().nrows(), 2u);
  EXPECT_EQ(m.model_summary().string_at("converged", 0), "yes");
}
TEST(GLMDiagnostics, InitialWorkingStateAndTraceReproduceActualUpdates) {
  const GLM  m(binary_groups(), "y ~ x", {.family = GLMFamily::Binomial});
  const auto w = m.working_data();
  EXPECT_DOUBLE_EQ(w.double_at("mu", 0), .25);
  EXPECT_DOUBLE_EQ(w.double_at("mu", 3), .75);
  EXPECT_DOUBLE_EQ(w.double_at("weight", 0), .1875);
  EXPECT_NEAR(w.double_at("z", 0), -std::log(3.0) - 4.0 / 3, 1e-12);
  const auto trace = m.iteration_history();
  EXPECT_NEAR(trace.double_at("beta:(Intercept)", 0), (-std::log(3.0) - 4.0 / 3) / 2, 1e-12);
  EXPECT_EQ(trace.nrows(), m.iterations());
  for (std::size_t i = 0; i < m.iterations(); ++i) {
    const auto input     = m.working_data(i);
    double     numerator = 0.0, denominator = 0.0;
    for (std::size_t j = 0; j < 4; ++j) {
      numerator += input.double_at("weight", j) * input.double_at("z", j);
      denominator += input.double_at("weight", j);
    }
    EXPECT_NEAR(trace.double_at("beta:(Intercept)", i), numerator / denominator, 1e-10);
    if (i)
      EXPECT_LE(trace.double_at("deviance", i), trace.double_at("deviance", i - 1) + 1e-10);
  }
  EXPECT_THROW((void)m.working_data(m.iterations()), std::out_of_range);
}
TEST(GLMDiagnostics, IterationLimitIsReportedAndCovarianceUsesReturnedMean) {
  const GLM m(binary_groups(), "y ~ x", {.family = GLMFamily::Binomial, .max_iter = 1});
  EXPECT_FALSE(m.converged());
  EXPECT_EQ(m.iterations(), 1u);
  const double p = m.fitted_values()[0], w = p * (1 - p);
  EXPECT_NEAR(m.covariance()(0, 0), 1 / (4 * w), 1e-12);
  EXPECT_GT(std::abs(m.covariance()(0, 0) - 1 / .75), .001);
  EXPECT_NE(m.summary().find("converged: no"), std::string::npos);
}
TEST(GLMDiagnostics, PoissonMeanInformationAndDevianceAreClosedForm) {
  DataFrame d;
  d.add_column("y", std::vector<double>{0, 1, 2, 5});
  const GLM m(d, "y ~ 1", {.family = GLMFamily::Poisson});
  EXPECT_TRUE(m.converged());
  EXPECT_NEAR(m.coefficients()[0], std::log(2.0), 1e-10);
  EXPECT_NEAR(m.covariance()(0, 0), 1.0 / 8, 1e-10);
  EXPECT_NEAR(m.deviance(), 2 * (-std::log(2.0) + 5 * std::log(2.5)), 1e-10);
  const auto pred = m.predict(d, GLMPredictionInterval::Confidence);
  EXPECT_NEAR(pred.se_fit[0], std::sqrt(.5), 1e-10);
}
TEST(GLMDiagnostics, GaussianWeightedAicAndDevianceResidualsMatchLikelihood) {
  DataFrame d;
  d.add_column("y", std::vector<double>{1, 2, 4, 7});
  d.add_column("w", std::vector<double>{1, 2, 3, 4});
  const GLM m(d, "y ~ 1", {.weights_column = "w"});
  EXPECT_NEAR(m.coefficients()[0], 4.5, 1e-12);
  double rss = 0.0;
  for (std::size_t i = 0; i < 4; ++i) {
    double r = std::sqrt(d.double_at("w", i)) * (d.double_at("y", i) - 4.5);
    EXPECT_NEAR(m.residuals()[i], r, 1e-12);
    rss += r * r;
  }
  EXPECT_NEAR(m.deviance(), rss, 1e-12);
  EXPECT_NEAR(
      m.aic(), 4 * (std::log(rss / 4 * 2 * std::acos(-1.0)) + 1) + 4 - std::log(24.0), 1e-12);
  EXPECT_NEAR(
      std::inner_product(m.residuals().begin(), m.residuals().end(), m.residuals().begin(), 0.0),
      rss,
      1e-12);
}
TEST(GLMDiagnostics, WeightedBinomialInformationAndScoresAreConsistent) {
  auto d = binary_groups();
  d.add_column("w", std::vector<double>{1, 2, 3, 4, 4, 3, 2, 1});
  const GLM m(d, "y ~ x", {.family = GLMFamily::Binomial, .weights_column = "w", .tol = 1e-12});
  EXPECT_NEAR(m.fitted_values()[0], .4, 1e-10);
  EXPECT_NEAR(m.fitted_values()[7], .6, 1e-10);
  EXPECT_NEAR(m.covariance()(0, 0), 1 / 2.4, 1e-10);
  double sum = 0;
  for (double r : m.residuals())
    sum += r * r;
  EXPECT_NEAR(sum, m.deviance(), 1e-11);
  const auto score = m.binomial_score(d);
  EXPECT_DOUBLE_EQ(score.double_at("accuracy", 0), .75);
  EXPECT_NEAR(score.double_at("brier", 0), .21, 1e-10); // deliberately unweighted
}
TEST(GLMDiagnostics, NullDevianceForNoInterceptUsesZeroLink) {
  const auto d = binary_groups();
  const GLM  b(d, "y ~ x - 1", {.family = GLMFamily::Binomial});
  EXPECT_NEAR(b.null_deviance(), 16 * std::log(2.0), 1e-10);
  DataFrame p;
  p.add_column("x", std::vector<double>{1, 2, 3, 4});
  p.add_column("y", std::vector<double>{1, 2, 2, 3});
  const GLM m(p, "y ~ x - 1", {.family = GLMFamily::Poisson});
  double    expected = 0;
  for (double y : std::vector<double>{1, 2, 2, 3})
    expected += 2 * (y * std::log(y) - y + 1);
  EXPECT_NEAR(m.null_deviance(), expected, 1e-10);
}
TEST(GLMDiagnostics, SeparatedDataRemainFiniteAndCarryBoundaryWarning) {
  DataFrame d;
  d.add_column("x", std::vector<double>{-2, -1, 1, 2});
  d.add_column("y", std::vector<double>{0, 0, 1, 1});
  const GLM m(d, "y ~ x", {.family = GLMFamily::Binomial, .max_iter = 50});
  EXPECT_TRUE(m.boundary());
  for (double b : m.coefficients())
    EXPECT_TRUE(std::isfinite(b));
  EXPECT_TRUE(std::isfinite(m.deviance()));
  EXPECT_NE(m.summary().find("check for separation"), std::string::npos);
  EXPECT_GT(m.iteration_history().double_at("beta:x", m.iterations() - 1), 10.0);
}
TEST(GLMDiagnostics, MissingRowsHaveOriginalTrainingIndicesAndAlignedPredictions) {
  DataFrame d;
  d.add_column("x", std::vector<std::optional<double>>{0, 1, std::nullopt, 3, 4});
  d.add_column("y", std::vector<double>{1, 3, 3, 6, 10});
  const GLM m(d, "y ~ x");
  EXPECT_EQ(m.kept_rows(), (std::vector<std::size_t>{0, 1, 3, 4}));
  const auto p = m.predict(d, GLMPredictionInterval::Confidence);
  EXPECT_EQ(p.fit.size(), 5u);
  EXPECT_TRUE(std::isnan(p.fit[2]));
  EXPECT_TRUE(std::isnan(p.lower[2]));
  EXPECT_DOUBLE_EQ(m.training_details().double_at("row", 2), 3);
}
TEST(GLMDiagnostics, InvalidControlsAndNonfiniteInputsAreRejected) {
  const auto d = binary_groups();
  for (double tol :
       {0.0,
        -1.0,
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::quiet_NaN()})
    EXPECT_THROW(GLM(d, "y ~ x", {.tol = tol}), std::invalid_argument);
  EXPECT_THROW(GLM(d, "y ~ x", {.family = static_cast<GLMFamily>(99)}), std::invalid_argument);
  auto bad = d;
  bad.remove_column("x");
  bad.add_column(
      "x", std::vector<double>{0, 0, 0, 0, 1, 1, 1, std::numeric_limits<double>::infinity()});
  EXPECT_THROW(GLM(bad, "y ~ x"), std::invalid_argument);
  bad = d;
  bad.remove_column("y");
  bad.add_column(
      "y", std::vector<double>{0, 0, 0, 0, 1, 1, 1, std::numeric_limits<double>::quiet_NaN()});
  EXPECT_THROW(GLM(bad, "y ~ x"), std::invalid_argument);
  bad = d;
  bad.add_column("w", std::vector<double>(8, std::numeric_limits<double>::infinity()));
  EXPECT_THROW(GLM(bad, "y ~ x", {.weights_column = "w"}), std::invalid_argument);
}
TEST(GLMDiagnostics, PredictionAndCoefficientLevelsAreValidated) {
  const auto d = binary_groups();
  const GLM  m(d, "y ~ x", {.family = GLMFamily::Binomial});
  for (double level : {0.0, 1.0, -.5, std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_THROW((void)m.confidence_intervals(level), std::invalid_argument);
    EXPECT_THROW((void)m.predict(d, GLMPredictionInterval::None, level), std::invalid_argument);
  }
  EXPECT_THROW((void)m.predict(d, static_cast<GLMPredictionInterval>(99)), std::invalid_argument);
  auto bad = d;
  bad.remove_column("x");
  bad.add_column("x", std::vector<double>(8, std::numeric_limits<double>::infinity()));
  EXPECT_THROW((void)m.predict(bad), std::invalid_argument);
}
TEST(GLMDiagnostics, BinaryScoresMatchBernoulliLogLikelihoodAndHandleNullResponses) {
  auto       d = binary_groups();
  const GLM  m(d, "y ~ x", {.family = GLMFamily::Binomial});
  const auto s = m.binomial_score(d);
  EXPECT_DOUBLE_EQ(s.double_at("accuracy", 0), .75);
  EXPECT_NEAR(s.double_at("brier", 0), .1875, 1e-10);
  EXPECT_NEAR(s.double_at("log_loss", 0), m.deviance() / 16, 1e-10);
  d.remove_column("y");
  d.add_column("y", std::vector<std::optional<double>>{std::nullopt, 0, 0, 1, 0, 1, 1, 1});
  EXPECT_DOUBLE_EQ(m.binomial_score(d).double_at("n", 0), 7);
  EXPECT_THROW((void)m.binomial_score(d, 0), std::invalid_argument);
  const GLM gaussian(binary_groups(), "y ~ x");
  EXPECT_THROW((void)gaussian.binomial_score(d), std::invalid_argument);
}
TEST(GLMDiagnostics, GammaInverseIntervalsRespectPositiveMeanDomain) {
  DataFrame d;
  d.add_column("y", std::vector<double>{.1, .2, .3, 20});
  const GLM  m(d, "y ~ 1", {.family = GLMFamily::Gamma, .tol = 1e-12});
  const auto p = m.predict(d, GLMPredictionInterval::Confidence);
  EXPECT_GT(p.lower[0], 0);
  EXPECT_TRUE(std::isinf(p.upper[0]));
  EXPECT_NEAR(m.fitted_values()[0], 5.15, 1e-9);
}
TEST(GLMDiagnostics, GammaQueriesOutsideInverseLinkDomainAreRejected) {
  const GLM m(
      datamunge::datasets::iris(),
      "Petal.Length ~ Sepal.Length + Sepal.Width",
      {.family = GLMFamily::Gamma});
  DataFrame q;
  q.add_column("Sepal.Length", std::vector<double>{100});
  q.add_column("Sepal.Width", std::vector<double>{3});
  EXPECT_THROW((void)m.predict(q), std::invalid_argument);
}
TEST(GLMDiagnostics, FractionalPoissonFitsDoNotClaimAProbabilityLikelihood) {
  DataFrame d;
  d.add_column("y", std::vector<double>{.1, .5, 1.2, 2.2});
  const GLM m(d, "y ~ 1", {.family = GLMFamily::Poisson});
  EXPECT_NEAR(m.fitted_values()[0], 1, 1e-10);
  EXPECT_TRUE(std::isnan(m.aic()));
}
