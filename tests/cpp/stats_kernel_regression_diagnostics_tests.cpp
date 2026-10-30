#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/stats/kernel_regression.hpp>
#include <limits>
using datamunge::dstruct::DataFrame;
using namespace datamunge::stats;
namespace {
DataFrame points(const std::vector<double>& x, const std::vector<double>& y) {
  DataFrame d;
  d.add_column("x", x);
  d.add_column("y", y);
  return d;
}
KernelRegressionOptions fixed(
    double                 h,
    KernelRegressionKernel kernel = KernelRegressionKernel::Gaussian) {
  KernelRegressionOptions o;
  o.bandwidth   = h;
  o.kernel      = kernel;
  o.standardize = false;
  return o;
}
} // namespace
TEST(KernelRegressionDiagnostics, ThreePointGaussianWeightsAndSmootherHaveClosedForms) {
  const auto             d = points({-1, 0, 1}, {0, 1, 0});
  const KernelRegression m(d, "y ~ x", fixed(1));
  const double           a = std::exp(-.5), b = std::exp(-2);
  const auto             s = m.smoothing_matrix();
  EXPECT_NEAR(s(1, 1), 1 / (1 + 2 * a), 1e-15);
  EXPECT_NEAR(s(0, 1), a / (1 + a + b), 1e-15);
  for (std::size_t i = 0; i < 3; ++i) {
    double sum = 0;
    for (std::size_t j = 0; j < 3; ++j)
      sum += s(i, j);
    EXPECT_NEAR(sum, 1, 1e-15);
    EXPECT_NEAR(m.full_fitted_values()[i], s(i, 1), 1e-15);
    const double deleted =
        (m.full_fitted_values()[i] - s(i, i) * d.double_at("y", i)) / (1 - s(i, i));
    EXPECT_NEAR(m.fitted_values()[i], deleted, 1e-15);
  }
  EXPECT_NEAR(m.fitted_values()[1], 0, 1e-15);
}
TEST(KernelRegressionDiagnostics, LocalConstantBoundaryBiasAndConvexHullAreExplicit) {
  const auto             d = points({0, 1, 2}, {0, 1, 2});
  const KernelRegression m(d, "y ~ x", fixed(1));
  DataFrame              q;
  q.add_column("x", std::vector<double>{0, 1, 2, 1000, -1000});
  const auto   pred = m.predict(q);
  const double a = std::exp(-.5), b = std::exp(-2);
  EXPECT_NEAR(pred[0], (a + 2 * b) / (1 + a + b), 1e-15);
  EXPECT_NEAR(pred[1], 1, 1e-15);
  EXPECT_GT(pred[0], 0);
  for (double value : pred) {
    EXPECT_GE(value, 0);
    EXPECT_LE(value, 2);
  }
  EXPECT_DOUBLE_EQ(pred[3], 2);
  EXPECT_DOUBLE_EQ(pred[4], 0);
}
TEST(KernelRegressionDiagnostics, GaussianLogWeightsRetainRelativeInformationFarFromData) {
  const KernelRegression m(points({0, 1, 2}, {0, 10, 20}), "y ~ x", fixed(.001));
  DataFrame              q;
  q.add_column("x", std::vector<double>{.5, 1.5, 1000, 1e100, -1e100});
  const auto pred = m.predict(q);
  EXPECT_DOUBLE_EQ(pred[0], 5);
  EXPECT_DOUBLE_EQ(pred[1], 15);
  EXPECT_DOUBLE_EQ(pred[2], 20);
  EXPECT_DOUBLE_EQ(pred[3], 20);
  EXPECT_DOUBLE_EQ(pred[4], 0);
  const auto detail = m.prediction_details(q);
  EXPECT_DOUBLE_EQ(detail.double_at("effective_n", 0), 2);
  EXPECT_DOUBLE_EQ(detail.double_at("effective_n", 2), 1);
}
TEST(KernelRegressionDiagnostics, DuplicatesAverageAndDeletionExcludesOneResponseOnly) {
  const auto             d = points({0, 0, 1}, {0, 2, 4});
  const KernelRegression m(d, "y ~ x", fixed(.001));
  EXPECT_DOUBLE_EQ(m.full_fitted_values()[0], 1);
  EXPECT_DOUBLE_EQ(m.full_fitted_values()[1], 1);
  EXPECT_DOUBLE_EQ(m.fitted_values()[0], 2);
  EXPECT_DOUBLE_EQ(m.fitted_values()[1], 0);
  EXPECT_DOUBLE_EQ(m.fitted_values()[2], 1);
  EXPECT_NEAR(m.rmse(), std::sqrt(17.0 / 3), 1e-15);
  EXPECT_NEAR(m.model_summary().double_at("effective_df", 0), 2, 1e-15);
}
TEST(KernelRegressionDiagnostics, CompactSupportEndpointsDistinguishUniformFromTaperedKernels) {
  const auto d = points({0, 2}, {0, 2});
  DataFrame  q;
  q.add_column("x", std::vector<double>{1, 100});
  const KernelRegression uniform(d, "y ~ x", fixed(1, KernelRegressionKernel::Uniform));
  EXPECT_DOUBLE_EQ(uniform.predict(q)[0], 1);
  EXPECT_TRUE(std::isnan(uniform.predict(q)[1]));
  for (auto k : {KernelRegressionKernel::Triangular, KernelRegressionKernel::Epanechnikov}) {
    const KernelRegression m(d, "y ~ x", fixed(1, k));
    EXPECT_TRUE(std::isnan(m.predict(q)[0]));
    EXPECT_TRUE(std::isnan(m.predict(q)[1]));
  }
}
TEST(KernelRegressionDiagnostics, CoveragePreventsAnOptimisticPartialCvWinner) {
  const auto             d = points({0, 0, 10}, {0, 2, 100});
  auto                   o = fixed(.1, KernelRegressionKernel::Uniform);
  const KernelRegression small(d, "y ~ x", o);
  const auto             diag = small.bandwidth_diagnostics();
  EXPECT_DOUBLE_EQ(diag.double_at("loo_n", 0), 2);
  EXPECT_DOUBLE_EQ(diag.double_at("conditional_mse", 0), 4);
  EXPECT_TRUE(std::isinf(diag.double_at("objective_mse", 0)));
  o.bandwidth            = -1;
  o.bandwidth_candidates = {.1, 11};
  const KernelRegression selected(d, "y ~ x", o);
  EXPECT_DOUBLE_EQ(selected.bandwidth(), 11);
  const auto path = selected.bandwidth_diagnostics();
  EXPECT_DOUBLE_EQ(path.double_at("eligible", 0), 0);
  EXPECT_DOUBLE_EQ(path.double_at("eligible", 1), 1);
  o.bandwidth_candidates = {.1, 2};
  EXPECT_THROW(KernelRegression(d, "y ~ x", o), std::runtime_error);
}
TEST(KernelRegressionDiagnostics, AutomaticTiesUseCallerGridOrderAndMarkBoundary) {
  auto o                 = fixed(-1);
  o.bandwidth_candidates = {2, .2, 1};
  const KernelRegression m(points({0, 1, 2}, {5, 5, 5}), "y ~ x", o);
  EXPECT_DOUBLE_EQ(m.bandwidth(), 2);
  EXPECT_DOUBLE_EQ(m.model_summary().double_at("search_boundary", 0), 1);
  EXPECT_TRUE(std::isnan(m.r_squared()));
  EXPECT_DOUBLE_EQ(m.rmse(), 0);
}
TEST(KernelRegressionDiagnostics, StandardizationUsesPopulationSdAndPreservesSmallUnitRescaling) {
  const auto d  = points({0, 1, 2}, {0, 1, 3});
  auto       o  = fixed(.5);
  o.standardize = true;
  const KernelRegression m(d, "y ~ x", o);
  EXPECT_DOUBLE_EQ(m.feature_means()[0], 1);
  EXPECT_NEAR(m.feature_scales()[0], std::sqrt(2.0 / 3), 1e-15);
  const KernelRegression tiny(points({0, 1e-100, 2e-100}, {0, 1, 3}), "y ~ x", o);
  for (std::size_t i = 0; i < 3; ++i)
    EXPECT_NEAR(tiny.fitted_values()[i], m.fitted_values()[i], 1e-14);
}
TEST(KernelRegressionDiagnostics, JointOmissionPreservesOriginalTrainingAndQueryRows) {
  DataFrame d;
  d.add_column("x", std::vector<std::optional<double>>{0, std::nullopt, 1, 2});
  d.add_column("y", std::vector<double>{0, 99, 1, 3});
  const KernelRegression m(d, "y ~ x", fixed(1));
  EXPECT_EQ(m.kept_row_indices(), (std::vector<std::size_t>{0, 2, 3}));
  EXPECT_DOUBLE_EQ(m.training_details().double_at("row", 1), 2);
  DataFrame q;
  q.add_column(
      "x",
      std::vector<std::optional<double>>{0, std::nullopt, std::numeric_limits<double>::infinity()});
  const auto detail = m.prediction_details(q);
  const auto w      = m.query_weights(q);
  EXPECT_EQ(detail.string_at("status", 0), "ok");
  EXPECT_EQ(detail.string_at("status", 1), "missing_predictor");
  EXPECT_EQ(detail.string_at("status", 2), "nonfinite_predictor");
  EXPECT_TRUE(std::isnan(w(1, 0)));
  EXPECT_TRUE(std::isnan(m.predict(q)[2]));
  DataFrame missing;
  missing.add_column("x", std::vector<std::optional<double>>{std::nullopt, std::nullopt});
  EXPECT_EQ(m.predict(missing).size(), 2u);
  EXPECT_TRUE(std::isnan(m.predict(missing)[0]));
}
TEST(KernelRegressionDiagnostics, EmptyQueriesAndUnseenLevelsHaveExplicitContracts) {
  const auto             d = points({0, 1, 2}, {0, 1, 3});
  const KernelRegression m(d, "y ~ x", fixed(1));
  EXPECT_TRUE(m.predict(DataFrame{}).empty());
  EXPECT_EQ(m.prediction_details(DataFrame{}).nrows(), 0u);
  DataFrame categorical;
  categorical.add_column("g", std::vector<std::string>{"A", "A", "B", "B"});
  categorical.add_column("y", std::vector<double>{0, 1, 2, 3});
  const KernelRegression grouped(categorical, "y ~ g", fixed(1));
  DataFrame              unknown;
  unknown.add_column("g", std::vector<std::string>{"C"});
  EXPECT_THROW((void)grouped.predict(unknown), std::runtime_error);
}
TEST(KernelRegressionDiagnostics, ScoreUsesTrainingMeanAndOnlyCoveredResponsePairs) {
  const KernelRegression m(
      points({0, 1, 2}, {0, 1, 2}), "y ~ x", fixed(1, KernelRegressionKernel::Uniform));
  const auto d     = points({1, 100}, {1, 100});
  const auto score = m.score(d);
  EXPECT_DOUBLE_EQ(score.double_at("n_predicted", 0), 1);
  EXPECT_DOUBLE_EQ(score.double_at("coverage", 0), .5);
  EXPECT_DOUBLE_EQ(score.double_at("rmse", 0), 0);
  EXPECT_DOUBLE_EQ(score.double_at("mean_baseline_rmse", 0), 0);
  EXPECT_TRUE(std::isnan(score.double_at("r_squared", 0)));
  EXPECT_DOUBLE_EQ(m.score(points({1}, {4})).double_at("mean_baseline_rmse", 0), 3);
  const auto none = m.score(points({100}, {2}));
  EXPECT_DOUBLE_EQ(none.double_at("coverage", 0), 0);
  EXPECT_TRUE(std::isnan(none.double_at("rmse", 0)));
  DataFrame no_response;
  no_response.add_column("y", std::vector<std::optional<double>>{std::nullopt});
  EXPECT_DOUBLE_EQ(m.score(no_response).double_at("n_response", 0), 0);
}
TEST(KernelRegressionDiagnostics, NoLooCoverageDoesNotReportPerfectR2OrInventPlotPoints) {
  const KernelRegression m(
      points({0, 10, 20}, {0, 1, 2}), "y ~ x", fixed(.1, KernelRegressionKernel::Uniform));
  EXPECT_TRUE(std::isnan(m.r_squared()));
  EXPECT_TRUE(std::isnan(m.rmse()));
  EXPECT_DOUBLE_EQ(m.model_summary().double_at("loo_coverage", 0), 0);
  EXPECT_THROW((void)m.plot_predicted_vs_actual(), std::invalid_argument);
  EXPECT_THROW((void)m.plot_residuals_vs_fitted(), std::invalid_argument);
}
TEST(KernelRegressionDiagnostics, InvalidOptionsFiniteDomainsAndPlotContractsAreRejected) {
  const auto d = points({0, 1, 2}, {0, 1, 2});
  auto       o = fixed(1);
  o.bandwidth  = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(KernelRegression(d, "y ~ x", o), std::invalid_argument);
  o        = fixed(1);
  o.kernel = static_cast<KernelRegressionKernel>(99);
  EXPECT_THROW(KernelRegression(d, "y ~ x", o), std::invalid_argument);
  o                      = fixed(-1);
  o.bandwidth_candidates = {0};
  EXPECT_THROW(KernelRegression(d, "y ~ x", o), std::invalid_argument);
  EXPECT_THROW(KernelRegression(points({1}, {2}), "y ~ x", fixed(1)), std::invalid_argument);
  EXPECT_THROW(
      KernelRegression(
          points({0, 1, 2}, {0, 1, std::numeric_limits<double>::infinity()}), "y ~ x", fixed(1)),
      std::invalid_argument);
  const KernelRegression m(d, "y ~ x", fixed(1));
  EXPECT_THROW((void)m.plot_fit(d, 1), std::invalid_argument);
  const KernelRegression transformed(d, "y ~ I(x^2)", fixed(1));
  EXPECT_THROW((void)transformed.plot_fit(d), std::invalid_argument);
  DataFrame wrong;
  wrong.add_column("z", std::vector<double>{1});
  EXPECT_THROW((void)m.predict(wrong), std::invalid_argument);
}
