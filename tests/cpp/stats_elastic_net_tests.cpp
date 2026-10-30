#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <datamunge/datasets/datasets.hpp>
#include <datamunge/dstruct/dstruct.hpp>
#include <datamunge/stats/stats.hpp>
#include <limits>

using datamunge::dstruct::DataFrame;
using datamunge::stats::ElasticNet;
using datamunge::stats::ElasticNetOptions;
using datamunge::stats::Lasso;
using datamunge::stats::LassoOptions;
using datamunge::stats::Ridge;
using datamunge::stats::RidgeOptions;

namespace {
constexpr const char* kFormula = "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width";
}

// Reference values computed with an independent numpy coordinate-descent
// implementation of the identical glmnet-style objective
// ((1/2n)*RSS + lambda*alpha*L1 + lambda*(1-alpha)/2*L2, predictors
// standardized to population mean 0 / variance 1, response centered), and
// cross-checked to ~1e-9 against scikit-learn's ElasticNet fit directly on
// the same standardized data with fit_intercept=False (sklearn uses the
// identical parameterization: its `alpha` is glmnet's lambda, its
// `l1_ratio` is glmnet's alpha). Fit on the embedded iris dataset:
// Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width.

TEST(ElasticNet, RidgeFixedLambdaMatchesOracle) {
  ElasticNetOptions options;
  options.alpha  = 0.0;
  options.lambda = 0.5;
  ElasticNet model(datamunge::datasets::iris(), kFormula, options);

  const auto& coef = model.coefficients();
  ASSERT_EQ(coef.size(), 3u);
  EXPECT_NEAR(coef[0], 0.7273280846, 1e-6);
  EXPECT_NEAR(coef[1], -0.6357292217, 1e-6);
  EXPECT_NEAR(coef[2], 0.9670366388, 1e-6);
  EXPECT_NEAR(model.intercept(), 0.2918164237787053, 1e-6);
  EXPECT_NEAR(model.r_squared(), 0.9244183748297947, 1e-6);
  EXPECT_FALSE(model.lambda_was_selected());
}

TEST(ElasticNet, LassoFixedLambdaMatchesOracleAndZeroesOutAPredictor) {
  ElasticNetOptions options;
  options.alpha  = 1.0;
  options.lambda = 0.3;
  ElasticNet model(datamunge::datasets::iris(), kFormula, options);

  const auto& coef = model.coefficients();
  ASSERT_EQ(coef.size(), 3u);
  EXPECT_NEAR(coef[0], 0.3423021657, 1e-6);
  EXPECT_NEAR(coef[1], 0.0, 1e-9); // Sepal.Width is shrunk exactly to zero
  EXPECT_NEAR(coef[2], 1.530880997, 1e-6);
  EXPECT_NEAR(model.intercept(), -0.0782222637780996, 1e-6);
  EXPECT_NEAR(model.r_squared(), 0.9165375189333077, 1e-6);
  EXPECT_EQ(model.non_zero_coefficients(), 2u);
}

TEST(ElasticNet, MixedAlphaFixedLambdaMatchesOracle) {
  ElasticNetOptions options;
  options.alpha  = 0.5;
  options.lambda = 0.2;
  ElasticNet model(datamunge::datasets::iris(), kFormula, options);

  const auto& coef = model.coefficients();
  ASSERT_EQ(coef.size(), 3u);
  EXPECT_NEAR(coef[0], 0.6946763436, 1e-6);
  EXPECT_NEAR(coef[1], -0.4986995754, 1e-6);
  EXPECT_NEAR(coef[2], 1.2514772454, 1e-6);
  EXPECT_NEAR(model.intercept(), -0.2774729754454719, 1e-6);
  EXPECT_NEAR(model.r_squared(), 0.9547003034586754, 1e-6);
}

TEST(ElasticNet, RejectsAlphaOutOfRange) {
  ElasticNetOptions options;
  options.alpha = 1.5;
  EXPECT_THROW(ElasticNet(datamunge::datasets::iris(), kFormula, options), std::invalid_argument);
}

TEST(ElasticNet, LargerLambdaShrinksCoefficientNormTowardZero) {
  ElasticNetOptions small, large;
  small.alpha  = 0.5;
  small.lambda = 0.05;
  large.alpha  = 0.5;
  large.lambda = 1.0;

  ElasticNet loose(datamunge::datasets::iris(), kFormula, small);
  ElasticNet tight(datamunge::datasets::iris(), kFormula, large);

  auto l2norm = [](const std::vector<double>& v) {
    double s = 0.0;
    for (const double x : v)
      s += x * x;
    return std::sqrt(s);
  };
  EXPECT_LT(l2norm(tight.coefficients()), l2norm(loose.coefficients()));
}

TEST(ElasticNet, LassoPathReachesFullSparsityAtLargeLambda) {
  ElasticNetOptions options;
  options.alpha  = 1.0;
  options.lambda = 10.0;
  ElasticNet model(datamunge::datasets::iris(), kFormula, options);
  EXPECT_EQ(model.non_zero_coefficients(), 0u);
  for (const double c : model.coefficients())
    EXPECT_NEAR(c, 0.0, 1e-9);
}

TEST(ElasticNet, AutoSelectedLambdaLiesWithinSweptPathAndImprovesOnEndpoints) {
  ElasticNetOptions options; // lambda left at -1 (auto)
  options.alpha = 0.5;
  ElasticNet model(datamunge::datasets::iris(), kFormula, options);

  EXPECT_TRUE(model.lambda_was_selected());
  ASSERT_FALSE(model.lambda_path().empty());
  ASSERT_EQ(model.lambda_path().size(), model.cv_mean_squared_error().size());

  const auto&  path = model.lambda_path();
  const double lo   = *std::min_element(path.begin(), path.end());
  const double hi   = *std::max_element(path.begin(), path.end());
  EXPECT_GE(model.lambda(), lo);
  EXPECT_LE(model.lambda(), hi);

  // The selected lambda's CV error should be at least as good as the
  // largest-lambda (most-shrunk, closest-to-null-model) endpoint.
  const auto& cv          = model.cv_mean_squared_error();
  const auto  selected_it = std::find(path.begin(), path.end(), model.lambda());
  ASSERT_NE(selected_it, path.end());
  const auto selected_index = static_cast<std::size_t>(selected_it - path.begin());
  EXPECT_LE(cv[selected_index], cv.back() + 1e-9);
}

TEST(ElasticNet, PredictMatchesFittedValuesOnTrainingData) {
  ElasticNetOptions options;
  options.alpha   = 0.5;
  options.lambda  = 0.2;
  const auto iris = datamunge::datasets::iris();
  ElasticNet model(iris, kFormula, options);

  const auto preds = model.predict(iris);
  ASSERT_EQ(preds.size(), model.fitted_values().size());
  for (std::size_t i = 0; i < preds.size(); ++i)
    EXPECT_NEAR(preds[i], model.fitted_values()[i], 1e-9);
}

TEST(ElasticNet, SummaryContainsKeyInformation) {
  ElasticNetOptions options;
  options.alpha  = 0.5;
  options.lambda = 0.2;
  ElasticNet model(datamunge::datasets::iris(), kFormula, options);
  const auto text = model.summary();
  EXPECT_NE(text.find("Lambda"), std::string::npos);
  EXPECT_NE(text.find("R-squared"), std::string::npos);
  EXPECT_NE(text.find("Sepal.Length"), std::string::npos);
}

TEST(ElasticNet, PlotPathAndCvCurveThrowForFixedLambdaFit) {
  ElasticNetOptions options;
  options.alpha  = 0.5;
  options.lambda = 0.2;
  ElasticNet model(datamunge::datasets::iris(), kFormula, options);
  EXPECT_THROW(model.plot_coefficient_path(), std::invalid_argument);
  EXPECT_THROW(model.plot_cv_curve(), std::invalid_argument);
}

// --- Ridge / Lasso wrapper classes ------------------------------------

TEST(Ridge, FixesAlphaToZeroAndMatchesElasticNetEquivalent) {
  RidgeOptions options;
  options.lambda = 0.5;
  Ridge ridge(datamunge::datasets::iris(), kFormula, options);

  ElasticNetOptions eq;
  eq.alpha  = 0.0;
  eq.lambda = 0.5;
  ElasticNet equivalent(datamunge::datasets::iris(), kFormula, eq);

  ASSERT_EQ(ridge.coefficients().size(), equivalent.coefficients().size());
  for (std::size_t j = 0; j < ridge.coefficients().size(); ++j)
    EXPECT_NEAR(ridge.coefficients()[j], equivalent.coefficients()[j], 1e-9);
  EXPECT_NEAR(ridge.intercept(), equivalent.intercept(), 1e-9);
}

TEST(Ridge, NeverProducesExactZeroCoefficients) {
  RidgeOptions options;
  options.lambda = 5.0; // aggressive shrinkage
  Ridge ridge(datamunge::datasets::iris(), kFormula, options);
  for (const double c : ridge.coefficients())
    EXPECT_NE(c, 0.0);
}

TEST(Lasso, FixesAlphaToOneAndMatchesElasticNetEquivalent) {
  LassoOptions options;
  options.lambda = 0.3;
  Lasso lasso(datamunge::datasets::iris(), kFormula, options);

  ElasticNetOptions eq;
  eq.alpha  = 1.0;
  eq.lambda = 0.3;
  ElasticNet equivalent(datamunge::datasets::iris(), kFormula, eq);

  ASSERT_EQ(lasso.coefficients().size(), equivalent.coefficients().size());
  for (std::size_t j = 0; j < lasso.coefficients().size(); ++j)
    EXPECT_NEAR(lasso.coefficients()[j], equivalent.coefficients()[j], 1e-9);
  EXPECT_EQ(lasso.non_zero_coefficients(), equivalent.non_zero_coefficients());
}

TEST(Lasso, PerformsVariableSelectionAtModerateLambda) {
  LassoOptions options;
  options.lambda = 0.3;
  Lasso lasso(datamunge::datasets::iris(), kFormula, options);
  EXPECT_LT(lasso.non_zero_coefficients(), lasso.predictor_names().size());
}

TEST(ElasticNet, OrthogonalSoftThresholdAndActualColumnNorms) {
  DataFrame d;
  d.add_column("x", std::vector<double>{-1, -1, 1, 1});
  d.add_column("z", std::vector<double>{-1, 1, -1, 1});
  d.add_column("y", std::vector<double>{.8, 1.2, 4.8, 5.2});
  const ElasticNet m(d, "y ~ x + z", {.5, .5});
  EXPECT_NEAR(m.coefficients()[0], 1.4, 1e-12);
  EXPECT_EQ(m.coefficients()[1], 0);
  EXPECT_NEAR(m.intercept(), 3, 1e-12);
  EXPECT_TRUE(m.converged());
  EXPECT_LT(m.diagnostics().double_at("max_kkt_violation", 0), 1e-12);
  EXPECT_NEAR(m.diagnostics().double_at("objective", 0), .795, 1e-12);
  DataFrame raw;
  raw.add_column("x", std::vector<double>{-2, -2, 2, 2});
  raw.add_column("y", std::vector<double>{-1, -1, 7, 7});
  ElasticNetOptions o{.5, .5};
  o.standardize = false;
  const ElasticNet unscaled(raw, "y ~ x", o);
  EXPECT_NEAR(unscaled.coefficients()[0], 31. / 17, 1e-12);
  EXPECT_EQ(unscaled.feature_scales()[0], 1);
}
TEST(ElasticNet, NoInterceptUsesRawOriginAndRidgeNormalEquations) {
  DataFrame d;
  d.add_column("x", std::vector<double>{1, 2, 3, 4});
  d.add_column("y", std::vector<double>{3, 6, 9, 12});
  ElasticNetOptions o{0, .5};
  o.standardize = false;
  const ElasticNet m(d, "y ~ 0 + x", o);
  EXPECT_EQ(m.intercept(), 0);
  EXPECT_EQ(m.feature_means()[0], 0);
  EXPECT_NEAR(m.coefficients()[0], 22.5 / 8, 1e-12);
  o.standardize = true;
  const ElasticNet scaled(d, "y ~ 0 + x", o);
  EXPECT_NEAR(scaled.feature_scales()[0], std::sqrt(7.5), 1e-12);
  EXPECT_NEAR(scaled.coefficients()[0], 2, 1e-12);
}
TEST(ElasticNet, IdenticalPredictorsGroupWithL2ButLassoNeedNotSplit) {
  DataFrame d;
  d.add_column("x", std::vector<double>{-1, -1, 1, 1});
  d.add_column("z", std::vector<double>{-1, -1, 1, 1});
  d.add_column("y", std::vector<double>{-2, -2, 2, 2});
  ElasticNetOptions o{.5, 1};
  o.tol = 1e-12;
  const ElasticNet mixed(d, "y ~ x + z", o);
  EXPECT_NEAR(mixed.coefficients()[0], .6, 2e-12);
  EXPECT_NEAR(mixed.coefficients()[1], .6, 2e-12);
  o.alpha = 1;
  const ElasticNet lasso(d, "y ~ x + z", o), reverse(d, "y ~ z + x", o);
  EXPECT_NEAR(lasso.coefficients()[0] + lasso.coefficients()[1], 1, 1e-12);
  EXPECT_NEAR(lasso.fitted_values()[3], reverse.fitted_values()[3], 1e-12);
  EXPECT_NEAR(lasso.coefficients()[0], 1, 1e-12);
  EXPECT_EQ(lasso.coefficients()[1], 0);
  o.alpha    = .5;
  o.max_iter = 1;
  const ElasticNet partial(d, "y ~ x + z", o);
  EXPECT_FALSE(partial.converged());
  EXPECT_EQ(partial.iterations(), 1);
  EXPECT_GT(partial.diagnostics().double_at("max_kkt_violation", 0), .1);
}
TEST(ElasticNet, ConstantAndTinyPredictorsHaveDefinedPenaltyCoordinates) {
  DataFrame d;
  d.add_column("x", std::vector<double>{-1e-15, -1e-15, 1e-15, 1e-15});
  d.add_column("constant", std::vector<double>{7, 7, 7, 7});
  d.add_column("y", std::vector<double>{1, 1, 5, 5});
  const ElasticNet m(d, "y ~ x + constant", {.5, .5});
  EXPECT_NEAR(m.coefficient_table().double_at("penalty_coefficient", 0), 1.4, 1e-12);
  EXPECT_NEAR(m.coefficients()[0] / 1e15, 1.4, 1e-12);
  EXPECT_EQ(m.coefficients()[1], 0);
  const ElasticNet only(d, "y ~ 1", {.5, .5});
  EXPECT_TRUE(only.converged());
  EXPECT_EQ(only.iterations(), 0);
  EXPECT_NEAR(only.intercept(), 3, 1e-12);
}
TEST(ElasticNet, CrossValidationMatchesRefitsWithFoldLocalPreprocessing) {
  DataFrame d;
  d.add_column("x", std::vector<double>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100});
  d.add_column("y", std::vector<double>{1, 4, 4, 8, 8, 12, 12, 15, 16, 20, 20, 80});
  ElasticNetOptions o{.5, -1, 7, 3};
  o.tol = 1e-11;
  const ElasticNet m(d, "y ~ x", o);
  ASSERT_EQ(m.fold_ids().size(), 12);
  const auto& folds = m.cv_fold_results();
  ASSERT_EQ(folds.nrows(), 21);
  std::vector<double> pooled(7, 0);
  for (std::size_t f = 0; f < 3; ++f) {
    std::vector<std::size_t> train, test;
    for (std::size_t i = 0; i < 12; ++i)
      (m.fold_ids()[i] == f ? test : train).push_back(i);
    ASSERT_EQ(test.size(), 4);
    for (std::size_t k = 0; k < 7; ++k) {
      o.lambda = m.lambda_path()[k];
      const ElasticNet ref(d.take_rows(train), "y ~ x", o);
      const auto       pred = ref.predict(d.take_rows(test));
      double           sse  = 0;
      for (std::size_t i = 0; i < test.size(); ++i)
        sse += std::pow(d.double_at("y", test[i]) - pred[i], 2);
      EXPECT_NEAR(folds.double_at("mse", f * 7 + k), sse / test.size(), 1e-8);
      pooled[k] += sse / 12;
    }
  }
  for (std::size_t k = 0; k < 7; ++k)
    EXPECT_NEAR(m.cv_mean_squared_error()[k], pooled[k], 1e-8);
  const ElasticNet again(d, "y ~ x", ElasticNetOptions{.5, -1, 7, 3});
  EXPECT_EQ(again.fold_ids(), m.fold_ids());
}
TEST(ElasticNet, MissingRowsAndFrozenQueryCoordinatesAreAligned) {
  DataFrame d;
  d.add_column("x", std::vector<std::optional<double>>{-1., std::nullopt, -1., 1., 1.});
  d.add_column("y", std::vector<double>{1, 100, 1, 5, 5});
  const ElasticNet m(d, "y ~ x", {.5, .5});
  EXPECT_EQ(m.kept_row_indices(), (std::vector<std::size_t>{0, 2, 3, 4}));
  const auto p = m.predict(d);
  EXPECT_TRUE(std::isnan(p[1]));
  EXPECT_EQ(p.size(), 5);
  EXPECT_NEAR(m.predict(d.take_rows({4}))[0], p[4], 1e-15);
  EXPECT_TRUE(m.predict(DataFrame{}).empty());
}
TEST(ElasticNet, InvalidControlsAndRetainedNonfiniteValuesAreRejected) {
  auto              d = datamunge::datasets::iris();
  ElasticNetOptions o{.5, .2};
  for (double bad :
       {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
    o.alpha = bad;
    EXPECT_THROW(ElasticNet(d, kFormula, o), std::invalid_argument);
    o.alpha  = .5;
    o.lambda = bad;
    EXPECT_THROW(ElasticNet(d, kFormula, o), std::invalid_argument);
    o.lambda = .2;
    o.tol    = bad;
    EXPECT_THROW(ElasticNet(d, kFormula, o), std::invalid_argument);
    o.tol = 1e-7;
  }
  o.max_iter = 0;
  EXPECT_THROW(ElasticNet(d, kFormula, o), std::invalid_argument);
  o.max_iter = 1000;
  o.tol      = 0;
  EXPECT_THROW(ElasticNet(d, kFormula, o), std::invalid_argument);
  o = {.5, -1, 5, 151};
  EXPECT_THROW(ElasticNet(d, kFormula, o), std::invalid_argument);
  o = {.5, -1, 0, 5};
  EXPECT_THROW(ElasticNet(d, kFormula, o), std::invalid_argument);
  DataFrame bad;
  bad.add_column("x", std::vector<double>{1, std::numeric_limits<double>::infinity(), 3});
  bad.add_column("y", std::vector<double>{1, 2, 3});
  EXPECT_THROW(ElasticNet(bad, "y ~ x", ElasticNetOptions{.5, .2}), std::invalid_argument);
}

TEST(ElasticNet, ScoringUsesCompleteRowsAndTrainingMeanBaseline) {
  DataFrame d;
  d.add_column("x", std::vector<double>{-1, -1, 1, 1});
  d.add_column("y", std::vector<double>{1, 1, 5, 5});
  const ElasticNet m(d, "y ~ x", {.5, .5});
  EXPECT_NEAR(m.score(d).double_at("rmse", 0), .6, 1e-12);
  EXPECT_NEAR(m.score(d).double_at("mean_baseline_rmse", 0), 2, 1e-12);
  DataFrame query;
  query.add_column("x", std::vector<std::optional<double>>{1., std::nullopt, -1.});
  query.add_column("y", std::vector<std::optional<double>>{5., 4., std::nullopt});
  const auto score = m.score(query);
  EXPECT_EQ(score.double_at("n_input", 0), 3);
  EXPECT_EQ(score.double_at("n_scored", 0), 1);
  EXPECT_NEAR(score.double_at("rmse", 0), .6, 1e-12);
  EXPECT_TRUE(std::isnan(score.double_at("r_squared", 0)));
}
TEST(Ridge, OrthogonalIrrelevantPredictorCanHaveExactZeroSlope) {
  DataFrame d;
  d.add_column("x", std::vector<double>{-1, -1, 1, 1});
  d.add_column("z", std::vector<double>{-1, 1, -1, 1});
  d.add_column("y", std::vector<double>{-2, -2, 2, 2});
  const Ridge m(d, "y ~ x + z", {1});
  EXPECT_NEAR(m.coefficients()[0], 1, 1e-12);
  EXPECT_EQ(m.coefficients()[1], 0);
}
