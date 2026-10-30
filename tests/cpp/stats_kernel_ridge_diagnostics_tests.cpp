#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/datasets/datasets.hpp>
#include <datamunge/stats/kernel_ridge_regression.hpp>
#include <limits>
namespace s = datamunge::stats;
using DF    = datamunge::dstruct::DataFrame;
namespace {
DF pair() {
  DF d;
  d.add_column("x", std::vector<double>{0, 1});
  d.add_column("y", std::vector<double>{0, 2});
  return d;
}
s::KernelRidgeRegressionOptions exact() {
  s::KernelRidgeRegressionOptions o;
  o.lambda      = .5;
  o.gamma       = std::log(2.);
  o.standardize = false;
  return o;
}
} // namespace
TEST(KernelRidgeDiagnostics, ExactTwoPointRbfAndProperInterceptLOO) {
  const auto                     d = pair();
  const s::KernelRidgeRegression m(d, "y ~ x", exact());
  EXPECT_NEAR(m.gram_matrix()(0, 1), .5, 1e-14);
  EXPECT_NEAR(m.centered_gram_matrix()(0, 0), .25, 1e-14);
  EXPECT_NEAR(m.dual_coefficients()[0], -1, 1e-14);
  EXPECT_NEAR(m.dual_coefficients()[1], 1, 1e-14);
  EXPECT_NEAR(m.training_predictions()[0], .5, 1e-14);
  EXPECT_NEAR(m.training_predictions()[1], 1.5, 1e-14);
  EXPECT_NEAR(m.smoothing_matrix()(0, 0), .75, 1e-14);
  EXPECT_NEAR(m.smoothing_matrix()(0, 1), .25, 1e-14);
  EXPECT_NEAR(m.effective_degrees_of_freedom(), 1.5, 1e-14);
  EXPECT_NEAR(m.fitted_values()[0], 2, 1e-14);
  EXPECT_NEAR(m.fitted_values()[1], 0, 1e-14);
  EXPECT_NEAR(m.rmse(), 2, 1e-14);
  EXPECT_NEAR(m.r_squared(), -3, 1e-14);
  EXPECT_NEAR(m.intercept(), 1, 1e-14);
  EXPECT_NEAR(m.score(d).double_at("rmse", 0), .5, 1e-14);
  EXPECT_NEAR(m.diagnostics().double_at("objective", 0), 1, 1e-13);
}
TEST(KernelRidgeDiagnostics, UnequalKernelRowsEnforceTheInterceptNormalEquation) {
  DF d;
  d.add_column("x", std::vector<double>{0, 1, 3, 4});
  d.add_column("y", std::vector<double>{0, 1, 4, 2});
  auto                           o = exact();
  const s::KernelRidgeRegression m(d, "y ~ x", o);
  double                         sum_alpha = 0, sum_residual = 0;
  const auto                     pred = m.predict(d);
  for (std::size_t i = 0; i < d.nrows(); ++i) {
    sum_alpha += m.dual_coefficients()[i];
    sum_residual += d.double_at("y", i) - pred[i];
    EXPECT_NEAR(pred[i], m.training_predictions()[i], 1e-12);
  }
  EXPECT_NEAR(sum_alpha, 0, 1e-13);
  EXPECT_NEAR(sum_residual, 0, 1e-12);
  EXPECT_LT(m.diagnostics().double_at("system_residual", 0), 1e-12);
  const auto weights = m.prediction_weights(d);
  for (std::size_t i = 0; i < d.nrows(); ++i) {
    double sum = 0, p = 0;
    for (std::size_t j = 0; j < d.nrows(); ++j) {
      sum += weights(i, j);
      p += weights(i, j) * d.double_at("y", j);
    }
    EXPECT_NEAR(sum, 1, 1e-12);
    EXPECT_NEAR(p, pred[i], 1e-12);
  }
}
TEST(KernelRidgeDiagnostics, SignedWeightsAndConstantReproduction) {
  auto o                           = exact();
  o.kernel                         = "linear";
  const auto                     d = pair();
  const s::KernelRidgeRegression m(d, "y ~ x", o);
  DF                             q;
  q.add_column("x", std::vector<double>{3});
  const auto w = m.prediction_weights(q);
  EXPECT_NEAR(w(0, 0), -.75, 1e-14);
  EXPECT_NEAR(w(0, 1), 1.75, 1e-14);
  EXPECT_NEAR(m.predict(q)[0], 3.5, 1e-14);
  const auto                     constant = d.mutate("y", std::vector<double>{7, 7});
  const s::KernelRidgeRegression c(constant, "y ~ x", o);
  EXPECT_NEAR(c.predict(q)[0], 7, 1e-14);
  EXPECT_EQ(c.r_squared(), 0);
}
TEST(KernelRidgeDiagnostics, OptionalZeroInterceptChangesTheEstimator) {
  auto o          = exact();
  o.kernel        = "linear";
  o.fit_intercept = false;
  const s::KernelRidgeRegression m(pair(), "y ~ x", o);
  EXPECT_EQ(m.intercept(), 0);
  EXPECT_NEAR(m.predict(pair())[0], 0, 1e-14);
  EXPECT_NEAR(m.predict(pair())[1], 4. / 3, 1e-14);
  EXPECT_NEAR(m.effective_degrees_of_freedom(), 2. / 3, 1e-14);
}
TEST(KernelRidgeDiagnostics, ZeroPenaltyUsesActualDeletionsWithoutLeverageFloors) {
  auto o   = exact();
  o.lambda = 0;
  const s::KernelRidgeRegression m(pair(), "y ~ x", o);
  EXPECT_NEAR(m.predict(pair())[0], 0, 1e-13);
  EXPECT_NEAR(m.predict(pair())[1], 2, 1e-13);
  EXPECT_NEAR(m.fitted_values()[0], 2, 1e-13);
  EXPECT_NEAR(m.fitted_values()[1], 0, 1e-13);
  EXPECT_NEAR(m.effective_degrees_of_freedom(), 2, 1e-13);
  DF duplicate;
  duplicate.add_column("x", std::vector<double>{0, 0, 0});
  duplicate.add_column("y", std::vector<double>{0, 1, 2});
  const s::KernelRidgeRegression c(duplicate, "y ~ x", o);
  EXPECT_NEAR(c.training_predictions()[0], 1, 1e-13);
  EXPECT_NEAR(c.fitted_values()[0], 1.5, 1e-13);
  EXPECT_NEAR(c.fitted_values()[1], 1, 1e-13);
  EXPECT_NEAR(c.effective_degrees_of_freedom(), 1, 1e-13);
}
TEST(KernelRidgeDiagnostics, PositivePenaltyIdentityMatchesExplicitFixedMetricRefits) {
  DF d;
  d.add_column("x", std::vector<double>{0, 1, 3, 4});
  d.add_column("y", std::vector<double>{0, 1, 4, 2});
  auto o = exact();
  for (bool intercept : {false, true}) {
    o.fit_intercept = intercept;
    const s::KernelRidgeRegression m(d, "y ~ x", o);
    for (std::size_t i = 0; i < 4; ++i) {
      std::vector<std::size_t> kept;
      for (std::size_t j = 0; j < 4; ++j)
        if (i != j)
          kept.push_back(j);
      const s::KernelRidgeRegression sub(d.take_rows(kept), "y ~ x", o);
      EXPECT_NEAR(m.fitted_values()[i], sub.predict(d.take_rows({i}))[0], 1e-11);
    }
  }
}
TEST(KernelRidgeDiagnostics, PopulationScalingSurvivesTinyUnitsAndFreezesQueries) {
  DF d;
  d.add_column("x", std::vector<double>{0, 1, 3, 4});
  d.add_column("y", std::vector<double>{0, 1, 4, 2});
  auto o        = exact();
  o.standardize = true;
  const s::KernelRidgeRegression a(d, "y ~ x", o),
      b(d.mutate("x", std::vector<double>{0, 1e-20, 3e-20, 4e-20}), "y ~ x", o);
  EXPECT_NEAR(a.feature_mean()[0], 2, 1e-14);
  EXPECT_NEAR(a.feature_scale()[0], std::sqrt(2.5), 1e-14);
  for (std::size_t i = 0; i < 4; ++i)
    EXPECT_NEAR(a.training_predictions()[i], b.training_predictions()[i], 1e-12);
  const auto before = a.feature_mean();
  DF         q;
  q.add_column("x", std::vector<double>{100});
  (void)a.predict(q);
  EXPECT_EQ(a.feature_mean(), before);
}
TEST(KernelRidgeDiagnostics, PolynomialDegreeOneMatchesScaledLinearRidge) {
  DF d;
  d.add_column("x", std::vector<double>{-2, 0, 1, 3});
  d.add_column("y", std::vector<double>{0, 1, 4, 2});
  auto o   = exact();
  o.kernel = "polynomial";
  o.gamma  = 2;
  o.degree = 1;
  o.coef0  = .3;
  o.lambda = 1;
  const s::KernelRidgeRegression poly(d, "y ~ x", o);
  o.kernel = "linear";
  o.lambda = .5;
  const s::KernelRidgeRegression linear(d, "y ~ x", o);
  for (std::size_t i = 0; i < 4; ++i)
    EXPECT_NEAR(poly.training_predictions()[i], linear.training_predictions()[i], 1e-12);
}
TEST(KernelRidgeDiagnostics, SearchTableIncludesEveryPairAndFirstStrictTie) {
  auto d              = pair().mutate("y", std::vector<double>{7, 7});
  auto o              = exact();
  o.lambda            = -1;
  o.gamma             = -1;
  o.lambda_candidates = {1, .1};
  o.gamma_candidates  = {2, .5};
  const s::KernelRidgeRegression m(d, "y ~ x", o);
  EXPECT_EQ(m.selection_table().nrows(), 4);
  EXPECT_EQ(m.lambda(), 1);
  EXPECT_EQ(m.gamma(), 2);
  EXPECT_EQ(m.lambda_grid(), o.lambda_candidates);
  EXPECT_EQ(m.selection_table().double_at("selected", 0), 1);
  EXPECT_EQ(m.selection_table().double_at("selected", 1), 0);
}
TEST(KernelRidgeDiagnostics, RowAlignmentAndMissingQueryHandling) {
  DF d;
  d.add_column("x", std::vector<std::optional<double>>{0, std::nullopt, 1, 2, 3});
  d.add_column("y", std::vector<std::optional<double>>{0, 4, 1, std::nullopt, 2});
  const s::KernelRidgeRegression m(d, "y ~ x", exact());
  EXPECT_EQ(m.kept_row_indices(), (std::vector<std::size_t>{0, 2, 4}));
  EXPECT_EQ(m.training_diagnostics().nrows(), 3);
  const auto p = m.predict(d);
  EXPECT_TRUE(std::isnan(p[1]));
  EXPECT_TRUE(std::isfinite(p[3]));
  EXPECT_TRUE(m.predict(d.head(0)).empty());
  auto q = d.head(1);
  q.set_value("x", 0, std::optional<double>{});
  EXPECT_TRUE(std::isnan(m.predict(q)[0]));
  EXPECT_TRUE(std::isnan(m.prediction_weights(q)(0, 0)));
  EXPECT_THROW(m.score(d), std::invalid_argument);
  EXPECT_THROW(m.score(d.head(0)), std::invalid_argument);
}
TEST(KernelRidgeDiagnostics, StrictControlsAndFiniteInputs) {
  auto       o = exact();
  const auto d = pair();
  for (double value : {NAN, INFINITY}) {
    o.lambda = value;
    EXPECT_THROW((s::KernelRidgeRegression(d, "y ~ x", o)), std::invalid_argument);
  }
  o       = exact();
  o.gamma = 0;
  EXPECT_THROW((s::KernelRidgeRegression(d, "y ~ x", o)), std::invalid_argument);
  o        = exact();
  o.kernel = "polynomial";
  o.degree = 1.5;
  EXPECT_THROW((s::KernelRidgeRegression(d, "y ~ x", o)), std::invalid_argument);
  o.degree = 2;
  o.coef0  = -1;
  EXPECT_THROW((s::KernelRidgeRegression(d, "y ~ x", o)), std::invalid_argument);
  o              = exact();
  o.n_gamma_grid = 0;
  EXPECT_THROW((s::KernelRidgeRegression(d, "y ~ x", o)), std::invalid_argument);
  auto bad = d.mutate("x", std::vector<double>{0, INFINITY});
  EXPECT_THROW((s::KernelRidgeRegression(bad, "y ~ x", exact())), std::invalid_argument);
  o                   = exact();
  o.lambda            = -1;
  o.lambda_candidates = {-.1, 1};
  EXPECT_THROW((s::KernelRidgeRegression(d, "y ~ x", o)), std::invalid_argument);
  EXPECT_THROW(s::kernel_ridge_wave(1), std::invalid_argument);
}
TEST(KernelRidgeDiagnostics, PlotGuardsAndExpandedDesignInference) {
  const auto                     d = datamunge::datasets::iris();
  auto                           o = exact();
  const s::KernelRidgeRegression m(d, "Petal.Length ~ log(Petal.Width)", o);
  EXPECT_EQ(m.predict(d).size(), 150);
  EXPECT_THROW(m.plot_fit(d), std::invalid_argument);
  const s::KernelRidgeRegression simple(pair(), "y ~ x", o);
  EXPECT_THROW(simple.plot_fit(pair(), 1), std::invalid_argument);
}
TEST(KernelRidgeCV, RefitsScalingAndInterceptWithinEachFold) {
  DF d;
  d.add_column("x", std::vector<double>{0, 1, 2, 10});
  d.add_column("y", std::vector<double>{0, 1, 0, 2});
  auto o            = exact();
  o.standardize     = true;
  o.lambda          = .1;
  o.gamma           = 1;
  const auto result = s::kernel_ridge_cross_validate(d, "y ~ x", {o}, {0, 1, 2, 3});
  EXPECT_EQ(result.folds.nrows(), 4);
  EXPECT_EQ(result.predictions.nrows(), 4);
  EXPECT_EQ(result.best_candidate, 0);
  EXPECT_NEAR(result.folds.double_at("mean_0", 3), 1, 1e-14);
  EXPECT_NEAR(result.folds.double_at("scale_0", 3), std::sqrt(2. / 3), 1e-14);
  double mse = 0;
  for (std::size_t i = 0; i < 4; ++i) {
    std::vector<std::size_t> kept;
    for (std::size_t j = 0; j < 4; ++j)
      if (i != j)
        kept.push_back(j);
    const s::KernelRidgeRegression fold(d.take_rows(kept), "y ~ x", o);
    const double                   r = d.double_at("y", i) - fold.predict(d.take_rows({i}))[0];
    mse += r * r / 4;
    EXPECT_NEAR(result.folds.double_at("mse", i), r * r, 1e-12);
    EXPECT_EQ(result.predictions.double_at("row", i), i);
    EXPECT_NEAR(result.predictions.double_at("prediction", i), d.double_at("y", i) - r, 1e-12);
  }
  EXPECT_NEAR(result.candidates.double_at("mse", 0), mse, 1e-12);
  const s::KernelRidgeRegression full(d, "y ~ x", o);
  EXPECT_GT(std::abs(full.rmse() * full.rmse() - mse), .01);
}
TEST(KernelRidgeCV, RejectsRecursiveSelectionBadFoldsAndHandlesOmittedRows) {
  auto d = pair();
  auto o = exact();
  EXPECT_THROW(s::kernel_ridge_cross_validate(d, "y ~ x", {o}, {0, 1}), std::invalid_argument);
  d = s::kernel_ridge_wave(8);
  EXPECT_THROW(
      s::kernel_ridge_cross_validate(d, "y ~ x", {o}, {0, 0, 0, 0, 2, 2, 2, 2}),
      std::invalid_argument);
  o.lambda = -1;
  EXPECT_THROW(
      s::kernel_ridge_cross_validate(d, "y ~ x", {o}, {0, 0, 0, 0, 1, 1, 1, 1}),
      std::invalid_argument);
  o = exact();
  d.set_value("y", 0, std::optional<double>{});
  const auto r = s::kernel_ridge_cross_validate(d, "y ~ x", {o}, {99, 0, 0, 0, 1, 1, 1, 1});
  EXPECT_EQ(r.folds.nrows(), 2);
}
