#include <gtest/gtest.h>

#include <datamunge/datasets/datasets.hpp>
#include <datamunge/dstruct/dstruct.hpp>
#include <datamunge/stats/stats.hpp>

#include <algorithm>
#include <cmath>

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
    options.alpha = 0.0;
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
    options.alpha = 1.0;
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
    options.alpha = 0.5;
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
        for (const double x : v) s += x * x;
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
    for (const double c : model.coefficients()) EXPECT_NEAR(c, 0.0, 1e-9);
}

TEST(ElasticNet, AutoSelectedLambdaLiesWithinSweptPathAndImprovesOnEndpoints) {
    ElasticNetOptions options; // lambda left at -1 (auto)
    options.alpha = 0.5;
    ElasticNet model(datamunge::datasets::iris(), kFormula, options);

    EXPECT_TRUE(model.lambda_was_selected());
    ASSERT_FALSE(model.lambda_path().empty());
    ASSERT_EQ(model.lambda_path().size(), model.cv_mean_squared_error().size());

    const auto& path = model.lambda_path();
    const double lo  = *std::min_element(path.begin(), path.end());
    const double hi  = *std::max_element(path.begin(), path.end());
    EXPECT_GE(model.lambda(), lo);
    EXPECT_LE(model.lambda(), hi);

    // The selected lambda's CV error should be at least as good as the
    // largest-lambda (most-shrunk, closest-to-null-model) endpoint.
    const auto& cv = model.cv_mean_squared_error();
    const auto selected_it = std::find(path.begin(), path.end(), model.lambda());
    ASSERT_NE(selected_it, path.end());
    const auto selected_index = static_cast<std::size_t>(selected_it - path.begin());
    EXPECT_LE(cv[selected_index], cv.back() + 1e-9);
}

TEST(ElasticNet, PredictMatchesFittedValuesOnTrainingData) {
    ElasticNetOptions options;
    options.alpha  = 0.5;
    options.lambda = 0.2;
    const auto iris = datamunge::datasets::iris();
    ElasticNet model(iris, kFormula, options);

    const auto preds = model.predict(iris);
    ASSERT_EQ(preds.size(), model.fitted_values().size());
    for (std::size_t i = 0; i < preds.size(); ++i) EXPECT_NEAR(preds[i], model.fitted_values()[i], 1e-9);
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
    for (const double c : ridge.coefficients()) EXPECT_NE(c, 0.0);
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
