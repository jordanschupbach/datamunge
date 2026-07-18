#include <gtest/gtest.h>

#include <datamunge/datasets/datasets.hpp>
#include <datamunge/dstruct/dstruct.hpp>
#include <datamunge/stats/stats.hpp>

#include <cmath>

using datamunge::dstruct::DataFrame;
using datamunge::stats::GBMRegressor;
using datamunge::stats::GBMRegressorOptions;
using datamunge::stats::XGBoostClassifier;
using datamunge::stats::XGBoostClassifierOptions;
using datamunge::stats::XGBoostRegressor;
using datamunge::stats::XGBoostRegressorOptions;

// For squared-error loss the Hessian is uniformly 1, which makes XGBoost's
// Newton leaf weight -G/(H+lambda) reduce exactly to the CART mean
// residual when lambda=0, and its regularized gain formula reduce exactly
// to ordinary SSE-reduction (verified independently with a numpy oracle:
// both approaches pick the same root split on this dataset, feature=2
// (Petal.Width) threshold=0.8, with identical Newton/mean leaf values
// -2.296 / 1.148). So with lambda=alpha=gamma=0 and matching
// learning_rate/max_depth, XGBoostRegressor should track the
// already-validated GBMRegressor closely round to round.
TEST(XGBoostRegressor, MatchesGBMWhenRegularizationIsZero) {
    GBMRegressorOptions gbm_options;
    gbm_options.n_trees       = 20;
    gbm_options.learning_rate = 0.1;
    gbm_options.max_depth     = 3;
    GBMRegressor gbm(datamunge::datasets::iris(), "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width",
                     gbm_options);

    XGBoostRegressorOptions xgb_options;
    xgb_options.n_trees       = 20;
    xgb_options.learning_rate = 0.1;
    xgb_options.max_depth     = 3;
    xgb_options.lambda        = 0.0;
    xgb_options.alpha         = 0.0;
    xgb_options.gamma         = 0.0;
    XGBoostRegressor xgb(datamunge::datasets::iris(), "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width",
                         xgb_options);

    ASSERT_EQ(gbm.fitted_values().size(), xgb.fitted_values().size());
    double max_diff = 0.0;
    for (std::size_t i = 0; i < gbm.fitted_values().size(); ++i)
        max_diff = std::max(max_diff, std::fabs(gbm.fitted_values()[i] - xgb.fitted_values()[i]));
    // Not bit-identical (independent split-search tie-breaking), but should
    // track very closely since the underlying math is the same.
    EXPECT_LT(max_diff, 0.05);
    EXPECT_NEAR(gbm.r_squared(), xgb.r_squared(), 0.02);
}

TEST(XGBoostRegressor, StrongL2RegularizationShrinksFitTowardBaseScore) {
    XGBoostRegressorOptions loose, tight;
    loose.lambda = 0.0;
    tight.lambda = 1000.0;
    XGBoostRegressor loose_model(datamunge::datasets::iris(),
                                "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width", loose);
    XGBoostRegressor tight_model(datamunge::datasets::iris(),
                                "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width", tight);
    EXPECT_GT(loose_model.r_squared(), tight_model.r_squared());
}

TEST(XGBoostRegressor, LargeGammaPreventsSplitsAndHurtsFit) {
    XGBoostRegressorOptions huge_gamma;
    huge_gamma.gamma = 1e6;
    XGBoostRegressor model(datamunge::datasets::iris(), "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width",
                           huge_gamma);
    // No split can ever clear a gain threshold this large, so every tree
    // stays a single root leaf and the ensemble barely moves from the base
    // score -- feature importance should reflect that nothing was ever
    // split on.
    const auto importance = model.feature_importance();
    double     sum        = 0.0;
    for (const double v : importance) sum += v;
    EXPECT_NEAR(sum, 0.0, 1e-9);
}

TEST(XGBoostRegressor, TrainingDevianceDecreasesOverBoostingRounds) {
    XGBoostRegressor model(datamunge::datasets::iris(), "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width");
    const auto& deviance = model.training_deviance();
    ASSERT_EQ(deviance.size(), 100u);
    EXPECT_LT(deviance.back(), deviance.front() * 0.2);
}

TEST(XGBoostRegressor, IrisAchievesHighRSquared) {
    XGBoostRegressor model(datamunge::datasets::iris(), "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width");
    EXPECT_GT(model.r_squared(), 0.95);
}

TEST(XGBoostRegressor, PredictorNamesExcludeIntercept) {
    XGBoostRegressor model(datamunge::datasets::iris(), "Petal.Length ~ Sepal.Length + Sepal.Width");
    const auto& names = model.predictor_names();
    ASSERT_EQ(names.size(), 2u);
    EXPECT_EQ(names[0], "Sepal.Length");
    EXPECT_EQ(names[1], "Sepal.Width");
}

TEST(XGBoostRegressor, PredictMatchesFittedValuesOnTrainingData) {
    const auto iris = datamunge::datasets::iris();
    XGBoostRegressor model(iris, "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width");
    const auto preds = model.predict(iris);
    ASSERT_EQ(preds.size(), model.fitted_values().size());
    for (std::size_t i = 0; i < preds.size(); ++i) EXPECT_NEAR(preds[i], model.fitted_values()[i], 1e-9);
}

TEST(XGBoostRegressor, PredictReturnsNanForMissingPredictors) {
    XGBoostRegressor model(datamunge::datasets::iris(), "Petal.Length ~ Sepal.Length + Sepal.Width");

    DataFrame newdata;
    newdata.add_column("Sepal.Length", std::vector<std::optional<double>>{5.0, std::nullopt});
    newdata.add_column("Sepal.Width", std::vector<std::optional<double>>{3.0, 3.0});
    const auto preds = model.predict(newdata);
    ASSERT_EQ(preds.size(), 2u);
    EXPECT_FALSE(std::isnan(preds[0]));
    EXPECT_TRUE(std::isnan(preds[1]));
}

TEST(XGBoostRegressor, RejectsZeroTrees) {
    XGBoostRegressorOptions options;
    options.n_trees = 0;
    EXPECT_THROW(
        XGBoostRegressor(datamunge::datasets::iris(), "Petal.Length ~ Sepal.Length + Sepal.Width", options),
        std::invalid_argument);
}

TEST(XGBoostRegressor, RejectsSubsampleOutOfRange) {
    XGBoostRegressorOptions options;
    options.subsample = 0.0;
    EXPECT_THROW(
        XGBoostRegressor(datamunge::datasets::iris(), "Petal.Length ~ Sepal.Length + Sepal.Width", options),
        std::invalid_argument);
}

TEST(XGBoostRegressor, RejectsColsampleOutOfRange) {
    XGBoostRegressorOptions options;
    options.colsample_bytree = 1.5;
    EXPECT_THROW(
        XGBoostRegressor(datamunge::datasets::iris(), "Petal.Length ~ Sepal.Length + Sepal.Width", options),
        std::invalid_argument);
}

// --- Classifier --------------------------------------------------------

TEST(XGBoostClassifier, IrisAchievesHighTrainAccuracy) {
    XGBoostClassifier model(datamunge::datasets::iris(), "Species ~ Petal.Length + Petal.Width");
    EXPECT_GT(model.training_accuracy(), 0.95);
}

TEST(XGBoostClassifier, TrainingDevianceDecreasesOverBoostingRounds) {
    XGBoostClassifier model(datamunge::datasets::iris(), "Species ~ Petal.Length + Petal.Width");
    const auto& deviance = model.training_deviance();
    ASSERT_EQ(deviance.size(), 100u);
    EXPECT_LT(deviance.back(), deviance.front() * 0.2);
}

TEST(XGBoostClassifier, FeatureImportanceSumsToOneAndIsNonNegative) {
    XGBoostClassifier model(datamunge::datasets::iris(), "Species ~ Petal.Length + Petal.Width");
    const auto importance = model.feature_importance();
    ASSERT_EQ(importance.size(), 2u);
    double sum = 0.0;
    for (const double v : importance) {
        EXPECT_GE(v, 0.0);
        sum += v;
    }
    EXPECT_NEAR(sum, 1.0, 1e-9);
}

TEST(XGBoostClassifier, ConfusionMatrixDiagonalDominatesForIris) {
    XGBoostClassifier model(datamunge::datasets::iris(), "Species ~ Petal.Length + Petal.Width");
    const auto cm = model.confusion_matrix();
    ASSERT_EQ(cm.rows(), 3u);
    for (std::size_t i = 0; i < 3; ++i) {
        double row_total = 0.0;
        for (std::size_t j = 0; j < 3; ++j) row_total += cm(i, j);
        EXPECT_GT(cm(i, i) / row_total, 0.85);
    }
}

TEST(XGBoostClassifier, ProbabilitiesSumToOne) {
    const auto iris = datamunge::datasets::iris();
    XGBoostClassifier model(iris, "Species ~ Petal.Length + Petal.Width");

    DataFrame newdata;
    newdata.add_column("Petal.Length", std::vector<double>{4.5});
    newdata.add_column("Petal.Width", std::vector<double>{1.5});
    const auto detail = model.predict_detail(newdata);
    ASSERT_EQ(detail.probability.size(), 1u);
    double sum = 0.0;
    for (const double v : detail.probability[0]) sum += v;
    EXPECT_NEAR(sum, 1.0, 1e-9);
}

TEST(XGBoostClassifier, RejectsZeroTrees) {
    XGBoostClassifierOptions options;
    options.n_trees = 0;
    EXPECT_THROW(XGBoostClassifier(datamunge::datasets::iris(), "Species ~ Petal.Length + Petal.Width", options),
                std::invalid_argument);
}

TEST(XGBoostClassifier, RejectsSingleClassResponse) {
    DataFrame df;
    df.add_column("x", std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    df.add_column("label", std::vector<std::string>{"a", "a", "a", "a", "a", "a"});
    EXPECT_THROW(XGBoostClassifier(df, "label ~ x"), std::invalid_argument);
}

TEST(XGBoostClassifier, PredictThrowsOnUnseenCategoricalLevel) {
    auto penguins = datamunge::datasets::penguins().drop_nulls({"bill_length_mm", "bill_depth_mm", "island"});
    XGBoostClassifier model(penguins, "species ~ bill_length_mm + bill_depth_mm + island");

    DataFrame bad;
    bad.add_column("bill_length_mm", std::vector<double>{40.0});
    bad.add_column("bill_depth_mm", std::vector<double>{18.0});
    bad.add_column("island", std::vector<std::string>{"Atlantis"});
    EXPECT_THROW((void)model.predict(bad), std::runtime_error);
}

TEST(XGBoostClassifier, PlotDecisionRegionsRequiresExactlyTwoPredictors) {
    XGBoostClassifierOptions options;
    options.n_trees = 5;
    XGBoostClassifier model(datamunge::datasets::iris(),
                            "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width", options);
    EXPECT_THROW(model.plot_decision_regions("Sepal.Length", "Sepal.Width"), std::invalid_argument);
}
