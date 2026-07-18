#include <gtest/gtest.h>

#include <datamunge/datasets/datasets.hpp>
#include <datamunge/dstruct/dstruct.hpp>
#include <datamunge/stats/stats.hpp>

#include <algorithm>
#include <cmath>

using datamunge::dstruct::DataFrame;
using datamunge::stats::GBMClassifier;
using datamunge::stats::GBMClassifierOptions;
using datamunge::stats::GBMRegressor;
using datamunge::stats::GBMRegressorOptions;

// Reference ballpark computed with scikit-learn's GradientBoostingClassifier
// / GradientBoostingRegressor (n_estimators=100, learning_rate=0.1,
// max_depth=3, random_state=42) on the same embedded iris dataset:
// classifier train accuracy 0.9933, deviance 0.917 -> 0.013; regressor
// train R^2 0.9932, deviance (MSE) 2.526 -> 0.021. sklearn's tree-building
// and multiclass initialization details are not identical to this
// implementation's, so exact agreement isn't expected -- these tests check
// the same ballpark and the boosting procedure's defining property
// (training loss goes down) instead.

TEST(GBMClassifier, IrisAchievesHighTrainAccuracy) {
    GBMClassifier model(datamunge::datasets::iris(), "Species ~ Petal.Length + Petal.Width");
    EXPECT_GT(model.training_accuracy(), 0.95);
}

TEST(GBMClassifier, TrainingDevianceDecreasesOverBoostingRounds) {
    GBMClassifier model(datamunge::datasets::iris(), "Species ~ Petal.Length + Petal.Width");
    const auto& deviance = model.training_deviance();
    ASSERT_EQ(deviance.size(), 100u);
    // Not strictly monotonic round to round, but the ensemble should fit
    // the training data dramatically better by the last round than the first.
    EXPECT_LT(deviance.back(), deviance.front() * 0.1);
    EXPECT_LT(deviance.back(), deviance[deviance.size() / 2]);
}

TEST(GBMClassifier, FeatureImportanceSumsToOneAndIsNonNegative) {
    GBMClassifier model(datamunge::datasets::iris(), "Species ~ Petal.Length + Petal.Width");
    const auto importance = model.feature_importance();
    ASSERT_EQ(importance.size(), 2u);
    double sum = 0.0;
    for (const double v : importance) {
        EXPECT_GE(v, 0.0);
        sum += v;
    }
    EXPECT_NEAR(sum, 1.0, 1e-9);
}

TEST(GBMClassifier, ConfusionMatrixDiagonalDominatesForIris) {
    GBMClassifier model(datamunge::datasets::iris(), "Species ~ Petal.Length + Petal.Width");
    const auto cm = model.confusion_matrix();
    ASSERT_EQ(cm.rows(), 3u);
    for (std::size_t i = 0; i < 3; ++i) {
        double row_total = 0.0;
        for (std::size_t j = 0; j < 3; ++j) row_total += cm(i, j);
        EXPECT_GT(cm(i, i) / row_total, 0.85);
    }
}

TEST(GBMClassifier, PredictMatchesFittedClassesOnTrainingData) {
    const auto iris = datamunge::datasets::iris();
    GBMClassifier model(iris, "Species ~ Petal.Length + Petal.Width");
    const auto preds = model.predict(iris);
    ASSERT_EQ(preds, model.fitted_classes());
}

TEST(GBMClassifier, FewerTreesFitsTrainingDataLessWell) {
    GBMClassifierOptions few;
    few.n_trees = 5;
    GBMClassifier small(datamunge::datasets::iris(), "Species ~ Petal.Length + Petal.Width", few);
    GBMClassifier large(datamunge::datasets::iris(), "Species ~ Petal.Length + Petal.Width");
    EXPECT_LE(small.training_accuracy(), large.training_accuracy() + 1e-9);
    EXPECT_GT(small.training_deviance().back(), large.training_deviance().back());
}

TEST(GBMClassifier, RejectsZeroTrees) {
    GBMClassifierOptions options;
    options.n_trees = 0;
    EXPECT_THROW(GBMClassifier(datamunge::datasets::iris(), "Species ~ Petal.Length + Petal.Width", options),
                std::invalid_argument);
}

TEST(GBMClassifier, RejectsSubsampleOutOfRange) {
    GBMClassifierOptions options;
    options.subsample = 1.5;
    EXPECT_THROW(GBMClassifier(datamunge::datasets::iris(), "Species ~ Petal.Length + Petal.Width", options),
                std::invalid_argument);
}

TEST(GBMClassifier, RejectsSingleClassResponse) {
    DataFrame df;
    df.add_column("x", std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    df.add_column("label", std::vector<std::string>{"a", "a", "a", "a", "a", "a"});
    EXPECT_THROW(GBMClassifier(df, "label ~ x"), std::invalid_argument);
}

TEST(GBMClassifier, PredictThrowsOnUnseenCategoricalLevel) {
    auto penguins = datamunge::datasets::penguins().drop_nulls({"bill_length_mm", "bill_depth_mm", "island"});
    GBMClassifier model(penguins, "species ~ bill_length_mm + bill_depth_mm + island");

    DataFrame bad;
    bad.add_column("bill_length_mm", std::vector<double>{40.0});
    bad.add_column("bill_depth_mm", std::vector<double>{18.0});
    bad.add_column("island", std::vector<std::string>{"Atlantis"});
    EXPECT_THROW((void)model.predict(bad), std::runtime_error);
}

TEST(GBMClassifier, ProbabilitiesSumToOne) {
    const auto iris = datamunge::datasets::iris();
    GBMClassifier model(iris, "Species ~ Petal.Length + Petal.Width");

    DataFrame newdata;
    newdata.add_column("Petal.Length", std::vector<double>{4.5});
    newdata.add_column("Petal.Width", std::vector<double>{1.5});
    const auto detail = model.predict_detail(newdata);
    ASSERT_EQ(detail.probability.size(), 1u);
    double sum = 0.0;
    for (const double v : detail.probability[0]) sum += v;
    EXPECT_NEAR(sum, 1.0, 1e-9);
}

TEST(GBMClassifier, PlotDecisionRegionsRequiresExactlyTwoPredictors) {
    GBMClassifierOptions options;
    options.n_trees = 5;
    GBMClassifier model(datamunge::datasets::iris(),
                        "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width", options);
    EXPECT_THROW(model.plot_decision_regions("Sepal.Length", "Sepal.Width"), std::invalid_argument);
}

// --- Regressor -------------------------------------------------------

TEST(GBMRegressor, IrisAchievesHighRSquared) {
    GBMRegressor model(datamunge::datasets::iris(), "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width");
    EXPECT_GT(model.r_squared(), 0.95);
}

TEST(GBMRegressor, TrainingDevianceDecreasesOverBoostingRounds) {
    GBMRegressor model(datamunge::datasets::iris(), "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width");
    const auto& deviance = model.training_deviance();
    ASSERT_EQ(deviance.size(), 100u);
    EXPECT_LT(deviance.back(), deviance.front() * 0.1);
}

TEST(GBMRegressor, PredictorNamesExcludeIntercept) {
    GBMRegressor model(datamunge::datasets::iris(), "Petal.Length ~ Sepal.Length + Sepal.Width");
    const auto& names = model.predictor_names();
    ASSERT_EQ(names.size(), 2u);
    EXPECT_EQ(names[0], "Sepal.Length");
    EXPECT_EQ(names[1], "Sepal.Width");
}

TEST(GBMRegressor, FeatureImportanceSumsToOne) {
    GBMRegressor model(datamunge::datasets::iris(), "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width");
    const auto importance = model.feature_importance();
    double sum = 0.0;
    for (const double v : importance) {
        EXPECT_GE(v, 0.0);
        sum += v;
    }
    EXPECT_NEAR(sum, 1.0, 1e-9);
}

TEST(GBMRegressor, PredictMatchesFittedValuesOnTrainingData) {
    const auto iris = datamunge::datasets::iris();
    GBMRegressor model(iris, "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width");
    const auto preds = model.predict(iris);
    ASSERT_EQ(preds.size(), model.fitted_values().size());
    for (std::size_t i = 0; i < preds.size(); ++i) EXPECT_NEAR(preds[i], model.fitted_values()[i], 1e-9);
}

TEST(GBMRegressor, PredictReturnsNanForMissingPredictors) {
    GBMRegressor model(datamunge::datasets::iris(), "Petal.Length ~ Sepal.Length + Sepal.Width");

    DataFrame newdata;
    newdata.add_column("Sepal.Length", std::vector<std::optional<double>>{5.0, std::nullopt});
    newdata.add_column("Sepal.Width", std::vector<std::optional<double>>{3.0, 3.0});
    const auto preds = model.predict(newdata);
    ASSERT_EQ(preds.size(), 2u);
    EXPECT_FALSE(std::isnan(preds[0]));
    EXPECT_TRUE(std::isnan(preds[1]));
}

TEST(GBMRegressor, RejectsZeroTrees) {
    GBMRegressorOptions options;
    options.n_trees = 0;
    EXPECT_THROW(
        GBMRegressor(datamunge::datasets::iris(), "Petal.Length ~ Sepal.Length + Sepal.Width", options),
        std::invalid_argument);
}

TEST(GBMRegressor, SubsampleStillProducesAValidFit) {
    GBMRegressorOptions options;
    options.subsample = 0.7;
    options.seed       = 7;
    GBMRegressor model(datamunge::datasets::iris(), "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width",
                       options);
    EXPECT_GT(model.r_squared(), 0.8);
}
