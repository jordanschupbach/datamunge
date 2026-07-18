#include <gtest/gtest.h>

#include <datamunge/datasets/datasets.hpp>
#include <datamunge/dstruct/dstruct.hpp>
#include <datamunge/stats/stats.hpp>

using datamunge::dstruct::DataFrame;
using datamunge::stats::DecisionTreeClassifier;
using datamunge::stats::DecisionTreeClassifierOptions;
using datamunge::stats::DecisionTreeRegressor;
using datamunge::stats::DecisionTreeRegressorOptions;
using datamunge::stats::SplitCriterion;

// Reference values computed with scikit-learn's DecisionTreeClassifier /
// DecisionTreeRegressor (max_depth=5, min_samples_split=2,
// min_samples_leaf=1) on the same embedded iris dataset. sklearn's root
// split lands on Petal.Width<=0.8 while this implementation's tie-breaking
// (first feature in iteration order) lands on the equally-optimal
// Petal.Length<=2.45 — both perfectly separate setosa with identical Gini
// gain, so tree *shape* and per-feature importance legitimately differ,
// but accuracy/leaf-count/depth/predictions do not depend on which side of
// the tie was taken and match exactly.

TEST(DecisionTreeClassifier, IrisMatchesSklearnAccuracyAndStructure) {
    DecisionTreeClassifier model(datamunge::datasets::iris(), "Species ~ Petal.Length + Petal.Width");

    EXPECT_NEAR(model.training_accuracy(), 0.993333333, 1e-6);
    EXPECT_EQ(model.leaf_count(), 8u);
    EXPECT_EQ(model.depth(), 5u);

    const auto cm = model.confusion_matrix();
    const double expected[3][3] = {{50, 0, 0}, {0, 49, 1}, {0, 0, 50}};
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j) EXPECT_NEAR(cm(i, j), expected[i][j], 1e-9);
}

TEST(DecisionTreeClassifier, IrisPredictionsMatchSklearn) {
    DecisionTreeClassifier model(datamunge::datasets::iris(), "Species ~ Petal.Length + Petal.Width");

    DataFrame newdata;
    newdata.add_column("Petal.Length", std::vector<double>{1.4, 4.5, 5.5, 4.8});
    newdata.add_column("Petal.Width", std::vector<double>{0.2, 1.5, 2.0, 1.8});
    const auto preds = model.predict(newdata);
    ASSERT_EQ(preds.size(), 4u);
    EXPECT_EQ(preds[0], "setosa");
    EXPECT_EQ(preds[1], "versicolor");
    EXPECT_EQ(preds[2], "virginica");
    EXPECT_EQ(preds[3], "virginica");
}

TEST(DecisionTreeClassifier, FeatureImportanceSumsToOneAndIsNonNegative) {
    DecisionTreeClassifier model(datamunge::datasets::iris(), "Species ~ Petal.Length + Petal.Width");
    const auto importance = model.feature_importance();
    ASSERT_EQ(importance.size(), 2u);
    double sum = 0.0;
    for (const double v : importance) {
        EXPECT_GE(v, 0.0);
        sum += v;
    }
    EXPECT_NEAR(sum, 1.0, 1e-9);
}

TEST(DecisionTreeClassifier, ShallowerTreeIsLessAccurateOrEqual) {
    DecisionTreeClassifierOptions shallow;
    shallow.max_depth = 1;
    DecisionTreeClassifier stump(datamunge::datasets::iris(), "Species ~ Petal.Length + Petal.Width", shallow);
    DecisionTreeClassifier full(datamunge::datasets::iris(), "Species ~ Petal.Length + Petal.Width");

    EXPECT_LE(stump.training_accuracy(), full.training_accuracy());
    EXPECT_EQ(stump.depth(), 1u);
    // A depth-1 stump can only isolate one class perfectly (setosa); it
    // cannot also separate versicolor from virginica.
    EXPECT_LT(stump.training_accuracy(), 0.95);
}

TEST(DecisionTreeClassifier, EntropyCriterionAlsoAchievesHighAccuracy) {
    DecisionTreeClassifierOptions options;
    options.criterion = SplitCriterion::Entropy;
    DecisionTreeClassifier model(datamunge::datasets::iris(), "Species ~ Petal.Length + Petal.Width", options);
    EXPECT_GT(model.training_accuracy(), 0.95);
}

TEST(DecisionTreeClassifier, RejectsSingleClassResponse) {
    DataFrame df;
    df.add_column("x", std::vector<double>{1.0, 2.0, 3.0, 4.0});
    df.add_column("label", std::vector<std::string>{"a", "a", "a", "a"});
    EXPECT_THROW(DecisionTreeClassifier(df, "label ~ x"), std::invalid_argument);
}

TEST(DecisionTreeClassifier, PredictThrowsOnUnseenCategoricalLevel) {
    auto penguins = datamunge::datasets::penguins().drop_nulls({"bill_length_mm", "bill_depth_mm", "island"});
    DecisionTreeClassifier model(penguins, "species ~ bill_length_mm + bill_depth_mm + island");

    DataFrame bad;
    bad.add_column("bill_length_mm", std::vector<double>{40.0});
    bad.add_column("bill_depth_mm", std::vector<double>{18.0});
    bad.add_column("island", std::vector<std::string>{"Atlantis"});
    EXPECT_THROW((void)model.predict(bad), std::runtime_error);
}

TEST(DecisionTreeClassifier, PlotDecisionRegionsRequiresExactlyTwoPredictors) {
    DecisionTreeClassifier model(datamunge::datasets::iris(),
                                 "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width");
    EXPECT_THROW(model.plot_decision_regions("Sepal.Length", "Sepal.Width"), std::invalid_argument);
}

// --- Regressor -------------------------------------------------------

TEST(DecisionTreeRegressor, IrisMatchesSklearnRSquaredAndRmse) {
    DecisionTreeRegressor model(datamunge::datasets::iris(), "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width");
    EXPECT_NEAR(model.r_squared(), 0.9874497779, 1e-6);
    EXPECT_NEAR(model.rmse(), 0.1971021208, 1e-6);
}

TEST(DecisionTreeRegressor, PredictorNamesExcludeIntercept) {
    DecisionTreeRegressor model(datamunge::datasets::iris(), "Petal.Length ~ Sepal.Length + Sepal.Width");
    const auto& names = model.predictor_names();
    ASSERT_EQ(names.size(), 2u);
    EXPECT_EQ(names[0], "Sepal.Length");
    EXPECT_EQ(names[1], "Sepal.Width");
}

TEST(DecisionTreeRegressor, FeatureImportanceSumsToOne) {
    DecisionTreeRegressor model(datamunge::datasets::iris(), "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width");
    const auto importance = model.feature_importance();
    double sum = 0.0;
    for (const double v : importance) {
        EXPECT_GE(v, 0.0);
        sum += v;
    }
    EXPECT_NEAR(sum, 1.0, 1e-9);
}

TEST(DecisionTreeRegressor, DeeperTreeFitsTrainingDataAtLeastAsWell) {
    DecisionTreeRegressorOptions shallow;
    shallow.max_depth = 1;
    DecisionTreeRegressor stump(datamunge::datasets::iris(), "Petal.Length ~ Sepal.Length + Sepal.Width", shallow);
    DecisionTreeRegressor full(datamunge::datasets::iris(), "Petal.Length ~ Sepal.Length + Sepal.Width");
    EXPECT_LE(stump.r_squared(), full.r_squared());
}

TEST(DecisionTreeRegressor, PredictReturnsNanForMissingPredictors) {
    DecisionTreeRegressor model(datamunge::datasets::iris(), "Petal.Length ~ Sepal.Length + Sepal.Width");

    DataFrame newdata;
    newdata.add_column("Sepal.Length", std::vector<std::optional<double>>{5.0, std::nullopt});
    newdata.add_column("Sepal.Width", std::vector<std::optional<double>>{3.0, 3.0});
    const auto preds = model.predict(newdata);
    ASSERT_EQ(preds.size(), 2u);
    EXPECT_FALSE(std::isnan(preds[0]));
    EXPECT_TRUE(std::isnan(preds[1]));
}
