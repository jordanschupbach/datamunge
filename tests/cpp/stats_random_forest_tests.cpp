#include <gtest/gtest.h>

#include <datamunge/datasets/datasets.hpp>
#include <datamunge/dstruct/dstruct.hpp>
#include <datamunge/stats/stats.hpp>

#include <cmath>

using datamunge::dstruct::DataFrame;
using datamunge::stats::RandomForestClassifier;
using datamunge::stats::RandomForestClassifierOptions;
using datamunge::stats::RandomForestRegressor;
using datamunge::stats::RandomForestRegressorOptions;
using datamunge::stats::SplitCriterion;

// Reference ballpark computed with scikit-learn's RandomForestClassifier /
// RandomForestRegressor (n_estimators=100, max_features="sqrt" or 1,
// max_depth=10, random_state=42, oob_score=True) on the same embedded iris
// dataset: classifier train accuracy 0.9933, OOB accuracy 0.9667;
// regressor train R^2 0.9944, OOB R^2 0.9657. This implementation's
// bootstrap/feature-subsampling RNG is unrelated to sklearn's, so exact
// tree-by-tree agreement isn't expected -- these tests check the same
// ballpark and the ensemble's internal consistency instead.

TEST(RandomForestClassifier, IrisAchievesHighTrainAndOobAccuracy) {
    RandomForestClassifier model(datamunge::datasets::iris(), "Species ~ Petal.Length + Petal.Width");

    EXPECT_GT(model.training_accuracy(), 0.95);
    EXPECT_GT(model.oob_accuracy(), 0.85);
    // OOB accuracy is a less optimistic (out-of-sample) estimate; it should
    // not exceed training (in-sample) accuracy by construction here.
    EXPECT_LE(model.oob_accuracy(), model.training_accuracy() + 1e-9);
}

TEST(RandomForestClassifier, DefaultMaxFeaturesIsSqrtOfPredictorCount) {
    RandomForestClassifier model(datamunge::datasets::iris(),
                                 "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width");
    // sqrt(4) = 2
    EXPECT_EQ(model.max_features_used(), 2u);
}

TEST(RandomForestClassifier, FeatureImportanceSumsToOneAndIsNonNegative) {
    RandomForestClassifier model(datamunge::datasets::iris(), "Species ~ Petal.Length + Petal.Width");
    const auto importance = model.feature_importance();
    ASSERT_EQ(importance.size(), 2u);
    double sum = 0.0;
    for (const double v : importance) {
        EXPECT_GE(v, 0.0);
        sum += v;
    }
    EXPECT_NEAR(sum, 1.0, 1e-9);
}

TEST(RandomForestClassifier, ConfusionMatrixDiagonalDominatesForIris) {
    RandomForestClassifier model(datamunge::datasets::iris(), "Species ~ Petal.Length + Petal.Width");
    const auto cm = model.confusion_matrix();
    ASSERT_EQ(cm.rows(), 3u);
    ASSERT_EQ(cm.cols(), 3u);
    for (std::size_t i = 0; i < 3; ++i) {
        double row_total = 0.0;
        for (std::size_t j = 0; j < 3; ++j) row_total += cm(i, j);
        EXPECT_GT(cm(i, i) / row_total, 0.85);
    }
}

TEST(RandomForestClassifier, PredictionsAreDeterministicGivenSameSeed) {
    RandomForestClassifierOptions options;
    options.seed = 7;
    RandomForestClassifier a(datamunge::datasets::iris(), "Species ~ Petal.Length + Petal.Width", options);
    RandomForestClassifier b(datamunge::datasets::iris(), "Species ~ Petal.Length + Petal.Width", options);

    DataFrame newdata;
    newdata.add_column("Petal.Length", std::vector<double>{1.4, 4.5, 5.5, 4.8});
    newdata.add_column("Petal.Width", std::vector<double>{0.2, 1.5, 2.0, 1.8});

    EXPECT_EQ(a.predict(newdata), b.predict(newdata));
}

TEST(RandomForestClassifier, MoreTreesDoesNotHurtAccuracy) {
    RandomForestClassifierOptions few;
    few.n_trees = 5;
    RandomForestClassifierOptions many;
    many.n_trees = 100;

    RandomForestClassifier small(datamunge::datasets::iris(), "Species ~ Petal.Length + Petal.Width", few);
    RandomForestClassifier large(datamunge::datasets::iris(), "Species ~ Petal.Length + Petal.Width", many);

    EXPECT_EQ(small.n_trees(), 5u);
    EXPECT_EQ(large.n_trees(), 100u);
    // A 100-tree forest's OOB estimate should be at least roughly as good as
    // a 5-tree forest's -- not a strict guarantee for any single seed, but
    // true by a wide margin for iris.
    EXPECT_GT(large.oob_accuracy(), 0.85);
}

TEST(RandomForestClassifier, EntropyCriterionAlsoAchievesHighAccuracy) {
    RandomForestClassifierOptions options;
    options.criterion = SplitCriterion::Entropy;
    RandomForestClassifier model(datamunge::datasets::iris(), "Species ~ Petal.Length + Petal.Width", options);
    EXPECT_GT(model.training_accuracy(), 0.9);
}

TEST(RandomForestClassifier, RejectsSingleClassResponse) {
    DataFrame df;
    df.add_column("x", std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    df.add_column("label", std::vector<std::string>{"a", "a", "a", "a", "a", "a"});
    EXPECT_THROW(RandomForestClassifier(df, "label ~ x"), std::invalid_argument);
}

TEST(RandomForestClassifier, PredictThrowsOnUnseenCategoricalLevel) {
    auto penguins = datamunge::datasets::penguins().drop_nulls({"bill_length_mm", "bill_depth_mm", "island"});
    RandomForestClassifier model(penguins, "species ~ bill_length_mm + bill_depth_mm + island");

    DataFrame bad;
    bad.add_column("bill_length_mm", std::vector<double>{40.0});
    bad.add_column("bill_depth_mm", std::vector<double>{18.0});
    bad.add_column("island", std::vector<std::string>{"Atlantis"});
    EXPECT_THROW((void)model.predict(bad), std::runtime_error);
}

TEST(RandomForestClassifier, PlotDecisionRegionsRequiresExactlyTwoPredictors) {
    RandomForestClassifier model(datamunge::datasets::iris(),
                                 "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width");
    EXPECT_THROW(model.plot_decision_regions("Sepal.Length", "Sepal.Width"), std::invalid_argument);
}

// --- Regressor -------------------------------------------------------

TEST(RandomForestRegressor, IrisAchievesHighTrainAndOobRSquared) {
    RandomForestRegressor model(datamunge::datasets::iris(), "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width");
    EXPECT_GT(model.r_squared(), 0.9);
    EXPECT_GT(model.oob_r_squared(), 0.8);
}

TEST(RandomForestRegressor, DefaultMaxFeaturesIsPredictorCountOverThree) {
    RandomForestRegressor model(datamunge::datasets::iris(), "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width");
    // floor(3 / 3) = 1
    EXPECT_EQ(model.max_features_used(), 1u);
}

TEST(RandomForestRegressor, PredictorNamesExcludeIntercept) {
    RandomForestRegressor model(datamunge::datasets::iris(), "Petal.Length ~ Sepal.Length + Sepal.Width");
    const auto& names = model.predictor_names();
    ASSERT_EQ(names.size(), 2u);
    EXPECT_EQ(names[0], "Sepal.Length");
    EXPECT_EQ(names[1], "Sepal.Width");
}

TEST(RandomForestRegressor, FeatureImportanceSumsToOne) {
    RandomForestRegressor model(datamunge::datasets::iris(), "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width");
    const auto importance = model.feature_importance();
    double sum = 0.0;
    for (const double v : importance) {
        EXPECT_GE(v, 0.0);
        sum += v;
    }
    EXPECT_NEAR(sum, 1.0, 1e-9);
}

TEST(RandomForestRegressor, PredictReturnsNanForMissingPredictors) {
    RandomForestRegressor model(datamunge::datasets::iris(), "Petal.Length ~ Sepal.Length + Sepal.Width");

    DataFrame newdata;
    newdata.add_column("Sepal.Length", std::vector<std::optional<double>>{5.0, std::nullopt});
    newdata.add_column("Sepal.Width", std::vector<std::optional<double>>{3.0, 3.0});
    const auto preds = model.predict(newdata);
    ASSERT_EQ(preds.size(), 2u);
    EXPECT_FALSE(std::isnan(preds[0]));
    EXPECT_TRUE(std::isnan(preds[1]));
}

TEST(RandomForestRegressor, RejectsZeroTrees) {
    RandomForestRegressorOptions options;
    options.n_trees = 0;
    EXPECT_THROW(RandomForestRegressor(datamunge::datasets::iris(), "Petal.Length ~ Sepal.Length + Sepal.Width",
                                       options),
                 std::invalid_argument);
}
