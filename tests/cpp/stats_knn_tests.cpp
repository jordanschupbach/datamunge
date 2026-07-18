#include <gtest/gtest.h>

#include <datamunge/datasets/datasets.hpp>
#include <datamunge/dstruct/dstruct.hpp>
#include <datamunge/stats/stats.hpp>

#include <cmath>

using datamunge::dstruct::DataFrame;
using datamunge::stats::DistanceMetric;
using datamunge::stats::KNNClassifier;
using datamunge::stats::KNNClassifierOptions;
using datamunge::stats::KNNRegressor;
using datamunge::stats::KNNRegressorOptions;

// Reference values computed with scikit-learn's KNeighborsClassifier /
// KNeighborsRegressor, using sklearn's StandardScaler (population std,
// ddof=0 -- matching this implementation's standardization convention
// exactly) on the same embedded iris dataset. Leave-one-out figures were
// computed by refitting sklearn's estimator with each row excluded in
// turn, since sklearn has no built-in "exclude self" prediction mode.

TEST(KNNClassifier, FixedQueryPredictionsMatchSklearnUniformEuclidean) {
    KNNClassifierOptions options;
    options.k = 5;
    KNNClassifier model(datamunge::datasets::iris(), "Species ~ Petal.Length + Petal.Width", options);

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

TEST(KNNClassifier, FixedQueryPredictionsMatchSklearnWeightedManhattan) {
    KNNClassifierOptions options;
    options.k        = 7;
    options.metric   = DistanceMetric::Manhattan;
    options.weighted = true;
    KNNClassifier model(datamunge::datasets::iris(), "Species ~ Petal.Length + Petal.Width", options);

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

TEST(KNNClassifier, LeaveOneOutAccuracyMatchesSklearn) {
    KNNClassifierOptions options;
    options.k = 5;
    KNNClassifier model(datamunge::datasets::iris(), "Species ~ Petal.Length + Petal.Width", options);
    EXPECT_NEAR(model.training_accuracy(), 0.9666666666666667, 1e-9);
}

TEST(KNNClassifier, KEqualsOneOverfitsResubstitutionButLooIsHonest) {
    // A resubstitution fit (fit and predict on the very same points) with
    // k=1 always gets 100% on training data trivially -- the nearest
    // neighbor of a point is always itself. This implementation only
    // exposes the honest leave-one-out figure, which must therefore be
    // strictly below 100% whenever classes genuinely overlap.
    KNNClassifierOptions options;
    options.k = 1;
    KNNClassifier model(datamunge::datasets::iris(), "Species ~ Petal.Length + Petal.Width", options);
    EXPECT_LT(model.training_accuracy(), 1.0);
    EXPECT_GT(model.training_accuracy(), 0.9);
}

TEST(KNNClassifier, RejectsZeroK) {
    KNNClassifierOptions options;
    options.k = 0;
    EXPECT_THROW(KNNClassifier(datamunge::datasets::iris(), "Species ~ Petal.Length + Petal.Width", options),
                std::invalid_argument);
}

TEST(KNNClassifier, RejectsSingleClassResponse) {
    DataFrame df;
    df.add_column("x", std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    df.add_column("label", std::vector<std::string>{"a", "a", "a", "a", "a", "a"});
    EXPECT_THROW(KNNClassifier(df, "label ~ x"), std::invalid_argument);
}

TEST(KNNClassifier, PredictThrowsOnUnseenCategoricalLevel) {
    auto penguins = datamunge::datasets::penguins().drop_nulls({"bill_length_mm", "bill_depth_mm", "island"});
    KNNClassifier model(penguins, "species ~ bill_length_mm + bill_depth_mm + island");

    DataFrame bad;
    bad.add_column("bill_length_mm", std::vector<double>{40.0});
    bad.add_column("bill_depth_mm", std::vector<double>{18.0});
    bad.add_column("island", std::vector<std::string>{"Atlantis"});
    EXPECT_THROW((void)model.predict(bad), std::runtime_error);
}

TEST(KNNClassifier, VoteSharesSumToOne) {
    KNNClassifierOptions options;
    options.k = 5;
    KNNClassifier model(datamunge::datasets::iris(), "Species ~ Petal.Length + Petal.Width", options);

    DataFrame newdata;
    newdata.add_column("Petal.Length", std::vector<double>{4.5});
    newdata.add_column("Petal.Width", std::vector<double>{1.5});
    const auto detail = model.predict_detail(newdata);
    ASSERT_EQ(detail.vote_share.size(), 1u);
    double sum = 0.0;
    for (const double v : detail.vote_share[0]) sum += v;
    EXPECT_NEAR(sum, 1.0, 1e-9);
}

TEST(KNNClassifier, PlotDecisionRegionsRequiresExactlyTwoPredictors) {
    KNNClassifier model(datamunge::datasets::iris(),
                        "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width");
    EXPECT_THROW(model.plot_decision_regions("Sepal.Length", "Sepal.Width"), std::invalid_argument);
}

// --- Regressor -------------------------------------------------------

TEST(KNNRegressor, FixedQueryPredictionsMatchSklearn) {
    KNNRegressorOptions options;
    options.k = 5;
    KNNRegressor model(datamunge::datasets::iris(), "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width",
                       options);

    DataFrame newdata;
    newdata.add_column("Sepal.Length", std::vector<double>{5.0, 6.5});
    newdata.add_column("Sepal.Width", std::vector<double>{3.0, 3.0});
    newdata.add_column("Petal.Width", std::vector<double>{0.3, 1.8});
    const auto preds = model.predict(newdata);
    ASSERT_EQ(preds.size(), 2u);
    EXPECT_NEAR(preds[0], 1.46, 1e-9);
    EXPECT_NEAR(preds[1], 5.36, 1e-9);
}

TEST(KNNRegressor, LeaveOneOutRSquaredAndRmseMatchSklearn) {
    // iris has several exact/near-duplicate rows: 23 of 150 rows have an
    // exact tie at the k=5 neighbor-distance boundary. This implementation
    // breaks ties by ascending original row index; sklearn's internal
    // algorithm breaks them differently (and LOO refitting reindexes the
    // remaining rows each time, changing which index "wins" a tie), so the
    // two leave-one-out figures are close but not bit-identical -- a
    // legitimate tie-breaking difference, not a bug (matches this
    // implementation's DecisionTree root-split tie-breaking precedent).
    KNNRegressorOptions options;
    options.k = 5;
    KNNRegressor model(datamunge::datasets::iris(), "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width",
                       options);
    EXPECT_NEAR(model.r_squared(), 0.9632464646560365, 5e-3);
    EXPECT_NEAR(model.rmse(), 0.3372990759943861, 5e-3);
}

TEST(KNNRegressor, PredictorNamesExcludeIntercept) {
    KNNRegressor model(datamunge::datasets::iris(), "Petal.Length ~ Sepal.Length + Sepal.Width");
    const auto& names = model.predictor_names();
    ASSERT_EQ(names.size(), 2u);
    EXPECT_EQ(names[0], "Sepal.Length");
    EXPECT_EQ(names[1], "Sepal.Width");
}

TEST(KNNRegressor, PredictReturnsNanForMissingPredictors) {
    KNNRegressor model(datamunge::datasets::iris(), "Petal.Length ~ Sepal.Length + Sepal.Width");

    DataFrame newdata;
    newdata.add_column("Sepal.Length", std::vector<std::optional<double>>{5.0, std::nullopt});
    newdata.add_column("Sepal.Width", std::vector<std::optional<double>>{3.0, 3.0});
    const auto preds = model.predict(newdata);
    ASSERT_EQ(preds.size(), 2u);
    EXPECT_FALSE(std::isnan(preds[0]));
    EXPECT_TRUE(std::isnan(preds[1]));
}

TEST(KNNRegressor, RejectsZeroK) {
    KNNRegressorOptions options;
    options.k = 0;
    EXPECT_THROW(
        KNNRegressor(datamunge::datasets::iris(), "Petal.Length ~ Sepal.Length + Sepal.Width", options),
        std::invalid_argument);
}

TEST(KNNRegressor, WeightedAndUniformGiveDifferentPredictionsInGeneral) {
    KNNRegressorOptions uniform, weighted;
    uniform.k  = 5;
    weighted.k = 5;
    weighted.weighted = true;
    KNNRegressor u(datamunge::datasets::iris(), "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width", uniform);
    KNNRegressor w(datamunge::datasets::iris(), "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width", weighted);
    // Not identical in general, since neighbor distances within a k-set are rarely all equal.
    EXPECT_NE(u.rmse(), w.rmse());
}
