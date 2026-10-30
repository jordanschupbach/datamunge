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

namespace {

// Small two-predictor data set with unequal scales, three classes and a numeric response.
DataFrame small_frame(std::size_t skip = SIZE_MAX) {
    const std::vector<double> x1{0.1, 0.4, 1.3, 2.2, 2.9, 3.1, 4.0, 4.4, 5.2, 6.1, 6.3, 7.5};
    const std::vector<double> x2{10, 35, 20, 80, 55, 90, 40, 75, 15, 60, 95, 30};
    const std::vector<std::string> c{"a", "a", "b", "a", "b", "c", "b", "c", "a", "c", "c", "b"};
    std::vector<double> a, b, y;
    std::vector<std::string> cls;
    for (std::size_t i = 0; i < x1.size(); ++i) {
        if (i == skip) continue;
        a.push_back(x1[i]), b.push_back(x2[i]), cls.push_back(c[i]), y.push_back(x1[i] * 2.0 + x2[i] / 10.0);
    }
    DataFrame d;
    d.add_column("x1", a);
    d.add_column("x2", b);
    d.add_column("cls", cls);
    d.add_column("y", y);
    return d;
}

DataFrame row_frame(std::size_t i) {
    const auto all = small_frame();
    DataFrame d;
    d.add_column("x1", std::vector<double>{all.get_column<double>("x1").values()[i]});
    d.add_column("x2", std::vector<double>{all.get_column<double>("x2").values()[i]});
    return d;
}

} // namespace

TEST(KNNClassifier, LeaveOneOutEqualsRefittingWithoutTheRow) {
    for (const bool weighted : {false, true})
        for (const std::size_t k : {1u, 3u, 4u}) {
            const KNNClassifierOptions options{.k = k, .weighted = weighted};
            const KNNClassifier full(small_frame(), "cls ~ x1 + x2", options);
            for (std::size_t i = 0; i < 12; ++i) {
                const KNNClassifier refit(small_frame(i), "cls ~ x1 + x2", options);
                EXPECT_EQ(full.fitted_classes()[i], refit.predict(row_frame(i))[0]) << "row " << i << " k " << k;
            }
        }
}

TEST(KNNRegressor, LeaveOneOutEqualsRefittingWithoutTheRow) {
    for (const auto metric : {DistanceMetric::Euclidean, DistanceMetric::Manhattan}) {
        const KNNRegressorOptions options{.k = 3, .metric = metric, .weighted = true};
        const KNNRegressor full(small_frame(), "y ~ x1 + x2", options);
        for (std::size_t i = 0; i < 12; ++i) {
            const KNNRegressor refit(small_frame(i), "y ~ x1 + x2", options);
            EXPECT_NEAR(full.fitted_values()[i], refit.predict(row_frame(i))[0], 1e-12) << "row " << i;
        }
    }
}

TEST(KNNClassifier, LooAccuracyPathMatchesSeparateFits) {
    const auto iris = datamunge::datasets::iris();
    for (const bool weighted : {false, true}) {
        const KNNClassifier base(iris, "Species ~ Sepal.Length + Sepal.Width", {.k = 1, .weighted = weighted});
        const auto path = base.loo_accuracy_path(12);
        ASSERT_EQ(path.size(), 12u);
        for (std::size_t k = 1; k <= 12; ++k) {
            const KNNClassifier fit(iris, "Species ~ Sepal.Length + Sepal.Width", {.k = k, .weighted = weighted});
            EXPECT_DOUBLE_EQ(path[k - 1], fit.training_accuracy()) << "k " << k;
        }
    }
}

TEST(KNNRegressor, LooRmsePathMatchesSeparateFits) {
    const auto iris = datamunge::datasets::iris();
    const KNNRegressor base(iris, "Petal.Length ~ Sepal.Length + Sepal.Width", {.k = 1});
    const auto path = base.loo_rmse_path(10);
    for (std::size_t k = 1; k <= 10; ++k) {
        const KNNRegressor fit(iris, "Petal.Length ~ Sepal.Length + Sepal.Width", {.k = k});
        EXPECT_NEAR(path[k - 1], fit.rmse(), 1e-12) << "k " << k;
    }
}

TEST(KNNClassifier, KneighborsByHand) {
    DataFrame d;
    d.add_column("x", std::vector<double>{0, 3, 1, 4});
    d.add_column("y", std::vector<double>{0, 4, 1, 0});
    d.add_column("c", std::vector<std::string>{"a", "b", "a", "b"});
    const KNNClassifier knn(d, "c ~ x + y", {.k = 3, .metric = DistanceMetric::Manhattan, .standardize = false});
    DataFrame q;
    q.add_column("x", std::vector<double>{2.0});
    q.add_column("y", std::vector<double>{1.0});
    const auto nb = knn.kneighbors(q);
    ASSERT_EQ(nb.index[0].size(), 3u);
    EXPECT_EQ(nb.index[0][0], 2u);  // (1, 1): distance 1
    EXPECT_EQ(nb.index[0][1], 0u);  // (0, 0): distance 3, tie with (4, 0) broken by row order
    EXPECT_EQ(nb.index[0][2], 3u);
    EXPECT_DOUBLE_EQ(nb.distance[0][0], 1.0);
    EXPECT_DOUBLE_EQ(nb.distance[0][1], 3.0);
    EXPECT_DOUBLE_EQ(nb.distance[0][2], 3.0);
    EXPECT_EQ(knn.predict(q)[0], "a");
}

TEST(KNNAsymptoticRisk, ClosedForms) {
    using datamunge::stats::knn_asymptotic_risk;
    for (const double eta : {0.0, 0.1, 0.3, 0.5, 0.8}) {
        EXPECT_NEAR(knn_asymptotic_risk(eta, 1), 2 * eta * (1 - eta), 1e-14);
        EXPECT_NEAR(knn_asymptotic_risk(eta, 2), 2 * eta * (1 - eta), 1e-14);  // a tie is a coin flip
        EXPECT_NEAR(knn_asymptotic_risk(eta, 7), knn_asymptotic_risk(1 - eta, 7), 1e-14);
    }
    // k = 3: eta P(B <= 1) + (1 - eta) P(B >= 2).
    const double e = 0.2, p_le1 = std::pow(0.8, 3) + 3 * 0.2 * 0.64, p_ge2 = 1 - p_le1;
    EXPECT_NEAR(knn_asymptotic_risk(e, 3), e * p_le1 + (1 - e) * p_ge2, 1e-14);
    EXPECT_NEAR(knn_asymptotic_risk(0.5, 9), 0.5, 1e-14);
    EXPECT_NEAR(knn_asymptotic_risk(0.2, 1001), 0.2, 1e-10);  // tends to the Bayes risk min(eta, 1 - eta)
    EXPECT_THROW((void)knn_asymptotic_risk(1.5, 3), std::invalid_argument);
    EXPECT_THROW((void)knn_asymptotic_risk(0.5, 0), std::invalid_argument);
}

TEST(KNNRegressor, RawDistanceTiesSurviveStandardisation) {
    // x = 2 is equally far from x = 1 and x = 3; standardised values carry rounding error, but
    // differences are taken on raw values, so the tie stands and row order picks x = 1.
    DataFrame d;
    d.add_column("x", std::vector<double>{1, 2, 3, 4, 5, 6});
    d.add_column("y", std::vector<double>{2, 4, 3, 7, 6, 9});
    const KNNRegressor knn(d, "y ~ x", {.k = 1});
    const std::vector<double> expected{4, 2, 4, 3, 7, 6};
    for (std::size_t i = 0; i < 6; ++i) EXPECT_DOUBLE_EQ(knn.fitted_values()[i], expected[i]) << "row " << i;
}
