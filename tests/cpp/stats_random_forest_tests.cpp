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

namespace {

// Ten points on a line with overlapping classes: small enough to check by hand.
DataFrame ten_points() {
    DataFrame d;
    d.add_column("x", std::vector<double>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    d.add_column("y", std::vector<std::string>{"a", "a", "a", "b", "a", "b", "b", "a", "b", "b"});
    return d;
}

} // namespace

TEST(DecisionTreeClassifier, SweepSplitsMatchTheKnownIrisTree) {
    // The full-depth CART tree of iris (scikit-learn finds the same tree, up to the tie between
    // the two petal measurements at the root).
    const datamunge::stats::DecisionTreeClassifier tree(datamunge::datasets::iris(), "Species ~ .", {.max_depth = 30});
    EXPECT_EQ(tree.node_count(), 17u);
    const auto frame = tree.node_frame();
    EXPECT_EQ(frame.string_at("feature", 0), "Petal.Length");
    EXPECT_DOUBLE_EQ(frame.double_at("threshold", 0), 2.45);
    EXPECT_DOUBLE_EQ(frame.double_at("threshold", 2), 1.75);
    EXPECT_FALSE(frame.optional_double_at("threshold", 1).has_value()); // a leaf
    EXPECT_DOUBLE_EQ(frame.double_at("n", 2), 100.0);
    const auto importance = tree.feature_importance();
    EXPECT_NEAR(importance[2], 0.5507226247987119, 1e-12);
    EXPECT_NEAR(importance[3], 0.4226107085346215, 1e-12);
    EXPECT_EQ(tree.classes()[tree.predict_row({5.0, 3.4, 1.5, 0.2})], "setosa");
    EXPECT_THROW((void)tree.predict_row({1.0}), std::invalid_argument);
}

TEST(RandomForestClassifier, OutOfBagBookkeepingByHand) {
    const RandomForestClassifier forest(ten_points(), "y ~ x", {.n_trees = 3, .max_depth = 1, .seed = 1});
    std::vector<int> oob(10, 0);
    for (std::size_t t = 0; t < 3; ++t) {
        const auto counts = forest.in_bag_counts(t);
        int        total  = 0;
        for (std::size_t i = 0; i < 10; ++i) total += counts[i], oob[i] += counts[i] == 0;
        EXPECT_EQ(total, 10); // a bootstrap sample has n rows
    }
    EXPECT_EQ(forest.oob_tree_counts(), oob);
    const auto predictions = forest.oob_predictions();
    const auto margins     = forest.oob_margins();
    int        right = 0, used = 0;
    for (std::size_t i = 0; i < 10; ++i) {
        if (oob[i] == 0) {
            EXPECT_EQ(predictions[i], "");
            EXPECT_TRUE(std::isnan(margins[i]));
            continue;
        }
        ++used;
        const bool correct = predictions[i] == ten_points().string_at("y", i);
        right += correct;
        if (margins[i] > 0) EXPECT_TRUE(correct);
        if (margins[i] < 0) EXPECT_FALSE(correct);
    }
    EXPECT_DOUBLE_EQ(forest.oob_accuracy(), static_cast<double>(right) / used);
    EXPECT_DOUBLE_EQ(forest.oob_accuracy_path().back(), forest.oob_accuracy());
    EXPECT_THROW((void)forest.tree(3), std::out_of_range);
}

TEST(RandomForestClassifier, FastVotesMatchTheTreesOwnPredictions) {
    const auto                   iris = datamunge::datasets::iris();
    const RandomForestClassifier forest(iris, "Species ~ .", {.n_trees = 30});
    const auto                   detail = forest.predict_detail(iris);
    for (std::size_t i = 0; i < iris.nrows(); i += 7) {
        const std::vector<double> x{iris.double_at("Sepal.Length", i), iris.double_at("Sepal.Width", i),
                                    iris.double_at("Petal.Length", i), iris.double_at("Petal.Width", i)};
        const auto votes = forest.tree_votes(x);
        const auto row   = iris.take_rows({i});
        for (std::size_t t = 0; t < forest.n_trees(); ++t)
            EXPECT_EQ(forest.classes()[votes[t]], forest.tree(t).predict(row)[0]);
        const auto share = forest.vote_share(x);
        for (std::size_t c = 0; c < 3; ++c) EXPECT_NEAR(share[c], detail.vote_share[i][c], 1e-12);
    }
}

TEST(RandomForestClassifier, PermutationImportanceIgnoresNoiseButImpurityDoesNot) {
    auto                           iris = datamunge::datasets::iris();
    datamunge::random::SplitMix64 rng(4);
    std::vector<double>           noise(iris.nrows());
    for (double& v : noise) v = static_cast<double>(rng.next_u64() % 1000000) / 1e6;
    iris.add_column("noise", noise);
    const RandomForestClassifier forest(iris, "Species ~ .", {.n_trees = 300});
    const auto                   permutation = forest.permutation_importance();
    const auto                   impurity    = forest.feature_importance();
    ASSERT_EQ(permutation.predictor_names.back(), "noise");
    EXPECT_LT(std::abs(permutation.importance[4]), 3.0 * permutation.standard_error[4] + 1e-3);
    EXPECT_GT(permutation.importance[2], 0.1); // Petal.Length
    EXPECT_GT(impurity[4], 0.01);               // the impurity bias towards continuous noise
}

TEST(RandomForestClassifier, StrengthAndCorrelationSatisfyTheirDefinitions) {
    const RandomForestClassifier forest(datamunge::datasets::iris(), "Species ~ .", {.n_trees = 200});
    const auto                   sc      = forest.strength_correlation();
    const auto                   margins = forest.oob_margins();
    double                       mean    = 0.0;
    for (const double m : margins) mean += m / static_cast<double>(margins.size());
    EXPECT_EQ(sc.rows, 150u);
    EXPECT_NEAR(sc.strength, mean, 1e-12);
    EXPECT_NEAR(sc.correlation * sc.mean_tree_sd * sc.mean_tree_sd, sc.margin_variance, 1e-12);
    EXPECT_NEAR(sc.error_bound, sc.correlation * (1 - sc.strength * sc.strength) / (sc.strength * sc.strength), 1e-12);
    EXPECT_GT(sc.error_bound, 1.0 - forest.oob_accuracy());
}

TEST(RandomForestClassifier, RowsWithMissingValuesAreDroppedConsistently) {
    // penguins has two rows without bill measurements; the forest must use the same 342 rows
    // for its samples, its out-of-bag votes and its fitted classes.
    const auto               penguins = datamunge::datasets::penguins();
    std::vector<std::size_t> keep;
    for (std::size_t i = 0; i < penguins.nrows(); ++i)
        if (penguins.optional_double_at("bill_length_mm", i)) keep.push_back(i);
    const std::string            f = "species ~ bill_length_mm + bill_depth_mm";
    const RandomForestClassifier all(penguins, f), complete(penguins.take_rows(keep), f);
    EXPECT_EQ(all.fitted_classes().size(), 342u);
    EXPECT_DOUBLE_EQ(all.training_accuracy(), complete.training_accuracy());
    EXPECT_DOUBLE_EQ(all.oob_accuracy(), complete.oob_accuracy());

    const std::string            g = "body_mass_g ~ bill_length_mm + bill_depth_mm";
    const RandomForestRegressor  rall(penguins, g), rcomplete(penguins.take_rows(keep), g);
    EXPECT_DOUBLE_EQ(rall.oob_r_squared(), rcomplete.oob_r_squared());
}

TEST(RandomForestClassifier, DecisionRegionsShadeByVoteAgreement) {
    const RandomForestClassifier forest(datamunge::datasets::iris(), "Species ~ Petal.Length + Petal.Width", {.n_trees = 20});
    const auto plot = forest.plot_decision_regions("Petal.Length", "Petal.Width", 40, std::array<double, 4>{1, 7, 0, 2.6});
    EXPECT_NE(plot.to_svg().find("<svg"), std::string::npos);
}
