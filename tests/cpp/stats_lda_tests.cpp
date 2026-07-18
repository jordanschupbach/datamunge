#include <gtest/gtest.h>

#include <datamunge/datasets/datasets.hpp>
#include <datamunge/dstruct/dstruct.hpp>
#include <datamunge/stats/stats.hpp>

#include <cmath>

using datamunge::dstruct::DataFrame;
using datamunge::stats::LDA;
using datamunge::stats::LDAOptions;

// Reference values below were computed with scikit-learn's
// LinearDiscriminantAnalysis(solver='eigen') on the same embedded iris
// dataset, which is itself byte-identical to R's built-in `iris`.

TEST(LDA, IrisGroupMeansAndPriorsMatchSklearn) {
    LDA model(datamunge::datasets::iris(),
              "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width");

    ASSERT_EQ(model.classes().size(), 3u);
    EXPECT_EQ(model.classes()[0], "setosa");
    EXPECT_EQ(model.classes()[1], "versicolor");
    EXPECT_EQ(model.classes()[2], "virginica");

    for (const double p : model.priors()) EXPECT_NEAR(p, 1.0 / 3.0, 1e-9);

    const std::vector<std::vector<double>> expected_means = {
        {5.006, 3.428, 1.462, 0.246},
        {5.936, 2.770, 4.260, 1.326},
        {6.588, 2.974, 5.552, 2.026},
    };
    ASSERT_EQ(model.group_means().size(), 3u);
    for (std::size_t c = 0; c < 3; ++c)
        for (std::size_t j = 0; j < 4; ++j)
            EXPECT_NEAR(model.group_means()[c][j], expected_means[c][j], 1e-9);
}

TEST(LDA, IrisTrainingAccuracyAndConfusionMatrixMatchSklearn) {
    LDA model(datamunge::datasets::iris(),
              "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width");

    EXPECT_NEAR(model.training_accuracy(), 0.98, 1e-9);

    const auto cm = model.confusion_matrix();
    ASSERT_EQ(cm.rows(), 3u);
    ASSERT_EQ(cm.cols(), 3u);
    const double expected[3][3] = {{50, 0, 0}, {0, 48, 2}, {0, 1, 49}};
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j) EXPECT_NEAR(cm(i, j), expected[i][j], 1e-9);
}

TEST(LDA, IrisProportionOfTraceMatchesSklearnExplainedVarianceRatio) {
    LDA model(datamunge::datasets::iris(),
              "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width");

    ASSERT_EQ(model.proportion_of_trace().size(), 2u);
    EXPECT_NEAR(model.proportion_of_trace()[0], 0.9912126, 1e-4);
    EXPECT_NEAR(model.proportion_of_trace()[1], 0.0087874, 1e-4);
}

TEST(LDA, IrisPredictionsMatchSklearn) {
    LDA model(datamunge::datasets::iris(),
              "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width");

    DataFrame newdata;
    newdata.add_column("Sepal.Length", std::vector<double>{5.1, 6.0, 6.5});
    newdata.add_column("Sepal.Width", std::vector<double>{3.5, 2.7, 3.0});
    newdata.add_column("Petal.Length", std::vector<double>{1.4, 4.5, 5.5});
    newdata.add_column("Petal.Width", std::vector<double>{0.2, 1.5, 2.0});

    const auto detail = model.predict_detail(newdata);
    ASSERT_EQ(detail.class_label.size(), 3u);
    EXPECT_EQ(detail.class_label[0], "setosa");
    EXPECT_EQ(detail.class_label[1], "versicolor");
    EXPECT_EQ(detail.class_label[2], "virginica");

    // sklearn's predict_proba() pools the within-class covariance with a
    // divide-by-n estimator; this class uses the unbiased divide-by-(n-k)
    // estimator R's MASS::lda uses, so posteriors differ very slightly from
    // sklearn (labels, means, and proportion-of-trace above are unaffected
    // by that scale choice and match sklearn exactly). These reference
    // values come from an independent from-scratch numpy re-derivation of
    // the same divide-by-(n-k) Bayes discriminant formula this class
    // implements, cross-checked to agree with sklearn to within ~0.1
    // percentage points as expected from the (n-k)/n = 147/150 scale ratio.
    ASSERT_EQ(detail.posterior[0].size(), 3u);
    EXPECT_NEAR(detail.posterior[0][0], 1.0, 1e-6);
    EXPECT_NEAR(detail.posterior[1][1], 0.9853729233, 1e-8);
    EXPECT_NEAR(detail.posterior[1][2], 0.0146270767, 1e-8);
    EXPECT_NEAR(detail.posterior[2][2], 0.999722335, 1e-4);

    const auto simple = model.predict(newdata);
    ASSERT_EQ(simple.size(), 3u);
    EXPECT_EQ(simple[0], "setosa");
    EXPECT_EQ(simple[1], "versicolor");
    EXPECT_EQ(simple[2], "virginica");
}

TEST(LDA, CustomPriorsChangeClassification) {
    // A point right on the versicolor/virginica boundary should flip
    // toward whichever class is given overwhelming prior probability.
    LDAOptions heavy_virginica;
    heavy_virginica.priors = std::vector<double>{0.01, 0.01, 0.98};
    LDA model(datamunge::datasets::iris(),
              "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width", heavy_virginica);

    DataFrame boundary;
    boundary.add_column("Sepal.Length", std::vector<double>{6.2});
    boundary.add_column("Sepal.Width", std::vector<double>{2.8});
    boundary.add_column("Petal.Length", std::vector<double>{4.8});
    boundary.add_column("Petal.Width", std::vector<double>{1.8});

    EXPECT_EQ(model.predict(boundary)[0], "virginica");
}

TEST(LDA, RejectsMismatchedPriors) {
    LDAOptions bad_size;
    bad_size.priors = std::vector<double>{0.5, 0.5};
    EXPECT_THROW(LDA(datamunge::datasets::iris(),
                     "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width", bad_size),
                std::invalid_argument);

    LDAOptions bad_sum;
    bad_sum.priors = std::vector<double>{0.5, 0.5, 0.5};
    EXPECT_THROW(LDA(datamunge::datasets::iris(),
                     "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width", bad_sum),
                std::invalid_argument);
}

TEST(LDA, RejectsSingleClassResponse) {
    DataFrame df;
    df.add_column("x", std::vector<double>{1.0, 2.0, 3.0, 4.0});
    df.add_column("label", std::vector<std::string>{"a", "a", "a", "a"});
    EXPECT_THROW(LDA(df, "label ~ x"), std::invalid_argument);
}

TEST(LDA, DotFormulaMatchesExplicitFormula) {
    DataFrame df;
    df.add_column("x1", std::vector<double>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    df.add_column("x2", std::vector<double>{2, 1, 4, 3, 6, 5, 8, 7, 10, 9});
    df.add_column("label", std::vector<std::string>{"a", "a", "a", "a", "a", "b", "b", "b", "b", "b"});

    LDA explicit_model(df, "label ~ x1 + x2");
    LDA dot_model(df, "label ~ .");

    ASSERT_EQ(explicit_model.predictor_names().size(), dot_model.predictor_names().size());
    for (std::size_t c = 0; c < explicit_model.classes().size(); ++c)
        for (std::size_t j = 0; j < explicit_model.group_means()[c].size(); ++j)
            EXPECT_NEAR(explicit_model.group_means()[c][j], dot_model.group_means()[c][j], 1e-9);
}

TEST(LDA, CategoricalPredictorAndUnseenLevelHandling) {
    auto penguins = datamunge::datasets::penguins().drop_nulls({"bill_length_mm", "bill_depth_mm", "island"});
    LDA  model(penguins, "species ~ bill_length_mm + bill_depth_mm + island");

    EXPECT_GT(model.training_accuracy(), 0.9);

    DataFrame bad;
    bad.add_column("bill_length_mm", std::vector<double>{40.0});
    bad.add_column("bill_depth_mm", std::vector<double>{18.0});
    bad.add_column("island", std::vector<std::string>{"Atlantis"});
    EXPECT_THROW((void)model.predict(bad), std::runtime_error);
}
