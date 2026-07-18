#include <gtest/gtest.h>

#include <datamunge/datasets/datasets.hpp>
#include <datamunge/dstruct/dstruct.hpp>
#include <datamunge/stats/stats.hpp>

#include <cmath>

using datamunge::dstruct::DataFrame;
using datamunge::stats::NaiveBayesClassifier;
using datamunge::stats::NaiveBayesClassifierOptions;

// Reference values computed with scikit-learn's GaussianNB (all-numeric
// case, on the embedded iris dataset -- var_smoothing default 1e-9
// matches this implementation's default exactly) and CategoricalNB
// (all-categorical case, on the embedded penguins dataset's island/sex
// columns predicting species -- alpha default 1.0 matches this
// implementation's laplace_smoothing default exactly). Both are
// well-defined closed-form algorithms with no tie-breaking ambiguity, so
// exact agreement is expected.

TEST(NaiveBayesClassifier, GaussianOnlyMatchesSklearnGaussianNB) {
    NaiveBayesClassifier model(datamunge::datasets::iris(),
                               "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width");

    ASSERT_EQ(model.classes().size(), 3u);
    EXPECT_EQ(model.classes()[0], "setosa");
    EXPECT_EQ(model.classes()[1], "versicolor");
    EXPECT_EQ(model.classes()[2], "virginica");

    for (const double p : model.class_priors()) EXPECT_NEAR(p, 1.0 / 3.0, 1e-9);
    EXPECT_NEAR(model.training_accuracy(), 0.96, 1e-6);

    DataFrame newdata;
    newdata.add_column("Sepal.Length", std::vector<double>{5.0, 6.0, 7.2});
    newdata.add_column("Sepal.Width", std::vector<double>{3.5, 2.7, 3.0});
    newdata.add_column("Petal.Length", std::vector<double>{1.3, 5.1, 6.0});
    newdata.add_column("Petal.Width", std::vector<double>{0.3, 1.6, 1.8});
    const auto detail = model.predict_detail(newdata);

    EXPECT_EQ(detail.class_label[0], "setosa");
    EXPECT_EQ(detail.class_label[1], "versicolor");
    EXPECT_EQ(detail.class_label[2], "virginica");

    const double expected_p0[3] = {1.00000000e+000, 6.10454787e-018, 1.90690879e-025};
    const double expected_p1[3] = {2.14069731e-135, 6.12159845e-001, 3.87840155e-001};
    const double expected_p2[3] = {1.82145519e-207, 1.14505557e-005, 9.99988549e-001};
    for (int c = 0; c < 3; ++c) {
        EXPECT_NEAR(detail.probability[0][c], expected_p0[c], 1e-6);
        EXPECT_NEAR(detail.probability[1][c], expected_p1[c], 1e-6);
        EXPECT_NEAR(detail.probability[2][c], expected_p2[c], 1e-6);
    }
}

TEST(NaiveBayesClassifier, CategoricalOnlyMatchesSklearnCategoricalNB) {
    auto penguins = datamunge::datasets::penguins().drop_nulls(
        {"species", "island", "sex", "bill_length_mm", "bill_depth_mm"});
    NaiveBayesClassifier model(penguins, "species ~ island + sex");

    ASSERT_EQ(model.classes().size(), 3u);
    EXPECT_EQ(model.classes()[0], "Adelie");
    EXPECT_EQ(model.classes()[1], "Chinstrap");
    EXPECT_EQ(model.classes()[2], "Gentoo");

    const double expected_priors[3] = {0.43843844, 0.2042042, 0.35735736};
    for (std::size_t k = 0; k < 3; ++k) EXPECT_NEAR(model.class_priors()[k], expected_priors[k], 1e-6);
    EXPECT_NEAR(model.training_accuracy(), 0.7027027027027027, 1e-6);

    DataFrame newdata;
    newdata.add_column("island", std::vector<std::string>{"Biscoe", "Dream", "Torgersen"});
    newdata.add_column("sex", std::vector<std::string>{"male", "female", "male"});
    const auto detail = model.predict_detail(newdata);

    EXPECT_EQ(detail.class_label[0], "Gentoo");
    EXPECT_EQ(detail.class_label[1], "Chinstrap");
    EXPECT_EQ(detail.class_label[2], "Adelie");

    const double expected_p0[3] = {0.26723137, 0.00580442, 0.72696421};
    const double expected_p1[3] = {0.45011308, 0.54208411, 0.00780281};
    const double expected_p2[3] = {0.96004686, 0.01954948, 0.02040366};
    for (int c = 0; c < 3; ++c) {
        EXPECT_NEAR(detail.probability[0][c], expected_p0[c], 1e-6);
        EXPECT_NEAR(detail.probability[1][c], expected_p1[c], 1e-6);
        EXPECT_NEAR(detail.probability[2][c], expected_p2[c], 1e-6);
    }
}

TEST(NaiveBayesClassifier, MixedNumericAndCategoricalFormulaFitsAndPredicts) {
    auto penguins = datamunge::datasets::penguins().drop_nulls(
        {"species", "island", "sex", "bill_length_mm", "bill_depth_mm"});
    NaiveBayesClassifier model(penguins, "species ~ bill_length_mm + bill_depth_mm + island + sex");

    EXPECT_GT(model.training_accuracy(), 0.9); // mixing in the strong numeric bill measurements should help a lot
    ASSERT_EQ(model.predictor_names().size(), 4u);

    const auto preds = model.predict(penguins);
    EXPECT_EQ(preds.size(), penguins.nrows());
}

TEST(NaiveBayesClassifier, ProbabilitiesSumToOne) {
    NaiveBayesClassifier model(datamunge::datasets::iris(), "Species ~ Petal.Length + Petal.Width");

    DataFrame newdata;
    newdata.add_column("Petal.Length", std::vector<double>{4.5});
    newdata.add_column("Petal.Width", std::vector<double>{1.5});
    const auto detail = model.predict_detail(newdata);
    ASSERT_EQ(detail.probability.size(), 1u);
    double sum = 0.0;
    for (const double v : detail.probability[0]) sum += v;
    EXPECT_NEAR(sum, 1.0, 1e-9);
}

TEST(NaiveBayesClassifier, RejectsInteractionTerms) {
    EXPECT_THROW(NaiveBayesClassifier(datamunge::datasets::iris(), "Species ~ Petal.Length:Petal.Width"),
                std::invalid_argument);
}

TEST(NaiveBayesClassifier, RejectsFunctionTerms) {
    EXPECT_THROW(NaiveBayesClassifier(datamunge::datasets::iris(), "Species ~ log(Petal.Length)"),
                std::invalid_argument);
}

TEST(NaiveBayesClassifier, RejectsSingleClassResponse) {
    DataFrame df;
    df.add_column("x", std::vector<double>{1.0, 2.0, 3.0, 4.0});
    df.add_column("label", std::vector<std::string>{"a", "a", "a", "a"});
    EXPECT_THROW(NaiveBayesClassifier(df, "label ~ x"), std::invalid_argument);
}

TEST(NaiveBayesClassifier, PredictThrowsOnUnseenCategoricalLevel) {
    auto penguins = datamunge::datasets::penguins().drop_nulls({"species", "bill_length_mm", "island"});
    NaiveBayesClassifier model(penguins, "species ~ bill_length_mm + island");

    DataFrame bad;
    bad.add_column("bill_length_mm", std::vector<double>{40.0});
    bad.add_column("island", std::vector<std::string>{"Atlantis"});
    EXPECT_THROW((void)model.predict(bad), std::runtime_error);
}

TEST(NaiveBayesClassifier, PlotDecisionRegionsRequiresTwoNumericPredictors) {
    auto penguins = datamunge::datasets::penguins().drop_nulls({"species", "bill_length_mm", "island"});
    NaiveBayesClassifier mixed(penguins, "species ~ bill_length_mm + island");
    EXPECT_THROW(mixed.plot_decision_regions("bill_length_mm", "island"), std::invalid_argument);

    NaiveBayesClassifier model(datamunge::datasets::iris(),
                               "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width");
    EXPECT_THROW(model.plot_decision_regions("Sepal.Length", "Sepal.Width"), std::invalid_argument);
}

TEST(NaiveBayesClassifier, SummaryContainsKeyInformation) {
    NaiveBayesClassifier model(datamunge::datasets::iris(), "Species ~ Petal.Length + Petal.Width");
    const auto text = model.summary();
    EXPECT_NE(text.find("Class priors"), std::string::npos);
    EXPECT_NE(text.find("Petal.Length"), std::string::npos);
    EXPECT_NE(text.find("Training accuracy"), std::string::npos);
}
