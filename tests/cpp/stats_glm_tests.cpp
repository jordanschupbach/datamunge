#include <gtest/gtest.h>

#include <datamunge/datasets/datasets.hpp>
#include <datamunge/dstruct/dstruct.hpp>
#include <datamunge/stats/stats.hpp>

#include <cmath>

using datamunge::dstruct::DataFrame;
using datamunge::stats::GLM;
using datamunge::stats::GLMFamily;
using datamunge::stats::GLMOptions;
using datamunge::stats::GLMPredictionInterval;

// Reference values computed with R's base glm() (an exact, authoritative
// oracle -- no extra packages needed) on the embedded iris dataset. IRLS is
// a deterministic algorithm, so agreement with R's glm.fit is expected to
// close numerical precision.

TEST(GLM, BinomialLogitMatchesR) {
    auto iris = datamunge::datasets::iris();
    // versicolor/virginica subset only: the setosa-vs-rest split is
    // perfectly separable on these predictors, which sends both R's and
    // this implementation's IRLS coefficients toward +/-infinity -- a
    // genuine degenerate-data behavior in logistic regression, not
    // something either implementation should be validated against (the
    // divergent path depends on convergence-tolerance details neither
    // implementation is obligated to match exactly).
    std::vector<double> is_virginica;
    std::vector<double> sepal_length, sepal_width;
    for (std::size_t i = 0; i < iris.nrows(); ++i) {
        const auto species = iris.string_at("Species", i);
        if (species != "versicolor" && species != "virginica") continue;
        is_virginica.push_back(species == "virginica" ? 1.0 : 0.0);
        sepal_length.push_back(iris.double_at("Sepal.Length", i));
        sepal_width.push_back(iris.double_at("Sepal.Width", i));
    }
    DataFrame sub;
    sub.add_column("Sepal.Length", sepal_length);
    sub.add_column("Sepal.Width", sepal_width);
    sub.add_column("is_virginica", is_virginica);

    GLMOptions options;
    options.family = GLMFamily::Binomial;
    GLM model(sub, "is_virginica ~ Sepal.Length + Sepal.Width", options);

    ASSERT_EQ(model.coefficients().size(), 3u);
    EXPECT_NEAR(model.coefficients()[0], -13.046029650693, 1e-4);
    EXPECT_NEAR(model.coefficients()[1], 1.902375218511, 1e-4);
    EXPECT_NEAR(model.coefficients()[2], 0.404659412253, 1e-4);
    EXPECT_NEAR(model.standard_errors()[0], 3.097360539369, 1e-3);
    EXPECT_NEAR(model.standard_errors()[1], 0.516912518687, 1e-3);
    EXPECT_NEAR(model.standard_errors()[2], 0.862831624630, 1e-3);
    EXPECT_NEAR(model.deviance(), 110.325708079242, 1e-4);
    EXPECT_NEAR(model.null_deviance(), 138.629436111989, 1e-4);
    EXPECT_NEAR(model.aic(), 116.325708079242, 1e-4);
    EXPECT_NEAR(model.dispersion(), 1.0, 1e-12);

    DataFrame newdata;
    newdata.add_column("Sepal.Length", std::vector<double>{6.0, 7.0});
    newdata.add_column("Sepal.Width", std::vector<double>{3.0, 3.2});
    const auto preds = model.predict(newdata);
    EXPECT_NEAR(preds[0], 0.397043285851898, 1e-6);
    EXPECT_NEAR(preds[1], 0.827142151707129, 1e-6);

    const auto detail = model.predict(newdata, GLMPredictionInterval::Confidence);
    // Response-scale CI must stay inside the family's natural [0, 1] range.
    for (std::size_t i = 0; i < 2; ++i) {
        EXPECT_GE(detail.lower[i], 0.0);
        EXPECT_LE(detail.upper[i], 1.0);
        EXPECT_LE(detail.lower[i], detail.fit[i]);
        EXPECT_GE(detail.upper[i], detail.fit[i]);
    }
}

TEST(GLM, PoissonLogMatchesR) {
    auto iris = datamunge::datasets::iris();
    std::vector<double> count(iris.nrows());
    for (std::size_t i = 0; i < iris.nrows(); ++i) count[i] = std::nearbyint(iris.double_at("Sepal.Length", i));
    iris.add_column("count", count);

    GLMOptions options;
    options.family = GLMFamily::Poisson;
    GLM model(iris, "count ~ Sepal.Width + Petal.Length", options);

    ASSERT_EQ(model.coefficients().size(), 3u);
    EXPECT_NEAR(model.coefficients()[0], 1.2034409062539, 1e-5);
    EXPECT_NEAR(model.coefficients()[1], 0.0850675760671, 1e-5);
    EXPECT_NEAR(model.coefficients()[2], 0.0789034548015, 1e-5);
    EXPECT_NEAR(model.deviance(), 4.79227675016613, 1e-5);
    EXPECT_NEAR(model.null_deviance(), 19.0085151534416, 1e-5);
    EXPECT_NEAR(model.aic(), 554.422665609849, 1e-4);
    EXPECT_NEAR(model.dispersion(), 1.0, 1e-12);

    DataFrame newdata;
    newdata.add_column("Sepal.Width", std::vector<double>{3.0, 3.5});
    newdata.add_column("Petal.Length", std::vector<double>{4.0, 1.5});
    const auto preds = model.predict(newdata);
    EXPECT_NEAR(preds[0], 5.89590153039469, 1e-5);
    EXPECT_NEAR(preds[1], 5.05072907771298, 1e-5);
}

TEST(GLM, GammaInverseMatchesR) {
    GLMOptions options;
    options.family = GLMFamily::Gamma;
    GLM model(datamunge::datasets::iris(), "Petal.Length ~ Sepal.Length + Sepal.Width", options);

    ASSERT_EQ(model.coefficients().size(), 3u);
    EXPECT_NEAR(model.coefficients()[0], 0.5762350651021, 1e-5);
    EXPECT_NEAR(model.coefficients()[1], -0.1106524274803, 1e-5);
    EXPECT_NEAR(model.coefficients()[2], 0.1258214146574, 1e-5);
    EXPECT_NEAR(model.dispersion(), 0.105447456187738, 1e-6);
    EXPECT_NEAR(model.deviance(), 16.4993252731534, 1e-4);
    EXPECT_NEAR(model.null_deviance(), 44.6545916530366, 1e-4);
    EXPECT_NEAR(model.aic(), 457.844449935925, 1e-3);

    DataFrame newdata;
    newdata.add_column("Sepal.Length", std::vector<double>{5.0, 6.5});
    newdata.add_column("Sepal.Width", std::vector<double>{3.0, 3.0});
    const auto preds = model.predict(newdata);
    EXPECT_NEAR(preds[0], 2.49727066002181, 1e-5);
    EXPECT_NEAR(preds[1], 4.26514658294584, 1e-5);
}

TEST(GLM, GaussianMatchesRAndReducesToOls) {
    GLM model(datamunge::datasets::iris(), "Petal.Length ~ Sepal.Length + Sepal.Width"); // family defaults to Gaussian

    ASSERT_EQ(model.coefficients().size(), 3u);
    EXPECT_NEAR(model.coefficients()[0], -2.52476151183341, 1e-6);
    EXPECT_NEAR(model.coefficients()[1], 1.77559254648113, 1e-6);
    EXPECT_NEAR(model.coefficients()[2], -1.33862328873899, 1e-6);
    EXPECT_NEAR(model.deviance(), 61.4367468270757, 1e-4); // == RSS for gaussian
    EXPECT_NEAR(model.aic(), 299.787486645891, 1e-3);
    // Gaussian dispersion (== sigma^2) is estimated as RSS/df, not fixed at 1 like binomial/poisson.
    EXPECT_NEAR(model.dispersion(), model.deviance() / static_cast<double>(model.degrees_of_freedom()), 1e-9);
}

TEST(GLM, RejectsInvalidBinomialResponse) {
    DataFrame df;
    df.add_column("x", std::vector<double>{1.0, 2.0, 3.0, 4.0});
    df.add_column("y", std::vector<double>{0.0, 1.0, 2.0, 0.0}); // 2.0 is not a valid binomial response
    GLMOptions options;
    options.family = GLMFamily::Binomial;
    EXPECT_THROW(GLM(df, "y ~ x", options), std::invalid_argument);
}

TEST(GLM, RejectsNegativePoissonResponse) {
    DataFrame df;
    df.add_column("x", std::vector<double>{1.0, 2.0, 3.0, 4.0});
    df.add_column("y", std::vector<double>{0.0, 1.0, -2.0, 0.0});
    GLMOptions options;
    options.family = GLMFamily::Poisson;
    EXPECT_THROW(GLM(df, "y ~ x", options), std::invalid_argument);
}

TEST(GLM, RejectsNonPositiveGammaResponse) {
    DataFrame df;
    df.add_column("x", std::vector<double>{1.0, 2.0, 3.0, 4.0});
    df.add_column("y", std::vector<double>{1.0, 2.0, 0.0, 4.0});
    GLMOptions options;
    options.family = GLMFamily::Gamma;
    EXPECT_THROW(GLM(df, "y ~ x", options), std::invalid_argument);
}

TEST(GLM, RejectsZeroMaxIter) {
    GLMOptions options;
    options.max_iter = 0;
    EXPECT_THROW(GLM(datamunge::datasets::iris(), "Petal.Length ~ Sepal.Length", options), std::invalid_argument);
}

TEST(GLM, SummaryContainsKeyStatistics) {
    GLMOptions options;
    options.family = GLMFamily::Poisson;
    auto iris = datamunge::datasets::iris();
    std::vector<double> count(iris.nrows());
    for (std::size_t i = 0; i < iris.nrows(); ++i) count[i] = std::nearbyint(iris.double_at("Sepal.Length", i));
    iris.add_column("count", count);

    GLM model(iris, "count ~ Sepal.Width + Petal.Length", options);
    const auto text = model.summary();
    EXPECT_NE(text.find("poisson"), std::string::npos);
    EXPECT_NE(text.find("Null deviance"), std::string::npos);
    EXPECT_NE(text.find("AIC"), std::string::npos);
    EXPECT_NE(text.find("z value"), std::string::npos); // fixed dispersion -> z, not t
}
