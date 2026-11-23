#include <datamunge/stats/glm.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <gtest/gtest.h>

namespace {

using datamunge::dstruct::DataFrame;
using datamunge::stats::GLM;
using datamunge::stats::GLMBinomialLink;
using datamunge::stats::GLMFamily;
using datamunge::stats::GLMOptions;
using datamunge::stats::GLMPredictionInterval;

TEST(GroupedBinomialGlm, MatchesRAndExpandedBernoulliFitWithCaseWeights) {
    const std::vector<double> x{-2, -1, -.5, 0, 1, 2, 3};
    const std::vector<double> successes{0, 3, 7, 0, 12, 15, 5};
    const std::vector<double> trials{8, 11, 12, 0, 19, 18, 5};
    const std::vector<double> weights{1, .5, 1.4, 1, 1.25, .7, 1.5};
    DataFrame grouped, expanded, query;
    grouped.add_column("x", x);
    grouped.add_column("wins", successes);
    grouped.add_column("trials", trials);
    grouped.add_column("w", weights);
    query.add_column("x", std::vector<double>{.75});
    std::vector<double> trial_x, trial_y, trial_weight;
    for (std::size_t i = 0; i < x.size(); ++i)
        for (int j = 0; j < trials[i]; ++j) {
            trial_x.push_back(x[i]);
            trial_y.push_back(j < successes[i] ? 1 : 0);
            trial_weight.push_back(weights[i]);
        }
    expanded.add_column("x", trial_x);
    expanded.add_column("wins", trial_y);
    expanded.add_column("w", trial_weight);
    const std::array links{GLMBinomialLink::Logit, GLMBinomialLink::Probit, GLMBinomialLink::Cloglog};
    const std::array r_deviance{6.740138919883041, 6.362392295111444, 6.836680278738823};
    const std::array r_aic{22.371210036154963, 21.993463411383367, 22.467751395010740};
    for (std::size_t l = 0; l < links.size(); ++l) {
        GLMOptions options{.family = GLMFamily::Binomial, .weights_column = "w", .max_iter = 100,
                           .tol = 1e-13, .trials_column = "trials", .binomial_link = links[l]};
        const GLM grouped_fit(grouped, "wins ~ x", options);
        options.trials_column.reset();
        const GLM expanded_fit(expanded, "wins ~ x", options);
        EXPECT_TRUE(grouped_fit.converged());
        EXPECT_TRUE(expanded_fit.converged());
        EXPECT_EQ(grouped_fit.kept_rows(), (std::vector<std::size_t>{0, 1, 2, 4, 5, 6}));
        EXPECT_EQ(grouped_fit.degrees_of_freedom(), 4u);
        EXPECT_NEAR(grouped_fit.coefficients()[0], expanded_fit.coefficients()[0], 4e-7);
        EXPECT_NEAR(grouped_fit.coefficients()[1], expanded_fit.coefficients()[1], 4e-7);
        EXPECT_NEAR(grouped_fit.covariance()(0, 1), expanded_fit.covariance()(0, 1), 5e-8);
        EXPECT_NEAR(grouped_fit.covariance()(1, 1), expanded_fit.covariance()(1, 1), 5e-8);
        EXPECT_NEAR(grouped_fit.predict(query)[0], expanded_fit.predict(query)[0], 4e-7);
        EXPECT_NEAR(grouped_fit.deviance(), r_deviance[l], 1e-11);
        EXPECT_NEAR(grouped_fit.aic(), r_aic[l], 1e-11);
        EXPECT_NEAR(grouped_fit.working_data().double_at("response", 1), 3.0 / 11, 1e-14);
    }
}

TEST(GroupedBinomialGlm, ComputesAicForHugeCountsAndAllowsSaturatedGroups) {
    DataFrame data;
    data.add_column("s", std::vector<double>{.25e18, .75e18});
    data.add_column("n", std::vector<double>{1e18, 1e18});
    data.add_column("x", std::vector<double>{0, 1});
    const GLM fit(data, "s ~ x", {.family = GLMFamily::Binomial, .trials_column = "n"});
    EXPECT_TRUE(fit.converged());
    EXPECT_EQ(fit.degrees_of_freedom(), 0u);
    EXPECT_NEAR(fit.coefficients()[1], 2 * std::log(3.0), 1e-12);
    EXPECT_NEAR(fit.aic(), 87.22086461346112, 1e-10);
}

TEST(GroupedBinomialGlm, RemovesZeroTrialRowsBeforeResolvingFactorLevels) {
    DataFrame data;
    data.add_column("s", std::vector<double>{1, 0, 2, 1});
    data.add_column("n", std::vector<double>{4, 0, 4, 2});
    data.add_column("treatment", std::vector<std::string>{"A", "unused", "B", "B"});
    const GLM fit(data, "s ~ treatment", {.family = GLMFamily::Binomial, .trials_column = "n"});
    EXPECT_EQ(fit.kept_rows(), (std::vector<std::size_t>{0, 2, 3}));
    EXPECT_EQ(fit.coefficient_names(), (std::vector<std::string>{"(Intercept)", "treatmentB"}));
    EXPECT_NEAR(fit.fitted_values()[0], .25, 1e-12);
    EXPECT_NEAR(fit.fitted_values()[1], .5, 1e-12);
}

TEST(GroupedBinomialGlm, RejectsInvalidCountsAndLinks) {
    const GLMOptions grouped{.family = GLMFamily::Binomial, .trials_column = "n"};
    for (const double invalid : {-1.0, 1.5, 5.0, std::numeric_limits<double>::infinity()}) {
        DataFrame data;
        data.add_column("s", std::vector<double>{1, 2, invalid});
        data.add_column("n", std::vector<double>{4, 4, 4});
        EXPECT_THROW((void)GLM(data, "s ~ 1", grouped), std::invalid_argument);
    }
    DataFrame data;
    data.add_column("s", std::vector<double>{0, 0, 1});
    data.add_column("n", std::vector<double>{4, 4, 0});
    EXPECT_THROW((void)GLM(data, "s ~ 1", grouped), std::invalid_argument);
    EXPECT_THROW((void)GLM(data, "s ~ 1", {.binomial_link = GLMBinomialLink::Probit}), std::invalid_argument);
    EXPECT_THROW((void)GLM(data, "s ~ 1", {.family = GLMFamily::Binomial,
        .binomial_link = static_cast<GLMBinomialLink>(99)}), std::invalid_argument);
}

TEST(AlternateBinomialLink, BinaryFitsAndMeanIntervalsMatchR) {
    std::vector<double> x, y;
    const std::array predictor{-2.0, -1.2, -.4, .3, .9, 1.7};
    const std::array successes{1, 1, 2, 3, 4, 5};
    for (std::size_t j = 0; j < predictor.size(); ++j)
        for (int i = 0; i < 5; ++i) {
            x.push_back(predictor[j]);
            y.push_back(i >= 5 - successes[j] ? 1 : 0);
        }
    DataFrame data, query, tails;
    data.add_column("x", x);
    data.add_column("y", y);
    query.add_column("x", std::vector<double>{.65});
    tails.add_column("x", std::vector<double>{-1e4, 1e4});
    struct Reference {
        GLMBinomialLink link;
        const char* name;
        double intercept, slope, covariance, deviance, mean, se, lower, upper;
    };
    const std::array reference{
        Reference{GLMBinomialLink::Probit, "probit", .215412078779682, .730217099050852,
                  .010429509023498, 29.884690668680001, .754919631633046,
                  .103816025726631, .517112304787158, .909421161950189},
        Reference{GLMBinomialLink::Cloglog, "cloglog", -.246272648396595, .894029968620917,
                  -.025181915647075, 29.097707174482704, .752841705889739,
                  .106733814467632, .533659827647851, .922773102852572},
    };
    for (const auto& r : reference) {
        const GLM fit(data, "y ~ x", {.family = GLMFamily::Binomial, .max_iter = 100,
                                      .tol = 1e-13, .binomial_link = r.link});
        EXPECT_TRUE(fit.converged());
        EXPECT_FALSE(fit.boundary());
        EXPECT_EQ(fit.binomial_link(), r.link);
        EXPECT_NE(fit.summary().find(r.name), std::string::npos);
        EXPECT_NEAR(fit.coefficients()[0], r.intercept, 4e-7);
        EXPECT_NEAR(fit.coefficients()[1], r.slope, 4e-7);
        EXPECT_NEAR(fit.covariance()(0, 1), r.covariance, 1e-7);
        EXPECT_NEAR(fit.deviance(), r.deviance, 1e-11);
        const auto prediction = fit.predict(query, GLMPredictionInterval::Confidence);
        EXPECT_NEAR(prediction.fit[0], r.mean, 3e-7);
        EXPECT_NEAR(prediction.se_fit[0], r.se, 3e-7);
        EXPECT_NEAR(prediction.lower[0], r.lower, 4e-7);
        EXPECT_NEAR(prediction.upper[0], r.upper, 4e-7);
        const auto extreme = fit.predict(tails);
        EXPECT_GT(extreme[0], 0);
        EXPECT_LT(extreme[0], 1e-12);
        EXPECT_GT(extreme[1], 1 - 1e-12);
        EXPECT_LT(extreme[1], 1);
    }
}

} // namespace
