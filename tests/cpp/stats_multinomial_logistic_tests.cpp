#include <datamunge/stats/multinomial_logistic.hpp>

#include <cmath>
#include <stdexcept>
#include <gtest/gtest.h>

namespace {

using datamunge::linalg::DenseMatrix;
using datamunge::stats::multinomial_logistic;
using datamunge::stats::MultinomialLogisticStatus;

TEST(MultinomialLogistic, MatchesRWeightedLogOddsCrossClassCovarianceAndPrediction) {
    const DenseMatrix<double> design(8, 3, {1, 0, 0, 1, .2, .6, 1, .3, .2, 1, .45, .9,
                                           1, .6, .55, 1, .8, 1, 1, .9, .15, 1, 1, .7});
    const DenseMatrix<double> counts(8, 3, {8,2,1, 7,5,2, 6,4,3, 4,5,2, 4,6,6, 3,8,5, 2,5,9, 2,6,11});
    const auto fit = multinomial_logistic(design, counts, {.weights = {1,2,.6,1,1.4,.75,1.3,1.2}});
    ASSERT_EQ(fit.status, MultinomialLogisticStatus::Converged);
    EXPECT_EQ(fit.classes, (std::vector<std::int64_t>{0, 1, 2}));
    EXPECT_EQ(fit.baseline, 0);
    // R nnet::multinom() with multinomial counts and a full Hessian.
    EXPECT_NEAR(fit.coefficients(0, 0), -1.160540300573088, 2e-7);
    EXPECT_NEAR(fit.coefficients(0, 1), 2.042258106431802, 2e-7);
    EXPECT_NEAR(fit.coefficients(1, 1), 3.842266846526659, 2e-7);
    EXPECT_NEAR(fit.coefficients(1, 2), -.380971674268742, 2e-7);
    ASSERT_EQ(fit.covariance.rows(), 6u);
    EXPECT_NEAR(fit.covariance(0, 0), .271773656038475, 3e-8);
    EXPECT_NEAR(fit.covariance(1, 4), .349918930701083, 3e-8);
    EXPECT_NEAR(fit.covariance(2, 5), .404735002360137, 3e-8);
    EXPECT_DOUBLE_EQ(fit.covariance(4, 1), fit.covariance(1, 4));
    EXPECT_NEAR(fit.log_likelihood, -133.36204759652838, 1e-11);
    const auto p = fit.predict(DenseMatrix<double>(3, 3, {1,.55,.4, 1,-1e6,.4, 1,1e6,.4}));
    EXPECT_NEAR(p(0, 0), .302745443729088, 2e-8);
    EXPECT_NEAR(p(0, 1), .365782288522896, 2e-8);
    EXPECT_NEAR(p(0, 2), .331472267748016, 2e-8);
    EXPECT_DOUBLE_EQ(p(1, 0), 1);
    EXPECT_DOUBLE_EQ(p(2, 2), 1);
}

TEST(MultinomialLogistic, ExpandsCountsAndTransformsInferenceWhenChangingBaseline) {
    const DenseMatrix<double> x(4, 2, {1,0, 1,1, 1,2, 1,100});
    const DenseMatrix<double> y(4, 3, {6,3,2, 3,5,4, 2,4,9, 0,0,0});
    const auto original = multinomial_logistic(x, y);
    const auto recoded = multinomial_logistic(x, y, {.baseline = 1});
    ASSERT_EQ(original.status, MultinomialLogisticStatus::Converged);
    ASSERT_EQ(recoded.status, MultinomialLogisticStatus::Converged);
    EXPECT_EQ(recoded.baseline, 1);
    EXPECT_NEAR(recoded.log_likelihood, original.log_likelihood, 1e-12);
    for (std::size_t j = 0; j < x.cols(); ++j) {
        EXPECT_NEAR(recoded.coefficients(0, j), -original.coefficients(0, j), 1e-8);
        EXPECT_NEAR(recoded.coefficients(1, j), original.coefficients(1, j) - original.coefficients(0, j), 1e-8);
    }
    EXPECT_NEAR(recoded.covariance(1, 3), original.covariance(1, 1) - original.covariance(1, 3), 1e-8);
    for (std::size_t i = 0; i < x.rows(); ++i)
        for (std::size_t c = 0; c < y.cols(); ++c)
            EXPECT_NEAR(recoded.fitted_probabilities(i, c), original.fitted_probabilities(i, c), 1e-8);

    std::vector<std::int64_t> labels;
    std::vector<double> rows;
    const std::vector<std::int64_t> names{-4, 3, 99};
    for (std::size_t i = 0; i < x.rows(); ++i)
        for (std::size_t c = 0; c < y.cols(); ++c)
            for (std::size_t k = 0; k < y(i, c); ++k) {
                rows.insert(rows.end(), {x(i, 0), x(i, 1)});
                labels.push_back(names[c]);
            }
    const auto individual = multinomial_logistic(DenseMatrix<double>(labels.size(), 2, rows), labels, {.baseline = 3});
    ASSERT_EQ(individual.status, MultinomialLogisticStatus::Converged);
    EXPECT_EQ(individual.classes, names);
    EXPECT_NEAR(individual.log_likelihood, recoded.log_likelihood, 1e-12);
    EXPECT_NEAR(individual.covariance(0, 3), recoded.covariance(0, 3), 1e-9);
}

TEST(MultinomialLogistic, BinaryInterceptHasKnownOddsAndInformation) {
    const auto fit = multinomial_logistic(DenseMatrix<double>(1, 1, 1),
                                           DenseMatrix<double>(1, 2, {3, 7}));
    ASSERT_EQ(fit.status, MultinomialLogisticStatus::Converged);
    EXPECT_NEAR(fit.coefficients(0, 0), std::log(7.0 / 3), 1e-9);
    EXPECT_NEAR(fit.covariance(0, 0), 10.0 / 21, 1e-9);
    EXPECT_NEAR(fit.fitted_probabilities(0, 1), .7, 1e-10);
}

TEST(MultinomialLogistic, ReportsNonconvergenceAndRejectsInvalidOrAliasedData) {
    const DenseMatrix<double> design(6, 2, {1,-1, 1,-1, 1,0, 1,0, 1,1, 1,1});
    const std::vector<std::int64_t> separated{1,1,0,0,2,2};
    const auto fit = multinomial_logistic(design, separated);
    EXPECT_EQ(fit.status, MultinomialLogisticStatus::Separated);
    EXPECT_EQ(fit.covariance.rows(), 0u);
    const auto limited = multinomial_logistic(DenseMatrix<double>(1, 1, 1), DenseMatrix<double>(1, 3, {3,4,8}),
                                               {.max_iterations = 1});
    EXPECT_EQ(limited.status, MultinomialLogisticStatus::IterationLimit);
    EXPECT_EQ(limited.iterations, 1u);
    EXPECT_EQ(limited.covariance.rows(), 0u);
    EXPECT_THROW((void)multinomial_logistic(design, separated, {.baseline = 99}), std::invalid_argument);
    EXPECT_THROW((void)multinomial_logistic(DenseMatrix<double>(6, 2, 1), separated), std::invalid_argument);
    EXPECT_THROW((void)multinomial_logistic(DenseMatrix<double>(1, 1, 1), DenseMatrix<double>(1, 3, {1,0,2})),
                 std::invalid_argument);
    EXPECT_THROW((void)multinomial_logistic(DenseMatrix<double>(1, 1, 1), DenseMatrix<double>(1, 2, {1,1.5})),
                 std::invalid_argument);
}

} // namespace
