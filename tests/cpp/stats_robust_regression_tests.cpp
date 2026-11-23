#include <datamunge/stats/robust_regression.hpp>
#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>

using datamunge::linalg::DenseMatrix;
using namespace datamunge::stats;

TEST(RobustRegression, HuberAndBisquareMatchIndependentWeightedQrIterations) {
    const DenseMatrix<double> design(10, 2, {1, 0, 1, 1, 1, 2, 1, 3, 1, 4, 1, 5, 1, 6, 1, 7, 1, 8, 1, 11});
    const std::vector<double> y{2.1, 3.3, 5.5, 6.2, 8.4, 9.3, 11.5, 12.4, 13.6, 38};
    const auto huber = robust_regression(design, y, {.tolerance = 1e-11});
    const auto tukey = robust_regression(design, y, {.loss = RobustRegressionLoss::TukeyBisquare, .tolerance = 1e-11});
    ASSERT_EQ(huber.status, RobustRegressionStatus::Converged);
    EXPECT_NEAR(huber.coefficients[0], 1.874943315739015, 1e-10);
    EXPECT_NEAR(huber.coefficients[1], 1.560730481217538, 1e-10);
    EXPECT_NEAR(huber.scale, 0.5453978218382500, 1e-10);
    EXPECT_NEAR(huber.weights.back(), 0.03869595625005811, 1e-10);
    ASSERT_EQ(tukey.status, RobustRegressionStatus::Converged);
    EXPECT_NEAR(tukey.coefficients[0], 2.137205092022654, 1e-10);
    EXPECT_NEAR(tukey.coefficients[1], 1.472466175931749, 1e-10);
    EXPECT_NEAR(tukey.scale, 0.4978119527385826, 1e-10);
    EXPECT_DOUBLE_EQ(tukey.weights.back(), 0);
    EXPECT_NEAR(tukey.residuals.back(), 38 - tukey.coefficients[0] - 11 * tukey.coefficients[1], 1e-13);
    EXPECT_LE(tukey.relative_change, 1e-11);
}

TEST(RobustRegression, FixedScaleHuberSolvesTheClippedLocationEquation) {
    const DenseMatrix<double> design(4, 1, 1.0);
    const auto fit = robust_regression(design, {0, 0, 0, 9}, {.tuning_constant = 1, .fixed_scale = 1});
    ASSERT_EQ(fit.status, RobustRegressionStatus::Converged);
    EXPECT_NEAR(fit.coefficients.front(), 1.0 / 3, 1e-8);
    EXPECT_DOUBLE_EQ(fit.scale, 1);
    EXPECT_NEAR(fit.weights.back(), 3.0 / 26, 1e-9);
    EXPECT_DOUBLE_EQ(fit.weights.front(), 1);
}

TEST(RobustRegression, ReportsIterationLimitZeroScaleAndReweightingRankLoss) {
    const DenseMatrix<double> design(4, 1, 1.0);
    const auto limited = robust_regression(design, {0, 0, 0, 9}, {.fixed_scale = 1, .max_iterations = 1});
    EXPECT_EQ(limited.status, RobustRegressionStatus::IterationLimit);
    EXPECT_EQ(limited.iterations, 1U);
    const auto exact = robust_regression(design, {2, 2, 2, 2});
    EXPECT_EQ(exact.status, RobustRegressionStatus::ZeroScale);
    EXPECT_DOUBLE_EQ(exact.scale, 0);
    EXPECT_EQ(exact.weights, (std::vector<double>{1, 1, 1, 1}));
    const auto rejected = robust_regression(design, {0, 0, 0, 9},
        {.loss = RobustRegressionLoss::TukeyBisquare, .fixed_scale = 1, .initial_coefficients = {100}});
    EXPECT_EQ(rejected.status, RobustRegressionStatus::RankDeficient);
    EXPECT_EQ(rejected.iterations, 0U);
    EXPECT_DOUBLE_EQ(rejected.coefficients.front(), 100);
    EXPECT_EQ(rejected.weights, (std::vector<double>{0, 0, 0, 0}));
}

TEST(RobustRegression, ScalesPredictorUnitsAndChecksInvalidInputs) {
    const DenseMatrix<double> design(6, 2, {1, 0, 1, 1, 1, 2, 1, 3, 1, 4, 1, 5});
    const std::vector<double> y{1.2, 2.9, 5.2, 6.8, 9, 24};
    auto scaled = design;
    for (std::size_t i = 0; i < scaled.rows(); ++i)
        scaled(i, 1) *= 1e-150;
    const auto a = robust_regression(design, y, {.fixed_scale = 0.5});
    const auto b = robust_regression(scaled, y, {.fixed_scale = 0.5});
    EXPECT_DOUBLE_EQ(a.coefficients[0], b.coefficients[0]);
    EXPECT_NEAR(a.coefficients[1], b.coefficients[1] * 1e-150, 1e-14);
    EXPECT_THROW(robust_regression(design, {1, 2}), std::invalid_argument);
    EXPECT_THROW(robust_regression(design, y, {.fixed_scale = 0}), std::invalid_argument);
    EXPECT_THROW(robust_regression(design, y, {.initial_coefficients = {1}}), std::invalid_argument);
    EXPECT_THROW(robust_regression(DenseMatrix<double>(6, 2, 1), y), std::invalid_argument);
    scaled(0, 0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(robust_regression(scaled, y), std::invalid_argument);
}
