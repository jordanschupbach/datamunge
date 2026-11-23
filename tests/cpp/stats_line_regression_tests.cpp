#include <datamunge/stats/line_regression.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

using namespace datamunge::stats;

TEST(LineRegression, DistinctPredictorPairingsMatchRWithTiesAndEvenMedians) {
    std::vector<double> x{0, 0, 1, 2, 3, 4, 5, 5};
    std::vector<double> y{7, 10, 9, 11, 19, 6, 1, 15};
    const auto theil = theil_sen_regression(x, y);
    const auto repeated = repeated_median_regression(x, y);
    EXPECT_DOUBLE_EQ(theil.slope, 0.125);
    EXPECT_DOUBLE_EQ(theil.intercept, 9.4375);
    EXPECT_DOUBLE_EQ(repeated.slope, 1.375);
    EXPECT_DOUBLE_EQ(repeated.intercept, 7.875);
    std::reverse(x.begin(), x.end());
    std::reverse(y.begin(), y.end());
    EXPECT_DOUBLE_EQ(theil_sen_regression(x, y).slope, theil.slope);
    EXPECT_DOUBLE_EQ(repeated_median_regression(x, y).slope, repeated.slope);
}

TEST(LineRegression, RobustFitsHandleOutliersFlatResponsesAndExtremeScales) {
    const std::vector<double> x{0, 1, 2, 3, 4, 5};
    const std::vector<double> y{1, 3, 5, 7, 9, 1000};
    EXPECT_DOUBLE_EQ(theil_sen_regression(x, y).intercept, 1);
    EXPECT_DOUBLE_EQ(repeated_median_regression(x, y).slope, 2);
    EXPECT_DOUBLE_EQ(repeated_median_regression({0, 1, 2}, {3, 3, 3}).slope, 0);
    constexpr double huge = std::numeric_limits<double>::max();
    EXPECT_DOUBLE_EQ(theil_sen_regression({-huge, huge}, {huge / 2, -huge / 2}).slope, -0.5);
    EXPECT_DOUBLE_EQ(repeated_median_regression({-huge, 0, huge}, {huge / 2, 0, -huge / 2}).slope, -0.5);
}

TEST(LineRegression, DemingAgreesWithAWhitenedREigenanalysis) {
    const std::vector<double> x{0, 1, 2, 3, 4, 7, 9};
    const std::vector<double> y{3, 2, 4, 5, 7, 6, 12};
    const auto first = deming_regression(x, y, 0.25);
    ASSERT_TRUE(first.coefficients);
    EXPECT_NEAR(first.coefficients->intercept, 1.5325141147366423, 1e-14);
    EXPECT_NEAR(first.coefficients->slope, 1.0874000460324422, 1e-14);
    EXPECT_NEAR(first.direction_x, 0.6769069248207753, 1e-14);
    EXPECT_NEAR(first.direction_y, 0.7360686212097900, 1e-14);
    const auto second = deming_regression(x, y, 4);
    ASSERT_TRUE(second.coefficients);
    EXPECT_NEAR(second.coefficients->intercept, 2.0242398928597343, 1e-14);
    EXPECT_NEAR(second.coefficients->slope, 0.9550123365377637, 1e-14);
    const auto swapped = deming_regression(y, x, 0.25);
    ASSERT_TRUE(swapped.coefficients);
    EXPECT_NEAR(swapped.coefficients->slope * second.coefficients->slope, 1, 1e-14);
    EXPECT_NEAR(swapped.coefficients->intercept * second.coefficients->slope + second.coefficients->intercept,
                0, 1e-14);
}

TEST(LineRegression, DemingReturnsVerticalAndHorizontalLinesAndRejectsTiedAxes) {
    const auto vertical = deming_regression({2, 2, 2}, {1, 2, 3});
    EXPECT_FALSE(vertical.coefficients);
    EXPECT_DOUBLE_EQ(vertical.center_x, 2);
    EXPECT_DOUBLE_EQ(vertical.direction_x, 0);
    EXPECT_DOUBLE_EQ(vertical.direction_y, 1);
    const auto horizontal = deming_regression({1, 2, 3}, {7, 7, 7});
    ASSERT_TRUE(horizontal.coefficients);
    EXPECT_DOUBLE_EQ(horizontal.coefficients->intercept, 7);
    EXPECT_DOUBLE_EQ(horizontal.coefficients->slope, 0);
    const auto uncorrelated = deming_regression({-1, 1, 0, 0}, {0, 0, -2, 2});
    EXPECT_FALSE(uncorrelated.coefficients);
    EXPECT_DOUBLE_EQ(uncorrelated.direction_y, 1);
    EXPECT_THROW(deming_regression({-1, 1, 0, 0}, {0, 0, -1, 1}), std::domain_error);
    EXPECT_THROW(deming_regression({2, 2}, {3, 3}), std::domain_error);
}

TEST(LineRegression, DemingPreservesThePointDirectionFormForVerySteepLines) {
    const auto fit = deming_regression({0, 1e-150, 2e-150}, {0, -1e160, -2e160});
    EXPECT_FALSE(fit.coefficients);
    EXPECT_NEAR(fit.center_x, 1e-150, 1e-160);
    EXPECT_DOUBLE_EQ(fit.center_y, -1e160);
    EXPECT_LT(fit.direction_x, 1e-308);
    EXPECT_DOUBLE_EQ(fit.direction_y, -1);
    constexpr double huge = 1e300;
    const auto diagonal = deming_regression({-huge, 0, huge}, {huge / 2, 0, -huge / 2}, 4);
    ASSERT_TRUE(diagonal.coefficients);
    EXPECT_DOUBLE_EQ(diagonal.coefficients->slope, -0.5);
    EXPECT_NEAR(std::hypot(diagonal.direction_x, diagonal.direction_y), 1, 1e-15);
}

TEST(LineRegression, RejectsInvalidPairsAndUnrepresentableMedianCoefficients) {
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(theil_sen_regression({}, {}), std::invalid_argument);
    EXPECT_THROW(theil_sen_regression({2, 2}, {0, 1}), std::invalid_argument);
    EXPECT_THROW(repeated_median_regression({2, 2}, {0, 1}), std::invalid_argument);
    EXPECT_THROW(repeated_median_regression({0, 1}, {nan, 1}), std::invalid_argument);
    EXPECT_THROW(deming_regression({1, 2}, {1}), std::invalid_argument);
    EXPECT_THROW(deming_regression({1, 2}, {1, 2}, 0), std::invalid_argument);
    EXPECT_THROW(deming_regression({1, 2}, {1, 2}, nan), std::invalid_argument);
    EXPECT_THROW(theil_sen_regression({0, 1e-300}, {0, 1e300}), std::overflow_error);
    EXPECT_THROW(repeated_median_regression({0, 1e-300}, {0, 1e300}), std::overflow_error);
}
