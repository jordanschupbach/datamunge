#include <gtest/gtest.h>

#include <datamunge/dstruct/dstruct.hpp>
#include <datamunge/stats/quantile_regression.hpp>

#include <stdexcept>
#include <vector>

using datamunge::dstruct::DataFrame;
using datamunge::stats::QuantileRegression;
using datamunge::stats::QuantileRegressionOptions;

namespace {

DataFrame exact_line_frame() {
    DataFrame data;
    data.add_column("x", std::vector<double>{0, 1, 2, 3, 4, 5, 6, 7});
    data.add_column("y", std::vector<double>{1, 3, 5, 7, 9, 11, 13, 15});
    return data;
}

} // namespace

TEST(QuantileRegression, RecoversAnExactLineAtSeveralQuantiles) {
    const auto data = exact_line_frame();
    for (const double tau : {0.1, 0.5, 0.9}) {
        QuantileRegressionOptions options;
        options.quantile = tau;
        QuantileRegression model(data, "y ~ x", options);
        ASSERT_TRUE(model.converged());
        ASSERT_EQ(model.coefficients().size(), 2u);
        EXPECT_NEAR(model.coefficients()[0], 1.0, 1e-5);
        EXPECT_NEAR(model.coefficients()[1], 2.0, 1e-5);
        EXPECT_NEAR(model.objective(), 0.0, 1e-5);
        EXPECT_NEAR(model.pseudo_r_squared(), 1.0, 1e-5);
    }
}

TEST(QuantileRegression, InterceptOnlyFitEqualsTheSampleMedian) {
    DataFrame data;
    data.add_column("y", std::vector<double>{1, 2, 4, 8, 16});
    QuantileRegression model(data, "y ~ 1");
    ASSERT_TRUE(model.converged());
    ASSERT_EQ(model.coefficients().size(), 1u);
    EXPECT_NEAR(model.coefficients()[0], 4.0, 1e-5);
}

TEST(QuantileRegression, DifferentQuantilesCaptureHeterogeneousSlopes) {
    DataFrame data;
    std::vector<double> x;
    std::vector<double> y;
    for (int i = 1; i <= 30; ++i) {
        for (const double scale : {-1.0, 0.0, 1.0}) {
            x.push_back(static_cast<double>(i));
            y.push_back(2.0 + 3.0 * i + scale * i);
        }
    }
    data.add_column("x", x);
    data.add_column("y", y);

    QuantileRegressionOptions low_options;
    low_options.quantile = 0.2;
    QuantileRegressionOptions high_options;
    high_options.quantile = 0.8;
    QuantileRegression low(data, "y ~ x", low_options);
    QuantileRegression high(data, "y ~ x", high_options);

    ASSERT_TRUE(low.converged());
    ASSERT_TRUE(high.converged());
    EXPECT_LT(low.coefficients()[1], high.coefficients()[1]);
    EXPECT_NEAR(low.coefficients()[1], 2.0, 2e-2);
    EXPECT_NEAR(high.coefficients()[1], 4.0, 2e-2);
}

TEST(QuantileRegression, PredictUsesTheResolvedFormula) {
    const auto data = exact_line_frame();
    QuantileRegression model(data, "y ~ x");
    DataFrame newdata;
    newdata.add_column("x", std::vector<double>{8, 10});
    const auto predictions = model.predict(newdata);
    ASSERT_EQ(predictions.size(), 2u);
    EXPECT_NEAR(predictions[0], 17.0, 1e-5);
    EXPECT_NEAR(predictions[1], 21.0, 1e-5);
}

TEST(QuantileRegression, ObservationWeightsChangeTheFittedQuantile) {
    DataFrame data;
    data.add_column("y", std::vector<double>{0, 10, 20});
    data.add_column("w", std::vector<double>{1, 1, 10});

    QuantileRegression unweighted(data, "y ~ 1");
    QuantileRegressionOptions options;
    options.weights_column = "w";
    QuantileRegression weighted(data, "y ~ 1", options);

    ASSERT_TRUE(unweighted.converged());
    ASSERT_TRUE(weighted.converged());
    EXPECT_NEAR(unweighted.coefficients()[0], 10.0, 1e-3);
    EXPECT_NEAR(weighted.coefficients()[0], 20.0, 1e-3);
}

TEST(QuantileRegression, ReportsWhenIterationLimitIsReached) {
    DataFrame data;
    data.add_column("x", std::vector<double>{0, 1, 2, 3, 4});
    data.add_column("y", std::vector<double>{0, 1, 4, 9, 16});
    QuantileRegressionOptions options;
    options.max_iterations = 1;
    options.adaptive_rho = false;
    QuantileRegression model(data, "y ~ x", options);

    EXPECT_FALSE(model.converged());
    EXPECT_EQ(model.iterations(), 1u);
}

TEST(QuantileRegression, RejectsInvalidOptionsAndRankDeficiency) {
    const auto data = exact_line_frame();
    QuantileRegressionOptions invalid;
    invalid.quantile = 1.0;
    EXPECT_THROW(QuantileRegression(data, "y ~ x", invalid), std::invalid_argument);

    DataFrame collinear;
    collinear.add_column("x", std::vector<double>{1, 2, 3, 4});
    collinear.add_column("z", std::vector<double>{2, 4, 6, 8});
    collinear.add_column("y", std::vector<double>{1, 2, 3, 4});
    EXPECT_THROW(QuantileRegression(collinear, "y ~ x + z"), std::runtime_error);
}
