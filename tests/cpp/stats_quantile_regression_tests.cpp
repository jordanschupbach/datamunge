#include <gtest/gtest.h>

#include <datamunge/dstruct/dstruct.hpp>
#include <datamunge/stats/quantile_regression.hpp>

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

using datamunge::dstruct::DataFrame;
using datamunge::stats::QuantileRegression;
using datamunge::stats::QuantileRegressionOptions;
using datamunge::stats::QuantileRegressionMethod;
using datamunge::stats::QuantileStandardErrors;

namespace {

DataFrame exact_line_frame() {
    DataFrame data;
    data.add_column("x", std::vector<double>{0, 1, 2, 3, 4, 5, 6, 7});
    data.add_column("y", std::vector<double>{1, 3, 5, 7, 9, 11, 13, 15});
    return data;
}

DataFrame fan_frame(std::size_t n, unsigned seed) {
    std::mt19937_64                  gen(seed);
    std::normal_distribution<double> z;
    std::uniform_real_distribution<double> u(0.0, 10.0);
    std::vector<double> x(n), y(n);
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = u(gen);
        y[i] = 1.0 + 2.0 * x[i] + (1.0 + 0.5 * x[i]) * z(gen);
    }
    DataFrame data;
    data.add_column("x", x);
    data.add_column("y", y);
    return data;
}

QuantileRegressionOptions simplex(double tau) {
    QuantileRegressionOptions options;
    options.quantile = tau;
    options.method   = QuantileRegressionMethod::Simplex;
    return options;
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
    options.method = QuantileRegressionMethod::ADMM;
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

TEST(QuantileRegression, SimplexInterpolatesExactlyPRowsOfAnExactLine) {
    const auto data = exact_line_frame();
    const QuantileRegression model(data, "y ~ x", simplex(0.3));
    ASSERT_TRUE(model.converged());
    EXPECT_EQ(model.basis().size(), 2u);
    EXPECT_NEAR(model.coefficients()[0], 1.0, 1e-12);
    EXPECT_NEAR(model.coefficients()[1], 2.0, 1e-12);
    EXPECT_NEAR(model.objective(), 0.0, 1e-12);
}

TEST(QuantileRegression, SimplexInterceptOnlyFitIsTheSampleQuantile) {
    DataFrame data;
    data.add_column("y", std::vector<double>{16, 1, 8, 2, 4});
    EXPECT_DOUBLE_EQ(QuantileRegression(data, "y ~ 1", simplex(0.3)).coefficients()[0], 2.0);
    EXPECT_DOUBLE_EQ(QuantileRegression(data, "y ~ 1", simplex(0.5)).coefficients()[0], 4.0);
    EXPECT_DOUBLE_EQ(QuantileRegression(data, "y ~ 1", simplex(0.9)).coefficients()[0], 16.0);
}

TEST(QuantileRegression, SimplexSolvesTheFiveRowHandExample) {
    DataFrame data;
    data.add_column("x", std::vector<double>{1, 2, 3, 4, 5});
    data.add_column("y", std::vector<double>{1, 3, 2, 5, 4});
    const QuantileRegression model(data, "y ~ x", simplex(0.5));
    ASSERT_TRUE(model.converged());
    EXPECT_DOUBLE_EQ(model.coefficients()[0], 0.25);
    EXPECT_DOUBLE_EQ(model.coefficients()[1], 0.75);
    EXPECT_NEAR(model.objective(), 1.75, 1e-12);
    EXPECT_EQ(model.basis(), (std::vector<std::size_t>{0, 4}));
    const std::vector<double> dual{-0.25, 0.5, -0.5, 0.5, -0.25};
    for (std::size_t i = 0; i < 5; ++i) EXPECT_NEAR(model.dual_solution()[i], dual[i], 1e-12);
}

TEST(QuantileRegression, SimplexCertifiesDegenerateVerticesWithTiedRows) {
    // Every row duplicated: the fit passes through 2p rows, and the dual needs the bounded solve.
    DataFrame data;
    data.add_column("x", std::vector<double>{1, 2, 3, 4, 5, 1, 2, 3, 4, 5});
    data.add_column("y", std::vector<double>{1, 3, 2, 5, 4, 1, 3, 2, 5, 4});
    const QuantileRegression model(data, "y ~ x", simplex(0.5));
    ASSERT_TRUE(model.converged());
    EXPECT_NEAR(model.objective(), 3.5, 1e-12);
    double s0 = 0.0;
    for (const double a : model.dual_solution()) s0 += a;
    EXPECT_NEAR(s0, 0.0, 1e-9);
}

TEST(QuantileRegression, SimplexAndAdmmReachTheSameOptimum) {
    const auto data = fan_frame(300, 7);
    for (const double tau : {0.1, 0.5, 0.9}) {
        const QuantileRegression exact(data, "y ~ x", simplex(tau));
        QuantileRegressionOptions admm;
        admm.quantile           = tau;
        admm.method             = QuantileRegressionMethod::ADMM;
        admm.absolute_tolerance = 1e-9;
        admm.relative_tolerance = 1e-9;
        admm.max_iterations     = 200000;
        const QuantileRegression approx(data, "y ~ x", admm);
        ASSERT_TRUE(exact.converged());
        EXPECT_LE(exact.objective(), approx.objective() + 1e-9);
        EXPECT_NEAR(exact.objective(), approx.objective(), 1e-5 * exact.objective());
        EXPECT_NEAR(exact.coefficients()[1], approx.coefficients()[1], 1e-3);
    }
}

TEST(QuantileRegression, DualSolutionIsFeasibleAndClosesTheDualityGap) {
    const auto data = fan_frame(200, 11);
    const double tau = 0.25;
    const QuantileRegression model(data, "y ~ x", simplex(tau));
    const auto& a = model.dual_solution();
    const auto& x = data.numeric_column("x").values();
    const auto& y = data.numeric_column("y").values();
    double s0 = 0.0, s1 = 0.0, dual_value = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_GE(a[i], tau - 1.0 - 1e-9);
        EXPECT_LE(a[i], tau + 1e-9);
        s0 += a[i];
        s1 += a[i] * x[i];
        dual_value += a[i] * y[i];
    }
    EXPECT_NEAR(s0, 0.0, 1e-9);
    EXPECT_NEAR(s1, 0.0, 1e-8);
    EXPECT_NEAR(dual_value, model.objective(), 1e-8);
    // Exactly floor or ceil of n * tau rows lie below the fit.
    std::size_t below = 0;
    for (const double r : model.residuals()) below += r < 0.0;
    EXPECT_LE(below, 50u);
    EXPECT_GE(below + 2, 50u);
}

TEST(QuantileRegression, BandwidthsMatchTheirFormulas) {
    EXPECT_NEAR(datamunge::stats::hall_sheather_bandwidth(0.5, 100), 0.20931604694700326, 1e-8);
    EXPECT_NEAR(datamunge::stats::hall_sheather_bandwidth(0.9, 250), 0.05492322324526707, 1e-8);
    EXPECT_NEAR(datamunge::stats::bofinger_bandwidth(0.5, 100), 0.257852030339281, 1e-8);
    EXPECT_NEAR(datamunge::stats::bofinger_bandwidth(0.9, 250), 0.06218744984597973, 1e-8);
    EXPECT_THROW((void)datamunge::stats::hall_sheather_bandwidth(1.0, 10), std::invalid_argument);
}

TEST(QuantileRegression, StandardErrorsAgreeWithTheAsymptoticFormula) {
    // Homoskedastic normal errors: Var(beta) = tau (1 - tau) / phi(0)^2 (X^T X)^{-1}.
    std::mt19937_64                  gen(3);
    std::normal_distribution<double> z;
    const std::size_t n = 2000;
    std::vector<double> x(n), y(n);
    double sx = 0.0, sxx = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = static_cast<double>(i % 10);
        y[i] = 1.0 + 0.5 * x[i] + z(gen);
        sx += x[i];
        sxx += x[i] * x[i];
    }
    DataFrame data;
    data.add_column("x", x);
    data.add_column("y", y);
    const QuantileRegression model(data, "y ~ x", simplex(0.5));
    const double det = n * sxx - sx * sx;
    const double phi0 = 1.0 / std::sqrt(2.0 * M_PI);
    const double slope_se = std::sqrt(0.25 / (phi0 * phi0) * n / det);
    for (const auto method : {QuantileStandardErrors::IID, QuantileStandardErrors::NID, QuantileStandardErrors::Bootstrap}) {
        const auto table = model.coefficient_table(method, 0.95, 100);
        EXPECT_NEAR(table.std_error[1], slope_se, 0.2 * slope_se);
        EXPECT_LT(table.lower[1], table.estimate[1]);
        EXPECT_GT(table.upper[1], table.estimate[1]);
        EXPECT_EQ(table.degrees_of_freedom, n - 2);
    }
    const auto b1 = model.coefficient_table(QuantileStandardErrors::Bootstrap, 0.95, 50, 9);
    const auto b2 = model.coefficient_table(QuantileStandardErrors::Bootstrap, 0.95, 50, 9);
    EXPECT_EQ(b1.std_error, b2.std_error);
    EXPECT_EQ(b1.replications, 50u);
}

TEST(QuantileRegression, QuantileProcessCollectsOneRowPerLevel) {
    const auto data = fan_frame(300, 5);
    const auto process = datamunge::stats::quantile_process(data, "y ~ x", {0.1, 0.5, 0.9}, simplex(0.5));
    ASSERT_EQ(process.coefficients.size(), 3u);
    ASSERT_EQ(process.names.size(), 2u);
    // The fan opens with x, so the slope grows with tau (true slopes 2 + 0.5 z_tau).
    EXPECT_LT(process.coefficients[0][1], process.coefficients[1][1]);
    EXPECT_LT(process.coefficients[1][1], process.coefficients[2][1]);
    EXPECT_LT(process.lower[2][1], process.coefficients[2][1]);
    EXPECT_THROW((void)datamunge::stats::quantile_process(data, "y ~ x", {}), std::invalid_argument);
}
