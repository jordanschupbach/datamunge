#include <gtest/gtest.h>

#include <datamunge/filter/kalman_filter.hpp>

#include <cmath>
#include <random>

using namespace datamunge::filter;

namespace {
struct ConstantVelocityFixture {
    double dt = 1.0;
    std::vector<std::vector<double>> F = {{1.0, 1.0}, {0.0, 1.0}};
    std::vector<std::vector<double>> H = {{1.0, 0.0}};
    std::vector<std::vector<double>> Q = {{0.01, 0.0}, {0.0, 0.01}};
    std::vector<std::vector<double>> R = {{4.0}};
    std::vector<double> x0 = {0.0, 0.0};
    std::vector<std::vector<double>> P0 = {{100.0, 0.0}, {0.0, 100.0}};

    std::vector<std::vector<double>> measurements;
    std::vector<double> true_positions;
    double true_velocity = 2.0;

    ConstantVelocityFixture() {
        std::mt19937 rng(42);
        std::normal_distribution<double> noise(0.0, 2.0);
        double pos = 0.0;
        for (int i = 0; i < 50; ++i) {
            pos += true_velocity * dt;
            true_positions.push_back(pos);
            measurements.push_back({pos + noise(rng)});
        }
    }
};
} // namespace

TEST(KalmanFilter, ConvergesToTheTrueVelocityOnAConstantVelocityModel) {
    ConstantVelocityFixture fx;
    KalmanFilter kf(fx.F, fx.H, fx.Q, fx.R, fx.x0, fx.P0);
    const auto result = kf.filter(fx.measurements);
    EXPECT_NEAR(result.back().x[1], fx.true_velocity, 0.5);
}

TEST(KalmanFilter, FilteredEstimateHasLowerMseThanTheRawNoisyMeasurement) {
    ConstantVelocityFixture fx;
    KalmanFilter kf(fx.F, fx.H, fx.Q, fx.R, fx.x0, fx.P0);
    const auto result = kf.filter(fx.measurements);

    double mse_filtered = 0.0, mse_raw = 0.0;
    const int start = 25;
    for (int i = start; i < 50; ++i) {
        const double err_f = result[i].x[0] - fx.true_positions[i];
        const double err_raw = fx.measurements[i][0] - fx.true_positions[i];
        mse_filtered += err_f * err_f;
        mse_raw += err_raw * err_raw;
    }
    EXPECT_LT(mse_filtered, mse_raw);
}

TEST(KalmanFilter, CovarianceShrinksFromTheInitialPrior) {
    ConstantVelocityFixture fx;
    KalmanFilter kf(fx.F, fx.H, fx.Q, fx.R, fx.x0, fx.P0);
    const auto result = kf.filter(fx.measurements);
    EXPECT_LT(result.back().P[0][0], 10.0);
}

TEST(KalmanFilter, RejectsMismatchedMatrixDimensions) {
    EXPECT_THROW(KalmanFilter({{1.0}}, {{1.0, 0.0}}, {{1.0}}, {{1.0}}, {0.0}, {{1.0}}), std::invalid_argument);
}

TEST(KalmanFilterSmooth, SmoothedMatchesFilteredExactlyAtTheFinalStep) {
    ConstantVelocityFixture fx;
    KalmanFilter kf(fx.F, fx.H, fx.Q, fx.R, fx.x0, fx.P0);
    const auto result = kf.smooth(fx.measurements);
    ASSERT_EQ(result.filtered.size(), result.smoothed.size());
    for (std::size_t i = 0; i < result.filtered.back().x.size(); ++i) {
        EXPECT_NEAR(result.smoothed.back().x[i], result.filtered.back().x[i], 1e-9);
    }
}

TEST(KalmanFilterSmooth, SmoothedCovarianceNeverExceedsFilteredCovariance) {
    ConstantVelocityFixture fx;
    KalmanFilter kf(fx.F, fx.H, fx.Q, fx.R, fx.x0, fx.P0);
    const auto result = kf.smooth(fx.measurements);
    for (std::size_t i = 0; i < result.filtered.size(); ++i) {
        EXPECT_LE(result.smoothed[i].P[0][0], result.filtered[i].P[0][0] + 1e-9);
    }
}

TEST(KalmanFilterSmooth, SmoothedMseIsAtMostTheFilteredMse) {
    ConstantVelocityFixture fx;
    KalmanFilter kf(fx.F, fx.H, fx.Q, fx.R, fx.x0, fx.P0);
    const auto result = kf.smooth(fx.measurements);

    double mse_smoothed = 0.0, mse_filtered = 0.0;
    for (std::size_t i = 0; i < result.filtered.size(); ++i) {
        const double err_s = result.smoothed[i].x[0] - fx.true_positions[i];
        const double err_f = result.filtered[i].x[0] - fx.true_positions[i];
        mse_smoothed += err_s * err_s;
        mse_filtered += err_f * err_f;
    }
    EXPECT_LE(mse_smoothed, mse_filtered);
}
