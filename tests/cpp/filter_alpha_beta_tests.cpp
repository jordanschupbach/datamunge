#include <gtest/gtest.h>

#include <datamunge/filter/alpha_beta_filter.hpp>

#include <cmath>

using namespace datamunge::filter;

TEST(AlphaBetaFilter, ConvergesExactlyOnANoiseFreeConstantVelocityTrajectory) {
    AlphaBetaFilter ab(0.5, 0.3, 1.0);
    double true_pos = 0.0;
    const double true_velocity = 2.0;
    for (int i = 0; i < 15; ++i) {
        true_pos += true_velocity;
        ab.update(true_pos);
    }
    EXPECT_NEAR(ab.velocity(), true_velocity, 0.05);
    EXPECT_NEAR(ab.position(), true_pos, 0.5);
}

TEST(AlphaBetaFilter, TracksReasonablyUnderMeasurementNoise) {
    AlphaBetaFilter ab(0.5, 0.3, 1.0);
    std::vector<double> measurements = {2.1, 3.8, 6.2, 7.9, 10.3, 11.8, 14.2, 15.9, 18.1, 19.8};
    const auto result = ab.filter(measurements);
    ASSERT_EQ(result.size(), measurements.size());
    EXPECT_NEAR(result.back().velocity, 2.0, 1.0);
}

TEST(AlphaBetaFilter, RejectsOutOfRangeGains) {
    EXPECT_THROW(AlphaBetaFilter(1.5, 0.3, 1.0), std::invalid_argument);
    EXPECT_THROW(AlphaBetaFilter(0.5, -0.1, 1.0), std::invalid_argument);
    EXPECT_THROW(AlphaBetaFilter(0.5, 0.3, 0.0), std::invalid_argument);
}

TEST(AlphaBetaGammaFilter, ConvergesExactlyOnANoiseFreeConstantAccelerationTrajectory) {
    // Gains must satisfy the g-h-k filter's own stability relation (roughly, gamma small
    // relative to beta^2/alpha) or the filter oscillates instead of converging -- this is a
    // real property of the alpha-beta-gamma recursion, not a tuning nicety specific to this
    // implementation. (0.5, 0.4, 0.2) from an earlier draft of this test was NOT such a choice
    // and oscillated indefinitely; (0.5, 0.1, 0.01) is comfortably stable.
    AlphaBetaGammaFilter abg(0.5, 0.1, 0.01, 1.0);
    double true_pos = 0.0, true_vel = 0.0;
    const double true_accel = 0.5;
    for (int i = 0; i < 40; ++i) {
        true_pos += true_vel + 0.5 * true_accel;
        true_vel += true_accel;
        abg.update(true_pos);
    }
    EXPECT_NEAR(abg.acceleration(), true_accel, 0.05);
    EXPECT_NEAR(abg.velocity(), true_vel, 0.5);
}

TEST(AlphaBetaGammaFilter, RejectsOutOfRangeGains) {
    EXPECT_THROW(AlphaBetaGammaFilter(1.5, 0.3, 0.1, 1.0), std::invalid_argument);
    EXPECT_THROW(AlphaBetaGammaFilter(0.5, -0.1, 0.1, 1.0), std::invalid_argument);
    EXPECT_THROW(AlphaBetaGammaFilter(0.5, 0.3, -0.1, 1.0), std::invalid_argument);
    EXPECT_THROW(AlphaBetaGammaFilter(0.5, 0.3, 0.1, 0.0), std::invalid_argument);
}
