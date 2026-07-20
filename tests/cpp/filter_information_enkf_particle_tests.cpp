#include <gtest/gtest.h>

#include <datamunge/filter/ensemble_kalman_filter.hpp>
#include <datamunge/filter/information_filter.hpp>
#include <datamunge/filter/kalman_filter.hpp>
#include <datamunge/filter/particle_filter.hpp>

#include <cmath>
#include <random>

using namespace datamunge::filter;

namespace {
struct LinearFn : VectorFunction {
    std::vector<std::vector<double>> M;
    explicit LinearFn(std::vector<std::vector<double>> m) : M(std::move(m)) {}
    std::vector<double> evaluate(const std::vector<double>& x) override {
        std::vector<double> out(M.size(), 0.0);
        for (std::size_t i = 0; i < M.size(); ++i)
            for (std::size_t j = 0; j < x.size(); ++j) out[i] += M[i][j] * x[j];
        return out;
    }
};

struct ConstantVelocityFixture {
    std::vector<std::vector<double>> F = {{1.0, 1.0}, {0.0, 1.0}};
    std::vector<std::vector<double>> H = {{1.0, 0.0}};
    std::vector<std::vector<double>> Q = {{0.01, 0.0}, {0.0, 0.01}};
    std::vector<std::vector<double>> R = {{4.0}};
    std::vector<double> x0 = {0.0, 0.0};
    std::vector<std::vector<double>> P0 = {{100.0, 0.0}, {0.0, 100.0}};
    std::vector<std::vector<double>> measurements;
    double true_pos = 0.0;

    ConstantVelocityFixture() {
        std::mt19937 rng(42);
        std::normal_distribution<double> noise(0.0, 2.0);
        for (int i = 0; i < 50; ++i) {
            true_pos += 2.0;
            measurements.push_back({true_pos + noise(rng)});
        }
    }
};
} // namespace

TEST(InformationFilter, MatchesKalmanFilterExactly) {
    ConstantVelocityFixture fx;
    KalmanFilter kf(fx.F, fx.H, fx.Q, fx.R, fx.x0, fx.P0);
    const auto kf_result = kf.filter(fx.measurements);

    InformationFilter inf(fx.F, fx.H, fx.Q, fx.R, fx.x0, fx.P0);
    const auto inf_result = inf.filter(fx.measurements);

    ASSERT_EQ(inf_result.size(), kf_result.size());
    for (std::size_t i = 0; i < kf_result.size(); ++i) {
        EXPECT_NEAR(inf_result[i].x[0], kf_result[i].x[0], 1e-6);
        EXPECT_NEAR(inf_result[i].x[1], kf_result[i].x[1], 1e-6);
        EXPECT_NEAR(inf_result[i].P[0][0], kf_result[i].P[0][0], 1e-6);
    }
}

TEST(InformationFilter, RejectsMismatchedDimensions) {
    EXPECT_THROW(InformationFilter({{1.0}}, {{1.0, 0.0}}, {{1.0}}, {{1.0}}, {0.0}, {{1.0}}), std::invalid_argument);
}

TEST(EnsembleKalmanFilter, ApproachesTheKalmanFilterResultWithALargeEnsemble) {
    ConstantVelocityFixture fx;
    KalmanFilter kf(fx.F, fx.H, fx.Q, fx.R, fx.x0, fx.P0);
    const auto kf_result = kf.filter(fx.measurements);

    LinearFn f_lin(fx.F), h_lin(fx.H);
    EnsembleKalmanFilter enkf(f_lin, h_lin, fx.Q, fx.R, fx.x0, fx.P0, 500, 7);
    const auto enkf_result = enkf.filter(fx.measurements);

    EXPECT_NEAR(enkf_result.back().x[0], kf_result.back().x[0], 3.0);
}

TEST(EnsembleKalmanFilter, RejectsTooSmallAnEnsemble) {
    LinearFn f({{1.0, 1.0}, {0.0, 1.0}}), h({{1.0, 0.0}});
    EXPECT_THROW(EnsembleKalmanFilter(f, h, {{1.0, 0.0}, {0.0, 1.0}}, {{1.0}}, {0.0, 0.0}, {{1.0, 0.0}, {0.0, 1.0}}, 1), std::invalid_argument);
}

TEST(EnsembleKalmanFilter, RejectsMismatchedDimensions) {
    LinearFn f({{1.0, 1.0}, {0.0, 1.0}}), h({{1.0, 0.0}});
    EXPECT_THROW(EnsembleKalmanFilter(f, h, {{1.0}}, {{1.0}}, {0.0, 0.0}, {{1.0, 0.0}, {0.0, 1.0}}), std::invalid_argument);
}

TEST(ParticleFilter, TracksTheTrueTrajectoryReasonablyWell) {
    ConstantVelocityFixture fx;
    LinearFn f_lin(fx.F), h_lin(fx.H);
    ParticleFilter pf(f_lin, h_lin, fx.Q, fx.R, fx.x0, fx.P0, 500, 7);
    const auto result = pf.filter(fx.measurements);
    EXPECT_NEAR(result.back().x[0], fx.true_pos, 10.0);
}

TEST(ParticleFilter, RejectsTooFewParticles) {
    LinearFn f({{1.0, 1.0}, {0.0, 1.0}}), h({{1.0, 0.0}});
    EXPECT_THROW(ParticleFilter(f, h, {{1.0, 0.0}, {0.0, 1.0}}, {{1.0}}, {0.0, 0.0}, {{1.0, 0.0}, {0.0, 1.0}}, 1), std::invalid_argument);
}
