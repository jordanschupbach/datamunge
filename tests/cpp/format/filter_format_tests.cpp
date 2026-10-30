// std::format / std::print support for datamunge::filter types.
#include <dm/dm.hpp>

#include <gtest/gtest.h>

#include <format>
#include <string>
#include <vector>

namespace fl = dm::filter;

static_assert(std::formattable<fl::AlphaBetaState, char>);
static_assert(std::formattable<fl::AlphaBetaGammaState, char>);
static_assert(std::formattable<fl::AlphaBetaFilter, char>);
static_assert(std::formattable<fl::AlphaBetaGammaFilter, char>);
static_assert(std::formattable<fl::KalmanState, char>);
static_assert(std::formattable<fl::SmoothResult, char>);
static_assert(std::formattable<fl::KalmanFilter, char>);
static_assert(std::formattable<fl::InformationFilter, char>);
static_assert(std::formattable<fl::ExtendedKalmanFilter, char>);
static_assert(std::formattable<fl::UnscentedKalmanFilter, char>);
static_assert(std::formattable<fl::EnsembleKalmanFilter, char>);
static_assert(std::formattable<fl::ParticleFilter, char>);

namespace {
struct Identity : fl::VectorFunction {
    std::vector<double> evaluate(const std::vector<double>& x) override { return x; }
};
const std::vector<std::vector<double>> I2{{1, 0}, {0, 1}};
const std::vector<std::vector<double>> Q2{{0.01, 0}, {0, 0.01}};
}  // namespace

TEST(FilterFormat, AlphaBeta) {
    EXPECT_EQ(std::format("{}", fl::AlphaBetaState{1.5, 2}), "AlphaBetaState(position=1.5, velocity=2)");
    EXPECT_EQ(std::format("{}", fl::AlphaBetaGammaState{1, 2, 3}),
              "AlphaBetaGammaState(position=1, velocity=2, acceleration=3)");
    EXPECT_EQ(std::format("{}", fl::AlphaBetaFilter(0.5, 0.1, 1.0, 2.0, 1.0)),
              "AlphaBetaFilter(alpha=0.5, beta=0.1, dt=1, position=2, velocity=1)");
    EXPECT_EQ(std::format("{}", fl::AlphaBetaGammaFilter(0.5, 0.1, 0.01, 1.0)),
              "AlphaBetaGammaFilter(alpha=0.5, beta=0.1, gamma=0.01, dt=1, position=0, velocity=0, acceleration=0)");
}

TEST(FilterFormat, KalmanStateAndSmoothResult) {
    const fl::KalmanState s{{1, 2}, {{0.5, 0.1}, {0.1, 0.25}}};
    EXPECT_EQ(std::format("{}", s), "KalmanState(x=[1, 2], variances=[0.5, 0.25])");
    EXPECT_EQ(std::format("{}", fl::SmoothResult{{s, s}, {s, s}}),
              "SmoothResult(filtered=2, smoothed=2, final=KalmanState(x=[1, 2], variances=[0.5, 0.25]))");
    EXPECT_EQ(std::format("{}", fl::SmoothResult{}), "SmoothResult(filtered=0, smoothed=0)");
}

TEST(FilterFormat, LinearFilters) {
    const fl::KalmanFilter kf({{1, 1}, {0, 1}}, {{1, 0}}, Q2, {{1}}, {0, 1}, I2);
    EXPECT_EQ(std::format("{}", kf), "KalmanFilter(n=2, state=[0, 1], variances=[1, 1])");
    const fl::InformationFilter inf({{1, 1}, {0, 1}}, {{1, 0}}, Q2, {{1}}, {0, 1}, I2);
    EXPECT_EQ(std::format("{}", inf), "InformationFilter(n=2, state=[0, 1], variances=[1, 1])");
}

TEST(FilterFormat, NonlinearFilters) {
    Identity f, h;
    EXPECT_EQ(std::format("{}", fl::ExtendedKalmanFilter(f, h, Q2, Q2, {1, 2}, I2)),
              "ExtendedKalmanFilter(n=2, state=[1, 2], variances=[1, 1])");
    EXPECT_EQ(std::format("{}", fl::UnscentedKalmanFilter(f, h, Q2, Q2, {1, 2}, I2)),
              "UnscentedKalmanFilter(n=2, state=[1, 2], variances=[1, 1])");
    EXPECT_EQ(std::format("{}", fl::EnsembleKalmanFilter(f, h, Q2, Q2, {1, 2}, I2, 10)).find("EnsembleKalmanFilter(n=2, state=["),
              0u);
    EXPECT_EQ(std::format("{}", fl::ParticleFilter(f, h, Q2, Q2, {1, 2}, I2, 10)).find("ParticleFilter(n=2, state=["), 0u);
}
