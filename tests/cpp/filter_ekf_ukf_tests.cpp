#include <gtest/gtest.h>

#include <datamunge/filter/extended_kalman_filter.hpp>
#include <datamunge/filter/kalman_filter.hpp>
#include <datamunge/filter/unscented_kalman_filter.hpp>

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

// Monotonic nonlinear observation (derivative 2+cos(x) >= 1 > 0 everywhere -- no sign
// ambiguity, unlike a symmetric function such as sqrt(x^2+c)).
struct MonotonicNonlinearObservation : VectorFunction {
    std::vector<double> evaluate(const std::vector<double>& x) override { return {2.0 * x[0] + std::sin(x[0])}; }
};
} // namespace

TEST(ExtendedKalmanFilter, MatchesKalmanFilterExactlyWhenGivenLinearFunctions) {
    std::vector<std::vector<double>> F = {{1.0, 1.0}, {0.0, 1.0}};
    std::vector<std::vector<double>> H = {{1.0, 0.0}};
    std::vector<std::vector<double>> Q = {{0.01, 0.0}, {0.0, 0.01}};
    std::vector<std::vector<double>> R = {{4.0}};
    std::vector<double> x0 = {0.0, 0.0};
    std::vector<std::vector<double>> P0 = {{100.0, 0.0}, {0.0, 100.0}};

    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 2.0);
    std::vector<std::vector<double>> measurements;
    double pos = 0.0;
    for (int i = 0; i < 30; ++i) {
        pos += 2.0;
        measurements.push_back({pos + noise(rng)});
    }

    KalmanFilter kf(F, H, Q, R, x0, P0);
    const auto kf_result = kf.filter(measurements);

    LinearFn f_lin(F), h_lin(H);
    ExtendedKalmanFilter ekf(f_lin, h_lin, Q, R, x0, P0);
    const auto ekf_result = ekf.filter(measurements);

    for (std::size_t i = 0; i < kf_result.size(); ++i) EXPECT_NEAR(ekf_result[i].x[0], kf_result[i].x[0], 1e-3);
}

TEST(ExtendedKalmanFilter, TracksATrueNonlinearObservationModel) {
    std::vector<std::vector<double>> F = {{1.0, 1.0}, {0.0, 1.0}};
    std::vector<std::vector<double>> Q = {{0.01, 0.0}, {0.0, 0.01}};
    std::vector<std::vector<double>> R = {{0.25}};
    std::vector<double> x0 = {0.0, 0.0};
    std::vector<std::vector<double>> P0 = {{100.0, 0.0}, {0.0, 100.0}};

    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.5);
    std::vector<std::vector<double>> measurements;
    double pos = 0.0;
    for (int i = 0; i < 50; ++i) {
        pos += 2.0;
        measurements.push_back({2.0 * pos + std::sin(pos) + noise(rng)});
    }

    LinearFn f_lin(F);
    MonotonicNonlinearObservation h_nl;
    ExtendedKalmanFilter ekf(f_lin, h_nl, Q, R, x0, P0);
    const auto result = ekf.filter(measurements);
    EXPECT_NEAR(result.back().x[0], pos, 5.0);
}

TEST(ExtendedKalmanFilter, RejectsMismatchedDimensions) {
    LinearFn f({{1.0, 0.0}, {0.0, 1.0}});
    LinearFn h({{1.0, 0.0}});
    EXPECT_THROW(ExtendedKalmanFilter(f, h, {{1.0}}, {{1.0}}, {0.0, 0.0}, {{1.0, 0.0}, {0.0, 1.0}}), std::invalid_argument);
}

TEST(UnscentedKalmanFilter, MatchesKalmanFilterExactlyWhenGivenLinearFunctions) {
    std::vector<std::vector<double>> F = {{1.0, 1.0}, {0.0, 1.0}};
    std::vector<std::vector<double>> H = {{1.0, 0.0}};
    std::vector<std::vector<double>> Q = {{0.01, 0.0}, {0.0, 0.01}};
    std::vector<std::vector<double>> R = {{4.0}};
    std::vector<double> x0 = {0.0, 0.0};
    std::vector<std::vector<double>> P0 = {{100.0, 0.0}, {0.0, 100.0}};

    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 2.0);
    std::vector<std::vector<double>> measurements;
    double pos = 0.0;
    for (int i = 0; i < 30; ++i) {
        pos += 2.0;
        measurements.push_back({pos + noise(rng)});
    }

    KalmanFilter kf(F, H, Q, R, x0, P0);
    const auto kf_result = kf.filter(measurements);

    LinearFn f_lin(F), h_lin(H);
    UnscentedKalmanFilter ukf(f_lin, h_lin, Q, R, x0, P0);
    const auto ukf_result = ukf.filter(measurements);

    for (std::size_t i = 0; i < kf_result.size(); ++i) EXPECT_NEAR(ukf_result[i].x[0], kf_result[i].x[0], 1e-3);
}

TEST(UnscentedKalmanFilter, TracksATrueNonlinearObservationModel) {
    std::vector<std::vector<double>> F = {{1.0, 1.0}, {0.0, 1.0}};
    std::vector<std::vector<double>> Q = {{0.01, 0.0}, {0.0, 0.01}};
    std::vector<std::vector<double>> R = {{0.25}};
    std::vector<double> x0 = {0.0, 0.0};
    std::vector<std::vector<double>> P0 = {{100.0, 0.0}, {0.0, 100.0}};

    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.5);
    std::vector<std::vector<double>> measurements;
    double pos = 0.0;
    for (int i = 0; i < 50; ++i) {
        pos += 2.0;
        measurements.push_back({2.0 * pos + std::sin(pos) + noise(rng)});
    }

    LinearFn f_lin(F);
    MonotonicNonlinearObservation h_nl;
    UnscentedKalmanFilter ukf(f_lin, h_nl, Q, R, x0, P0);
    const auto result = ukf.filter(measurements);
    EXPECT_NEAR(result.back().x[0], pos, 5.0);
}

TEST(UnscentedKalmanFilter, RejectsMismatchedDimensions) {
    LinearFn f({{1.0, 0.0}, {0.0, 1.0}});
    LinearFn h({{1.0, 0.0}});
    EXPECT_THROW(UnscentedKalmanFilter(f, h, {{1.0}}, {{1.0}}, {0.0, 0.0}, {{1.0, 0.0}, {0.0, 1.0}}), std::invalid_argument);
}
