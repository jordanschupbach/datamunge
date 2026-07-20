#include <datamunge/filter/filter.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
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

// A monotonic nonlinear sensor model (derivative 2+cos(x) >= 1 > 0 everywhere, so -- unlike a
// symmetric function such as sqrt(x^2+c) -- there's no sign ambiguity to trip up EKF/UKF).
struct MonotonicSensor : VectorFunction {
    std::vector<double> evaluate(const std::vector<double>& x) override { return {2.0 * x[0] + std::sin(x[0])}; }
};

} // namespace

int main() {
    std::cout << std::fixed << std::setprecision(3);

    // A constant-velocity target: position advances by velocity*dt each step, and only
    // position is observed (with noise) -- the running example for every filter below.
    const double dt = 1.0;
    std::vector<std::vector<double>> F = {{1.0, dt}, {0.0, 1.0}};
    std::vector<std::vector<double>> H = {{1.0, 0.0}};
    std::vector<std::vector<double>> Q = {{0.01, 0.0}, {0.0, 0.01}};
    std::vector<std::vector<double>> R = {{4.0}};
    std::vector<double> x0 = {0.0, 0.0};
    std::vector<std::vector<double>> P0 = {{100.0, 0.0}, {0.0, 100.0}};

    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 2.0);
    const double true_velocity = 2.0;
    std::vector<std::vector<double>> measurements;
    double true_pos = 0.0;
    for (int i = 0; i < 50; ++i) {
        true_pos += true_velocity * dt;
        measurements.push_back({true_pos + noise(rng)});
    }

    std::cout << "=================== Kalman Filter + RTS smoother ===================\n";
    KalmanFilter kf(F, H, Q, R, x0, P0);
    const auto kf_result = kf.filter(measurements);
    std::cout << "filtered final estimate: position=" << kf_result.back().x[0] << " velocity=" << kf_result.back().x[1]
              << " (true position=" << true_pos << ", true velocity=" << true_velocity << ")\n";

    KalmanFilter kf2(F, H, Q, R, x0, P0);
    const auto smoothed = kf2.smooth(measurements);
    double filtered_mse = 0.0, smoothed_mse = 0.0;
    for (std::size_t i = 0; i < measurements.size(); ++i) {
        const double true_i = true_velocity * dt * static_cast<double>(i + 1);
        filtered_mse += std::pow(smoothed.filtered[i].x[0] - true_i, 2);
        smoothed_mse += std::pow(smoothed.smoothed[i].x[0] - true_i, 2);
    }
    std::cout << "RTS smoother MSE=" << smoothed_mse / measurements.size() << " vs. filter-only MSE="
              << filtered_mse / measurements.size() << " (smoother uses the whole sequence, so it should be lower)\n\n";

    std::cout << "=================== Extended and Unscented Kalman Filters (nonlinear sensor) ===================\n";
    std::vector<std::vector<double>> R2 = {{0.25}};
    std::normal_distribution<double> small_noise(0.0, 0.5);
    std::vector<std::vector<double>> nl_measurements;
    double pos2 = 0.0;
    for (int i = 0; i < 50; ++i) {
        pos2 += true_velocity * dt;
        nl_measurements.push_back({2.0 * pos2 + std::sin(pos2) + small_noise(rng)});
    }

    LinearFn f_lin(F);
    MonotonicSensor h_nl;
    ExtendedKalmanFilter ekf(f_lin, h_nl, Q, R2, x0, P0);
    const auto ekf_result = ekf.filter(nl_measurements);
    std::cout << "EKF final position estimate: " << ekf_result.back().x[0] << " (true: " << pos2 << ")\n";

    UnscentedKalmanFilter ukf(f_lin, h_nl, Q, R2, x0, P0);
    const auto ukf_result = ukf.filter(nl_measurements);
    std::cout << "UKF final position estimate: " << ukf_result.back().x[0] << " (true: " << pos2 << ")\n\n";

    std::cout << "=================== Information Filter (KalmanFilter's algebraic dual) ===================\n";
    InformationFilter inf(F, H, Q, R, x0, P0);
    const auto inf_result = inf.filter(measurements);
    double max_diff = 0.0;
    for (std::size_t i = 0; i < kf_result.size(); ++i) max_diff = std::max(max_diff, std::abs(inf_result[i].x[0] - kf_result[i].x[0]));
    std::cout << "max difference from KalmanFilter's result: " << max_diff << " (should be ~0 -- same estimator, different form)\n\n";

    std::cout << "=================== Ensemble Kalman Filter and Particle Filter ===================\n";
    EnsembleKalmanFilter enkf(f_lin, h_nl, Q, R2, x0, P0, /*ensemble_size=*/300, /*seed=*/7);
    const auto enkf_result = enkf.filter(nl_measurements);
    std::cout << "EnKF final position estimate: " << enkf_result.back().x[0] << " (true: " << pos2 << ")\n";

    ParticleFilter pf(f_lin, h_nl, Q, R2, x0, P0, /*num_particles=*/300, /*seed=*/7);
    const auto pf_result = pf.filter(nl_measurements);
    std::cout << "Particle filter final position estimate: " << pf_result.back().x[0] << " (true: " << pos2 << ")\n\n";

    std::cout << "=================== Alpha-Beta and Alpha-Beta-Gamma trackers ===================\n";
    AlphaBetaFilter ab(0.5, 0.3, dt);
    std::vector<double> scalar_measurements;
    for (const auto& z : measurements) scalar_measurements.push_back(z[0]);
    const auto ab_result = ab.filter(scalar_measurements);
    std::cout << "alpha-beta final velocity estimate: " << ab_result.back().velocity << " (true: " << true_velocity << ")\n";

    // Gains chosen to satisfy the g-h-k filter's own stability relation (roughly gamma small
    // relative to beta^2/alpha) -- an arbitrary triple can leave the filter oscillating rather
    // than converging, a real property of this recursion, not a tuning nicety.
    AlphaBetaGammaFilter abg(0.5, 0.1, 0.01, dt);
    double accel_true_pos = 0.0, accel_true_vel = 0.0;
    const double true_accel = 0.5;
    std::vector<double> accel_measurements;
    for (int i = 0; i < 40; ++i) {
        accel_true_pos += accel_true_vel + 0.5 * true_accel;
        accel_true_vel += true_accel;
        accel_measurements.push_back(accel_true_pos);
    }
    const auto abg_result = abg.filter(accel_measurements);
    std::cout << "alpha-beta-gamma final acceleration estimate: " << abg_result.back().acceleration << " (true: " << true_accel
              << ")\n";

    return 0;
}
