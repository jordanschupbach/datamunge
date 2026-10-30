#include <gtest/gtest.h>

#include <datamunge/algorithms/ode.hpp>

#include <cmath>
#include <vector>

using namespace datamunge::algorithms;

namespace {

// Scalar exponential decay y' = -lambda y, exact y(t) = y0 e^{-lambda t}.
OdeSystem decay(double lambda) {
    return [lambda](double, const std::vector<double>& y) { return std::vector<double>{-lambda * y[0]}; };
}

double final_error(const OdeSolution& s, double exact) { return std::fabs(s.y.back()[0] - exact); }

} // namespace

TEST(Ode, AllMethodsMatchExponentialSolution) {
    const double lambda = 1.0, t1 = 2.0;
    const double exact  = std::exp(-lambda * t1);
    const auto   e  = euler_method(decay(lambda), {1.0}, 0.0, t1, 2000);
    const auto   be = backward_euler(decay(lambda), {1.0}, 0.0, t1, 2000);
    const auto   rk = runge_kutta4(decay(lambda), {1.0}, 0.0, t1, 2000);
    EXPECT_NEAR(e.y.back()[0], exact, 1e-3);
    EXPECT_NEAR(be.y.back()[0], exact, 1e-3);
    EXPECT_NEAR(rk.y.back()[0], exact, 1e-10);

    // RK4 solves y' = y to e at t = 1 very accurately.
    auto grow = [](double, const std::vector<double>& y) { return std::vector<double>{y[0]}; };
    const auto g = runge_kutta4(grow, {1.0}, 0.0, 1.0, 100);
    EXPECT_NEAR(g.y.back()[0], std::exp(1.0), 1e-9);
}

TEST(Ode, ConvergenceOrders) {
    const double exact = std::exp(-1.0);
    // Forward Euler is first order: halving h ~ halves the error.
    const double e1 = final_error(euler_method(decay(1.0), {1.0}, 0.0, 1.0, 100), exact);
    const double e2 = final_error(euler_method(decay(1.0), {1.0}, 0.0, 1.0, 200), exact);
    EXPECT_NEAR(e1 / e2, 2.0, 0.2);

    // Backward Euler is also first order.
    const double b1 = final_error(backward_euler(decay(1.0), {1.0}, 0.0, 1.0, 100), exact);
    const double b2 = final_error(backward_euler(decay(1.0), {1.0}, 0.0, 1.0, 200), exact);
    EXPECT_NEAR(b1 / b2, 2.0, 0.2);

    // RK4 is fourth order: halving h cuts the error by ~16x.
    const double r1 = final_error(runge_kutta4(decay(1.0), {1.0}, 0.0, 1.0, 8), exact);
    const double r2 = final_error(runge_kutta4(decay(1.0), {1.0}, 0.0, 1.0, 16), exact);
    EXPECT_GT(r1 / r2, 12.0);
    EXPECT_LT(r1 / r2, 20.0);
}

TEST(Ode, BackwardEulerIsStableWhereForwardEulerBlowsUp) {
    // Stiff decay with a large step: h*lambda = 5 >> 1.
    const double lambda = 50.0;
    const int    steps  = 10; // h = 0.1
    const auto   fe = euler_method(decay(lambda), {1.0}, 0.0, 1.0, steps);
    const auto   be = backward_euler(decay(lambda), {1.0}, 0.0, 1.0, steps);

    // Forward Euler amplifies by (1 - h*lambda) = -4 each step: it explodes and oscillates in sign.
    EXPECT_GT(std::fabs(fe.y.back()[0]), 1e3);

    // Backward Euler stays bounded, positive, and monotonically decaying toward 0.
    for (std::size_t i = 1; i < be.y.size(); ++i) {
        EXPECT_GT(be.y[i][0], 0.0);
        EXPECT_LT(be.y[i][0], be.y[i - 1][0]);
        EXPECT_TRUE(std::isfinite(be.y[i][0]));
    }
    EXPECT_LT(be.y.back()[0], 1e-3);
}

TEST(Ode, VelocityVerletConservesEnergy) {
    // Simple harmonic oscillator x'' = -x; energy E = (v^2 + x^2)/2 should stay ~constant.
    auto accel = [](const std::vector<double>& x) { return std::vector<double>{-x[0]}; };
    const auto s = velocity_verlet(accel, {1.0}, {0.0}, 0.0, 100.0, 10000); // ~16 periods

    const double e0 = 0.5 * (s.v[0][0] * s.v[0][0] + s.x[0][0] * s.x[0][0]);
    double       emin = e0, emax = e0;
    for (std::size_t i = 0; i < s.x.size(); ++i) {
        const double e = 0.5 * (s.v[i][0] * s.v[i][0] + s.x[i][0] * s.x[i][0]);
        emin = std::min(emin, e);
        emax = std::max(emax, e);
    }
    // Symplectic: energy oscillates within a tight bounded band, no secular drift.
    EXPECT_LT(emax - emin, 1e-2);
    EXPECT_NEAR(emax, e0, 1e-2);

    // Position tracks cos(t): check a late point.
    const std::size_t last = s.x.size() - 1;
    EXPECT_NEAR(s.x[last][0], std::cos(s.t[last]), 1e-2);
}

TEST(Ode, VerletVersusForwardEulerEnergyDrift) {
    // Forward Euler on the SHO written as a first-order system pumps energy in over time.
    auto sho = [](double, const std::vector<double>& z) {
        return std::vector<double>{z[1], -z[0]}; // z = (x, v), x' = v, v' = -x
    };
    const auto fe = euler_method(sho, {1.0, 0.0}, 0.0, 100.0, 10000);
    const double e0 = 0.5 * (fe.y.front()[1] * fe.y.front()[1] + fe.y.front()[0] * fe.y.front()[0]);
    const double eN = 0.5 * (fe.y.back()[1] * fe.y.back()[1] + fe.y.back()[0] * fe.y.back()[0]);
    EXPECT_GT(eN, 1.3 * e0); // forward Euler's energy has drifted noticeably upward
}

TEST(BackwardEulerAdaptive, MeetsToleranceOnDecay) {
    using datamunge::algorithms::backward_euler_adaptive;
    const auto f = [](double, const std::vector<double>& y) { return std::vector<double>{-y[0]}; };
    const auto r = backward_euler_adaptive(f, {1.0}, 0.0, 5.0, 1e-4, 1e-8, 1e-3);
    ASSERT_GT(r.accepted, 10);
    EXPECT_DOUBLE_EQ(r.solution.t.back(), 5.0);
    EXPECT_EQ(r.step_sizes.size(), static_cast<std::size_t>(r.accepted));
    EXPECT_NEAR(r.solution.y.back()[0], std::exp(-5.0), 5e-3 * std::exp(-5.0) + 1e-3);
    EXPECT_GT(r.f_evaluations, 0);
    // Tighter tolerance means more steps (local error O(h^2) -> steps ~ tol^-1/2).
    const auto tight = backward_euler_adaptive(f, {1.0}, 0.0, 5.0, 1e-6, 1e-10, 1e-3);
    EXPECT_GT(tight.accepted, 5 * r.accepted);
    EXPECT_THROW(backward_euler_adaptive(f, {1.0}, 1.0, 0.0, 1e-4, 1e-8, 1e-3), std::invalid_argument);
}

TEST(BackwardEulerAdaptive, StiffProblemTakesLargeSteps) {
    using datamunge::algorithms::backward_euler_adaptive;
    // y' = -1000 (y - cos t): after a fast transient y tracks cos t; steps must grow far beyond 2/1000.
    const auto f = [](double t, const std::vector<double>& y) { return std::vector<double>{-1000.0 * (y[0] - std::cos(t))}; };
    const auto r = backward_euler_adaptive(f, {0.0}, 0.0, 10.0, 1e-3, 1e-6, 1e-5);
    double hmax = 0;
    for (double h : r.step_sizes) hmax = std::max(hmax, h);
    EXPECT_GT(hmax, 0.02);
    EXPECT_NEAR(r.solution.y.back()[0], std::cos(10.0), 1e-2);
}

TEST(StabilityFunction, KnownValues) {
    using datamunge::algorithms::OdeMethod;
    using datamunge::algorithms::stability_function;
    const std::complex<double> z{-3.0, 0.0};
    EXPECT_NEAR(std::abs(stability_function(OdeMethod::ForwardEuler, z)), 2.0, 1e-15);
    EXPECT_NEAR(std::abs(stability_function(OdeMethod::BackwardEuler, z)), 0.25, 1e-15);
    EXPECT_NEAR(stability_function(OdeMethod::Trapezoidal, z).real(), -0.2, 1e-15);
    EXPECT_NEAR(stability_function(OdeMethod::RungeKutta4, {-1.0, 0.0}).real(), 1 - 1 + 0.5 - 1.0 / 6 + 1.0 / 24, 1e-15);
    // Backward Euler: |R| <= 1 on the whole left half-plane.
    for (double x = -50; x <= 0; x += 0.5)
        for (double y = -50; y <= 50; y += 0.5) EXPECT_LE(std::abs(stability_function(OdeMethod::BackwardEuler, {x, y})), 1.0 + 1e-15);
}
