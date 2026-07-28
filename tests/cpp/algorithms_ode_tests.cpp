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
