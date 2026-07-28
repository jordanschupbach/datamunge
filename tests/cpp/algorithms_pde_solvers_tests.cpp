#include <gtest/gtest.h>

#include <datamunge/algorithms/pde_solvers.hpp>

#include <cmath>
#include <vector>

using namespace datamunge::algorithms;

TEST(PdeSolvers, FiniteDifferencePoissonMatchesAnalytic) {
    // -u'' = pi^2 sin(pi x), u(0)=u(1)=0  ->  u = sin(pi x).
    auto f = [](double x) { return M_PI * M_PI * std::sin(M_PI * x); };
    // O(h^2): refining should cut the error ~4x.
    double prev = 1e9;
    for (int n : {20, 40, 80, 160}) {
        const auto u = finite_difference_poisson(f, 0.0, 0.0, n);
        const double h = 1.0 / (n + 1);
        double err = 0.0;
        for (int i = 0; i <= n + 1; ++i) err = std::max(err, std::fabs(u[i] - std::sin(M_PI * i * h)));
        EXPECT_LT(err, 5.0 / (n * n)); // O(h^2)
        if (n > 20) EXPECT_LT(err, prev); // monotone improvement
        prev = err;
    }
    // Nonzero Dirichlet data with f=0 -> linear solution u = alpha + (beta-alpha) x.
    const auto lin = finite_difference_poisson([](double) { return 0.0; }, 2.0, 5.0, 30);
    const double h = 1.0 / 31;
    for (int i = 0; i <= 31; ++i) EXPECT_NEAR(lin[i], 2.0 + 3.0 * i * h, 1e-10);
}

TEST(PdeSolvers, CrankNicolsonHeatMatchesAnalytic) {
    // u_t = u_xx on [0,1], u(x,0)=sin(pi x), Dirichlet 0  ->  u(x,t)=e^{-pi^2 t} sin(pi x).
    auto u0 = [](double x) { return std::sin(M_PI * x); };
    const int nx = 100; const double dt = 0.0005; const int steps = 200; // t = 0.1
    const auto u = crank_nicolson_heat(u0, 1.0, 1.0, nx, dt, steps);
    const double t = dt * steps, dx = 1.0 / (nx + 1);
    double err = 0.0;
    for (int i = 0; i <= nx + 1; ++i) err = std::max(err, std::fabs(u[i] - std::exp(-M_PI * M_PI * t) * std::sin(M_PI * i * dx)));
    EXPECT_LT(err, 1e-4) << "CN heat error " << err;
    // Unconditional stability: a huge time step must not blow up.
    const auto big = crank_nicolson_heat(u0, 1.0, 1.0, 50, 0.5, 4);
    for (double v : big) EXPECT_TRUE(std::isfinite(v));
    for (double v : big) EXPECT_LT(std::fabs(v), 1.0); // decays, stays bounded
}

TEST(PdeSolvers, LaxWendroffAdvectsAProfile) {
    // u_t + a u_x = 0, periodic; a smooth bump should translate by a*t with little distortion.
    const int    m = 200; const double L = 1.0, dx = L / m, a = 1.0;
    const double dt = 0.8 * dx / a; // Courant 0.8
    std::vector<double> u0(m);
    auto exact = [&](double x) { x = std::fmod(x, L); if (x < 0) x += L; return std::exp(-100.0 * (x - 0.5) * (x - 0.5)); };
    for (int j = 0; j < m; ++j) u0[j] = exact(j * dx);

    const int steps = 100; const double t = steps * dt;
    const auto u = lax_wendroff_advection(u0, a, dx, dt, steps);
    double err = 0.0;
    for (int j = 0; j < m; ++j) err = std::max(err, std::fabs(u[j] - exact(j * dx - a * t)));
    EXPECT_LT(err, 0.05) << "Lax-Wendroff advection error " << err;

    // Mass (integral) is conserved to high accuracy.
    double m0 = 0, m1 = 0; for (int j = 0; j < m; ++j) { m0 += u0[j]; m1 += u[j]; }
    EXPECT_NEAR(m1, m0, 1e-9);
}

TEST(PdeSolvers, TrapezoidalOdeIsSecondOrder) {
    // y' = -y, y(0)=1  ->  y = e^{-t}; trapezoidal is 2nd order (error ~ h^2, quarters per halving).
    auto f = [](double, double y) { return -y; };
    const double exact = std::exp(-1.0);
    double e1 = std::fabs(trapezoidal_ode(f, 1.0, 0.0, 1.0, 50).back().second - exact);
    double e2 = std::fabs(trapezoidal_ode(f, 1.0, 0.0, 1.0, 100).back().second - exact);
    EXPECT_NEAR(e1 / e2, 4.0, 0.3); // second order

    // A-stability: on stiff y'=-100y with a big step it stays bounded and decays.
    auto stiff = [](double, double y) { return -100.0 * y; };
    const auto traj = trapezoidal_ode(stiff, 1.0, 0.0, 1.0, 10); // h=0.1, h*lambda=10
    for (auto& p : traj) EXPECT_TRUE(std::isfinite(p.second));
    EXPECT_LT(std::fabs(traj.back().second), 1.0);

    // Nonlinear check: y'=y (growth) -> e at t=1.
    auto grow = [](double, double y) { return y; };
    EXPECT_NEAR(trapezoidal_ode(grow, 1.0, 0.0, 1.0, 1000).back().second, std::exp(1.0), 1e-4);
}
