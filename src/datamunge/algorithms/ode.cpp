/// @file ode.cpp
/// @brief One-step ODE integrators and the adaptive backward Euler.

#include <datamunge/algorithms/ode.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

/// @brief Small dense vector helpers and the implicit step.
namespace {

/// @brief A dense state vector.
using Vec = std::vector<double>;

/// @brief a + s·b.
Vec axpy(const Vec& a, double s, const Vec& b) {
    Vec r(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) r[i] = a[i] + s * b[i];
    return r;
}

/// @brief Largest absolute entry.
double norm_inf(const Vec& a) {
    double m = 0.0;
    for (double x : a) m = std::max(m, std::fabs(x));
    return m;
}

/// @brief Solves A x = b by Gaussian elimination with partial pivoting.
Vec solve_dense(std::vector<Vec> A, Vec b) {
    const std::size_t n = b.size();
    for (std::size_t c = 0; c < n; ++c) {
        std::size_t piv = c;
        for (std::size_t r = c + 1; r < n; ++r)
            if (std::fabs(A[r][c]) > std::fabs(A[piv][c])) piv = r;
        std::swap(A[c], A[piv]);
        std::swap(b[c], b[piv]);
        const double d = A[c][c];
        if (d == 0.0) continue; // singular; leave as-is (degenerate step)
        for (std::size_t r = 0; r < n; ++r) {
            if (r == c) continue;
            const double factor = A[r][c] / d;
            for (std::size_t k = c; k < n; ++k) A[r][k] -= factor * A[c][k];
            b[r] -= factor * b[c];
        }
    }
    Vec x(n);
    for (std::size_t i = 0; i < n; ++i) x[i] = (A[i][i] != 0.0) ? b[i] / A[i][i] : 0.0;
    return x;
}

/// @brief One backward Euler step from (t, y) of size h: solves Y = y + h f(t + h, Y) by Newton's
///        method with a central-difference Jacobian, starting from the forward Euler predictor.
/// @param iterations incremented once per Newton iteration, when not null.
Vec implicit_euler_step(const OdeSystem& f, double t, const Vec& y, double h, double tol, int max_iter,
                        long* iterations) {
    const std::size_t dim = y.size();
    const double      tn1 = t + h;
    Vec               Y   = axpy(y, h, f(t, y));
    for (int it = 0; it < max_iter; ++it) {
        const Vec fY = f(tn1, Y);
        Vec       G(dim);
        for (std::size_t i = 0; i < dim; ++i) G[i] = Y[i] - y[i] - h * fY[i];
        if (norm_inf(G) < tol) break;
        if (iterations) ++*iterations;
        // Central differences: exact for quadratic right-hand sides (as in chemical kinetics), so the
        // Jacobian stays right even for components far smaller than the step eps.
        std::vector<Vec> J(dim, Vec(dim, 0.0));
        for (std::size_t j = 0; j < dim; ++j) {
            const double eps = 6e-6 * std::max(1.0, std::fabs(Y[j]));
            Vec          Yp = Y, Ym = Y;
            Yp[j] += eps;
            Ym[j] -= eps;
            const Vec fp = f(tn1, Yp), fm = f(tn1, Ym);
            for (std::size_t i = 0; i < dim; ++i) J[i][j] = (i == j ? 1.0 : 0.0) - h * (fp[i] - fm[i]) / (2 * eps);
        }
        Vec neg_G(dim);
        for (std::size_t i = 0; i < dim; ++i) neg_G[i] = -G[i];
        const Vec dY = solve_dense(J, neg_G);
        for (std::size_t i = 0; i < dim; ++i) Y[i] += dY[i];
        if (norm_inf(dY) < tol) break;
    }
    return Y;
}

} // namespace

OdeSolution euler_method(const OdeSystem& f, std::vector<double> y0, double t0, double t1, int steps) {
    OdeSolution   sol;
    const double  h = (t1 - t0) / steps;
    Vec           y = std::move(y0);
    double        t = t0;
    sol.t.push_back(t);
    sol.y.push_back(y);
    for (int n = 0; n < steps; ++n) {
        const Vec k = f(t, y);
        y           = axpy(y, h, k); // y += h * f(t, y)
        t           = t0 + (n + 1) * h;
        sol.t.push_back(t);
        sol.y.push_back(y);
    }
    return sol;
}

OdeSolution runge_kutta4(const OdeSystem& f, std::vector<double> y0, double t0, double t1, int steps) {
    OdeSolution  sol;
    const double h = (t1 - t0) / steps;
    Vec          y = std::move(y0);
    double       t = t0;
    sol.t.push_back(t);
    sol.y.push_back(y);
    for (int n = 0; n < steps; ++n) {
        const Vec k1 = f(t, y);
        const Vec k2 = f(t + 0.5 * h, axpy(y, 0.5 * h, k1));
        const Vec k3 = f(t + 0.5 * h, axpy(y, 0.5 * h, k2));
        const Vec k4 = f(t + h, axpy(y, h, k3));
        for (std::size_t i = 0; i < y.size(); ++i)
            y[i] += (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        t = t0 + (n + 1) * h;
        sol.t.push_back(t);
        sol.y.push_back(y);
    }
    return sol;
}

OdeSolution backward_euler(const OdeSystem& f, std::vector<double> y0, double t0, double t1, int steps,
                           double newton_tol, int newton_max_iter) {
    OdeSolution  sol;
    const double h = (t1 - t0) / steps;
    Vec          y = std::move(y0);
    sol.t.push_back(t0);
    sol.y.push_back(y);
    for (int n = 0; n < steps; ++n) {
        y = implicit_euler_step(f, t0 + n * h, y, h, newton_tol, newton_max_iter, nullptr);
        sol.t.push_back(t0 + (n + 1) * h);
        sol.y.push_back(y);
    }
    return sol;
}

AdaptiveOdeResult backward_euler_adaptive(const OdeSystem& f, std::vector<double> y0, double t0, double t1,
                                          double rtol, double atol, double h0, double h_max) {
    if (!(t1 > t0)) throw std::invalid_argument("backward_euler_adaptive: t1 must exceed t0");
    if (!(rtol > 0) || !(atol > 0) || !(h0 > 0)) throw std::invalid_argument("backward_euler_adaptive: tolerances and h0 must be positive");
    AdaptiveOdeResult out;
    const OdeSystem   counted = [&](double t, const Vec& y) {
        ++out.f_evaluations;
        return f(t, y);
    };
    Vec    y = std::move(y0);
    double t = t0, h = std::min(h0, h_max);
    out.solution.t.push_back(t);
    out.solution.y.push_back(y);
    const double newton_tol = std::min(1e-12, 0.01 * atol);
    while (t < t1) {
        if (t + h > t1) h = t1 - t;
        const Vec full = implicit_euler_step(counted, t, y, h, newton_tol, 50, &out.newton_iterations);
        const Vec half = implicit_euler_step(counted, t, y, h / 2, newton_tol, 50, &out.newton_iterations);
        const Vec two  = implicit_euler_step(counted, t + h / 2, half, h / 2, newton_tol, 50, &out.newton_iterations);
        double err = 0.0;
        for (std::size_t i = 0; i < y.size(); ++i)
            err = std::max(err, std::fabs(two[i] - full[i]) / (atol + rtol * std::fabs(two[i])));
        if (err <= 1.0 || h < 1e-14 * std::max(1.0, std::fabs(t))) {
            t += h;
            y = two;
            out.solution.t.push_back(t);
            out.solution.y.push_back(y);
            out.step_sizes.push_back(h);
            ++out.accepted;
        } else {
            ++out.rejected;
        }
        const double scale = err > 0 ? 0.9 / std::sqrt(err) : 2.0;
        h = std::min(h_max, h * std::clamp(scale, 0.2, 2.0));
    }
    return out;
}

std::complex<double> stability_function(OdeMethod method, std::complex<double> z) {
    switch (method) {
        case OdeMethod::ForwardEuler: return 1.0 + z;
        case OdeMethod::BackwardEuler: return 1.0 / (1.0 - z);
        case OdeMethod::Trapezoidal: return (1.0 + z / 2.0) / (1.0 - z / 2.0);
        case OdeMethod::RungeKutta4: return 1.0 + z * (1.0 + z * (0.5 + z * (1.0 / 6.0 + z / 24.0)));
    }
    return 0.0;
}

VerletSolution velocity_verlet(const std::function<std::vector<double>(const std::vector<double>&)>& accel,
                               std::vector<double> x0, std::vector<double> v0, double t0, double t1,
                               int steps) {
    VerletSolution sol;
    const double   h = (t1 - t0) / steps;
    Vec            x = std::move(x0);
    Vec            v = std::move(v0);
    Vec            a = accel(x);
    sol.t.push_back(t0);
    sol.x.push_back(x);
    sol.v.push_back(v);
    for (int n = 0; n < steps; ++n) {
        for (std::size_t i = 0; i < x.size(); ++i) x[i] += v[i] * h + 0.5 * a[i] * h * h; // drift
        const Vec a_new = accel(x);
        for (std::size_t i = 0; i < v.size(); ++i) v[i] += 0.5 * (a[i] + a_new[i]) * h;   // kick
        a = a_new;
        sol.t.push_back(t0 + (n + 1) * h);
        sol.x.push_back(x);
        sol.v.push_back(v);
    }
    return sol;
}

} // namespace datamunge::algorithms
