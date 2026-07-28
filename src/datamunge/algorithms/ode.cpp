#include <datamunge/algorithms/ode.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

namespace {

using Vec = std::vector<double>;

// y = a + s*b
Vec axpy(const Vec& a, double s, const Vec& b) {
    Vec r(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) r[i] = a[i] + s * b[i];
    return r;
}

double norm_inf(const Vec& a) {
    double m = 0.0;
    for (double x : a) m = std::max(m, std::fabs(x));
    return m;
}

// Solve the dense linear system A x = b in place by Gaussian elimination with partial pivoting.
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
    double       t = t0;
    sol.t.push_back(t);
    sol.y.push_back(y);
    const std::size_t dim = y.size();

    for (int n = 0; n < steps; ++n) {
        const double tn1 = t0 + (n + 1) * h;
        // Solve G(Y) = Y - y_n - h f(tn1, Y) = 0 by Newton with a finite-difference Jacobian.
        Vec Y = axpy(y, h, f(t, y)); // forward-Euler predictor as the initial guess
        for (int it = 0; it < newton_max_iter; ++it) {
            const Vec fY = f(tn1, Y);
            Vec       G(dim);
            for (std::size_t i = 0; i < dim; ++i) G[i] = Y[i] - y[i] - h * fY[i];
            if (norm_inf(G) < newton_tol) break;

            // Jacobian J = I - h * df/dY, via forward differences.
            std::vector<Vec> J(dim, Vec(dim, 0.0));
            for (std::size_t j = 0; j < dim; ++j) {
                const double eps = std::sqrt(2.2e-16) * std::max(1.0, std::fabs(Y[j]));
                Vec          Yp  = Y;
                Yp[j] += eps;
                const Vec fp = f(tn1, Yp);
                for (std::size_t i = 0; i < dim; ++i)
                    J[i][j] = (i == j ? 1.0 : 0.0) - h * (fp[i] - fY[i]) / eps;
            }
            Vec neg_G(dim);
            for (std::size_t i = 0; i < dim; ++i) neg_G[i] = -G[i];
            const Vec dY = solve_dense(J, neg_G);
            for (std::size_t i = 0; i < dim; ++i) Y[i] += dY[i];
            if (norm_inf(dY) < newton_tol) break;
        }
        y = Y;
        t = tn1;
        sol.t.push_back(t);
        sol.y.push_back(y);
    }
    return sol;
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
