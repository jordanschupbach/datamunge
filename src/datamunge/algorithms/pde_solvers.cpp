#include <datamunge/algorithms/pde_solvers.hpp>

#include <cmath>
#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

namespace {

// Thomas algorithm: solve a tridiagonal system with sub-diagonal a, diagonal b, super-diagonal c.
// a[0] and c[n-1] are unused. Returns the solution.
std::vector<double> thomas(std::vector<double> a, std::vector<double> b, std::vector<double> c,
                           std::vector<double> d) {
    const std::size_t n = d.size();
    for (std::size_t i = 1; i < n; ++i) {
        const double m = a[i] / b[i - 1];
        b[i] -= m * c[i - 1];
        d[i] -= m * d[i - 1];
    }
    std::vector<double> x(n);
    x[n - 1] = d[n - 1] / b[n - 1];
    for (std::size_t i = n - 1; i-- > 0;) x[i] = (d[i] - c[i] * x[i + 1]) / b[i];
    return x;
}

} // namespace

std::vector<double> finite_difference_poisson(const std::function<double(double)>& f, double alpha,
                                              double beta, int n) {
    const double h  = 1.0 / (n + 1);
    const double h2 = h * h;
    std::vector<double> a(n, -1.0 / h2), b(n, 2.0 / h2), c(n, -1.0 / h2), d(n);
    for (int i = 0; i < n; ++i) d[i] = f((i + 1) * h);
    d[0] += alpha / h2;     // u_0 = alpha
    d[n - 1] += beta / h2;  // u_{n+1} = beta

    const std::vector<double> u = thomas(a, b, c, d);
    std::vector<double>       out(n + 2);
    out[0] = alpha;
    out[n + 1] = beta;
    for (int i = 0; i < n; ++i) out[i + 1] = u[i];
    return out;
}

std::vector<double> crank_nicolson_heat(const std::function<double(double)>& u0, double L, double D,
                                        int nx, double dt, int nsteps) {
    const double dx = L / (nx + 1);
    const double r  = D * dt / (2.0 * dx * dx);

    std::vector<double> u(nx); // interior values
    for (int i = 0; i < nx; ++i) u[i] = u0((i + 1) * dx);

    const std::vector<double> a(nx, -r), b(nx, 1.0 + 2.0 * r), c(nx, -r);
    for (int step = 0; step < nsteps; ++step) {
        std::vector<double> rhs(nx);
        for (int i = 0; i < nx; ++i) {
            const double left = (i > 0) ? u[i - 1] : 0.0;      // Dirichlet 0 boundary
            const double right = (i < nx - 1) ? u[i + 1] : 0.0;
            rhs[i] = r * left + (1.0 - 2.0 * r) * u[i] + r * right;
        }
        u = thomas(a, b, c, rhs);
    }

    std::vector<double> out(nx + 2, 0.0);
    for (int i = 0; i < nx; ++i) out[i + 1] = u[i];
    return out;
}

std::vector<double> lax_wendroff_advection(const std::vector<double>& u0, double a, double dx, double dt,
                                           int nsteps) {
    const int    m  = static_cast<int>(u0.size());
    const double nu = a * dt / dx; // Courant number
    std::vector<double> u = u0;
    for (int step = 0; step < nsteps; ++step) {
        std::vector<double> un(m);
        for (int j = 0; j < m; ++j) {
            const double up = u[(j + 1) % m];
            const double um = u[(j - 1 + m) % m];
            un[j] = u[j] - 0.5 * nu * (up - um) + 0.5 * nu * nu * (up - 2.0 * u[j] + um);
        }
        u.swap(un);
    }
    return u;
}

std::vector<std::pair<double, double>> trapezoidal_ode(const std::function<double(double, double)>& f,
                                                       double y0, double t0, double t1, int steps) {
    const double                            h = (t1 - t0) / steps;
    std::vector<std::pair<double, double>>  traj;
    double                                  t = t0, y = y0;
    traj.emplace_back(t, y);
    for (int n = 0; n < steps; ++n) {
        const double tn1 = t0 + (n + 1) * h;
        const double fn  = f(t, y);
        // Solve Y = y + (h/2)(fn + f(tn1, Y)) by Newton with a finite-difference derivative.
        double Y = y + h * fn; // forward-Euler predictor
        for (int it = 0; it < 60; ++it) {
            const double g  = Y - y - 0.5 * h * (fn + f(tn1, Y));
            const double eps = 1e-8 * (std::fabs(Y) + 1e-8);
            const double gp = 1.0 - 0.5 * h * (f(tn1, Y + eps) - f(tn1, Y)) / eps;
            const double Yn = Y - g / gp;
            if (std::fabs(Yn - Y) <= 1e-14 * (std::fabs(Yn) + 1e-14)) { Y = Yn; break; }
            Y = Yn;
        }
        y = Y;
        t = tn1;
        traj.emplace_back(t, y);
    }
    return traj;
}

} // namespace datamunge::algorithms
